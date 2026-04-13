"""Shared plotting helpers: sample loading, dmeq curve solves, generic figures.

Every plotting submodule draws its figure/axis setup, draw selection, sample
reshaping and curve solving from here rather than re-implementing them.
"""

from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dmeq import default_parameters, solve
from matplotlib.lines import Line2D

from ..age_grid import GRIFFIN_AGE_GRID, age_grid_array
from ..dtypes import FloatT, as_np_float, asfloat
from ..priors import DMEQ_OVERRIDE_PARAMS, build_griffin_priors, build_xi_priors

# dmeq solve(..) rows used here:
#   0 -> microscopy prevalence, 1 -> PCR prevalence, 2 -> incidence (pop-fraction/day),
#   6 -> population age proportion (binning weights).
PROP_ROW = 6

# Model age grid shared with the fit (fine, non-uniform; see msinf.age_grid).
AGE_GRID = age_grid_array()


# ---------------------------------------------------------------------------
# Sample-array helpers
# ---------------------------------------------------------------------------

def flat2(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"Expected (chain, draw), got {x.shape}")
    return x.reshape(-1)


def flat3(x: np.ndarray) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected (chain, draw, n), got {x.shape}")
    return x.reshape(-1, x.shape[-1])


def select_draw_indices(n_total: int, n_eval: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(n_eval, n_total)
    return rng.choice(n_total, size=n, replace=False)


def scalar_posterior_arrays(samples_npz: Path) -> tuple[dict[str, np.ndarray], int]:
    """Flatten every ``(chain, draw)`` array in an npz to ``(chain*draw,)``."""
    post = np.load(samples_npz)
    arrays: dict[str, np.ndarray] = {}
    for k in post.files:
        arr = post[k]
        if arr.ndim == 2:  # (chain, draw)
            arrays[k] = arr.reshape(-1)
    n_total = len(next(iter(arrays.values()))) if arrays else 0
    if n_total == 0:
        raise ValueError(f"No scalar posterior arrays found in {samples_npz}")
    return arrays, n_total


def prior_draws(param: str, n: int, rng: np.random.Generator, prior_family: str = "xi") -> np.ndarray:
    """Draw from a core parameter's prior on its native scale."""
    builders = {"xi": build_xi_priors, "griffin": build_griffin_priors}
    family = prior_family.strip().lower()
    if family not in builders:
        raise ValueError("prior_family must be one of {'xi', 'griffin'}")
    sp = builders[family]()[param]
    if sp.family == "beta":
        return rng.beta(sp.param1, sp.param2, size=n)
    return rng.lognormal(sp.param1, sp.param2, size=n)


# ---------------------------------------------------------------------------
# dmeq curve solve
# ---------------------------------------------------------------------------

def solve_curves(
    log_eir: np.ndarray,       # (D, n_site)
    ft_study: np.ndarray,      # (D, n_study)
    site_to_study: np.ndarray,
    curve_row: int,
    core_params: dict[str, np.ndarray] | None = None,  # name -> (D,) per-draw values
) -> tuple[np.ndarray, np.ndarray]:
    """Solve dmeq for each (draw, site). Returns (target_curve, prop_curve).

    ``core_params`` supplies per-draw core/immunity parameter values (e.g. the
    posterior draws). When omitted, dmeq ``default_parameters`` are used. Passing
    the posterior draws makes the overlay reproduce exactly the curves the
    likelihood evaluated, instead of a default-parameter approximation.
    """
    p0 = default_parameters()
    p0_j = {k: asfloat(v) for k, v in p0.items()}
    age_grid = asfloat(GRIFFIN_AGE_GRID)
    site_to_study_j = jnp.asarray(site_to_study, dtype=jnp.int32)

    cp = {
        k: asfloat(v)
        for k, v in (core_params or {}).items()
        if k in DMEQ_OVERRIDE_PARAMS
    }

    def curve_for_site(p_draw, leir, ft):
        p = dict(p_draw)
        p["EIR"] = jnp.exp(leir)
        p["ft"] = ft
        out = solve(p, dtype=FloatT(), age_bins_years=age_grid)
        return out[curve_row, :], out[PROP_ROW, :]

    def curves_for_draw(cp_draw, leir_draw, ft_draw):
        p_draw = dict(p0_j)
        for k, v in cp_draw.items():
            p_draw[k] = v
        # ad0 is sampled on a years scale; the ODE uses days (matches the model).
        if "ad0" in cp_draw:
            p_draw["ad0"] = p_draw["ad0"] * 365.0
        ft_site = ft_draw[site_to_study_j]
        return jax.vmap(curve_for_site, in_axes=(None, 0, 0))(p_draw, leir_draw, ft_site)

    leir_sel = asfloat(log_eir)
    ft_sel = asfloat(ft_study)
    target, prop = jax.vmap(curves_for_draw, in_axes=(0, 0, 0))(cp, leir_sel, ft_sel)
    return as_np_float(target), as_np_float(prop)


# ---------------------------------------------------------------------------
# Generic figure helpers
# ---------------------------------------------------------------------------

def facet_grid(n_panels: int, ncols: int, panel_w: float, panel_h: float):
    """Create a flattened facet grid sized for ``n_panels``."""
    nrows = int(np.ceil(max(n_panels, 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, panel_h * nrows), sharex=False, sharey=False)
    return fig, np.asarray(axes).reshape(-1)


def blank_unused(axes: np.ndarray, n_used: int) -> None:
    for i in range(n_used, len(axes)):
        axes[i].axis("off")


def save_figure(fig, out_png: Path, dpi: int = 180) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png


def placeholder_png(out_png: Path, title: str, figsize: tuple[float, float] = (10.0, 2.6)) -> Path:
    """A figure carrying only a message, for outputs a run cannot produce."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=12)
    fig.tight_layout()
    return save_figure(fig, out_png)


# ---------------------------------------------------------------------------
# Overlay spec + generic overlay/calibration plotters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OverlaySpec:
    band_color: str
    pred_label: str
    figure_title: str
    calibration_title: str
    y_label: str
    y_min: float | None
    y_max: float | None
    cal_axis_max: float | None  # None -> auto


def plot_faceted_overlay(
    *,
    summary: pd.DataFrame,
    site_label_by_i: dict[int, object],
    spec: OverlaySpec,
    out_png: Path,
    group_col: str | None = None,
    group_colors: dict | None = None,
    obs_color_default: str = "#444444",
) -> Path:
    sites_with_obs = sorted(pd.unique(summary["site_i"]).tolist())
    n_site_plot = len(sites_with_obs)
    fig, axes = facet_grid(n_site_plot, ncols=6, panel_w=3.0, panel_h=2.2)

    for j, s in enumerate(sites_with_obs):
        ax = axes[j]
        ss = summary[summary["site_i"] == s]
        if not ss.empty:
            x_min = float(ss["age_start"].min())
            x_max = float(ss["age_end"].max())
            iter_groups = ss.groupby(group_col) if group_col else [(None, ss)]
            for grp_key, g in iter_groups:
                x_obs = as_np_float(g["age_mid"].to_numpy()) - 0.08
                x_pred = as_np_float(g["age_mid"].to_numpy()) + 0.08
                y_obs = as_np_float(g["obs_value"].to_numpy())
                y_pred = as_np_float(g["pred_obs_median"].to_numpy())
                y_lo = as_np_float(g["pred_obs_lower95"].to_numpy())
                y_hi = as_np_float(g["pred_obs_upper95"].to_numpy())

                obs_color = (group_colors or {}).get(int(grp_key), obs_color_default) if grp_key is not None else obs_color_default

                ax.scatter(x_obs, y_obs, s=20, marker="o", edgecolor="white", linewidth=0.3, color=obs_color, alpha=0.9, zorder=3)
                ax.vlines(x_pred, y_lo, y_hi, color=spec.band_color, alpha=0.85, linewidth=1.0, zorder=2)
                ax.scatter(x_pred, y_pred, s=24, marker="s", edgecolor="white", linewidth=0.3, color=spec.band_color, alpha=0.95, zorder=4)
                for xo, yo, xp, yp in zip(x_obs, y_obs, x_pred, y_pred, strict=False):
                    ax.plot([xo, xp], [yo, yp], color="#999999", alpha=0.35, linewidth=0.5, zorder=1)

            ax.set_xlim(x_min - 0.25, x_max + 0.25)
            if spec.y_min is not None or spec.y_max is not None:
                lo, hi = ax.get_ylim()
                ax.set_ylim(spec.y_min if spec.y_min is not None else lo,
                            spec.y_max if spec.y_max is not None else hi)
        ax.set_title(f"site {site_label_by_i[s]}", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    blank_unused(axes, n_site_plot)

    fig.text(0.5, 0.01, "age (years)", ha="center", fontsize=10)
    fig.text(0.01, 0.5, spec.y_label, va="center", rotation="vertical", fontsize=10)
    fig.suptitle(spec.figure_title, y=0.995, fontsize=12)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=obs_color_default, markersize=5, label="obs (numer/denom)"),
        Line2D([0], [0], marker="s", color=spec.band_color, markersize=5, label=f"{spec.pred_label} median"),
        Line2D([0], [0], color=spec.band_color, lw=1.0, label=f"{spec.pred_label} 95% CI"),
    ]
    fig.legend(handles=handles, loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.02, 0.03, 1.0, 0.97))
    return save_figure(fig, out_png)


def plot_calibration(
    *,
    summary: pd.DataFrame,
    spec: OverlaySpec,
    out_png: Path,
    x_label: str,
    y_label: str,
    group_col: str | None = None,
    group_colors: dict | None = None,
    group_labels: dict | None = None,
    default_color: str = "#444444",
    default_label: str = "points",
) -> Path:
    if spec.cal_axis_max is not None:
        xmax = spec.cal_axis_max
    else:
        xmax = max(float(summary["obs_value"].max()), float(summary["pred_obs_median"].max()), 1e-6)

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    iter_groups = summary.groupby(group_col) if group_col else [(None, summary)]
    for grp_key, g in iter_groups:
        color = (group_colors or {}).get(int(grp_key), default_color) if grp_key is not None else default_color
        label = (group_labels or {}).get(int(grp_key), f"{group_col}={grp_key}") if grp_key is not None else default_label
        ax.errorbar(
            g["obs_value"],
            g["pred_obs_median"],
            yerr=[g["pred_obs_median"] - g["pred_obs_lower95"], g["pred_obs_upper95"] - g["pred_obs_median"]],
            fmt="o", ms=3.5, alpha=0.7, color=color, label=label,
        )

    ax.plot([0, xmax], [0, xmax], linestyle="--", color="black", linewidth=1.0, label="y = x")
    ax.set_xlim(0, xmax * 1.05)
    ax.set_ylim(0, xmax * 1.05)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(spec.calibration_title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return save_figure(fig, out_png)


def plot_eir_ft_panel_grid(
    *,
    age: np.ndarray,
    eir_values: tuple[float, ...],
    ft_values: tuple[float, ...],
    panel_lookup: dict[tuple[float, float], int],
    series: list[dict],
    suptitle: str,
    out_png: Path,
) -> Path:
    """EIR x ft panel grid used by the incidence-age-grid figures.

    Each entry of ``series`` is ``{"y": (n_panel, n_age), "color", "label"}`` and
    may carry ``"band": (lo, hi)`` for a filled interval.
    """
    nrows = len(ft_values)
    ncols = len(eir_values)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.9 * ncols, 2.1 * nrows), sharex=False, sharey=False)
    axes = np.asarray(axes).reshape(nrows, ncols)

    for r, ft in enumerate(ft_values):
        for c, eir in enumerate(eir_values):
            ax = axes[r, c]
            j = panel_lookup[(eir, ft)]
            for s in series:
                band = s.get("band")
                if band is not None:
                    lo, hi = band
                    ax.fill_between(age, lo[j], hi[j], color=s["color"], alpha=0.22, linewidth=0)
                ax.plot(age, s["y"][j], color=s["color"], linewidth=s.get("linewidth", 1.2), label=s["label"])
            ax.set_title(f"EIR={eir}, ft={ft}", fontsize=9)
            ax.set_xlabel("age", fontsize=8)
            ax.set_ylabel("inc", fontsize=8)
            ax.tick_params(axis="both", labelsize=7)

    handles, labels = axes[-1, -1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower right", frameon=False)
    fig.suptitle(suptitle, fontsize=12, y=0.995)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.97))
    return save_figure(fig, out_png)
