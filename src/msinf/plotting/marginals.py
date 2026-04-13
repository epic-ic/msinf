from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from ..data import (
    aggregate_eir_priors,
    encode_canonical_sites,
    load_observed_data,
    load_reference_draws,
)
from ..priors import CORE_PARAMS, REFERENCE_PARAM_MAP, build_xi_priors
from ._common import blank_unused, facet_grid, prior_draws, save_figure


def _kde_xy(arr: np.ndarray, n: int = 220) -> tuple[np.ndarray, np.ndarray]:
    arr = arr[np.isfinite(arr)]
    if arr.size < 5:
        return np.array([]), np.array([])
    lo = np.quantile(arr, 0.01)
    hi = np.quantile(arr, 0.99)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.array([]), np.array([])
    grid = np.linspace(lo, hi, n)
    kde = gaussian_kde(arr)
    y = kde(grid)
    area = np.trapezoid(y, grid)
    if np.isfinite(area) and area > 0:
        y = y / area
    return grid, y


def _draw_curve(ax, arr: np.ndarray, label: str, color: str, alpha: float = 1.0, linestyle: str = "-") -> None:
    x, y = _kde_xy(arr)
    if x.size == 0:
        return
    ax.plot(x, y, label=label, color=color, alpha=alpha, linestyle=linestyle, linewidth=1.6)


def _to_plot_units(param: str, arr: np.ndarray, series: str) -> np.ndarray:
    # Align ad0 units to days for overlays against PF reference (`ad` is in days).
    # NumPyro/prior ad0 draws are in years-scale latent units.
    if param == "ad0" and series in {"prior", "numpyro_posterior"}:
        return arr * 365.0
    return arr



def plot_core_marginals_overlay(
    samples_npz: Path,
    out_png: Path,
    reference_csv: Path | None = None,
    include_prior: bool = True,
    include_numpyro_posterior: bool = True,
    include_reference_pf: bool = True,
    selected_params: list[str] | tuple[str, ...] | None = None,
    prior_family: str = "xi",
    seed: int = 42,
) -> Path:
    post = np.load(samples_npz)
    ref_df = load_reference_draws(reference_csv)

    ref_map = REFERENCE_PARAM_MAP

    rng = np.random.default_rng(seed)

    if selected_params is None:
        params_to_plot = list(CORE_PARAMS)
    else:
        allowed = set(CORE_PARAMS)
        params_to_plot = list(dict.fromkeys(selected_params))
        unknown = [p for p in params_to_plot if p not in allowed]
        if unknown:
            raise ValueError(f"Unknown selected_params: {unknown}. Allowed: {list(CORE_PARAMS)}")
    if len(params_to_plot) == 0:
        raise ValueError("No parameters selected for plotting.")

    fig, axes = facet_grid(len(params_to_plot), ncols=4, panel_w=4.0, panel_h=3.0)

    for i, p in enumerate(params_to_plot):
        ax = axes[i]

        series_data: list[tuple[str, np.ndarray, str]] = []

        if include_prior and p in build_xi_priors():
            # Draw a large pool, then downsample to a matched count across series.
            prior = _to_plot_units(p, prior_draws(p, n=6000, rng=rng, prior_family=prior_family), "prior")
            series_data.append(("prior", prior, "#b22222"))

        if include_numpyro_posterior and p in post.files:
            np_post = _to_plot_units(p, post[p].reshape(-1), "numpyro_posterior")
            series_data.append(("numpyro_posterior", np_post, "#1f77b4"))

        if include_reference_pf and ref_df is not None and p in ref_map and ref_map[p] in ref_df.columns:
            ref = _to_plot_units(p, ref_df[ref_map[p]].to_numpy(dtype=float), "reference_pf")
            series_data.append(("reference_pf", ref, "#2ca02c"))

        for label, arr, color in series_data:
            _draw_curve(ax, arr, label=label, color=color)

        ax.set_title(p, fontsize=10)
        ax.set_xlabel("value", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)

    blank_unused(axes, len(params_to_plot))

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Core marginals overlay", y=0.995, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    return save_figure(fig, out_png)


def plot_logeir_marginals_overlay(
    samples_npz: Path,
    eir_priors_csv: Path,
    data_csv: Path,
    out_panel_png: Path,
    out_pooled_png: Path,
    seed: int = 42,
) -> tuple[Path, Path]:
    post = np.load(samples_npz)
    if "log_eir" not in post.files:
        raise ValueError(f"samples file has no 'log_eir' variable: {samples_npz}")

    log_eir = post["log_eir"]
    if log_eir.ndim != 3:
        raise ValueError(f"Expected log_eir with shape (chain, draw, site), got {log_eir.shape}")
    post_draws = log_eir.reshape(-1, log_eir.shape[-1])
    n_site = post_draws.shape[1]

    df = load_observed_data(Path(data_csv))
    df, _ = encode_canonical_sites(df)
    mu, sigma = aggregate_eir_priors(df=df, eir_priors_csv=Path(eir_priors_csv), n_site=n_site)

    rng = np.random.default_rng(seed)
    n_draw = post_draws.shape[0]
    prior_draws = np.vstack([rng.normal(mu[i], sigma[i], size=n_draw) for i in range(n_site)]).T

    # Per-site panel
    fig, axes = facet_grid(n_site, ncols=6, panel_w=3.0, panel_h=2.1)

    for i in range(n_site):
        ax = axes[i]
        ax.hist(prior_draws[:, i], bins=35, density=True, histtype="step", linewidth=1.1, color="#b22222", label="prior")
        ax.hist(post_draws[:, i], bins=35, density=True, histtype="step", linewidth=1.1, color="#1f77b4", label="posterior")
        ax.set_title(f"log_eir[{i}]", fontsize=8)
        ax.tick_params(axis="both", labelsize=7)

    blank_unused(axes, n_site)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle("log_eir marginals overlay: prior vs posterior", y=0.995, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))

    save_figure(fig, out_panel_png)

    # Pooled quick-check
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(prior_draws.ravel(), bins=100, density=True, histtype="step", linewidth=1.8, color="#b22222", label="prior (pooled)")
    ax.hist(post_draws.ravel(), bins=100, density=True, histtype="step", linewidth=1.8, color="#1f77b4", label="posterior (pooled)")
    ax.set_title("Pooled log_eir marginal: prior vs posterior")
    ax.set_xlabel("log_eir")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    fig.tight_layout()

    save_figure(fig, out_pooled_png)

    return out_panel_png, out_pooled_png
