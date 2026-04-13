from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..data import load_reference_draws
from ..priors import REFERENCE_PARAM_MAP
from ._common import prior_draws, save_figure


class ImmunityPlotUnavailable(RuntimeError):
    """The fit exposes neither Griffin's immunity parameters nor its curves.

    This plot reads ``b``/``phi``/``q`` off Griffin's parameterization, so it
    means nothing for a candidate model that invented its own parameters. Raised
    rather than drawn so a caller can skip it cleanly instead of catching every
    exception the plotting code might throw.
    """


IMMUNITY_PARAMS: tuple[str, ...] = (
    "b0",
    "IB0",
    "kb",
    "phi0",
    "phi1",
    "IC0",
    "kc",
    "d1",
    "ID0",
    "fd0",
    "gd",
    "ad0",
    "kd",
)


def _sample_curves_from_params(params: dict[str, np.ndarray], n_curves: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    for p in IMMUNITY_PARAMS:
        if p not in params:
            raise ValueError(f"Missing required immunity parameter '{p}'")

    # Align ad0 to the notebook's units (days) for fd(a=5*365) scaling.
    ad0 = np.asarray(params["ad0"], dtype=float) * 365.0

    n_avail = min(np.asarray(v).shape[0] for v in params.values())
    if n_avail < 1:
        raise ValueError("No draws available for immunity curves")
    take = min(n_curves, n_avail)
    idx = rng.choice(n_avail, size=take, replace=False)

    exposures = np.arange(100, dtype=float)[None, :]
    b1 = 0.5
    a_days = 5.0 * 365.0

    b0 = np.asarray(params["b0"], dtype=float)[idx][:, None]
    IB0 = np.clip(np.asarray(params["IB0"], dtype=float)[idx][:, None], 1e-12, np.inf)
    kb = np.clip(np.asarray(params["kb"], dtype=float)[idx][:, None], 1e-12, np.inf)

    phi0 = np.asarray(params["phi0"], dtype=float)[idx][:, None]
    phi1 = np.asarray(params["phi1"], dtype=float)[idx][:, None]
    IC0 = np.clip(np.asarray(params["IC0"], dtype=float)[idx][:, None], 1e-12, np.inf)
    kc = np.clip(np.asarray(params["kc"], dtype=float)[idx][:, None], 1e-12, np.inf)

    d1 = np.asarray(params["d1"], dtype=float)[idx][:, None]
    ID0 = np.clip(np.asarray(params["ID0"], dtype=float)[idx][:, None], 1e-12, np.inf)
    fd0 = np.asarray(params["fd0"], dtype=float)[idx][:, None]
    gd = np.clip(np.asarray(params["gd"], dtype=float)[idx][:, None], 1e-12, np.inf)
    ad0_days = np.clip(ad0[idx][:, None], 1e-12, np.inf)
    kd = np.clip(np.asarray(params["kd"], dtype=float)[idx][:, None], 1e-12, np.inf)

    fd = 1.0 - (1.0 - fd0) / (1.0 + (a_days / ad0_days) ** gd)

    prob_b = b0 * ((1.0 - b1) / (1.0 + (exposures / IB0) ** kb) + b1)
    prob_c = phi0 * ((1.0 - phi1) / (1.0 + (exposures / IC0) ** kc) + phi1)
    prob_d = d1 + (1.0 - d1) / (1.0 + fd * (exposures / ID0) ** kd)

    return {
        "prob_b": np.clip(prob_b, 0.0, 1.0),
        "prob_c": np.clip(prob_c, 0.0, 1.0),
        "prob_d": np.clip(prob_d, 0.0, 1.0),
    }


def _posterior_draws_from_npz(samples_npz: Path, fixed_core_values: dict[str, float] | None = None) -> dict[str, np.ndarray]:
    post = np.load(samples_npz)
    out: dict[str, np.ndarray] = {}
    for p in IMMUNITY_PARAMS:
        if p in post.files:
            arr = np.asarray(post[p])
            if arr.ndim == 2:
                out[p] = arr.reshape(-1)
            elif arr.ndim == 1:
                out[p] = arr
    fixed_core_values = fixed_core_values or {}
    n = min((np.asarray(v).shape[0] for v in out.values()), default=0)
    for p in IMMUNITY_PARAMS:
        if p not in out and p in fixed_core_values and n > 0:
            out[p] = np.full(n, float(fixed_core_values[p]), dtype=float)
    return out


def missing_immunity_params(
    samples_npz: Path, fixed_core_values: dict[str, float] | None = None
) -> list[str]:
    """Griffin immunity parameters this fit does not expose.

    Empty means the immunity overlay can be drawn. Callers use this to make the
    plot opt-in rather than discovering the problem from an exception.
    """
    have = _posterior_draws_from_npz(samples_npz=samples_npz, fixed_core_values=fixed_core_values)
    return [p for p in IMMUNITY_PARAMS if p not in have]


def _reference_draws(reference_csv: Path | None) -> dict[str, np.ndarray]:
    df = load_reference_draws(reference_csv)
    if df is None:
        return {}
    out: dict[str, np.ndarray] = {}
    for p in IMMUNITY_PARAMS:
        ref_col = REFERENCE_PARAM_MAP.get(p)
        if ref_col and ref_col in df.columns:
            arr = pd.to_numeric(df[ref_col], errors="coerce").to_numpy(dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size:
                out[p] = arr
    return out


def plot_immunity_functions_overlay(
    samples_npz: Path,
    out_png: Path,
    reference_csv: Path | None = None,
    n_curves: int = 400,
    prior_draw_count: int = 6000,
    fixed_core_values: dict[str, float] | None = None,
    prior_family: str = "xi",
    seed: int = 42,
) -> Path:
    rng = np.random.default_rng(seed)

    prior_params = {p: prior_draws(p, n=prior_draw_count, rng=rng, prior_family=prior_family) for p in IMMUNITY_PARAMS}
    post_params = _posterior_draws_from_npz(samples_npz=samples_npz, fixed_core_values=fixed_core_values)
    ref_params = _reference_draws(reference_csv=reference_csv)

    missing_post = [p for p in IMMUNITY_PARAMS if p not in post_params]
    if missing_post:
        raise ImmunityPlotUnavailable(
            "This fit does not expose the Griffin immunity parameters this plot needs: "
            f"{missing_post}. If parameters were fixed, pass fixed_core_values from "
            "run_config_resolved.json; if the fit used a different curve family, skip this plot."
        )

    prior_curves = _sample_curves_from_params(prior_params, n_curves=n_curves, rng=rng)
    post_curves = _sample_curves_from_params(post_params, n_curves=n_curves, rng=rng)
    ref_curves = None
    if ref_params and all(p in ref_params for p in IMMUNITY_PARAMS):
        ref_curves = _sample_curves_from_params(ref_params, n_curves=n_curves, rng=rng)

    x = np.arange(100)
    labels = ["prob_b", "prob_c", "prob_d"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)

    for i, k in enumerate(labels):
        ax = axs[i]

        y_prior = prior_curves[k]
        y_post = post_curves[k]
        ax.plot(x, y_prior.T, color="#b22222", alpha=0.04, linewidth=0.8)
        ax.plot(x, np.median(y_prior, axis=0), color="#b22222", linewidth=2.0, label="prior median")

        ax.plot(x, y_post.T, color="#1f77b4", alpha=0.04, linewidth=0.8)
        ax.plot(x, np.median(y_post, axis=0), color="#1f77b4", linewidth=2.0, label="numpyro posterior median")

        if ref_curves is not None:
            y_ref = ref_curves[k]
            ax.plot(x, y_ref.T, color="#2ca02c", alpha=0.04, linewidth=0.8)
            ax.plot(x, np.median(y_ref, axis=0), color="#2ca02c", linewidth=2.0, label="reference posterior median")

        ax.set_title(k)
        ax.set_xlabel("Exposures (number)")
        ax.set_ylim(0.0, 1.0)

    axs[0].set_ylabel("Immunity probability")
    handles, labels_ = axs[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Immunity functions overlay (b, c, d)", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    return save_figure(fig, out_png)
