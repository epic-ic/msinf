from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from dmeq import default_parameters, solve

from ..age_grid import GRIFFIN_AGE_GRID, age_grid_array, class_fraction_weights
from ..data import latest_samples_npz, load_reference_draws
from ..dtypes import FloatT, as_np_float, asfloat
from ..priors import REFERENCE_PARAM_MAP
from ._common import (
    plot_eir_ft_panel_grid,
    scalar_posterior_arrays,
    select_draw_indices,
)


@dataclass(frozen=True)
class GridSpec:
    eir_values: tuple[float, ...] = (1, 10, 50, 100, 200, 300, 500)
    ft_values: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    max_age: int = 20


DEFAULT_GRID = GridSpec()

def compute_incidence_age_grid_bootstrap(
    samples_npz: Path,
    out_csv: Path,
    out_png: Path,
    n_eval_draws: int = 200,
    n_bootstrap: int = 500,
    seed: int = 42,
    grid: GridSpec = DEFAULT_GRID,
) -> tuple[Path, Path]:
    posterior_arrays, n_total = scalar_posterior_arrays(samples_npz)
    draw_idx = select_draw_indices(n_total=n_total, n_eval=n_eval_draws, seed=seed)

    # Solve on Griffin's fine grid; aggregate to a per-capita annual incidence
    # rate for each year-of-age bin [a, a+1) so the curve is comparable across
    # grids and consistent with the site overlay / likelihood.
    age = np.arange(0, grid.max_age + 1)
    n_age = len(age)
    age_grid_np = age_grid_array()
    age_grid_j = asfloat(GRIFFIN_AGE_GRID)
    # (n_age, n_grid) fraction-of-class weights for the 1-year plotting bins.
    W = jnp.asarray(
        class_fraction_weights(age.astype(float), (age + 1).astype(float), age_grid_np),
        dtype=FloatT(),
    )

    def _per_year_inc(out: jnp.ndarray) -> jnp.ndarray:
        inc = out[2, :]
        prop = out[6, :] if out.shape[0] > 6 else out[3, :]
        num = W @ inc
        den = jnp.clip(W @ prop, 1e-12)
        return 365.0 * num / den

    panel_pairs = [(eir, ft) for eir in grid.eir_values for ft in grid.ft_values]
    panel_eir = asfloat([p[0] for p in panel_pairs])
    panel_ft = asfloat([p[1] for p in panel_pairs])
    n_panel = len(panel_pairs)

    p0 = default_parameters()
    p0_j = {k: asfloat(v) for k, v in p0.items()}

    # Griffin baseline (default dmeq parameters)
    griffin = np.zeros((n_panel, n_age), dtype=float)
    for j, (eir, ft) in enumerate(panel_pairs):
        p = dict(p0_j)
        p["EIR"] = asfloat(float(eir))
        p["ft"] = asfloat(float(ft))
        out = solve(p, dtype=FloatT(), age_bins_years=age_grid_j)
        griffin[j, :] = as_np_float(_per_year_inc(out))

    dmeq_keys = set(p0.keys())
    usable_params = sorted([k for k in posterior_arrays if k in dmeq_keys])
    draw_param_arrays = [asfloat(posterior_arrays[k][draw_idx]) for k in usable_params]

    def curve_for_one(eir: jnp.ndarray, ft: jnp.ndarray, *param_vals: jnp.ndarray) -> jnp.ndarray:
        p = dict(p0_j)
        for name, val in zip(usable_params, param_vals, strict=False):
            p[name] = val
        if "ad0" in usable_params:
            p["ad0"] = p["ad0"] * 365.0
        p["EIR"] = eir
        p["ft"] = ft
        out = solve(p, dtype=FloatT(), age_bins_years=age_grid_j)
        return _per_year_inc(out)

    def curves_for_draw(*param_vals: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda eir, ft: curve_for_one(eir, ft, *param_vals), in_axes=(0, 0))(panel_eir, panel_ft)

    if draw_param_arrays:
        curves = jax.vmap(curves_for_draw, in_axes=tuple(0 for _ in draw_param_arrays))(*draw_param_arrays)
    else:
        # Fallback (unlikely): no scalar posterior params available
        one = curves_for_draw()
        curves = jnp.repeat(one[None, :, :], repeats=len(draw_idx), axis=0)

    curves_np = as_np_float(curves)  # (draw, panel, age)
    n_draw = curves_np.shape[0]

    # Vectorized bootstrap CI for posterior median curve (pointwise)
    rng = np.random.default_rng(seed + 1)
    boot_idx = rng.integers(0, n_draw, size=(n_bootstrap, n_draw))
    boot_sampled = curves_np[boot_idx, :, :]  # (B, draw, panel, age)
    boot_med = np.median(boot_sampled, axis=1)  # (B, panel, age)

    med = np.median(curves_np, axis=0)
    lo = np.quantile(boot_med, 0.025, axis=0)
    hi = np.quantile(boot_med, 0.975, axis=0)

    rows = []
    for j, (eir, ft) in enumerate(panel_pairs):
        for a_i, a in enumerate(age):
            rows.append(
                {
                    "eir": eir,
                    "ft": ft,
                    "age": int(a),
                    "griffin": griffin[j, a_i],
                    "posterior_median": med[j, a_i],
                    "lower95": lo[j, a_i],
                    "upper95": hi[j, a_i],
                }
            )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    plot_eir_ft_panel_grid(
        age=age,
        eir_values=grid.eir_values,
        ft_values=grid.ft_values,
        panel_lookup={(eir, ft): i for i, (eir, ft) in enumerate(panel_pairs)},
        series=[
            {"y": griffin, "color": "black", "label": "Griffin", "linewidth": 1.0},
            {"y": med, "color": "#1f77b4", "label": "Posterior median", "band": (lo, hi)},
        ],
        suptitle=f"Incidence by age grid with bootstrap CI (B={n_bootstrap}, n_draw={n_draw})",
        out_png=out_png,
    )

    return out_csv, out_png


def compute_incidence_age_grid_prior_predictive(
    samples_npz: Path,
    out_csv: Path,
    out_png: Path,
    n_eval_draws: int = 200,
    n_bootstrap: int = 500,
    seed: int = 42,
    grid: GridSpec = DEFAULT_GRID,
) -> tuple[Path, Path]:
    """Compute prior predictive incidence-age grid (before seeing data).
    
    This uses only the prior distributions for core parameters (from msinf.priors)
    to generate predictive incidence curves. The EIR and ft parameters use their
    prior distributions as well.
    
    Args:
        samples_npz: Path to samples file containing prior draws (from run with core=none)
        out_csv: Output CSV path
        out_png: Output PNG path
        n_eval_draws: Number of prior draws to evaluate
        n_bootstrap: Number of bootstrap iterations for CI
        seed: Random seed
        grid: GridSpec for EIR and ft values
    
    Returns:
        Tuple of (out_csv, out_png)
    """
    prior_arrays, n_total = scalar_posterior_arrays(samples_npz)
    draw_idx = select_draw_indices(n_total=n_total, n_eval=n_eval_draws, seed=seed)

    age = np.arange(0, grid.max_age + 1)
    n_age = len(age)

    panel_pairs = [(eir, ft) for eir in grid.eir_values for ft in grid.ft_values]
    panel_eir = asfloat([p[0] for p in panel_pairs])
    panel_ft = asfloat([p[1] for p in panel_pairs])
    n_panel = len(panel_pairs)

    p0 = default_parameters()
    p0_j = {k: asfloat(v) for k, v in p0.items()}

    # Baseline with default dmeq parameters (all priors at mode/median)
    griffin = np.zeros((n_panel, n_age), dtype=float)
    for j, (eir, ft) in enumerate(panel_pairs):
        p = dict(p0)
        p["EIR"] = float(eir)
        p["ft"] = float(ft)
        out = solve(p, dtype=FloatT())
        griffin[j, :] = as_np_float(out[2, :n_age])

    dmeq_keys = set(p0.keys())

    # Use only parameters that have prior draws in the samples
    usable_params = sorted([k for k in prior_arrays if k in dmeq_keys])
    
    if not usable_params:
        raise ValueError(f"No prior draws found in {samples_npz}. "
                        f"Available keys: {list(prior_arrays.keys())}")
    
    draw_param_arrays = [asfloat(prior_arrays[k][draw_idx]) for k in usable_params]

    def curve_for_one(eir: jnp.ndarray, ft: jnp.ndarray, *param_vals: jnp.ndarray) -> jnp.ndarray:
        p = dict(p0_j)
        for name, val in zip(usable_params, param_vals, strict=False):
            p[name] = val
        if "ad0" in usable_params:
            p["ad0"] = p["ad0"] * 365.0
        p["EIR"] = eir
        p["ft"] = ft
        out = solve(p, dtype=FloatT())
        return out[2, :n_age]

    def curves_for_draw(*param_vals: jnp.ndarray) -> jnp.ndarray:
        return jax.vmap(lambda eir, ft: curve_for_one(eir, ft, *param_vals), in_axes=(0, 0))(panel_eir, panel_ft)

    if draw_param_arrays:
        curves = jax.vmap(curves_for_draw, in_axes=tuple(0 for _ in draw_param_arrays))(*draw_param_arrays)
    else:
        one = curves_for_draw()
        curves = jnp.repeat(one[None, :, :], repeats=len(draw_idx), axis=0)

    curves_np = as_np_float(curves)  # (draw, panel, age)
    n_draw = curves_np.shape[0]

    # Bootstrap CI for prior predictive median
    rng = np.random.default_rng(seed + 1)
    boot_idx = rng.integers(0, n_draw, size=(n_bootstrap, n_draw))
    boot_sampled = curves_np[boot_idx, :, :]
    boot_med = np.median(boot_sampled, axis=1)

    med = np.median(curves_np, axis=0)
    lo = np.quantile(boot_med, 0.025, axis=0)
    hi = np.quantile(boot_med, 0.975, axis=0)

    rows = []
    for j, (eir, ft) in enumerate(panel_pairs):
        for a_i, a in enumerate(age):
            rows.append(
                {
                    "eir": eir,
                    "ft": ft,
                    "age": int(a),
                    "griffin": griffin[j, a_i],
                    "prior_median": med[j, a_i],
                    "prior_lower95": lo[j, a_i],
                    "prior_upper95": hi[j, a_i],
                }
            )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    plot_eir_ft_panel_grid(
        age=age,
        eir_values=grid.eir_values,
        ft_values=grid.ft_values,
        panel_lookup={(eir, ft): i for i, (eir, ft) in enumerate(panel_pairs)},
        series=[
            {"y": griffin, "color": "black", "label": "Griffin", "linewidth": 1.0},
            {"y": med, "color": "#ff7f0e", "label": "Prior median", "band": (lo, hi)},
        ],
        suptitle=f"Prior predictive incidence by age grid (B={n_bootstrap}, n_draw={n_draw})",
        out_png=out_png,
    )

    return out_csv, out_png


def run_from_artifacts_root(
    artifacts_root: Path,
    outdir: Path,
    n_eval_draws: int = 200,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    samples_npz = latest_samples_npz(artifacts_root)
    out_csv = outdir / "incidence_age_grid_bootstrap_latest.csv"
    out_png = outdir / "incidence_age_grid_bootstrap_latest.png"
    compute_incidence_age_grid_bootstrap(
        samples_npz=samples_npz,
        out_csv=out_csv,
        out_png=out_png,
        n_eval_draws=n_eval_draws,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return samples_npz, out_csv, out_png


def _select_reference_param_arrays(reference_csv: Path, p0_keys: set[str]) -> dict[str, np.ndarray]:
    df = load_reference_draws(reference_csv)
    if df is None:
        raise FileNotFoundError(f"reference draws CSV not found: {reference_csv}")
    arrays: dict[str, np.ndarray] = {}
    inv_map = {v: k for k, v in REFERENCE_PARAM_MAP.items()}
    for col in df.columns:
        key = inv_map.get(col, col)
        if key in p0_keys:
            vals = df[col].to_numpy(dtype=float)
            if np.isfinite(vals).any():
                arrays[key] = vals
    if not arrays:
        raise ValueError(f"No usable reference parameter columns found in {reference_csv}")
    return arrays


def _representative_median_draw_index(draw_matrix: np.ndarray) -> int:
    med = np.median(draw_matrix, axis=0)
    sd = np.std(draw_matrix, axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    z = (draw_matrix - med[None, :]) / sd[None, :]
    d2 = np.sum(z * z, axis=1)
    return int(np.argmin(d2))


def compute_incidence_age_grid_numpyro_vs_reference(
    samples_npz: Path,
    reference_csv: Path,
    out_png_median_draw: Path,
    out_png_bootstrap: Path,
    n_eval_draws: int = 200,
    n_bootstrap: int = 500,
    seed: int = 42,
    grid: GridSpec = DEFAULT_GRID,
) -> tuple[Path, Path]:
    posterior_arrays, n_total_post = scalar_posterior_arrays(samples_npz)

    p0 = default_parameters()
    p0_j = {k: asfloat(v) for k, v in p0.items()}
    dmeq_keys = set(p0.keys())

    reference_arrays = _select_reference_param_arrays(reference_csv=reference_csv, p0_keys=dmeq_keys)
    n_total_ref = len(next(iter(reference_arrays.values())))

    usable_numpyro = sorted([k for k in posterior_arrays if k in dmeq_keys])
    usable_reference = sorted([k for k in reference_arrays if k in dmeq_keys])
    common = sorted(set(usable_numpyro).intersection(usable_reference))
    if not common:
        raise ValueError("No common dmeq parameters between NumPyro posterior and reference draws.")

    idx_post = select_draw_indices(n_total=n_total_post, n_eval=n_eval_draws, seed=seed)
    idx_ref = select_draw_indices(n_total=n_total_ref, n_eval=n_eval_draws, seed=seed + 11)

    age = np.arange(0, grid.max_age + 1)
    len(age)
    age_grid_j = asfloat(GRIFFIN_AGE_GRID)
    W = jnp.asarray(
        class_fraction_weights(age.astype(float), (age + 1).astype(float), age_grid_array()),
        dtype=FloatT(),
    )

    def _per_year_inc(out: jnp.ndarray) -> jnp.ndarray:
        inc = out[2, :]
        prop = out[6, :] if out.shape[0] > 6 else out[3, :]
        return 365.0 * (W @ inc) / jnp.clip(W @ prop, 1e-12)

    panel_pairs = [(eir, ft) for eir in grid.eir_values for ft in grid.ft_values]
    panel_eir = asfloat([p[0] for p in panel_pairs])
    panel_ft = asfloat([p[1] for p in panel_pairs])
    len(panel_pairs)

    def curve_for_one(eir: jnp.ndarray, ft: jnp.ndarray, param_vals: dict[str, jnp.ndarray], scale_ad0: bool) -> jnp.ndarray:
        p = dict(p0_j)
        for name, val in param_vals.items():
            p[name] = val
        if scale_ad0 and "ad0" in param_vals:
            p["ad0"] = p["ad0"] * 365.0
        p["EIR"] = eir
        p["ft"] = ft
        out = solve(p, dtype=FloatT(), age_bins_years=age_grid_j)
        return _per_year_inc(out)

    def curves_for_draw(draw_vec: jnp.ndarray, names: list[str], scale_ad0: bool) -> jnp.ndarray:
        vals = {n: draw_vec[i] for i, n in enumerate(names)}
        return jax.vmap(lambda eir, ft: curve_for_one(eir, ft, vals, scale_ad0), in_axes=(0, 0))(panel_eir, panel_ft)

    post_draw_matrix = np.column_stack([posterior_arrays[k][idx_post] for k in common])
    ref_draw_matrix = np.column_stack([reference_arrays[k][idx_ref] for k in common])

    post_curves = jax.vmap(lambda dv: curves_for_draw(dv, common, True))(asfloat(post_draw_matrix))
    ref_curves = jax.vmap(lambda dv: curves_for_draw(dv, common, False))(asfloat(ref_draw_matrix))
    post_curves = as_np_float(post_curves)
    ref_curves = as_np_float(ref_curves)

    panel_lookup = {(eir, ft): i for i, (eir, ft) in enumerate(panel_pairs)}

    # Version A: representative median draw curves
    i_post = _representative_median_draw_index(post_draw_matrix)
    i_ref = _representative_median_draw_index(ref_draw_matrix)

    plot_eir_ft_panel_grid(
        age=age,
        eir_values=grid.eir_values,
        ft_values=grid.ft_values,
        panel_lookup=panel_lookup,
        series=[
            {"y": ref_curves[i_ref], "color": "#2ca02c", "label": "Griffin median draw"},
            {"y": post_curves[i_post], "color": "#1f77b4", "label": "NumPyro median draw"},
        ],
        suptitle="Incidence-age grid: NumPyro vs Griffin representative median draws",
        out_png=out_png_median_draw,
    )

    # Version B: median curves + bootstrap CIs for both
    rng = np.random.default_rng(seed + 1)
    boot_idx_post = rng.integers(0, post_curves.shape[0], size=(n_bootstrap, post_curves.shape[0]))
    boot_idx_ref = rng.integers(0, ref_curves.shape[0], size=(n_bootstrap, ref_curves.shape[0]))

    post_med = np.median(post_curves, axis=0)
    ref_med = np.median(ref_curves, axis=0)
    post_boot_med = np.median(post_curves[boot_idx_post], axis=1)
    ref_boot_med = np.median(ref_curves[boot_idx_ref], axis=1)
    post_lo = np.quantile(post_boot_med, 0.025, axis=0)
    post_hi = np.quantile(post_boot_med, 0.975, axis=0)
    ref_lo = np.quantile(ref_boot_med, 0.025, axis=0)
    ref_hi = np.quantile(ref_boot_med, 0.975, axis=0)

    plot_eir_ft_panel_grid(
        age=age,
        eir_values=grid.eir_values,
        ft_values=grid.ft_values,
        panel_lookup=panel_lookup,
        series=[
            {"y": ref_med, "color": "#2ca02c", "label": "Griffin median", "band": (ref_lo, ref_hi)},
            {"y": post_med, "color": "#1f77b4", "label": "NumPyro median", "band": (post_lo, post_hi)},
        ],
        suptitle=f"Incidence-age grid: NumPyro vs Griffin medians with bootstrap CI (B={n_bootstrap})",
        out_png=out_png_bootstrap,
    )

    return out_png_median_draw, out_png_bootstrap
