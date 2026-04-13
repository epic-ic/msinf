from pathlib import Path

import numpy as np
import pandas as pd

from ..age_grid import class_fraction_weights
from ..data import (
    aggregate_eir_priors,
    encode_canonical_sites,
    load_ft_prior_table,
    load_observed_data,
    site_label_map,
    site_to_study_from_df,
)
from ..dtypes import as_np_float
from ..priors import DMEQ_OVERRIDE_PARAMS
from ._common import (
    AGE_GRID,
    OverlaySpec,
    flat2,
    flat3,
    placeholder_png,
    plot_calibration,
    plot_faceted_overlay,
    select_draw_indices,
    solve_curves,
)

# Microscopy prevalence is the only prevalence stream: the study table carries no
# PCR (`type == 3`) rows, so the model does not fit it and there is nothing to
# overlay. TYPE_CODE selects the observed rows, CURVE_ROW the dmeq output row.
PREV_TYPE_CODE = 2
PREV_CURVE_ROW = 0
PREV_LABEL = "microscopy"


def _compute_summary(
    *,
    prev_curves: np.ndarray, prop_curves: np.ndarray,
    w_study_sel: np.ndarray,
    df: pd.DataFrame, canon_categories: list | None,
    type_code: int,
) -> pd.DataFrame:
    study_levels = sorted(pd.unique(df["study_index"]))
    study_map = {s: i for i, s in enumerate(study_levels)}
    site_label_by_i = site_label_map(df, canon_categories)

    prev_df = df[df["type"] == type_code].copy()
    if prev_df.empty:
        return pd.DataFrame()

    prev_df["study_i"] = prev_df["study_index"].map(study_map).astype(int)
    prev_df["obs_value"] = prev_df["numer"] / np.clip(prev_df["denom"], 1e-12, None)
    prev_df["age_mid"] = 0.5 * (prev_df["age0"] + prev_df["age1"])

    D = prev_curves.shape[0]
    eps = 1e-6
    rows = []
    for _, row in prev_df.iterrows():
        s = int(row["site_i"]); st = int(row["study_i"])
        a0 = float(row["age0"]); a1 = float(row["age1"])
        overlap = class_fraction_weights(np.array([a0]), np.array([a1]), AGE_GRID)[0]

        p_vals, q_vals = [], []
        for d in range(D):
            # dmeq row 0/1 is the joint proportion (positive AND in compartment),
            # already population-weighted. Band prevalence is positives-in-band /
            # population-in-band; do NOT multiply prev_curves by prop_curves again.
            den = max((overlap * prop_curves[d, s, :]).sum(), 1e-12)
            p_bin = float((overlap * prev_curves[d, s, :]).sum() / den)
            p_clip = min(max(p_bin, eps), 1.0 - eps)
            # Apply study-level logit offset (same as the numpyro model).
            q = 1.0 / (1.0 + np.exp(-(np.log(p_clip / (1.0 - p_clip)) + float(w_study_sel[d, st]))))
            p_vals.append(p_bin); q_vals.append(q)

        p_vals = np.asarray(p_vals); q_vals = np.asarray(q_vals)
        rows.append({
            "site_i": s, "site_index": site_label_by_i[s],
            "study_i": st, "study_index": row["study_index"],
            "age_start": a0, "age_end": a1, "age_mid": float(row["age_mid"]),
            "numer": float(row["numer"]), "denom": float(row["denom"]),
            "obs_value": float(row["obs_value"]),
            "pred_mu_median": float(np.median(p_vals)),
            "pred_mu_lower95": float(np.quantile(p_vals, 0.025)),
            "pred_mu_upper95": float(np.quantile(p_vals, 0.975)),
            "pred_obs_median": float(np.median(q_vals)),
            "pred_obs_lower95": float(np.quantile(q_vals, 0.025)),
            "pred_obs_upper95": float(np.quantile(q_vals, 0.975)),
        })
    return pd.DataFrame(rows)


def _emit(
    summary: pd.DataFrame, site_label_by_i: dict[int, object], pretty: str,
    band_color: str, pred_label: str, figure_title: str, calibration_title: str,
    out_png: Path, out_csv_summary: Path, out_png_calibration: Path,
) -> tuple[Path, Path, Path]:
    out_csv_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv_summary, index=False)

    if summary.empty:
        for path in (out_png, out_png_calibration):
            placeholder_png(path, f"No {pretty} prevalence rows in data", figsize=(8.0, 2.6))
        return out_png, out_csv_summary, out_png_calibration

    spec = OverlaySpec(
        band_color=band_color, pred_label=pred_label,
        figure_title=figure_title, calibration_title=calibration_title,
        y_label=f"{pretty} prevalence (proportion positive)",
        y_min=0.0, y_max=1.0, cal_axis_max=1.0,
    )
    plot_faceted_overlay(summary=summary, site_label_by_i=site_label_by_i, spec=spec, out_png=out_png)
    plot_calibration(
        summary=summary, spec=spec, out_png=out_png_calibration,
        x_label=f"Observed {pretty} prevalence (numer/denom)",
        y_label=f"Predicted {pretty} prevalence ({pred_label} median)",
        default_label=f"{pretty} prevalence points",
    )
    return out_png, out_csv_summary, out_png_calibration


def plot_site_prevalence_overlay(
    samples_npz: Path, data_csv: Path,
    out_png: Path, out_csv_summary: Path, out_png_calibration: Path,
    n_eval_draws: int = 80, seed: int = 42,
) -> tuple[Path, Path, Path]:
    pretty = PREV_LABEL

    post = np.load(samples_npz)
    required = {"log_eir", "ft_study"}
    missing = sorted(required - set(post.files))
    if missing:
        raise ValueError(f"samples missing required variables: {missing}")

    log_eir = flat3(post["log_eir"])
    ft_study = flat3(post["ft_study"])
    n_total = log_eir.shape[0]
    # The prevalence study effect w is integrated out by GH quadrature in the model
    # (logit Normal, mean 0), so it is absent from the posterior samples. Its median
    # offset is 0, so the obs-scale point prediction uses w = 0.
    if "w_study" in post.files:
        w_study = flat3(post["w_study"])
    else:
        w_study = np.zeros((n_total, log_eir.shape[1]), dtype=float)

    df = load_observed_data(data_csv)
    df, canon_categories = encode_canonical_sites(df)
    site_to_study, _sm, _sl = site_to_study_from_df(df)

    draw_idx = select_draw_indices(n_total=n_total, n_eval=n_eval_draws, seed=seed)
    log_eir_sel = log_eir[draw_idx, :]
    ft_study_sel = ft_study[draw_idx, :]
    w_study_sel = as_np_float(w_study[draw_idx, :])

    # Posterior draws of the core/immunity parameters, so the overlay reproduces the
    # curves the likelihood evaluated rather than dmeq defaults.
    core_params = {
        name: flat2(post[name])[draw_idx] for name in DMEQ_OVERRIDE_PARAMS if name in post.files
    }

    prev_curves, prop_curves = solve_curves(
        log_eir=log_eir_sel, ft_study=ft_study_sel,
        site_to_study=site_to_study, curve_row=PREV_CURVE_ROW,
        core_params=core_params,
    )

    summary = _compute_summary(
        prev_curves=prev_curves, prop_curves=prop_curves, w_study_sel=w_study_sel,
        df=df, canon_categories=canon_categories, type_code=PREV_TYPE_CODE,
    )

    return _emit(
        summary=summary, site_label_by_i=site_label_map(df, canon_categories), pretty=pretty,
        band_color="#1f77b4", pred_label="pred obs-scale",
        figure_title=f"Site-faceted {pretty} prevalence overlay (obs vs predicted obs-scale prevalence)",
        calibration_title=f"Calibration: predicted vs observed {pretty} prevalence",
        out_png=out_png, out_csv_summary=out_csv_summary, out_png_calibration=out_png_calibration,
    )


def plot_site_prevalence_overlay_prior_predictive(
    data_csv: Path, eir_priors_csv: Path,
    out_png: Path, out_csv_summary: Path, out_png_calibration: Path,
    ft_priors_csv: Path | None = None,
    n_eval_draws: int = 200, seed: int = 42,
) -> tuple[Path, Path, Path]:
    pretty = PREV_LABEL

    rng = np.random.default_rng(seed)

    df = load_observed_data(data_csv)
    df, canon_categories = encode_canonical_sites(df)
    site_to_study, _sm, study_levels = site_to_study_from_df(df)

    n_site = int(df["site_i"].max() + 1)
    n_study = len(study_levels)

    log_eir_mu, log_eir_sigma = aggregate_eir_priors(df=df, eir_priors_csv=Path(eir_priors_csv), n_site=n_site)

    ft_prior_df = load_ft_prior_table(df=df, ft_csv=Path(ft_priors_csv) if ft_priors_csv else None)
    ft_alpha = ft_prior_df["ft_alpha"].to_numpy(dtype=float)
    ft_beta = ft_prior_df["ft_beta"].to_numpy(dtype=float)
    if len(ft_alpha) != n_study:
        raise ValueError(f"FT prior rows ({len(ft_alpha)}) do not match n_study ({n_study})")

    D = int(n_eval_draws)
    log_eir = rng.normal(log_eir_mu, log_eir_sigma, size=(D, n_site))
    ft_study = np.empty((D, n_study), dtype=float)
    for j in range(n_study):
        ft_study[:, j] = rng.beta(float(ft_alpha[j]), float(ft_beta[j]), size=D)

    # sigma_p ~ HalfNormal(0.5); w_study | sigma_p ~ Normal(0, sigma_p).
    sigma_p = np.abs(rng.normal(0.0, 0.5, size=D))
    w_study = rng.normal(0.0, sigma_p[:, None], size=(D, n_study))

    prev_curves, prop_curves = solve_curves(
        log_eir=log_eir, ft_study=ft_study,
        site_to_study=site_to_study, curve_row=PREV_CURVE_ROW,
    )

    summary = _compute_summary(
        prev_curves=prev_curves, prop_curves=prop_curves, w_study_sel=as_np_float(w_study),
        df=df, canon_categories=canon_categories, type_code=PREV_TYPE_CODE,
    )

    return _emit(
        summary=summary, site_label_by_i=site_label_map(df, canon_categories), pretty=pretty,
        band_color="#ff7f0e", pred_label="prior predictive",
        figure_title=f"Site-faceted prior-predictive {pretty} prevalence overlay",
        calibration_title=f"Prior-predictive calibration: predicted vs observed {pretty} prevalence",
        out_png=out_png, out_csv_summary=out_csv_summary, out_png_calibration=out_png_calibration,
    )
