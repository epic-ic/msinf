from pathlib import Path

import numpy as np
import pandas as pd

from ..age_grid import class_fraction_weights
from ..data import (
    aggregate_eir_priors,
    encode_canonical_sites,
    latest_samples_npz,
    load_ft_prior_table,
    load_observed_data,
    site_label_map,
    site_to_study_from_df,
)
from ..dtypes import as_np_float
from ..priors import DMEQ_OVERRIDE_PARAMS, build_xi_priors
from ._common import (
    AGE_GRID,
    OverlaySpec,
    flat2,
    flat3,
    plot_calibration,
    plot_faceted_overlay,
    select_draw_indices,
    solve_curves,
)

_CD_COLORS = {1: "#444444", 2: "#d62728", 3: "#9467bd"}
_CD_LABELS = {1: "Daily ACD", 2: "Weekly ACD (rW)", 3: "PCD (rP)"}


def _compute_summary(
    *,
    inc_curves: np.ndarray, prop_curves: np.ndarray,
    u_study_sel: np.ndarray, v_site_age_sel: np.ndarray,
    rW_sel: np.ndarray, rP_sel: np.ndarray,
    df: pd.DataFrame, canon_categories: list | None,
    denom_unit_csv: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inc_df = df[df["type"] == 1].copy()
    study_levels = sorted(pd.unique(df["study_index"]))
    study_map = {s: i for i, s in enumerate(study_levels)}
    site_label_by_i = site_label_map(df, canon_categories)

    inc_df["study_i"] = inc_df["study_index"].map(study_map).astype(int)
    inc_df["agek"] = (inc_df["age_bracket"].astype(int) - 1).clip(lower=0, upper=3)
    inc_df["obs_inc_raw"] = inc_df["numer"] / np.clip(inc_df["denom"], 1e-12, None)
    inc_df["obs_time_scale"] = 1.0

    if denom_unit_csv is not None:
        tab = pd.read_csv(denom_unit_csv)
        req = {"study_index", "obs_to_model_time_scale"}
        miss = req - set(tab.columns)
        if miss:
            raise ValueError(f"denom_unit_csv missing columns: {sorted(miss)}")
        tab = tab[["study_index", "obs_to_model_time_scale"]].copy()
        tab["study_index"] = tab["study_index"].astype(inc_df["study_index"].dtype)
        tab["obs_to_model_time_scale"] = pd.to_numeric(tab["obs_to_model_time_scale"], errors="coerce")
        if bool(tab["obs_to_model_time_scale"].isna().any()) or bool((tab["obs_to_model_time_scale"] <= 0).any()):
            raise ValueError("obs_to_model_time_scale must be finite and > 0 for all rows")
        inc_df = inc_df.merge(tab, on="study_index", how="left")
        if bool(inc_df["obs_to_model_time_scale"].isna().any()):
            missing_studies = sorted(pd.unique(inc_df.loc[inc_df["obs_to_model_time_scale"].isna(), "study_index"]))
            raise ValueError(f"denom_unit_csv missing study_index rows: {missing_studies}")
        inc_df["obs_time_scale"] = inc_df["obs_to_model_time_scale"].astype(float)

    inc_df["obs_inc"] = inc_df["obs_inc_raw"] * inc_df["obs_time_scale"]
    inc_df["age_mid"] = 0.5 * (inc_df["age0"] + inc_df["age1"])

    D = inc_curves.shape[0]
    rows = []
    for _, row in inc_df.iterrows():
        s = int(row["site_i"]); st = int(row["study_i"])
        agek = int(row["agek"]); det = int(row["case_detection"])
        a0 = float(row["age0"]); a1 = float(row["age1"])
        overlap = class_fraction_weights(np.array([a0]), np.array([a1]), AGE_GRID)[0]

        vals_mu, vals_obs = [], []
        for d in range(D):
            den = max((overlap * prop_curves[d, s, :]).sum(), 1e-12)
            mu = float((overlap * inc_curves[d, s, :]).sum() / den) * 365.0
            r_inc = 1.0 if det <= 1 else (float(rW_sel[d]) if det == 2 else float(rP_sel[d]))
            scale = r_inc * float(u_study_sel[d, st]) * float(v_site_age_sel[d, s, agek])
            vals_mu.append(mu); vals_obs.append(scale * mu)

        vals_mu = np.asarray(vals_mu); vals_obs = np.asarray(vals_obs)
        rows.append({
            "site_i": s, "site_index": site_label_by_i[s],
            "study_i": st, "study_index": row["study_index"],
            "agek": agek, "age_start": a0, "age_end": a1, "age_mid": float(row["age_mid"]),
            "obs_inc_raw": float(row["obs_inc_raw"]),
            "obs_time_scale": float(row["obs_time_scale"]),
            "obs_value": float(row["obs_inc"]),
            "pred_mu_median": float(np.median(vals_mu)),
            "pred_mu_lower95": float(np.quantile(vals_mu, 0.025)),
            "pred_mu_upper95": float(np.quantile(vals_mu, 0.975)),
            "pred_obs_median": float(np.median(vals_obs)),
            "pred_obs_lower95": float(np.quantile(vals_obs, 0.025)),
            "pred_obs_upper95": float(np.quantile(vals_obs, 0.975)),
            "case_detection": det,
        })

    summary = pd.DataFrame(rows)

    ratio = summary["obs_value"] / np.clip(summary["pred_obs_median"], 1e-12, None)
    tmp = summary.assign(obs_pred_ratio=ratio)
    canon = as_np_float([1.0 / 365.0, 1.0 / 52.0, 1.0 / 12.0, 1.0, 12.0, 52.0, 365.0])

    def _nearest_canonical(x: float) -> float:
        return float(canon[np.argmin(np.abs(np.log(canon) - np.log(max(x, 1e-12))))])

    unit_audit = (
        tmp.groupby(["study_i", "study_index"], as_index=False)
        .agg(
            n=("obs_value", "size"),
            obs_mean=("obs_value", "mean"),
            pred_mean=("pred_obs_median", "mean"),
            ratio_median=("obs_pred_ratio", "median"),
            ratio_q10=("obs_pred_ratio", lambda x: float(np.quantile(x, 0.10))),
            ratio_q90=("obs_pred_ratio", lambda x: float(np.quantile(x, 0.90))),
            obs_time_scale_unique=("obs_time_scale", "nunique"),
        )
    )
    unit_audit["nearest_canonical_ratio"] = unit_audit["ratio_median"].apply(_nearest_canonical)
    unit_audit["canonical_gap_log"] = np.abs(
        np.log(np.clip(unit_audit["ratio_median"], 1e-12, None)) - np.log(unit_audit["nearest_canonical_ratio"])
    )
    assert isinstance(unit_audit, pd.DataFrame)
    return summary, unit_audit


def _emit(
    summary: pd.DataFrame, unit_audit: pd.DataFrame,
    site_label_by_i: dict[int, object],
    band_color: str, pred_label: str, figure_title: str, calibration_title: str,
    out_png: Path, out_csv_summary: Path, out_png_calibration: Path, out_csv_unit_audit: Path,
) -> tuple[Path, Path, Path, Path]:
    out_csv_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv_summary, index=False)
    out_csv_unit_audit.parent.mkdir(parents=True, exist_ok=True)
    unit_audit.to_csv(out_csv_unit_audit, index=False)

    ylabel = "incidence rate (cases per person-year)"
    if (summary["obs_time_scale"] != 1.0).any():
        ylabel = "incidence rate (cases per person-year; obs adjusted to model time unit)"

    spec = OverlaySpec(
        band_color=band_color, pred_label=pred_label,
        figure_title=figure_title, calibration_title=calibration_title,
        y_label=ylabel, y_min=None, y_max=None, cal_axis_max=None,
    )
    plot_faceted_overlay(
        summary=summary, site_label_by_i=site_label_by_i, spec=spec, out_png=out_png,
        group_col="case_detection", group_colors=_CD_COLORS,
    )
    plot_calibration(
        summary=summary, spec=spec, out_png=out_png_calibration,
        x_label="Observed incidence rate (cases per person-year; numer/denom)",
        y_label=f"Predicted incidence rate ({pred_label} median, cases per person-year)",
        group_col="case_detection", group_colors=_CD_COLORS, group_labels=_CD_LABELS,
    )
    return out_png, out_csv_summary, out_png_calibration, out_csv_unit_audit


def plot_site_incidence_overlay(
    samples_npz: Path, data_csv: Path,
    out_png: Path, out_csv_summary: Path, out_png_calibration: Path, out_csv_unit_audit: Path,
    n_eval_draws: int = 80, seed: int = 42, denom_unit_csv: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    post = np.load(samples_npz)
    required = {"log_eir", "ft_study", "rW", "rP"}
    missing = sorted(required - set(post.files))
    if missing:
        raise ValueError(f"samples missing required variables: {missing}")

    log_eir = flat3(post["log_eir"])
    # ft_study is the treatment fraction and only feeds the ODE solve (p["ft"]).
    ft_study = flat3(post["ft_study"])
    n_total = log_eir.shape[0]
    n_site = log_eir.shape[1]
    # The study effect u (incidence multiplier) and the site-age effect v_jk are
    # integrated out in the model (GH quadrature / negative-binomial), so they are
    # absent from the posterior samples. Both have marginal mean 1 on the rate, so
    # the obs-scale point prediction uses u = 1 and v = 1.
    if "u_study" in post.files:
        u_study = flat3(post["u_study"])
    else:
        # n_study <= n_site, so n_site columns of ones is enough for study indexing.
        u_study = np.ones((n_total, n_site), dtype=float)
    if "v_site_age" in post.files:
        v_site_age = flat3(post["v_site_age"])
    else:
        v_site_age = np.ones((n_total, n_site * 4), dtype=float)
    rW = flat2(post["rW"]); rP = flat2(post["rP"])

    df = load_observed_data(data_csv)
    df, canon_categories = encode_canonical_sites(df)
    site_to_study, _sm, _sl = site_to_study_from_df(df)

    draw_idx = select_draw_indices(n_total=n_total, n_eval=n_eval_draws, seed=seed)
    log_eir_sel = log_eir[draw_idx, :]
    ft_study_sel = ft_study[draw_idx, :]
    u_study_sel = u_study[draw_idx, :]
    v_site_age_sel = as_np_float(v_site_age[draw_idx, :]).reshape(len(draw_idx), -1, 4)
    rW_sel = as_np_float(rW[draw_idx])
    rP_sel = as_np_float(rP[draw_idx])

    # Use the posterior draws of the core/immunity parameters so the overlay solves
    # the same curves the likelihood evaluated (not dmeq defaults).
    core_params = {
        name: flat2(post[name])[draw_idx] for name in DMEQ_OVERRIDE_PARAMS if name in post.files
    }

    inc_curves, prop_curves = solve_curves(
        log_eir=log_eir_sel, ft_study=ft_study_sel, site_to_study=site_to_study, curve_row=2,
        core_params=core_params,
    )

    summary, unit_audit = _compute_summary(
        inc_curves=inc_curves, prop_curves=prop_curves,
        u_study_sel=as_np_float(u_study_sel),
        v_site_age_sel=v_site_age_sel, rW_sel=rW_sel, rP_sel=rP_sel,
        df=df, canon_categories=canon_categories, denom_unit_csv=denom_unit_csv,
    )

    return _emit(
        summary=summary, unit_audit=unit_audit,
        site_label_by_i=site_label_map(df, canon_categories),
        band_color="#2ca02c", pred_label="pred obs-scale",
        figure_title="Site-faceted incidence-rate overlay (obs vs predicted obs-scale rate)",
        calibration_title="Calibration: predicted vs observed incidence rate",
        out_png=out_png, out_csv_summary=out_csv_summary,
        out_png_calibration=out_png_calibration, out_csv_unit_audit=out_csv_unit_audit,
    )


def plot_site_incidence_overlay_prior_predictive(
    data_csv: Path, eir_priors_csv: Path,
    out_png: Path, out_csv_summary: Path, out_png_calibration: Path, out_csv_unit_audit: Path,
    ft_priors_csv: Path | None = None,
    n_eval_draws: int = 200, seed: int = 42, denom_unit_csv: Path | None = None,
) -> tuple[Path, Path, Path, Path]:
    rng = np.random.default_rng(seed)

    df = load_observed_data(data_csv)
    df, canon_categories = encode_canonical_sites(df)
    site_to_study, _sm, study_levels = site_to_study_from_df(df)

    n_site = int(df["site_i"].max() + 1)
    n_study = len(study_levels)
    n_age_groups = 4

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

    # Observation-scale incidence multiplier u_study (mean 1), matching the model:
    #   sigma_c ~ HalfNormal(0.5); u_study ~ LogNormal(-0.5*sigma_c^2, sigma_c).
    # This is the study correction applied to predicted incidence; ft_study is the
    # treatment fraction and feeds only the ODE solve below.
    sigma_c = np.abs(rng.normal(0.0, 0.5, size=D))
    u_study = rng.lognormal(mean=-0.5 * sigma_c[:, None] ** 2, sigma=sigma_c[:, None], size=(D, n_study))

    # v_site_age variance alpha_c ~ scale-0.25 half-t(6) (matches the model's prior).
    alpha_c_buf: list[np.ndarray] = []
    have = 0
    while have < D:
        block = rng.standard_t(df=6, size=max(D - have, 64)) * 0.25
        pos = block[block > 0]
        alpha_c_buf.append(pos); have += pos.size
    alpha_c = np.concatenate(alpha_c_buf)[:D]
    v_site_age = rng.gamma(shape=1.0 / alpha_c[:, None], scale=alpha_c[:, None], size=(D, n_site * n_age_groups))

    priors = build_xi_priors()
    rW = rng.beta(priors["rW"].param1, priors["rW"].param2, size=D)
    rP = rng.beta(priors["rP"].param1, priors["rP"].param2, size=D)

    inc_curves, prop_curves = solve_curves(
        log_eir=log_eir, ft_study=ft_study, site_to_study=site_to_study, curve_row=2,
    )
    v_site_age_arr = as_np_float(v_site_age.reshape(D, n_site, n_age_groups))

    summary, unit_audit = _compute_summary(
        inc_curves=inc_curves, prop_curves=prop_curves,
        u_study_sel=as_np_float(u_study), v_site_age_sel=v_site_age_arr,
        rW_sel=as_np_float(rW), rP_sel=as_np_float(rP),
        df=df, canon_categories=canon_categories, denom_unit_csv=denom_unit_csv,
    )

    return _emit(
        summary=summary, unit_audit=unit_audit,
        site_label_by_i=site_label_map(df, canon_categories),
        band_color="#ff7f0e", pred_label="prior predictive",
        figure_title="Site-faceted prior-predictive incidence-rate overlay (obs vs prior-predictive obs-scale rate)",
        calibration_title="Prior-predictive calibration: predicted vs observed incidence rate",
        out_png=out_png, out_csv_summary=out_csv_summary,
        out_png_calibration=out_png_calibration, out_csv_unit_audit=out_csv_unit_audit,
    )


def run_site_overlay_from_latest(
    artifacts_root: Path, data_csv: Path, outdir: Path,
    n_eval_draws: int = 80, seed: int = 42,
    samples_npz: Path | None = None, denom_unit_csv: Path | None = None,
) -> tuple[Path, Path, Path, Path, Path]:
    if samples_npz is None:
        samples_npz = latest_samples_npz(artifacts_root)
    out_png = outdir / "site_incidence_age_overlay_latest.png"
    out_csv = outdir / "site_incidence_age_overlay_summary_latest.csv"
    out_png_cal = outdir / "site_incidence_calibration_latest.png"
    out_csv_unit_audit = outdir / "site_incidence_unit_audit_latest.csv"
    plot_site_incidence_overlay(
        samples_npz=samples_npz, data_csv=data_csv,
        out_png=out_png, out_csv_summary=out_csv,
        out_png_calibration=out_png_cal, out_csv_unit_audit=out_csv_unit_audit,
        n_eval_draws=n_eval_draws, seed=seed, denom_unit_csv=denom_unit_csv,
    )
    return samples_npz, out_png, out_csv, out_png_cal, out_csv_unit_audit
