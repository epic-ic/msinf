import argparse
import json
from datetime import datetime
from pathlib import Path
from string import Template

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from msinf.data import SAMPLES_FILENAME
from msinf.plotting import (
    compute_incidence_age_grid_bootstrap,
    compute_incidence_age_grid_numpyro_vs_reference,
    missing_immunity_params,
    placeholder_png,
    plot_core_marginals_overlay,
    plot_immunity_functions_overlay,
    plot_logeir_marginals_overlay,
    plot_site_incidence_overlay,
    plot_site_incidence_overlay_prior_predictive,
    plot_site_prevalence_overlay,
    plot_site_prevalence_overlay_prior_predictive,
)
from msinf.priors import CORE_PARAMS


def _resolve_active_core_params(run_dir: Path, samples_npz: Path) -> list[str]:
    cfg_path = run_dir / "run_config_resolved.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        active = [p for p in cfg.get("active_core_params", []) if p in CORE_PARAMS]
        if active:
            return active

    post = np.load(samples_npz)
    active_core = [p for p in CORE_PARAMS if p in post.files and post[p].ndim == 2]
    return active_core


def _resolve_prior_family(run_dir: Path, run_cfg: dict) -> str:
    family = run_cfg.get("prior_family")
    if not family:
        hydra_cfg = run_dir / ".hydra" / "config.yaml"
        if hydra_cfg.exists():
            family = yaml.safe_load(hydra_cfg.read_text()).get("priors", {}).get("name")
    family = str(family or "xi").strip().lower()
    if family not in {"xi", "griffin"}:
        raise ValueError(f"Unsupported prior family recorded by run: {family!r}")
    return family


def _resolve_input_csv(
    cli_value: str | None,
    run_dir: Path,
    run_cfg: dict,
    key: str,
    fallback: Path,
) -> Path:
    """Resolve a fitting input CSV, so plots use the data the run was fitted to.

    Order: explicit CLI override, then the path recorded in
    run_config_resolved.json, then the Hydra config for runs predating that
    record, then the repo default.
    """
    if cli_value:
        return Path(cli_value).resolve()

    recorded = run_cfg.get(key)
    if not recorded:
        hydra_cfg = run_dir / ".hydra" / "config.yaml"
        if hydra_cfg.exists():
            recorded = yaml.safe_load(hydra_cfg.read_text()).get("data", {}).get(key)

    if not recorded:
        print(f"WARNING: run dir records no {key}; falling back to {fallback}")
        return fallback.resolve()

    # Hydra runs with chdir=True, so relative paths are relative to the repo root.
    path = Path(recorded)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        raise FileNotFoundError(f"{key} recorded by {run_dir} does not exist: {path}")
    print(f"Using {key} from run dir: {path}")
    return path.resolve()


def _plot_core_trace(samples_npz: Path, params: list[str], out_png: Path) -> Path:
    post = np.load(samples_npz)
    params = [p for p in params if p in post.files and post[p].ndim == 2]
    if not params:
        raise ValueError("No scalar core params available for trace plotting.")

    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 2.1 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for i, p in enumerate(params):
        ax = axes[i]
        arr = np.asarray(post[p])  # (chain, draw)
        n_chain = arr.shape[0]
        for c in range(n_chain):
            ax.plot(arr[c], linewidth=0.8, alpha=0.7, label=f"chain {c + 1}")
        ax.set_title(p, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)

    for j in range(len(params), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False)
    fig.suptitle("Core parameter trace plots", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def _plot_core_trace_warmup(samples_npz: Path, params: list[str], out_png: Path) -> Path | None:
    post = np.load(samples_npz)
    warmup_keys = {k for k in post.files if k.startswith("warmup__")}
    params = [p for p in params if f"warmup__{p}" in warmup_keys and post[f"warmup__{p}"].ndim == 2]
    if not params:
        return None

    n = len(params)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 2.1 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for i, p in enumerate(params):
        ax = axes[i]
        arr = np.asarray(post[f"warmup__{p}"])  # (chain, draw)
        n_chain = arr.shape[0]
        for c in range(n_chain):
            ax.plot(arr[c], linewidth=0.8, alpha=0.7, label=f"chain {c + 1}")
        ax.set_title(p, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)

    for j in range(len(params), len(axes)):
        axes[j].axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(4, len(handles)), frameon=False)
    fig.suptitle("Core parameter trace plots (warmup)", y=0.995, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return out_png


def _compute_arviz_summary(samples_npz: Path, active_core: list[str], out_csv: Path) -> pd.DataFrame:
    post = np.load(samples_npz)
    posterior = {k: post[k] for k in active_core if k in post.files and post[k].ndim == 2}
    if not posterior:
        empty = pd.DataFrame(columns=["mean", "sd", "hdi_3%", "hdi_97%", "ess_bulk", "ess_tail", "r_hat"])
        empty.to_csv(out_csv)
        return empty
    idata = az.from_dict(posterior=posterior)
    summary = az.summary(idata, var_names=list(posterior.keys()), round_to=6)
    assert isinstance(summary, pd.DataFrame)  # az.summary returns a Dataset only with fmt="xarray"
    summary.to_csv(out_csv)
    return summary


def _summary_markdown_table(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "_No active core parameters in this run (core=none), so core MCMC diagnostics are not applicable._"
    cols = ["r_hat", "ess_bulk", "ess_tail"]
    keep = summary[cols].copy()
    keep = keep.reset_index().rename(columns={"index": "parameter"})
    keep["r_hat"] = keep["r_hat"].map(lambda x: f"{float(x):.4f}")
    keep["ess_bulk"] = keep["ess_bulk"].map(lambda x: f"{float(x):.1f}")
    keep["ess_tail"] = keep["ess_tail"].map(lambda x: f"{float(x):.1f}")
    lines = [
        "| parameter | r_hat | ess_bulk | ess_tail |",
        "|---|---:|---:|---:|",
    ]
    for _, r in keep.iterrows():
        lines.append(f"| {r['parameter']} | {r['r_hat']} | {r['ess_bulk']} | {r['ess_tail']} |")
    return "\n".join(lines)


def _write_markdown_report(
    report_md: Path,
    template_md: Path,
    outputs: dict[str, Path | None],
    run_cfg: dict,
    summary_md_table: str,
) -> Path:
    rel = {k: v.name for k, v in outputs.items() if v is not None}
    warmup_trace = outputs.get("trace_core_warmup")
    warmup_trace_section = (
        f"![Core parameter warmup trace plots showing chain behavior during adaptation for active core parameters.]({warmup_trace.name})"
        if warmup_trace is not None
        else "_Warmup traces not available for this run (collect_warmup=false or warmup arrays missing)._"
    )
    now = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    template = Template(template_md.read_text())
    content = template.safe_substitute(
        generated_at=now,
        num_chains=run_cfg.get("num_chains", "NA"),
        num_warmup=run_cfg.get("num_warmup", "NA"),
        num_samples=run_cfg.get("num_samples", "NA"),
        summary_md_table=summary_md_table,
        warmup_trace_section=warmup_trace_section,
        **rel,
    )

    report_md.write_text(content)
    return report_md


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    template_md = root / "templates" / "diagnostics_report_template.md"

    ap = argparse.ArgumentParser(description="Generate full diagnostics plot bundle and markdown report from a NumPyro run directory.")
    ap.add_argument("--run-dir", required=True, help=f"Run directory containing {SAMPLES_FILENAME}")
    ap.add_argument("--outdir", default=None, help="Output directory (default: <run-dir>/diagnostics_bundle)")
    ap.add_argument("--reference-csv", default=str(root.parent / "malariasimulation" / "data-raw" / "parameter_draws_pf.csv"))
    ap.add_argument(
        "--eir-priors-csv",
        default=None,
        help="Override the EIR prior table; default is the one recorded by the run.",
    )
    ap.add_argument(
        "--data-csv",
        default=None,
        help="Override the observed dataset; default is the one the run was fitted to.",
    )
    ap.add_argument("--denom-unit-csv", default=None)
    ap.add_argument("--n-eval-draws", type=int, default=200)
    ap.add_argument("--n-bootstrap", type=int, default=500)
    ap.add_argument("--site-overlay-draws", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--immunity-plots",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Griffin-specific immunity overlay: 'auto' draws it when the fit exposes "
            "Griffin's immunity parameters and skips it otherwise, 'on' fails if it "
            "cannot be drawn, 'off' never draws it."
        ),
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    samples_npz = run_dir / SAMPLES_FILENAME
    if not samples_npz.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_npz}")

    outdir = Path(args.outdir).resolve() if args.outdir else (run_dir / "diagnostics_bundle")
    outdir.mkdir(parents=True, exist_ok=True)

    active_core = _resolve_active_core_params(run_dir=run_dir, samples_npz=samples_npz)
    run_cfg = {}
    cfg_path = run_dir / "run_config_resolved.json"
    if cfg_path.exists():
        run_cfg = json.loads(cfg_path.read_text())
    prior_family = _resolve_prior_family(run_dir, run_cfg)

    data_csv = _resolve_input_csv(
        cli_value=args.data_csv,
        run_dir=run_dir,
        run_cfg=run_cfg,
        key="data_csv",
        fallback=root / "artifacts" / "numpyro_jamie_100x4" / "study_data_2014.csv",
    )
    eir_priors_csv = _resolve_input_csv(
        cli_value=args.eir_priors_csv,
        run_dir=run_dir,
        run_cfg=run_cfg,
        key="eir_priors_csv",
        fallback=root / "artifacts" / "inputs" / "eir_site_priors_griffin2014.csv",
    )

    out_trace = outdir / "trace_core_params.png"
    out_trace_warmup = outdir / "trace_core_params_warmup.png"
    if active_core:
        warmup_trace_path = _plot_core_trace_warmup(samples_npz=samples_npz, params=active_core, out_png=out_trace_warmup)
        _plot_core_trace(samples_npz=samples_npz, params=active_core, out_png=out_trace)
    else:
        warmup_trace_path = placeholder_png(out_trace_warmup, "No active core parameters for this run")
        placeholder_png(out_trace, "No active core parameters for this run")

    out_eir_panel = outdir / "marginals_logeir_prior_vs_posterior_overlay_panel.png"
    out_eir_pooled = outdir / "marginals_logeir_prior_vs_posterior_overlay_pooled.png"
    plot_logeir_marginals_overlay(
        samples_npz=samples_npz,
        eir_priors_csv=eir_priors_csv,
        data_csv=data_csv,
        out_panel_png=out_eir_panel,
        out_pooled_png=out_eir_pooled,
        seed=args.seed,
    )

    out_core_prior_vs_numpyro = outdir / "core_marginals_overlay_prior_vs_numpyro_posterior.png"
    if active_core:
        plot_core_marginals_overlay(
            samples_npz=samples_npz,
            reference_csv=None,
            include_prior=True,
            include_numpyro_posterior=True,
            include_reference_pf=False,
            selected_params=active_core,
            prior_family=prior_family,
            out_png=out_core_prior_vs_numpyro,
            seed=args.seed,
        )
    else:
        placeholder_png(out_core_prior_vs_numpyro, "No active core parameters for this run")

    out_core_numpyro_vs_ref = outdir / "core_marginals_overlay_numpyro_vs_reference_pf.png"
    if active_core:
        plot_core_marginals_overlay(
            samples_npz=samples_npz,
            reference_csv=Path(args.reference_csv),
            include_prior=False,
            include_numpyro_posterior=True,
            include_reference_pf=True,
            selected_params=active_core,
            out_png=out_core_numpyro_vs_ref,
            seed=args.seed,
        )
    else:
        placeholder_png(out_core_numpyro_vs_ref, "No active core parameters for this run")

    out_immunity_overlay = outdir / "immunity_functions_overlay_prior_posterior_reference.png"
    fixed_core_values = run_cfg.get("fixed_core_values", {}) if isinstance(run_cfg, dict) else {}
    # The overlay reads b/phi/q off Griffin's parameterization, so it is opt-in:
    # a candidate curve family exposing neither gets a placeholder, not a crash.
    missing_immunity = missing_immunity_params(samples_npz, fixed_core_values)
    if args.immunity_plots == "off":
        placeholder_png(out_immunity_overlay, "Immunity overlay disabled (--immunity-plots off)")
    elif missing_immunity and args.immunity_plots == "auto":
        placeholder_png(
            out_immunity_overlay,
            "Immunity overlay skipped: this fit does not expose Griffin's immunity parameters",
        )
    else:
        plot_immunity_functions_overlay(
            samples_npz=samples_npz,
            reference_csv=Path(args.reference_csv) if Path(args.reference_csv).exists() else None,
            out_png=out_immunity_overlay,
            n_curves=400,
            prior_draw_count=6000,
            fixed_core_values=fixed_core_values,
            prior_family=prior_family,
            seed=args.seed,
        )

    out_inc_csv = outdir / "incidence_age_grid_bootstrap.csv"
    out_inc_png = outdir / "incidence_age_grid_bootstrap.png"
    compute_incidence_age_grid_bootstrap(
        samples_npz=samples_npz,
        out_csv=out_inc_csv,
        out_png=out_inc_png,
        n_eval_draws=args.n_eval_draws,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    # Site-faceted prior predictive incidence-rate overlay (analogous to the
    # posterior site-faceted overlay, but drawing latents from priors only).
    ft_priors_csv_cfg = run_cfg.get("ft_priors_csv") if isinstance(run_cfg, dict) else None

    # Site-faceted prior-predictive microscopy prevalence overlay (placed before the
    # incidence prior-predictive overlay in the report).
    out_site_prev_prior_png = outdir / "site_prevalence_age_overlay_prior_predictive.png"
    out_site_prev_prior_csv = outdir / "site_prevalence_age_overlay_prior_predictive_summary.csv"
    out_site_prev_prior_cal_png = outdir / "site_prevalence_calibration_prior_predictive.png"
    try:
        plot_site_prevalence_overlay_prior_predictive(
            data_csv=data_csv,
            eir_priors_csv=eir_priors_csv,
            ft_priors_csv=Path(ft_priors_csv_cfg) if ft_priors_csv_cfg else None,
            out_png=out_site_prev_prior_png,
            out_csv_summary=out_site_prev_prior_csv,
            out_png_calibration=out_site_prev_prior_cal_png,
            n_eval_draws=args.site_overlay_draws,
            seed=args.seed,
        )
    except Exception as e:  # noqa: BLE001 -- a failed figure becomes a placeholder
        print(f"Warning: Prior predictive prevalence overlay generation failed: {e}")
        placeholder_png(out_site_prev_prior_png, f"Prior predictive prevalence overlay unavailable: {e}")
        placeholder_png(out_site_prev_prior_cal_png, "Prior predictive prevalence calibration unavailable")

    out_site_prior_png = outdir / "site_incidence_age_overlay_prior_predictive.png"
    out_site_prior_csv = outdir / "site_incidence_age_overlay_prior_predictive_summary.csv"
    out_site_prior_cal_png = outdir / "site_incidence_calibration_prior_predictive.png"
    out_site_prior_unit_csv = outdir / "site_incidence_unit_audit_prior_predictive.csv"
    try:
        plot_site_incidence_overlay_prior_predictive(
            data_csv=data_csv,
            eir_priors_csv=eir_priors_csv,
            ft_priors_csv=Path(ft_priors_csv_cfg) if ft_priors_csv_cfg else None,
            out_png=out_site_prior_png,
            out_csv_summary=out_site_prior_csv,
            out_png_calibration=out_site_prior_cal_png,
            out_csv_unit_audit=out_site_prior_unit_csv,
            n_eval_draws=args.site_overlay_draws,
            seed=args.seed,
            denom_unit_csv=Path(args.denom_unit_csv) if args.denom_unit_csv else None,
        )
    except Exception as e:  # noqa: BLE001 -- a failed figure becomes a placeholder
        print(f"Warning: Prior predictive site overlay generation failed: {e}")
        placeholder_png(out_site_prior_png, f"Prior predictive site overlay unavailable: {e}")
        placeholder_png(out_site_prior_cal_png, "Prior predictive calibration unavailable")

    out_inc_median_draw_compare = outdir / "incidence_age_grid_numpyro_vs_griffin_median_draw_compare.png"
    out_inc_bootstrap_compare = outdir / "incidence_age_grid_numpyro_vs_griffin_bootstrap_compare.png"
    try:
        compute_incidence_age_grid_numpyro_vs_reference(
            samples_npz=samples_npz,
            reference_csv=Path(args.reference_csv),
            out_png_median_draw=out_inc_median_draw_compare,
            out_png_bootstrap=out_inc_bootstrap_compare,
            n_eval_draws=args.n_eval_draws,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed,
        )
    except Exception:  # noqa: BLE001 -- a failed figure becomes a placeholder
        placeholder_png(out_inc_median_draw_compare, "NumPyro vs reference comparison unavailable for this run")
        placeholder_png(out_inc_bootstrap_compare, "NumPyro vs reference comparison unavailable for this run")

    # Site-faceted posterior microscopy prevalence overlay (placed before the
    # incidence posterior overlay in the report).
    out_site_prev_png = outdir / "site_prevalence_age_overlay.png"
    out_site_prev_csv = outdir / "site_prevalence_age_overlay_summary.csv"
    out_site_prev_cal_png = outdir / "site_prevalence_calibration.png"
    try:
        plot_site_prevalence_overlay(
            samples_npz=samples_npz,
            data_csv=data_csv,
            out_png=out_site_prev_png,
            out_csv_summary=out_site_prev_csv,
            out_png_calibration=out_site_prev_cal_png,
            n_eval_draws=args.site_overlay_draws,
            seed=args.seed,
        )
    except Exception as e:  # noqa: BLE001 -- a failed figure becomes a placeholder
        print(f"Warning: Posterior prevalence overlay generation failed: {e}")
        placeholder_png(out_site_prev_png, f"Posterior prevalence overlay unavailable: {e}")
        placeholder_png(out_site_prev_cal_png, "Posterior prevalence calibration unavailable")

    out_site_png = outdir / "site_incidence_age_overlay.png"
    out_site_csv = outdir / "site_incidence_age_overlay_summary.csv"
    out_cal_png = outdir / "site_incidence_calibration.png"
    out_unit_csv = outdir / "site_incidence_unit_audit.csv"
    plot_site_incidence_overlay(
        samples_npz=samples_npz,
        data_csv=data_csv,
        out_png=out_site_png,
        out_csv_summary=out_site_csv,
        out_png_calibration=out_cal_png,
        out_csv_unit_audit=out_unit_csv,
        n_eval_draws=args.site_overlay_draws,
        seed=args.seed,
        denom_unit_csv=Path(args.denom_unit_csv) if args.denom_unit_csv else None,
    )

    out_summary_csv = outdir / "arviz_summary_active_core.csv"
    summary_df = _compute_arviz_summary(samples_npz=samples_npz, active_core=active_core, out_csv=out_summary_csv)
    summary_table_md = _summary_markdown_table(summary_df)

    report_md = outdir / "diagnostics_report.md"
    _write_markdown_report(
        report_md=report_md,
        template_md=template_md,
        outputs={
            "trace_core_warmup": warmup_trace_path,
            "trace_core": out_trace,
            "eir_panel": out_eir_panel,
            "eir_pooled": out_eir_pooled,
            "core_prior_vs_numpyro": out_core_prior_vs_numpyro,
            "core_numpyro_vs_ref": out_core_numpyro_vs_ref,
            "immunity_overlay": out_immunity_overlay,
            "inc_grid": out_inc_png,
            "site_overlay_prev_prior": out_site_prev_prior_png,
            "calibration_prev_prior": out_site_prev_prior_cal_png,
            "site_overlay_prior": out_site_prior_png,
            "calibration_prior": out_site_prior_cal_png,
            "inc_grid_median_draw_compare": out_inc_median_draw_compare,
            "inc_grid_bootstrap_compare": out_inc_bootstrap_compare,
            "site_overlay_prev": out_site_prev_png,
            "calibration_prev": out_site_prev_cal_png,
            "site_overlay": out_site_png,
            "calibration": out_cal_png,
        },
        run_cfg=run_cfg,
        summary_md_table=summary_table_md,
    )

    print(f"run_dir: {run_dir}")
    print(f"samples: {samples_npz}")
    print(f"outdir: {outdir}")
    print(f"report_md: {report_md}")


if __name__ == "__main__":
    main()
