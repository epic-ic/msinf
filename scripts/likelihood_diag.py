"""Compare data log-likelihood at the Griffin/prior init against posterior draws.

Diagnostic question: does the posterior have both lower predictive bias *and*
higher data log-likelihood than the prior? If the posterior log-likelihood is
higher while its predictions are *more* biased, the likelihood is rewarding
bias -- a formulation bug.

Works with any run directory produced by ``scripts/run_inference.py``: the run's
``run_config_resolved.json`` supplies the inputs and the active/fixed core
parameter split, and the model's observed sites are discovered from the trace
rather than hardcoded, so this keeps working when the likelihood is refactored.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numpyro import handlers

from msinf.config import NumpyroConfig, griffin_refit_init_values
from msinf.data import SAMPLES_FILENAME
from msinf.dtypes import asfloat, set_precision
from msinf.model import build_model_kwargs, hierarchical_model, model_kwargs_for_sampler


def _config_from_run_dir(run_dir: Path, overrides: dict[str, str | None]) -> NumpyroConfig:
    cfg_path = run_dir / "run_config_resolved.json"
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"{cfg_path} not found; this run predates resolved-config recording. "
            "Pass --data-csv and --eir-priors-csv explicitly."
        )
    raw = json.loads(cfg_path.read_text())

    def pick(key: str) -> str | None:
        return overrides.get(key) or raw.get(key)

    data_csv = pick("data_csv")
    eir_priors_csv = pick("eir_priors_csv")
    if not data_csv or not eir_priors_csv:
        raise ValueError("run config records no data_csv/eir_priors_csv; pass them explicitly")

    return NumpyroConfig(
        data_csv=str(data_csv),
        eir_priors_csv=str(eir_priors_csv),
        reference_draws_csv=pick("reference_draws_csv"),
        ft_priors_csv=pick("ft_priors_csv"),
        mortality_csv=pick("mortality_csv"),
        active_core_params=tuple(raw.get("active_core_params", ())),
        init_strategy=str(raw.get("init_strategy", "griffin")),
        prior_family=str(raw.get("prior_family", "xi")),
    )


def observed_site_logliks(values: dict, model_kwargs: dict) -> dict[str, float]:
    """Log-density contribution of every observed/factor site, keyed by site name.

    Discovers the sites from the trace, so it does not need to know whether the
    likelihood is expressed as observed samples or as ``numpyro.factor`` terms.
    """
    seeded = handlers.seed(hierarchical_model, rng_seed=0)
    substituted = handlers.substitute(seeded, data=values)
    tr = handlers.trace(substituted).get_trace(**model_kwargs_for_sampler(model_kwargs))

    out: dict[str, float] = {}
    for name, site in tr.items():
        if site.get("type") != "sample" or not site.get("is_observed"):
            continue
        out[name] = float(np.sum(np.asarray(site["fn"].log_prob(site["value"]))))
    if not out:
        raise RuntimeError("no observed sites found in the model trace")
    return out


def _latent_values_from_draw(npz, keys: list[str], index: int | None) -> dict:
    """Latent values for one draw, or the posterior mean when ``index`` is None."""
    vals = {}
    for k in keys:
        arr = npz[k]
        flat = arr.reshape(-1, *arr.shape[2:])  # (chain*draw, ...)
        vals[k] = asfloat(flat.mean(axis=0) if index is None else flat[index])
    return vals


def _report(label: str, logliks: dict[str, float]) -> float:
    total = sum(logliks.values())
    parts = "  ".join(f"{k}={v:.2f}" for k, v in sorted(logliks.items()))
    print(f"  {label:28s} total={total:12.2f}   {parts}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help=f"Run directory containing {SAMPLES_FILENAME}")
    ap.add_argument("--samples-npz", default=None, help=f"Override the samples file (default <run-dir>/{SAMPLES_FILENAME})")
    ap.add_argument("--data-csv", default=None, help="Override the observed dataset the run was fitted to")
    ap.add_argument("--eir-priors-csv", default=None, help="Override the EIR prior table")
    ap.add_argument("--ft-priors-csv", default=None, help="Override the treatment prior table")
    ap.add_argument("--reference-draws-csv", default=None, help="Override the reference draws CSV")
    ap.add_argument("--n-draws", type=int, default=20, help="Posterior draws to evaluate individually")
    ap.add_argument("--precision", default="f32", choices=("f32", "f64"),
                    help="Float precision; must match the run to reproduce its numbers")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_precision(args.precision)

    run_dir = Path(args.run_dir).resolve()
    samples_npz = Path(args.samples_npz) if args.samples_npz else run_dir / SAMPLES_FILENAME
    if not samples_npz.exists():
        raise FileNotFoundError(f"Missing samples file: {samples_npz}")

    cfg = _config_from_run_dir(
        run_dir,
        {
            "data_csv": args.data_csv,
            "eir_priors_csv": args.eir_priors_csv,
            "ft_priors_csv": args.ft_priors_csv,
            "reference_draws_csv": args.reference_draws_csv,
        },
    )
    model_kwargs, active_core_params, _fixed = build_model_kwargs(cfg)

    npz = np.load(samples_npz)
    # Only (chain, draw, ...) arrays are latent draws; skip warmup__* duplicates.
    latent_keys = [k for k in npz.files if not k.startswith("warmup__") and npz[k].ndim >= 2]
    if not latent_keys:
        raise ValueError(f"No posterior draw arrays found in {samples_npz}")

    print(f"run_dir:  {run_dir}")
    print(f"samples:  {samples_npz}  ({len(latent_keys)} latent arrays)")
    print(f"data_csv: {cfg.data_csv}")
    print(f"active core params: {list(active_core_params)}\n")

    print("=== data log-likelihood ===")

    ll_prior = None
    if cfg.init_strategy == "griffin":
        griffin_vals = griffin_refit_init_values(
            active_core_params=active_core_params,
            log_eir_hat=np.asarray(model_kwargs["log_eir_hat"]),
            ft_alpha=np.asarray(model_kwargs["ft_alpha"]),
            ft_beta=np.asarray(model_kwargs["ft_beta"]),
            n_study=model_kwargs["n_study"],
            n_site=model_kwargs["n_site"],
            n_age_groups=model_kwargs["n_age_groups"],
        )
        ll_prior = _report("PRIOR/Griffin init", observed_site_logliks(griffin_vals, model_kwargs))

    post_mean_vals = _latent_values_from_draw(npz, latent_keys, index=None)
    ll_post_mean = _report("POSTERIOR mean", observed_site_logliks(post_mean_vals, model_kwargs))

    if ll_prior is not None:
        print(f"\n  delta (posterior mean - prior) = {ll_post_mean - ll_prior:.2f}")

    # Log-likelihood across individual draws, not only at the (non-representative)
    # posterior mean.
    n_total = npz[latent_keys[0]].reshape(-1, *npz[latent_keys[0]].shape[2:]).shape[0]
    n_eval = min(args.n_draws, n_total)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(n_total, size=n_eval, replace=False)
    lls = [
        sum(observed_site_logliks(_latent_values_from_draw(npz, latent_keys, int(i)), model_kwargs).values())
        for i in idx
    ]
    print(
        f"\n  LL at {n_eval} random posterior draws: mean={np.mean(lls):.2f}  "
        f"median={np.median(lls):.2f}  min={np.min(lls):.2f}  max={np.max(lls):.2f}"
    )
    if ll_prior is not None:
        print(f"  cf prior/Griffin init: {ll_prior:.2f}")


if __name__ == "__main__":
    main()
