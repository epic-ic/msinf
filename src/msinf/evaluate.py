"""The importable evaluation API.

``msinf`` owns everything about what the numbers mean; a caller owns its own
subprocess and harness contract. :func:`evaluate` takes a resolved config, an
optional curve provider and an optional split, fits under a fixed budget, and
returns the metric dictionary and the figures as bytes.

No Hydra, no working-directory assumptions, no writes to disk, so it is safe to
call inside somebody else's child process; ``scripts/run_holdout.py`` is the
Hydra caller. One implementation means reported and searched-on numbers cannot
drift apart.

``elpd_holdout`` is held-out density with the random effects marginalised as the
fitted likelihood marginalises them: prevalence per row, incidence per
(site, age-group) cell (see :mod:`msinf.likelihood` for why). The two streams are
reported separately as well as summed. The per-unit vectors are returned in
``EvaluationResult.pointwise``: between two candidates on one fold, the paired
difference of those vectors is the comparison that can tell them apart, not the
totals.
"""

import io
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp
from numpyro import handlers
from numpyro.diagnostics import summary

from .age_grid import GRIFFIN_AGE_GRID
from .config import NumpyroConfig
from .dtypes import asfloat
from .fit import FitResult, fit
from .likelihood import (
    incidence_cell_loglik_nodes,
    mix_over_nodes,
    prevalence_row_loglik_nodes,
    study_effect_nodes,
)
from .model import bin_streams, build_model_kwargs, model_kwargs_for_sampler, site_curves
from .predictive import coverage, randomised_pit, simulate_incidence, simulate_prevalence
from .splits import Split

# Streams whose likelihood unit is the observation row. Microscopy prevalence
# is the only one: incidence is scored per cell, and there is no PCR stream.
PREVALENCE_STREAMS = ("prev_m",)
# Every stream, in report order.
STREAMS = ("inc",) + PREVALENCE_STREAMS


@dataclass
class EvaluationResult:
    """What a fit is worth, and the pictures that justify it."""

    metrics: dict[str, float]
    figures: dict[str, bytes] = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    #: the posterior draws the scoring pass ran on, flattened across chains and
    #: thinned to ``n_score_draws``. Returned because a caller that asks whether
    #: the data moved a candidate's parameters needs the draws themselves, and
    #: refitting to get them would be a diagnostic of a different fit.
    draws: dict[str, np.ndarray] = field(default_factory=dict)
    #: per-unit elpd, keyed ``"<tag>_<stream>"`` -- one entry per non-empty
    #: incidence cell or prevalence row. The order is a property of the split
    #: and the table, so two candidates under the same split line up elementwise.
    pointwise: dict[str, np.ndarray] = field(default_factory=dict)


def _observation_bundle(k: dict) -> dict:
    """Regroup the flat ``<field>_<stream>`` model kwargs into one dict per stream."""
    return {
        stream: {key.removesuffix(f"_{stream}"): v for key, v in k.items() if key.endswith(f"_{stream}")}
        for stream in STREAMS
    }


def _thin(draws: dict[str, np.ndarray], max_draws: int | None) -> dict[str, np.ndarray]:
    """Posterior draws thinned to at most ``max_draws``, evenly spaced."""
    n = int(next(iter(draws.values())).shape[0])
    if max_draws is not None and n > max_draws:
        idx = np.linspace(0, n - 1, max_draws).astype(int)
        return {k: np.asarray(v)[idx] for k, v in draws.items()}
    return {k: np.asarray(v) for k, v in draws.items()}


def _params_from_draw(parameters, draw: dict):
    """Re-run a candidate's ``parameters()`` with a posterior draw substituted.

    Works for any provider without adding to the contract: the draw's sample
    sites are substituted in and everything downstream of them is recomputed.
    """
    fn = handlers.substitute(handlers.seed(parameters, rng_seed=0), data=draw)
    return fn()


def _per_draw_terms(draw, *, parameters, curves, common, bundles):
    """Per-unit log-likelihoods and binned means for one posterior draw.

    The site curves are solved once and reused by every stream of every bundle,
    so scoring the training rows alongside the held-out ones is nearly free.
    """
    params = _params_from_draw(parameters, draw)
    ages = asfloat(GRIFFIN_AGE_GRID)
    ft_site = draw["ft_study"][common["site_to_study"]]
    site_pair = site_curves(
        params=params,
        curves=curves,
        log_eir=draw["log_eir"],
        ft_site=ft_site,
        ages=ages,
        prop=common["prop_site"],
        mu=common["mu_site"],
    )

    u_nodes, w_nodes = study_effect_nodes(draw["sigma_c"], draw["sigma_p"])
    theta_prev = 1.0 / jnp.clip(draw["alpha_p"], 1e-8)
    n_cells = common["n_cells"]

    out = {}
    for tag, bundle in bundles.items():
        binned = bin_streams(
            site_curve_pair=site_pair,
            prop=common["prop_site"],
            ages=ages,
            streams={
                "inc": ("inc", bundle["inc"]["site"], bundle["inc"]["age0"], bundle["inc"]["age1"]),
                "prev_m": ("prev_m", bundle["prev_m"]["site"], bundle["prev_m"]["age0"], bundle["prev_m"]["age1"]),
            },
        )
        terms = {"mean": binned}
        inc = bundle["inc"]
        cell_id = inc["site"] * common["n_age_groups"] + inc["agek"]
        terms["loglik_inc"] = mix_over_nodes(
            incidence_cell_loglik_nodes(
                mu_inc=binned["inc"], numer=inc["numer"], denom=inc["denom"], case_det=inc["case_det"],
                cell_id=cell_id, n_cells=n_cells, rW=draw["rW"], rP=draw["rP"],
                alpha_c=draw["alpha_c"], u_nodes=u_nodes,
            )
        )
        for stream in PREVALENCE_STREAMS:
            b = bundle[stream]
            terms[f"loglik_{stream}"] = mix_over_nodes(
                prevalence_row_loglik_nodes(
                    p=binned[stream], numer=b["numer"], denom=b["denom"],
                    theta_prev=theta_prev, w_nodes=w_nodes,
                )
            )
        out[tag] = terms
    return out


def _pointwise_elpd(loglik: np.ndarray) -> np.ndarray:
    """``log (1/S) sum_s p(y_unit | theta_s)``, one entry per scoring unit."""
    if loglik.size == 0:
        return np.zeros(0)
    s = loglik.shape[0]
    return np.asarray(logsumexp(jnp.asarray(loglik), axis=0) - np.log(s))


def _elpd_se(pointwise: np.ndarray) -> float:
    """Standard error of an elpd total: ``sqrt(n) * sd`` over the units.

    Says nothing about which studies were drawn into the fold, which dominates a
    small study split.
    """
    n = pointwise.size
    if n < 2:
        return float("nan")
    return float(np.sqrt(n) * np.std(pointwise, ddof=1))


def _nonempty_cells(cell_id: np.ndarray, n_cells: int) -> np.ndarray:
    """Indices of the cells that actually carry observations.

    Empty cells contribute zero at every quadrature node, which mixes to
    ``log sum_g w_g`` = 0; including them would inflate the unit count without
    changing the total, so they are dropped from the reported per-unit arrays.
    """
    if cell_id.size == 0:
        return np.zeros(0, dtype=int)
    return np.unique(np.asarray(cell_id, dtype=int))


def _simulate(key, draws, terms, bundle, common, n_replicates: int):
    """Predictive replicates of every stream, ``(n_rep, n_obs)`` per stream."""
    n_draws = int(next(iter(draws.values())).shape[0])
    idx = np.linspace(0, n_draws - 1, min(n_replicates, n_draws)).astype(int)
    keys = jax.random.split(key, len(idx))

    reps = {name: [] for name in STREAMS}
    for j, i in enumerate(idx):
        k_inc, k_prev = jax.random.split(keys[j], 2)
        inc = bundle["inc"]
        reps["inc"].append(
            simulate_incidence(
                k_inc, mu_inc=terms["mean"]["inc"][i], denom=inc["denom"], case_det=inc["case_det"],
                cell_id=inc["site"] * common["n_age_groups"] + inc["agek"], study_idx=inc["study"],
                n_cells=common["n_cells"], n_study=common["n_study"],
                rW=draws["rW"][i], rP=draws["rP"][i],
                sigma_c=draws["sigma_c"][i], alpha_c=draws["alpha_c"][i],
            )
        )
        for stream, k in zip(PREVALENCE_STREAMS, jax.random.split(k_prev, len(PREVALENCE_STREAMS))):
            b = bundle[stream]
            reps[stream].append(
                simulate_prevalence(
                    k, p=terms["mean"][stream][i], denom=b["denom"], study_idx=b["study"],
                    n_study=common["n_study"], sigma_p=draws["sigma_p"][i],
                    theta_prev=1.0 / np.clip(draws["alpha_p"][i], 1e-8, None),
                )
            )
    return {k: np.asarray(jnp.stack(v)) if len(v) else np.zeros((0, 0)) for k, v in reps.items()}


def _ppc_figure(reps: dict, bundle: dict, title: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    streams = [(s, bundle[s]) for s in STREAMS if bundle[s]["numer"].shape[0] > 0]
    fig, axes = plt.subplots(1, max(len(streams), 1), figsize=(4.5 * max(len(streams), 1), 4.0), squeeze=False)
    for ax, (name, b) in zip(axes[0], streams):
        obs = np.asarray(b["numer"], dtype=float)
        rep = reps[name]
        lo = np.quantile(rep, 0.05, axis=0)
        hi = np.quantile(rep, 0.95, axis=0)
        med = np.quantile(rep, 0.5, axis=0)
        order = np.argsort(obs)
        x = np.arange(len(obs))
        ax.fill_between(x, lo[order], hi[order], alpha=0.3, label="90% predictive", color="tab:blue")
        ax.plot(x, med[order], lw=1.0, color="tab:blue", label="predictive median")
        ax.plot(x, obs[order], ".", ms=3, color="black", label="observed")
        ax.set_yscale("symlog")
        ax.set_title(name)
        ax.set_xlabel("observation (sorted by observed count)")
    axes[0][0].set_ylabel("count")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    return _to_png(fig)


def _calibration_figure(pits: dict, title: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = [(k, v) for k, v in pits.items() if len(v)]
    fig, axes = plt.subplots(1, max(len(items), 1), figsize=(4.0 * max(len(items), 1), 3.5), squeeze=False)
    for ax, (name, pit) in zip(axes[0], items):
        ax.hist(pit, bins=20, range=(0, 1), density=True, color="tab:blue", alpha=0.75)
        ax.axhline(1.0, color="black", lw=1.0, ls="--")
        ax.set_title(f"{name} (n={len(pit)})")
        ax.set_xlabel("randomised PIT")
    axes[0][0].set_ylabel("density")
    fig.suptitle(title)
    fig.tight_layout()
    return _to_png(fig)


def _to_png(fig) -> bytes:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return buf.getvalue()


def _tail_ess(by_chain: dict[str, np.ndarray]) -> float:
    """The worst tail effective sample size over every scalar coordinate.

    Bulk ``n_eff`` -- what :func:`numpyro.diagnostics.summary` reports -- says
    nothing about the quantiles a predictive interval is read off, and a
    coverage number computed from chains whose tails have not mixed is not the
    coverage of the posterior. Reported alongside bulk rather than instead of
    it: the two fail for different reasons.
    """
    import arviz as az

    worst = np.inf
    for value in by_chain.values():
        draws = np.asarray(value)
        # (chain, draw, *event) -> one series per scalar coordinate.
        flat = draws.reshape(draws.shape[0], draws.shape[1], -1)
        for i in range(flat.shape[2]):
            with np.errstate(invalid="ignore"):
                ess = float(az.ess(flat[:, :, i], method="tail"))
            if np.isfinite(ess):
                worst = min(worst, ess)
    return float(worst) if np.isfinite(worst) else float("nan")


# fit_ok thresholds. Deliberately conventional rather than tuned: this is the
# "is this fit worth reading a score off" gate, not the analysis.
MAX_R_HAT = 1.01
MIN_ESS_TAIL = 100.0
#: Above this a variational approximation is not usable, importance weights or
#: not (Yao et al.; the same 0.7 the PSIS-LOO literature uses).
MAX_KHAT = 0.7


def fit_diagnostics(result: FitResult) -> dict[str, float]:
    """Name and threshold what a :class:`msinf.fit.FitResult` carries.

    The keys are backend-specific, because the quantities are: ``r_hat`` and
    ``ess`` describe a Markov chain and mean nothing for a variational fit,
    ``khat`` and the ELBO mean nothing for a chain. Emitting a union with holes
    in it would let a missing key read as a passing one.

    Two keys are always present and are the ones a caller can rely on without
    knowing the backend: ``fit_seconds``, and ``fit_ok`` -- one gate, computed
    from whichever diagnostics actually apply. ``info['fit_backend']`` says
    which set the rest came from.
    """
    out: dict[str, float] = {"fit_seconds": result.seconds}

    if result.draws_by_chain is not None:
        stats = summary(result.draws_by_chain, prob=0.9)
        rhat = [float(np.nanmax(v["r_hat"])) for v in stats.values() if "r_hat" in v]
        ess = [float(np.nanmin(v["n_eff"])) for v in stats.values() if "n_eff" in v]
        if rhat:
            out["max_r_hat"] = float(np.nanmax(rhat))
        if ess:
            out["min_ess"] = float(np.nanmin(ess))
        out["min_ess_tail"] = _tail_ess(result.draws_by_chain)
        if result.diverging is not None:
            out["n_divergent"] = float(np.sum(result.diverging))
        # Kept under its old name as well as under fit_seconds: it is what
        # discovery's report column reads, and for a Pathfinder-seeded fit the
        # two differ by the time the approximation took.
        out["mcmc_seconds"] = float(result.extra.get("mcmc_seconds", result.seconds))
        out["fit_ok"] = float(
            out.get("max_r_hat", np.inf) <= MAX_R_HAT
            and out.get("min_ess_tail", 0.0) >= MIN_ESS_TAIL
            and out.get("n_divergent", 0.0) == 0.0
        )
    else:
        # Every variational backend sets both.
        assert result.khat is not None and result.elbo is not None
        out["khat"] = float(result.khat)
        out["elbo"] = float(result.elbo)
        if result.dim is not None:
            out["vi_dim"] = float(result.dim)
        if result.elbo_trace is not None:
            out["vi_steps"] = float(len(result.elbo_trace))
        out["fit_ok"] = float(result.khat <= MAX_KHAT and np.isfinite(result.elbo))

    # Set whichever backend ran, when a Pathfinder approximation was involved:
    # as the posterior it is the fit, and as an initialiser it is a thing that
    # can silently be bad while the chains that started from it look fine.
    if result.pathfinder_seconds is not None:
        assert result.pathfinder_khat is not None and result.pathfinder_elbo is not None
        out["pathfinder_seconds"] = float(result.pathfinder_seconds)
        out["pathfinder_khat"] = float(result.pathfinder_khat)
        out["pathfinder_elbo"] = float(result.pathfinder_elbo)
    return out


def evaluate(
    cfg: NumpyroConfig,
    *,
    provider=None,
    split: Split | None = None,
    n_score_draws: int | None = 200,
    n_replicates: int = 200,
    seed: int = 0,
    figures: bool = True,
) -> EvaluationResult:
    """Fit ``cfg`` under ``split`` and score it.

    ``provider`` is a :class:`msinf.curves.CurveProvider`; ``None`` fits the
    incumbent Griffin/dmeq path resolved from ``cfg``. ``n_score_draws`` thins
    the posterior before the (ODE-bound) scoring pass; ``None`` uses every draw.
    """
    model_kwargs, active_core_params, _ = build_model_kwargs(cfg, provider=provider, split=split)
    result = fit(cfg, model_kwargs, active_core_params)

    sampler_kwargs = model_kwargs_for_sampler(model_kwargs)
    common = {
        "site_to_study": sampler_kwargs["site_to_study"],
        "prop_site": sampler_kwargs["prop_site"],
        "mu_site": sampler_kwargs["mu_site"],
        "n_study": sampler_kwargs["n_study"],
        "n_age_groups": sampler_kwargs["n_age_groups"],
        "n_cells": sampler_kwargs["n_site"] * sampler_kwargs["n_age_groups"],
    }
    bundles = {"train": _observation_bundle(sampler_kwargs)}
    if "holdout" in model_kwargs:
        bundles["holdout"] = _observation_bundle(model_kwargs["holdout"])

    draws = _thin(result.draws, n_score_draws)
    scored = jax.lax.map(
        lambda d: _per_draw_terms(
            d,
            parameters=sampler_kwargs["parameters"],
            curves=sampler_kwargs["curves"],
            common=common,
            bundles=bundles,
        ),
        {k: jnp.asarray(v) for k, v in draws.items()},
    )

    metrics: dict[str, float] = {}
    info: dict = {
        "split": (split.label() if split is not None else "none"),
        "fit_backend": result.backend,
        "init_strategy": str(cfg.init_strategy).strip().lower(),
    }
    figs: dict[str, bytes] = {}
    pointwise: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(seed)

    for tag, terms in scored.items():
        bundle = bundles[tag]
        keep = _nonempty_cells(np.asarray(bundle["inc"]["site"]) * common["n_age_groups"]
                               + np.asarray(bundle["inc"]["agek"]), common["n_cells"])
        pw_inc = _pointwise_elpd(np.asarray(terms["loglik_inc"])[:, keep])
        elpd_inc = float(np.sum(pw_inc))
        pw_prev = np.concatenate(
            [_pointwise_elpd(np.asarray(terms[f"loglik_{stream}"])) for stream in PREVALENCE_STREAMS]
        )
        elpd_prev = float(np.sum(pw_prev))
        n_prev = int(pw_prev.size)
        pointwise[f"{tag}_incidence"] = pw_inc
        pointwise[f"{tag}_prevalence"] = pw_prev
        metrics[f"elpd_{tag}_incidence"] = elpd_inc
        metrics[f"elpd_{tag}_prevalence"] = elpd_prev
        metrics[f"elpd_{tag}"] = elpd_inc + elpd_prev
        metrics[f"elpd_{tag}_se"] = _elpd_se(np.concatenate([pw_inc, pw_prev]))
        info[f"n_{tag}_incidence_cells"] = len(keep)
        info[f"n_{tag}_prevalence_rows"] = n_prev
        if len(keep):
            metrics[f"elpd_{tag}_incidence_per_cell"] = elpd_inc / len(keep)
        if n_prev:
            metrics[f"elpd_{tag}_prevalence_per_row"] = elpd_prev / n_prev

        reps = _simulate(jax.random.PRNGKey(seed), draws, terms, bundle, common, n_replicates)
        pits = {}
        for stream in STREAMS:
            obs = np.asarray(bundle[stream]["numer"])
            if obs.shape[0] == 0 or reps[stream].size == 0:
                continue
            pits[stream] = randomised_pit(reps[stream], obs, rng)
            metrics[f"coverage90_{tag}_{stream}"] = coverage(reps[stream], obs, level=0.9)
        if figures and pits:
            figs[f"ppc_{tag}.png"] = _ppc_figure(reps, bundle, f"posterior predictive ({tag}, {info['split']})")
            figs[f"calibration_{tag}.png"] = _calibration_figure(
                pits, f"randomised PIT ({tag}, {info['split']})"
            )

    metrics.update(fit_diagnostics(result))
    return EvaluationResult(
        metrics=metrics, figures=figs, info=info, draws=draws, pointwise=pointwise
    )
