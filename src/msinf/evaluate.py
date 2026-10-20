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
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from arviz.stats import psislw
from jax.scipy.special import logsumexp
from numpyro import handlers
from numpyro.diagnostics import summary

from .age_grid import GRIFFIN_AGE_GRID
from .config import NumpyroConfig
from .data import encode_canonical_sites, load_observed_data
from .dtypes import asfloat
from .fit import FitResult, fit
from .likelihood import (
    GH_WEIGHTS,
    case_detection_scaling,
    incidence_cell_loglik_nodes,
    mix_over_nodes,
    prevalence_row_loglik_nodes,
    study_effect_nodes,
)
from .model import (
    TYPE_INCIDENCE,
    TYPE_PREV_MICROSCOPY,
    bin_streams,
    build_model_kwargs,
    model_kwargs_for_sampler,
    site_curves,
)
from .predictive import coverage, randomised_pit, simulate_incidence, simulate_prevalence
from .splits import Split, apply_split

# Streams whose likelihood unit is the observation row. Microscopy prevalence
# is the only one: incidence is scored per cell, and there is no PCR stream.
PREVALENCE_STREAMS = ("prev_m",)
# Every stream, in report order.
STREAMS = ("inc",) + PREVALENCE_STREAMS

#: How the chains of one fit are combined into the predictive that is scored.
#:
#: ``pool`` weights every draw equally, which is the posterior only if the chains
#: mixed. ``stack`` weights each chain by stacking its leave-one-out predictive
#: on the training units (Yao, Vehtari & Gelman 2022, "Stacking for non-mixing
#: Bayesian computations"). This model's posterior has modes NUTS does not cross
#: -- the Doneguebougou EIR sits near 50 or near 320, each with its own immunity
#: parameters -- and a pooled score is then a mixture weighted by how many chains
#: happened to start in each basin, while a pooled PSIS-LOO mixes importance
#: weights across modes and falls below every chain's own. Stacking leaves each
#: unit out within each chain, gives no weight to a chain stuck where the training
#: rows reject it, and never looks at the held-out rows. It does not make a score
#: independent of which chains ran -- per-chain LOO is too noisy for that -- and
#: the ``_mcse`` metrics say how much that is.
CHAIN_COMBINATIONS = ("pool", "stack")

#: Dirichlet concentration on the stacking weights. Above 1 it breaks ties
#: between chains that sit in the same mode -- predictively interchangeable, so
#: plain stacking picks one of them arbitrarily and throws the others' draws away
#: -- toward sharing the weight, without keeping a mode the training rows reject.
STACK_CONCENTRATION = 1.5


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
    #: each of ``draws``' weight in the scored predictive, summing to one: equal
    #: under ``pool``, the chain's stacking weight shared over its draws under ``stack``
    draw_weights: np.ndarray | None = None
    #: per-unit elpd, keyed ``"<tag>_<stream>"`` -- one entry per non-empty
    #: incidence cell or prevalence row. The order is a property of the split
    #: and the table, so two candidates under the same split line up elementwise.
    #: ``loo_<stream>`` is the PSIS-LOO estimate on the training units.
    pointwise: dict[str, np.ndarray] = field(default_factory=dict)
    #: the fit itself, so a caller can archive the chains without fitting twice
    fit_result: FitResult | None = None
    #: per fold (``train``, ``holdout``), every observed row beside what the fit
    #: predicts for it -- see :func:`_fitted_frame`
    fitted: dict[str, pd.DataFrame] = field(default_factory=dict)


def _observation_bundle(k: dict) -> dict:
    """Regroup the flat ``<field>_<stream>`` model kwargs into one dict per stream."""
    return {
        stream: {key.removesuffix(f"_{stream}"): v for key, v in k.items() if key.endswith(f"_{stream}")}
        for stream in STREAMS
    }


def _n_chains(result: FitResult) -> int:
    """How many chains ``result.draws`` was flattened from; one for a variational fit."""
    if result.draws_by_chain is None:
        return 1
    return int(np.asarray(next(iter(result.draws_by_chain.values()))).shape[0])


def _score_draws(result: FitResult, max_draws: int | None) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The draws to score, thinned evenly *within* each chain, and each one's chain.

    Within each chain rather than across the flattened array, so every chain
    keeps the same number of draws and a per-chain leave-one-out can be taken
    from each. ``result.draws`` is chain-major (see :func:`msinf.fit._flatten_chains`).
    """
    n, n_chains = result.n_draws, _n_chains(result)
    per_chain = n // n_chains
    idx = np.arange(n)
    if max_draws is not None and n > max_draws:
        keep = max(1, max_draws // n_chains)
        within = np.linspace(0, per_chain - 1, keep).astype(int)
        idx = np.concatenate([c * per_chain + within for c in range(n_chains)])
    chain = idx // per_chain
    return {k: np.asarray(v)[idx] for k, v in result.draws.items()}, chain


def _params_from_draw(parameters, draw: dict):
    """Re-run a candidate's ``parameters()`` with a posterior draw substituted.

    Works for any provider without adding to the contract: the draw's sample
    sites are substituted in and everything downstream of them is recomputed.
    """
    fn = handlers.substitute(handlers.seed(parameters, rng_seed=0), data=draw)
    return fn()


def _conditional(nodes, unit_study, n_study: int):
    """Each unit's density given the rest of its study, and the study effect's posterior.

    ``nodes`` is ``(n_units, G)``: a unit's log-likelihood at each quadrature
    node of its study's effect. The fitted likelihood shares that one effect
    across every unit of a study and integrates it out once, over all of them, so
    a unit's own contribution to the posterior is not its marginal ``p(y_i |
    theta)`` but ``p(y_i | y_{study,-i}, theta) = p(y_study | theta) /
    p(y_{study,-i} | theta)``. That is the ratio that removes one unit, so it is
    what leave-one-out has to weight by (Burkner, Gabry & Vehtari 2021, on LOO
    for non-factorised models). A unit alone in its study gets its marginal.

    Also returns ``(n_study, G)`` log posterior weights over the nodes given the
    whole study, for a fitted value that carries the study's own effect.
    """
    log_w = jnp.log(asfloat(GH_WEIGHTS))
    total = jax.ops.segment_sum(nodes, unit_study, num_segments=n_study)
    whole = logsumexp(log_w + total, axis=-1)
    rest = logsumexp(log_w + total[unit_study] - nodes, axis=-1)
    return whole[unit_study] - rest, log_w + total - whole[:, None]


def _per_draw_terms(draw, *, parameters, curves, common, bundles):
    """Per-unit log-likelihoods and binned means for one posterior draw.

    The site curves are solved once and reused by every stream of every bundle,
    so scoring the training rows alongside the held-out ones is nearly free.

    ``loglik_<stream>`` is each unit's marginal, its study effect integrated out
    on its own -- the density of a unit from a study not yet seen, which is what
    a held-out unit is. ``cond_<stream>`` is the same unit given the rest of its
    study (see :func:`_conditional`), which is what leave-one-out on the
    training units needs. ``fit_<stream>`` is each row's expected observed rate
    with its study's effect at its posterior given that study's rows.
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
        nodes = incidence_cell_loglik_nodes(
            mu_inc=binned["inc"], numer=inc["numer"], denom=inc["denom"], case_det=inc["case_det"],
            cell_id=cell_id, n_cells=n_cells, rW=draw["rW"], rP=draw["rP"],
            alpha_c=draw["alpha_c"], u_nodes=u_nodes,
        )
        terms["loglik_inc"] = mix_over_nodes(nodes)
        terms["cond_inc"], log_post = _conditional(nodes, common["cell_to_study"], common["n_study"])
        # E[e^u | the study's rows]: the study effect the fit gave this row's study
        shift = jnp.sum(jnp.exp(log_post + u_nodes[None, :]), axis=-1)
        terms["fit_inc"] = (
            case_detection_scaling(inc["case_det"], draw["rW"], draw["rP"])
            * 365.0
            * jnp.clip(binned["inc"], 1e-12)
            * shift[inc["study"]]
        )
        for stream in PREVALENCE_STREAMS:
            b = bundle[stream]
            nodes = prevalence_row_loglik_nodes(
                p=binned[stream], numer=b["numer"], denom=b["denom"],
                theta_prev=theta_prev, w_nodes=w_nodes,
            )
            terms[f"loglik_{stream}"] = mix_over_nodes(nodes)
            terms[f"cond_{stream}"], log_post = _conditional(nodes, b["study"], common["n_study"])
            logit_p = jax.scipy.special.logit(jnp.clip(binned[stream], 1e-6, 1 - 1e-6))
            q = jax.nn.sigmoid(logit_p[:, None] + w_nodes[None, :])
            terms[f"fit_{stream}"] = jnp.sum(jnp.exp(log_post[b["study"]]) * q, axis=-1)
        out[tag] = terms
    return out


def _pointwise_elpd(loglik: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """``log sum_s w_s p(y_unit | theta_s)``, one entry per scoring unit.

    ``weights`` are the draws' weights in the predictive and sum to one; ``None``
    is ``1/S`` each, the plain posterior mean.
    """
    if loglik.size == 0:
        return np.zeros(0)
    s = loglik.shape[0]
    log_w = np.full(s, -np.log(s)) if weights is None else np.log(np.asarray(weights, dtype=float))
    return np.asarray(logsumexp(jnp.asarray(loglik) + jnp.asarray(log_w)[:, None], axis=0))


def _psis_loo(loglik: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PSIS-LOO per unit and its Pareto k, from ``(S, n_units)`` log-likelihoods.

    The in-sample ``elpd_train`` rewards every extra parameter; leaving each unit
    out in turn does not, which is what makes it the training score worth ranking
    on when the held-out fold must stay unseen.
    """
    if loglik.size == 0:
        return np.zeros(0), np.zeros(0)
    ll = np.asarray(loglik, dtype=float).T
    log_weights, khat = psislw(-ll.copy())
    return np.asarray(logsumexp(jnp.asarray(log_weights + ll), axis=1)), np.asarray(khat)


def _chain_loo(loglik: np.ndarray, chain: np.ndarray, n_chains: int) -> tuple[np.ndarray, np.ndarray]:
    """PSIS-LOO taken within each chain: ``(n_chains, n_units)`` estimates and Pareto k.

    Within, because importance weights computed across chains that sit in
    different modes are dominated by whichever mode fits the left-out unit
    worst: pooled, the incumbent's LOO comes out below every one of its chains'.
    """
    found = [_psis_loo(np.asarray(loglik)[chain == c]) for c in range(n_chains)]
    return np.stack([loo for loo, _ in found]), np.stack([k for _, k in found])


def _stack_weights(lpd: np.ndarray, concentration: float = STACK_CONCENTRATION, tol: float = 1e-10) -> np.ndarray:
    """Weights over chains maximising ``sum_i log sum_c w_c exp(lpd[c, i])``.

    ``lpd`` is ``(n_chains, n_units)`` leave-one-out log predictive densities.
    A Dirichlet(``concentration``) prior on the weights is added to the
    objective, which stays concave, and the maximum is found by the EM
    iteration for mixture weights: responsibilities, then weights in proportion
    to their sums plus ``concentration - 1``.
    """
    n_chains = lpd.shape[0]
    if n_chains == 1:
        return np.ones(1)
    lpd = np.asarray(lpd, dtype=float)
    lpd = lpd - lpd.max(axis=0, keepdims=True)  # per unit, so no column underflows
    density = np.exp(lpd)
    weights = np.full(n_chains, 1.0 / n_chains)
    prior = max(concentration - 1.0, 0.0)
    for _ in range(100_000):
        mixed = weights[:, None] * density
        responsibility = mixed / mixed.sum(axis=0, keepdims=True)
        updated = responsibility.sum(axis=1) + prior
        updated /= updated.sum()
        if np.max(np.abs(updated - weights)) < tol:
            return updated
        weights = updated
    return weights


def _replicate_index(weights: np.ndarray, n: int) -> np.ndarray:
    """``n`` draw indices taken in proportion to ``weights``, by systematic resampling.

    Equal weights give evenly spaced draws, which is what equal-weight scoring
    always used; unequal ones give each chain its share of the replicates.
    """
    cumulative = np.cumsum(np.asarray(weights, dtype=float))
    cumulative /= cumulative[-1]
    positions = (np.arange(n) + 0.5) / n
    return np.minimum(np.searchsorted(cumulative, positions), len(cumulative) - 1)


def _fitted_frame(
    frame: pd.DataFrame, reps: dict, pits: dict, terms: dict, n_age_groups: int, loo=None, weights=None
):
    """One fold's observed rows beside the posterior predictive for each.

    ``pred_q05/q50/q95`` are replicate counts over ``denom``, so they are in the
    units of ``observed`` -- prevalence, or episodes per person-year -- with the
    case-detection scaling and freshly drawn study effects already in them: the
    predictive a held-out study faces. ``fitted`` is the posterior mean of the
    row's expected rate with its *own* study's effect, as the fit inferred it
    from that study's rows, so ``observed / fitted`` is the misfit the mechanism
    is responsible for once the study's overall level has been absorbed.
    ``pit`` near 0 or 1 is a row in the tail of its predictive. Incidence
    ``elpd`` (and ``loo``) is the row's *cell*, which is the unit msinf scores.
    Rows are in the order the likelihood sees them, streams one after the other.
    ``weights`` are the scored draws' weights, as for :func:`_pointwise_elpd`.
    """
    parts = []
    for stream, code in (("inc", TYPE_INCIDENCE), ("prev_m", TYPE_PREV_MICROSCOPY)):
        rows = frame[frame["type"] == code].copy()
        if rows.empty:
            continue
        denom = rows["denom"].to_numpy(dtype=float)
        rows["stream"] = "incidence" if stream == "inc" else "prevalence"
        rows["observed"] = rows["numer"].to_numpy(dtype=float) / denom
        rate = np.asarray(reps[stream], dtype=float) / denom
        for q in (5, 50, 95):
            rows[f"pred_q{q:02d}"] = np.quantile(rate, q / 100, axis=0)
        if f"fit_{stream}" in terms:
            fit = np.asarray(terms[f"fit_{stream}"], dtype=float)
            w = np.full(fit.shape[0], 1.0 / fit.shape[0]) if weights is None else np.asarray(weights, dtype=float)
            rows["fitted"] = w @ fit
        rows["pit"] = pits.get(stream, np.full(len(rows), np.nan))
        per_unit = _pointwise_elpd(np.asarray(terms[f"loglik_{stream}"]), weights)
        unit = np.arange(len(rows))
        if stream == "inc":
            unit = rows["site_i"].to_numpy(dtype=int) * n_age_groups + rows["age_bracket"].to_numpy(dtype=int) - 1
        rows["elpd"] = per_unit[unit]
        if loo is not None:
            rows["loo"] = loo[stream][unit]
        rows["elpd_unit"] = "cell" if stream == "inc" else "row"
        parts.append(rows)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


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


def _simulate(key, draws, terms, bundle, common, idx: np.ndarray):
    """Predictive replicates of every stream, ``(n_rep, n_obs)`` per stream, one per draw in ``idx``."""
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
    combine_chains: str = "pool",
) -> EvaluationResult:
    """Fit ``cfg`` under ``split`` and score it.

    ``provider`` is a :class:`msinf.curves.CurveProvider`; ``None`` fits the
    incumbent Griffin/dmeq path resolved from ``cfg``. The rest is :func:`score`'s.
    """
    model_kwargs, active_core_params, _ = build_model_kwargs(cfg, provider=provider, split=split)
    result = fit(cfg, model_kwargs, active_core_params)
    return score(
        cfg,
        result,
        model_kwargs,
        split=split,
        n_score_draws=n_score_draws,
        n_replicates=n_replicates,
        seed=seed,
        figures=figures,
        combine_chains=combine_chains,
    )


def score(
    cfg: NumpyroConfig,
    result: FitResult,
    model_kwargs: dict,
    *,
    split: Split | None = None,
    n_score_draws: int | None = 200,
    n_replicates: int = 200,
    seed: int = 0,
    figures: bool = True,
    combine_chains: str = "pool",
) -> EvaluationResult:
    """Score a fit that has already run, on the folds ``model_kwargs`` was built with.

    Apart from :func:`evaluate` so that archived chains can be re-scored without
    being refitted. ``n_score_draws`` thins the posterior, evenly within each
    chain, before the (ODE-bound) scoring pass; ``None`` uses every draw.
    ``combine_chains`` is one of :data:`CHAIN_COMBINATIONS`: every score, the
    replicates behind the coverage, the PIT and the fitted frames all come from
    the same weighted predictive.

    ``elpd_loo_train`` leaves out one unit at a time *given the rest of its
    study* (see :func:`_conditional`), which is the leave-one-out of this
    likelihood. It is therefore larger than ``elpd_train``, which, like
    ``elpd_holdout``, scores each unit as if from a study not yet seen.
    """
    if combine_chains not in CHAIN_COMBINATIONS:
        raise ValueError(f"combine_chains must be one of {CHAIN_COMBINATIONS}, got {combine_chains!r}")
    sampler_kwargs = model_kwargs_for_sampler(model_kwargs)
    n_cells = sampler_kwargs["n_site"] * sampler_kwargs["n_age_groups"]
    common = {
        "site_to_study": sampler_kwargs["site_to_study"],
        "prop_site": sampler_kwargs["prop_site"],
        "mu_site": sampler_kwargs["mu_site"],
        "n_study": sampler_kwargs["n_study"],
        "n_age_groups": sampler_kwargs["n_age_groups"],
        "n_cells": n_cells,
        "cell_to_study": jnp.asarray(sampler_kwargs["site_to_study"])[
            jnp.arange(n_cells) // sampler_kwargs["n_age_groups"]
        ],
    }
    bundles = {"train": _observation_bundle(sampler_kwargs)}
    if "holdout" in model_kwargs:
        bundles["holdout"] = _observation_bundle(model_kwargs["holdout"])

    draws, chain = _score_draws(result, n_score_draws)
    n_chains = _n_chains(result)
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

    def kept(bundle) -> np.ndarray:
        return _nonempty_cells(
            np.asarray(bundle["inc"]["site"]) * common["n_age_groups"] + np.asarray(bundle["inc"]["agek"]),
            n_cells,
        )

    # Every training unit's density given the rest of its study, cells then
    # rows: what leave-one-out weights by, and what the chains are stacked on.
    keep_train = kept(bundles["train"])
    conditional = np.concatenate(
        [np.asarray(scored["train"]["cond_inc"])[:, keep_train]]
        + [np.asarray(scored["train"][f"cond_{stream}"]) for stream in PREVALENCE_STREAMS],
        axis=1,
    )
    chain_weights = np.full(n_chains, 1.0 / n_chains)
    if combine_chains == "stack" and n_chains > 1:
        chain_loo, chain_khat = _chain_loo(conditional, chain, n_chains)
        chain_weights = _stack_weights(chain_loo)

    def weights_of(subset: np.ndarray) -> np.ndarray:
        """The chain weights ``subset`` alone would have been given."""
        if combine_chains == "stack" and len(subset) > 1:
            return _stack_weights(chain_loo[subset])
        return np.full(len(subset), 1.0 / len(subset))

    def loo_of(subset: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Training leave-one-out from ``subset``'s chains, per unit."""
        if combine_chains == "stack" and n_chains > 1:
            return np.asarray(logsumexp(jnp.asarray(chain_loo[subset]) + jnp.log(weights)[:, None], axis=0))
        return _psis_loo(conditional[np.isin(chain, subset)])[0]

    #: each fold's marginal log-likelihoods, units in report order, filled below
    units: dict[str, np.ndarray] = {}
    draw_weights = chain_weights[chain] / np.bincount(chain, minlength=n_chains)[chain]
    replicates = _replicate_index(draw_weights, min(n_replicates, len(draw_weights)))

    metrics: dict[str, float] = {"effective_chains": float(1.0 / np.sum(chain_weights**2))}
    info: dict = {
        "split": (split.label() if split is not None else "none"),
        "fit_backend": result.backend,
        "init_strategy": str(cfg.init_strategy).strip().lower(),
        "combine_chains": combine_chains,
        "chain_weights": [float(w) for w in chain_weights],
    }
    figs: dict[str, bytes] = {}
    pointwise: dict[str, np.ndarray] = {}
    fitted: dict[str, pd.DataFrame] = {}
    rng = np.random.default_rng(seed)
    # The rows behind each bundle, re-read the way `build_model_kwargs` reads
    # them, so a fitted value can be put beside the study, site and age it is for.
    frame, _ = encode_canonical_sites(load_observed_data(Path(cfg.data_csv)))
    frames = dict(zip(("train", "holdout"), apply_split(frame, split)))

    for tag, terms in scored.items():
        bundle = bundles[tag]
        keep = kept(bundle)
        streams = [np.asarray(terms["loglik_inc"])[:, keep]] + [
            np.asarray(terms[f"loglik_{stream}"]) for stream in PREVALENCE_STREAMS
        ]
        units[tag] = np.concatenate(streams, axis=1)
        pw_inc = _pointwise_elpd(streams[0], draw_weights)
        elpd_inc = float(np.sum(pw_inc))
        pw_prev = np.concatenate([_pointwise_elpd(s, draw_weights) for s in streams[1:]])
        elpd_prev = float(np.sum(pw_prev))
        n_prev = int(pw_prev.size)
        pointwise[f"{tag}_incidence"] = pw_inc
        pointwise[f"{tag}_prevalence"] = pw_prev
        metrics[f"elpd_{tag}_incidence"] = elpd_inc
        metrics[f"elpd_{tag}_prevalence"] = elpd_prev
        metrics[f"elpd_{tag}"] = elpd_inc + elpd_prev
        metrics[f"elpd_{tag}_se"] = _elpd_se(np.concatenate([pw_inc, pw_prev]))
        if combine_chains != "pool":
            # What equal weight per draw would have scored, so the effect of
            # weighting the chains is on the record beside the weighted number.
            metrics[f"elpd_{tag}_pooled"] = float(sum(np.sum(_pointwise_elpd(s)) for s in streams))
        info[f"n_{tag}_incidence_cells"] = len(keep)
        info[f"n_{tag}_prevalence_rows"] = n_prev
        if len(keep):
            metrics[f"elpd_{tag}_incidence_per_cell"] = elpd_inc / len(keep)
        if n_prev:
            metrics[f"elpd_{tag}_prevalence_per_row"] = elpd_prev / n_prev

        reps = _simulate(jax.random.PRNGKey(seed), draws, terms, bundle, common, replicates)
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

        # Leave-one-out on the rows the fit saw, and only those: a held-out
        # unit is already out, and its elpd is the number LOO approximates.
        loo = None
        if tag == "train":
            pooled, pooled_khat = _psis_loo(conditional)
            if combine_chains == "stack" and n_chains > 1:
                # Each chain's own leave-one-out, mixed with the chains'
                # weights; a unit's k is its chains' k averaged the same way,
                # not their worst: each is estimated from one chain's draws and
                # the largest of eight noisy estimates is biased high.
                both = loo_of(np.arange(n_chains), chain_weights)
                khat = chain_weights @ chain_khat
                metrics["elpd_loo_train_pooled"] = float(np.sum(pooled))
            else:
                both, khat = pooled, pooled_khat
            loo_inc, loo_prev = both[: len(keep)], both[len(keep) :]
            pointwise["loo_incidence"], pointwise["loo_prevalence"] = loo_inc, loo_prev
            metrics["elpd_loo_train"] = float(np.sum(both))
            metrics["elpd_loo_train_se"] = _elpd_se(both)
            metrics["loo_khat_over_0p7"] = float(np.sum(khat > MAX_KHAT))
            per_cell = np.full(n_cells, np.nan)
            per_cell[keep] = loo_inc
            loo = {"inc": per_cell, "prev_m": loo_prev}
        if frames.get(tag) is not None:
            fitted[tag] = _fitted_frame(
                frames[tag], reps, pits, terms, common["n_age_groups"], loo, weights=draw_weights
            )

    # How much each total owes to which chains happened to run. A chain that
    # falls into one mode rather than another moves a score by nats on this
    # model, and the standard errors over units do not see it: they hold the
    # draws fixed. Leave-one-chain-out jackknife, re-weighting each time.
    if n_chains > 1:
        lpd = {tag: np.stack([_pointwise_elpd(ll[chain == c]) for c in range(n_chains)]) for tag, ll in units.items()}

        def totals(subset: np.ndarray) -> dict[str, float]:
            weights = weights_of(subset)
            out = {
                f"elpd_{tag}": float(np.sum(logsumexp(jnp.asarray(by[subset]) + jnp.log(weights)[:, None], axis=0)))
                for tag, by in lpd.items()
            }
            return out | {"elpd_loo_train": float(np.sum(loo_of(subset, weights)))}

        dropped = [totals(np.delete(np.arange(n_chains), c)) for c in range(n_chains)]
        for name in dropped[0]:
            values = np.array([d[name] for d in dropped])
            metrics[f"{name}_mcse"] = float(np.sqrt((n_chains - 1) / n_chains * np.sum((values - values.mean()) ** 2)))

    metrics.update(fit_diagnostics(result))
    return EvaluationResult(
        metrics=metrics, figures=figs, info=info, draws=draws, draw_weights=draw_weights,
        pointwise=pointwise, fit_result=result, fitted=fitted,
    )
