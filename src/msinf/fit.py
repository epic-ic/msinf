"""Fit backends, behind one result type.

Everything downstream of the fit -- the held-out ``elpd``, the pointwise
vectors, the predictive replicates, the PIT -- operates on a plain dictionary of
*constrained* posterior draws, so the backend is the only part that varies.

Three backends:

* ``mcmc`` -- NumPyro MCMC under the configured ``sampler`` (NUTS or SA). The
  reference.
* ``pathfinder`` -- BlackJAX Pathfinder. A quasi-Newton path to a Gaussian
  approximation; cheap enough to be rounding error next to a NUTS fit.
* ``fullrank`` -- BlackJAX full-rank Gaussian VI, optimised with Adam.

and, orthogonally, ``init_strategy='pathfinder'``, which runs Pathfinder first
and seeds each NUTS chain from a draw of the approximation. That is a *NUTS*
fit: it is exact in the limit and its diagnostics mean what they always meant.
Only the starting point changes.

``FitResult`` carries the raw material diagnostics are computed from; naming
and thresholding them is :func:`msinf.evaluate.fit_diagnostics`'s job. Fields a
backend cannot fill stay ``None``.

The variational diagnostic that matters
---------------------------------------
Not the ELBO. The ELBO is not comparable across models, cannot be compared to
anything absolute, and a converged ELBO says only that the optimiser stopped.
The diagnostic is the Pareto-k of the importance ratios ``log p(z) - log q(z)``
(Yao et al., *Yes, but Did It Work?*): under 0.7 the approximation is close
enough for importance weighting to fix what remains, and over it the
variational posterior is wrong in a way no amount of resampling repairs. Both
backends can produce it -- Pathfinder returns ``log q`` alongside its draws, and
full-rank VI exposes ``generate_fullrank_logdensity``.

``khat`` also buys the repair: with ``cfg.vi_psir`` the draws are resampled
against the smoothed weights (PSIR) before anything scores them, which costs one
extra pass and moves a mildly-wrong Gaussian noticeably closer to the posterior.
It is off by default, because the first thing this repo has to do with a
variational fit is compare the *raw* approximation against MCMC.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import jax
import numpy as np
import numpyro
import optax
from arviz.stats import psislw
from numpyro.diagnostics import print_summary
from numpyro.infer import MCMC, NUTS, SA

from .bridge import build_unconstrained
from .compare import BY_CHAIN_KEY
from .config import NumpyroConfig
from .data import SAMPLES_FILENAME
from .model import (
    build_model_kwargs,
    hierarchical_model,
    model_kwargs_for_sampler,
    resolve_init_strategy,
)
from .priors import CORE_PARAMS

#: Backends :func:`fit` dispatches on.
FIT_BACKENDS = ("mcmc", "pathfinder", "fullrank")


@dataclass
class FitResult:
    """A fitted posterior and the raw material its diagnostics are computed from."""

    #: what produced this: the MCMC kernel (``nuts``/``sa``) or the variational backend
    backend: str
    #: constrained posterior draws, ``(S, ...)`` per sample site, chains flattened.
    #: The same keys and the same shapes whatever the backend, which is what lets
    #: :mod:`msinf.evaluate` score any of them with one code path.
    draws: dict[str, np.ndarray]
    #: wall time for the fit, blocked on so it times the work and not the dispatch
    seconds: float
    #: ``(chain, draw, ...)`` draws, for the diagnostics that need chain structure
    #: -- ``r_hat``, bulk and tail ESS. ``None`` for the variational backends,
    #: which have no chains and for which those numbers are undefined.
    draws_by_chain: dict[str, np.ndarray] | None = None
    #: per-draw divergence flags from the HMC family; ``None`` where the kernel
    #: reports none (SA) or the concept does not apply (variational).
    diverging: np.ndarray | None = None
    #: normalised PSIS log weights over ``draws``, variational backends only.
    #: Carried rather than applied so a caller can score the raw approximation,
    #: the importance-weighted one, or both.
    log_weights: np.ndarray | None = None
    #: Pareto k of ``log p(z) - log q(z)``; the variational diagnostic that matters
    khat: float | None = None
    #: ELBO of the fitted approximation, sign-normalised so larger is better
    elbo: float | None = None
    #: ELBO per optimiser step, same sign convention as :attr:`elbo`
    elbo_trace: np.ndarray | None = None
    #: unconstrained dimension the approximation is over
    dim: int | None = None
    #: seconds spent in a Pathfinder run that only *seeded* an MCMC fit, and the
    #: khat/ELBO of that approximation. Set only by ``init_strategy='pathfinder'``.
    pathfinder_seconds: float | None = None
    pathfinder_khat: float | None = None
    pathfinder_elbo: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_draws(self) -> int:
        return int(next(iter(self.draws.values())).shape[0])


def psis_smooth(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    """Pareto-smoothed importance weights and the shape parameter ``khat``.

    ``arviz.stats.psislw`` rather than ``arviz.psislw``: 0.23 stopped exporting
    it at the top level.
    """
    lw = np.asarray(log_weights, dtype=float)
    smoothed, khat = psislw(lw.copy())
    return np.asarray(smoothed), float(np.ravel(khat)[0])


def _resample(draws: dict, log_weights: np.ndarray, *, seed: int) -> dict[str, np.ndarray]:
    """PSIR: draw ``S`` indices with replacement against the smoothed weights."""
    w = np.exp(log_weights - log_weights.max())
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        return draws
    idx = np.random.default_rng(seed).choice(len(w), size=len(w), replace=True, p=w / total)
    return {k: np.asarray(v)[idx] for k, v in draws.items()}


def _divergences(mcmc) -> np.ndarray | None:
    """Per-draw divergence flags, or ``None`` for a kernel that reports none."""
    extra = mcmc.get_extra_fields()
    return np.asarray(extra["diverging"]) if "diverging" in extra else None


def _flatten_chains(by_chain: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: np.asarray(v).reshape((-1,) + np.asarray(v).shape[2:]) for k, v in by_chain.items()}


def pathfinder_approximation(
    unconstrained,
    *,
    seed: int,
    num_draws: int,
    maxiter: int = 30,
    maxcor: int = 10,
):
    """Run Pathfinder and return ``(state, z_draws, log_q, khat, elbo)``.

    ``z_draws`` are unconstrained, which is the form both consumers want: NumPyro
    takes unconstrained ``init_params``, and :meth:`Unconstrained.constrain_batch`
    takes unconstrained draws.
    """
    import blackjax

    k_fit, k_sample = jax.random.split(jax.random.PRNGKey(seed))
    state, _ = blackjax.vi.pathfinder.approximate(
        k_fit,
        unconstrained.logdensity,
        unconstrained.z_init,
        num_samples=num_draws,
        maxiter=maxiter,
        maxcor=maxcor,
    )
    z_draws, log_q = blackjax.vi.pathfinder.sample(k_sample, state, num_draws)
    log_p = jax.vmap(unconstrained.logdensity)(z_draws)
    _, khat = psis_smooth(np.asarray(log_p) - np.asarray(log_q))
    # PathfinderState.elbo is the ELBO proper (unlike FRVIInfo.elbo; see below).
    return state, z_draws, np.asarray(log_q), khat, float(state.elbo)


def fit_mcmc(cfg: NumpyroConfig, model_kwargs: dict, active_core_params: tuple[str, ...]) -> FitResult:
    """A NumPyro MCMC fit, optionally seeded from a Pathfinder approximation.

    ``init_strategy='pathfinder'`` draws one unconstrained point per chain from
    the Pathfinder approximation and passes them as ``init_params``. NumPyro's
    ``init_params`` for a model-based kernel is in the *unconstrained* space
    (``infer/hmc.py:704``) and takes a leading chain dimension, so the draws go
    in with no round trip through the constrained space.
    """
    extra: dict = {}
    init_params = None
    pathfinder_seconds = pathfinder_khat = pathfinder_elbo = None

    if str(cfg.init_strategy).strip().lower() == "pathfinder":
        started = time.perf_counter()
        unconstrained = build_unconstrained(cfg, model_kwargs, active_core_params)
        _, z_draws, _, khat, elbo = pathfinder_approximation(
            unconstrained,
            seed=cfg.seed,
            num_draws=max(cfg.pathfinder_draws, cfg.num_chains),
            maxiter=cfg.pathfinder_maxiter,
            maxcor=cfg.pathfinder_maxcor,
        )
        jax.block_until_ready(z_draws)
        pathfinder_seconds = time.perf_counter() - started
        # One distinct start per chain: four chains from one point cannot
        # disagree about where they started, which is half of what r_hat checks.
        init_params = jax.tree.map(lambda v: v[: cfg.num_chains], z_draws)
        pathfinder_khat, pathfinder_elbo = khat, elbo
        extra["pathfinder_init"] = {k: np.asarray(v) for k, v in init_params.items()}

    started = time.perf_counter()
    mcmc, warmup_samples = run_mcmc(cfg, model_kwargs, active_core_params, init_params=init_params)
    # Blocked on, so the number is the fit and not the dispatch: JAX is
    # asynchronous, and an unblocked timer would charge the wait to whatever ran
    # next. Compilation is included, because a candidate that is slow to compile
    # is slow to fit.
    jax.block_until_ready(mcmc.get_samples())
    seconds = time.perf_counter() - started

    by_chain = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
    if warmup_samples:
        extra["warmup_samples"] = warmup_samples
    # The Pathfinder run, where there was one, is part of what the fit cost.
    extra["mcmc_seconds"] = seconds

    return FitResult(
        backend=str(cfg.sampler).strip().lower(),
        draws=_flatten_chains(by_chain),
        draws_by_chain=by_chain,
        diverging=_divergences(mcmc),
        seconds=seconds + (pathfinder_seconds or 0.0),
        pathfinder_seconds=pathfinder_seconds,
        pathfinder_khat=pathfinder_khat,
        pathfinder_elbo=pathfinder_elbo,
        extra=extra,
    )


def fit_pathfinder(cfg: NumpyroConfig, model_kwargs: dict, active_core_params: tuple[str, ...]) -> FitResult:
    """A Pathfinder approximation, used as the posterior rather than as a start."""
    started = time.perf_counter()
    unconstrained = build_unconstrained(cfg, model_kwargs, active_core_params)
    _, z_draws, log_q, khat, elbo = pathfinder_approximation(
        unconstrained,
        seed=cfg.seed,
        num_draws=cfg.vi_draws,
        maxiter=cfg.pathfinder_maxiter,
        maxcor=cfg.pathfinder_maxcor,
    )
    draws = unconstrained.constrain_batch(z_draws)
    jax.block_until_ready(z_draws)
    seconds = time.perf_counter() - started

    log_p = np.asarray(jax.vmap(unconstrained.logdensity)(z_draws))
    smoothed, _ = psis_smooth(log_p - log_q)
    if cfg.vi_psir:
        draws = _resample(draws, smoothed, seed=cfg.seed)

    return FitResult(
        backend="pathfinder",
        draws=draws,
        seconds=seconds,
        log_weights=smoothed,
        khat=khat,
        elbo=elbo,
        dim=unconstrained.dim,
        extra={"psir": bool(cfg.vi_psir)},
    )


def fit_fullrank(cfg: NumpyroConfig, model_kwargs: dict, active_core_params: tuple[str, ...]) -> FitResult:
    """Full-rank Gaussian VI, optimised with Adam.

    The whole optimisation is one ``lax.scan``, so it compiles once and the
    per-step cost is the model gradient rather than the dispatch. On this model
    that is the point: a step costs ``vi_elbo_samples`` gradients against NUTS's
    one-per-leapfrog, and there are far fewer steps than a NUTS fit has leapfrogs.
    """
    import blackjax

    started = time.perf_counter()
    unconstrained = build_unconstrained(cfg, model_kwargs, active_core_params)
    optimizer = optax.adam(cfg.vi_learning_rate)
    state = blackjax.vi.fullrank_vi.init(unconstrained.z_init, optimizer)

    def scan_step(carry, key):
        new_state, info = blackjax.vi.fullrank_vi.step(
            key, carry, unconstrained.logdensity, optimizer, num_samples=cfg.vi_elbo_samples
        )
        # FRVIInfo.elbo is the *loss* -- _gaussian_vi._elbo_step returns
        # mean(log q - log p), "the loss value to be minimized". Negated here so
        # this trace and PathfinderState.elbo mean the same thing and larger is
        # better in both.
        return new_state, -info.elbo

    keys = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.vi_steps)
    state, elbo_trace = jax.lax.scan(scan_step, state, keys)

    z_draws = blackjax.vi.fullrank_vi.sample(jax.random.PRNGKey(cfg.seed + 1), state, cfg.vi_draws)
    draws = unconstrained.constrain_batch(z_draws)
    jax.block_until_ready(z_draws)
    seconds = time.perf_counter() - started

    log_q_fn = blackjax.vi.fullrank_vi.generate_fullrank_logdensity(state.mu, state.chol_params)
    log_q = np.asarray(jax.vmap(log_q_fn)(z_draws))
    log_p = np.asarray(jax.vmap(unconstrained.logdensity)(z_draws))
    smoothed, khat = psis_smooth(log_p - log_q)
    if cfg.vi_psir:
        draws = _resample(draws, smoothed, seed=cfg.seed)

    elbo_trace = np.asarray(elbo_trace)
    # The last 1% of the trace rather than its final point: one Monte Carlo ELBO
    # estimate is noisy at vi_elbo_samples draws, and the tail mean is what says
    # where the optimiser settled.
    tail = elbo_trace[-max(1, len(elbo_trace) // 100) :]
    return FitResult(
        backend="fullrank",
        draws=draws,
        seconds=seconds,
        log_weights=smoothed,
        khat=khat,
        elbo=float(np.mean(tail)),
        elbo_trace=elbo_trace,
        dim=unconstrained.dim,
        extra={"psir": bool(cfg.vi_psir), "vi_steps": int(cfg.vi_steps)},
    )


def build_kernel(cfg: NumpyroConfig, model_kwargs: dict, active_core_params: tuple[str, ...]):
    """Construct the MCMC kernel, including the init strategy."""
    sampler_mode = str(cfg.sampler).strip().lower()
    if sampler_mode not in {"nuts", "sa"}:
        raise ValueError("sampler must be one of {'nuts', 'sa'}")
    if cfg.chain_method not in {"parallel", "sequential", "vectorized"}:
        raise ValueError("chain_method must be one of {'parallel','sequential','vectorized'}")

    init_strategy = resolve_init_strategy(cfg, model_kwargs, active_core_params)
    kwargs = {} if init_strategy is None else {"init_strategy": init_strategy}
    if sampler_mode == "nuts":
        return NUTS(hierarchical_model, max_tree_depth=cfg.max_tree_depth, **kwargs)
    return SA(
        hierarchical_model,
        adapt_state_size=cfg.sa_adapt_state_size,
        dense_mass=cfg.sa_dense_mass,
        **kwargs,
    )


def run_mcmc(
    cfg: NumpyroConfig,
    model_kwargs: dict,
    active_core_params: tuple[str, ...],
    init_params=None,
):
    """Fit under a fixed budget and return ``(mcmc, warmup_samples)``.

    Writes nothing: the caller decides what, if anything, lands on disk.

    ``init_params`` overrides the kernel's init strategy with explicit starting
    points, in the *unconstrained* space and carrying a leading chain dimension
    (``numpyro/infer/hmc.py:704``). :func:`msinf.fit.fit_mcmc` passes Pathfinder
    draws here; ``None`` leaves the configured init strategy in charge.
    """
    kernel = build_kernel(cfg, model_kwargs, active_core_params)
    sampler_kwargs = model_kwargs_for_sampler(model_kwargs)
    # Only the HMC family reports divergences.
    extra_fields = ("diverging",) if str(cfg.sampler).strip().lower() == "nuts" else ()

    mcmc = MCMC(
        kernel,
        num_warmup=cfg.num_warmup,
        num_samples=cfg.num_samples,
        num_chains=cfg.num_chains,
        chain_method=cfg.chain_method,
        progress_bar=cfg.progress_bar,
    )

    warmup_samples: dict[str, np.ndarray] = {}
    if cfg.collect_warmup:
        key_warmup, key_run = jax.random.split(jax.random.PRNGKey(cfg.seed))
        # num_samples=0 so the internal run() inside warmup() only executes the
        # warmup phase -- no wasted sampling pass.
        mcmc_w = MCMC(
            kernel,
            num_warmup=cfg.num_warmup,
            num_samples=0,
            num_chains=cfg.num_chains,
            chain_method=cfg.chain_method,
            progress_bar=cfg.progress_bar,
        )
        mcmc_w.warmup(key_warmup, collect_warmup=True, init_params=init_params, **sampler_kwargs)
        warmup_samples = {f"warmup__{k}": np.asarray(v) for k, v in mcmc_w.get_samples(group_by_chain=True).items()}
        # Seed the main MCMC from the post-warmup state so run() skips re-warmup.
        # init_params is deliberately not repeated here: warmup already consumed
        # it, and the post-warmup state is where the chains actually are.
        mcmc._warmup_state = mcmc_w._warmup_state
        mcmc.run(key_run, extra_fields=extra_fields, **sampler_kwargs)
    else:
        mcmc.run(
            jax.random.PRNGKey(cfg.seed),
            extra_fields=extra_fields,
            init_params=init_params,
            **sampler_kwargs,
        )
    return mcmc, warmup_samples


def run_inference(outdir: Path, cfg: NumpyroConfig) -> dict[str, str]:
    """Run the hierarchical NumPyro inference.

    Runtime policy: always set XLA host device count to match ``cfg.num_chains``
    so chain execution can parallelize across CPU host devices when possible.
    This has to happen before the first array is built, since that initializes
    the JAX backend and fixes the device count.
    """
    numpyro.set_host_device_count(cfg.num_chains)

    outdir.mkdir(parents=True, exist_ok=True)

    model_kwargs, active_core_params, fixed_core_values = build_model_kwargs(cfg)
    inactive_core_params = tuple([p for p in CORE_PARAMS if p not in set(active_core_params)])

    # Through fit() rather than run_mcmc directly, so this entry point gets every
    # backend for free and cannot disagree with what evaluate() fitted.
    result = fit(cfg, model_kwargs, active_core_params)

    # Grouped by chain where there are chains, so the file keeps the shape every
    # existing reader expects; a variational fit has none and saves flat draws.
    by_chain = result.draws_by_chain is not None
    samples = result.draws_by_chain if result.draws_by_chain is not None else result.draws
    sample_path = outdir / SAMPLES_FILENAME
    np.savez(
        sample_path,
        **{k: np.asarray(v) for k, v in samples.items()},
        **result.extra.get("warmup_samples", {}),
        # Which the leading axis is; the shapes cannot say. See
        # msinf.compare.load_draws_npz.
        **{BY_CHAIN_KEY: np.asarray(by_chain)},
    )

    config_path = outdir / "run_config_resolved.json"
    config_path.write_text(
        json.dumps(
            {
                **asdict(cfg),
                "active_core_params": list(active_core_params),
                "inactive_core_params": list(inactive_core_params),
                "fixed_core_values": fixed_core_values,
                "data_csv": str(Path(cfg.data_csv).resolve()),
                "eir_priors_csv": str(Path(cfg.eir_priors_csv).resolve()),
            },
            indent=2,
        )
    )

    # print_summary wants chains; a variational fit has none, so it is shown one
    # nominal chain rather than being skipped -- the means and quantiles are
    # meaningful, and the r_hat/n_eff columns are not and are labelled as such by
    # fit_diagnostics rather than here.
    by_chain = result.draws_by_chain or {k: v[None, ...] for k, v in result.draws.items()}
    print_summary(by_chain, prob=0.9)
    if result.draws_by_chain is None:
        print(
            f"[{result.backend}] r_hat/n_eff above are not meaningful for a "
            f"variational fit; khat={result.khat:.3f}, elbo={result.elbo:.2f}"
        )
    return {"samples": str(sample_path), "resolved_config": str(config_path)}


_BACKENDS: dict[str, Callable[..., FitResult]] = {
    "mcmc": fit_mcmc,
    "pathfinder": fit_pathfinder,
    "fullrank": fit_fullrank,
}


def fit(cfg: NumpyroConfig, model_kwargs: dict, active_core_params: tuple[str, ...]) -> FitResult:
    """Fit under ``cfg.fit_backend``."""
    backend = str(cfg.fit_backend).strip().lower()
    if backend not in _BACKENDS:
        raise ValueError(f"fit_backend must be one of {set(FIT_BACKENDS)}; got {cfg.fit_backend!r}")
    return _BACKENDS[backend](cfg, model_kwargs, active_core_params)
