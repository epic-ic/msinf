"""Run configuration, active-parameter resolution and MCMC initialisation."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from .dtypes import asfloat
from .priors import CORE_PARAMS, PriorSpec, build_griffin_priors, build_xi_priors

PRIOR_FAMILIES = {"xi": build_xi_priors, "griffin": build_griffin_priors}


@dataclass
class NumpyroConfig:
    data_csv: str
    eir_priors_csv: str
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    max_tree_depth: int = 10
    # Parameter-level core control:
    # - None: all core params active
    # - (): no core params active (EIR-only for core block)
    active_core_params: tuple[str, ...] | None = None
    # Optional inverse selector. If provided, active set is computed as
    # CORE_PARAMS - inactive_core_params.
    inactive_core_params: tuple[str, ...] | None = None
    # Required when any core params are inactive.
    reference_draws_csv: str | None = None
    # Study-level treatment prior table with columns: study_index,ft_alpha,ft_beta
    ft_priors_csv: str | None = None
    # Per-site WPP death-rate table on the model age grid, with columns
    # site_index,age_class,age_lower,mu_per_day. None keeps dmeq's uniform
    # default rate, which is what every run before these tables existed used.
    mortality_csv: str | None = None
    # MCMC initialization strategy.
    # - "griffin": seed with Griffin-refit-like fixed starts via init_to_value
    # - "median": init_to_median; the name-free option, for a candidate model
    #   whose parameters have no known-good values to init to
    # - "default": NumPyro default initializer
    # - "pathfinder": seed each chain from a draw of a Pathfinder approximation
    # - "griffin_jitter": the "griffin" starts for every site that has one,
    #   displaced per chain by init_jitter on the unconstrained scale, and
    #   init_to_median for the rest -- so it also serves a candidate model
    init_strategy: str = "griffin"
    # Standard deviation of the per-chain displacement "griffin_jitter" applies,
    # on the unconstrained scale (a log or logit for most sites).
    init_jitter: float = 0.5
    # Which backend produces the posterior: mcmc | pathfinder | fullrank.
    # "mcmc" runs whichever kernel `sampler` names; the variational backends
    # ignore `sampler`.
    fit_backend: str = "mcmc"
    # Pathfinder, used either as the posterior (fit_backend) or as an
    # initialiser (init_strategy). Defaults are BlackJAX's own.
    pathfinder_maxiter: int = 30
    pathfinder_maxcor: int = 10
    # Draws taken from the approximation when it is only seeding chains. Larger
    # than num_chains so khat is estimated from a usable sample before the first
    # num_chains of them become the starts.
    pathfinder_draws: int = 200
    # Full-rank Gaussian VI.
    vi_steps: int = 5000
    vi_learning_rate: float = 1e-2
    # Monte Carlo samples per ELBO gradient. This is the multiplier on the cost
    # of a step, and a step is one model gradient per sample.
    vi_elbo_samples: int = 8
    # Draws taken from the fitted approximation, and the size of the PSIS
    # weight vector khat is estimated from.
    vi_draws: int = 1000
    # Resample the variational draws against their PSIS weights before scoring
    # (see msinf.fit).
    vi_psir: bool = False
    # Core biological prior family: Xi refit or Griffin SI.
    prior_family: str = "xi"
    # Whether to collect warmup draws and save them alongside posterior draws.
    collect_warmup: bool = True
    # NumPyro chain execution mode: parallel | sequential | vectorized.
    chain_method: str = "parallel"
    # MCMC sampler kernel: "nuts" (default) or "sa" (NumPyro Sample Adaptive MCMC).
    sampler: str = "nuts"
    # SA-only options (ignored for NUTS).
    sa_adapt_state_size: int | None = None
    sa_dense_mass: bool = True
    seed: int = 42
    # Progress bars are noise in a child process; the Hydra path keeps them.
    progress_bar: bool = True


def resolve_priors(prior_family: str) -> dict[str, PriorSpec]:
    """The prior specs for a named family.

    Resolved by the caller and handed to the model as data: the model reads no
    configuration, so it cannot disagree with what the diagnostics resolved.
    """
    try:
        return PRIOR_FAMILIES[str(prior_family).strip().lower()]()
    except KeyError as exc:
        raise ValueError(f"prior_family must be one of {sorted(PRIOR_FAMILIES)}") from exc


def sample_param(name: str, spec: PriorSpec):
    p1 = asfloat(spec.param1)
    p2 = asfloat(spec.param2)
    if spec.family == "beta":
        return numpyro.sample(name, dist.Beta(p1, p2))
    if spec.family == "lognormal":
        return numpyro.sample(name, dist.LogNormal(p1, p2))
    raise ValueError(f"unsupported family for {name}: {spec.family}")


def resolve_active_core_params(cfg: NumpyroConfig) -> tuple[str, ...]:
    core_set = set(CORE_PARAMS)
    active_in = None if cfg.active_core_params is None else tuple(dict.fromkeys(cfg.active_core_params))
    inactive_in = None if cfg.inactive_core_params is None else tuple(dict.fromkeys(cfg.inactive_core_params))

    if active_in is not None and inactive_in is not None:
        overlap = sorted(set(active_in) & set(inactive_in))
        if overlap:
            raise ValueError(f"active_core_params and inactive_core_params overlap: {overlap}")

    if active_in is not None:
        unknown = sorted(set(active_in) - core_set)
        if unknown:
            raise ValueError(f"Unknown active_core_params: {unknown}. Allowed: {list(CORE_PARAMS)}")
        return active_in

    if inactive_in is not None:
        unknown = sorted(set(inactive_in) - core_set)
        if unknown:
            raise ValueError(f"Unknown inactive_core_params: {unknown}. Allowed: {list(CORE_PARAMS)}")
        return tuple([p for p in CORE_PARAMS if p not in set(inactive_in)])

    return CORE_PARAMS


def griffin_refit_init_values(
    active_core_params: tuple[str, ...],
    log_eir_hat: np.ndarray,
    ft_alpha: np.ndarray,
    ft_beta: np.ndarray,
    n_study: int,
    n_site: int,
    n_age_groups: int,
) -> dict[str, jnp.ndarray]:
    """Build NumPyro init_to_value payload approximating Griffin refit starts."""
    griffin_theta_init: dict[str, float] = {
        "d1": 0.160527,
        "ID0": 1.577533,
        "kd": 0.476614,
        "ud": 9.44512,
        "ad0": 8001.99 / 365.0,
        "fd0": 0.007055,
        "gd": 4.8183,
        "b0": 0.590076,
        "IB0": 43.8787,
        "kb": 2.15506,
        "ub": 7.19919,
        "phi0": 0.791666,
        "phi1": 0.000737,
        "IC0": 18.02366,
        "kc": 2.36949,
        "uc": 6.06349,
        "PM": 0.774368,
        "dm": 67.6952,
        "rW": 0.723,
        "rP": 0.342,
    }

    values: dict[str, jnp.ndarray] = {}
    active_set = set(active_core_params)

    for k, v in griffin_theta_init.items():
        if (k in CORE_PARAMS and k not in active_set):
            continue
        values[k] = asfloat(v)

    # Hyperparameters in Griffin eta-space: sigmac=-0.7, alphac=0.7, sigmap=-0.7, alphap=0.7.
    # NumPyro samples these on natural scales. alpha_c/alpha_p start in the bulk of the
    # tightened scale-0.25 half-t prior (median ~0.18) rather than Griffin's exp(0.7)~2,
    # which now sits far in the tail and would slow warmup.
    values["sigma_c"] = asfloat(np.exp(-0.7))
    values["alpha_c"] = asfloat(0.2)
    values["sigma_p"] = asfloat(np.exp(-0.7))
    values["alpha_p"] = asfloat(0.2)

    values["log_eir"] = asfloat(log_eir_hat)
    values["ft_study"] = asfloat(ft_alpha / (ft_alpha + ft_beta))

    # The study effects u (incidence) and w (prevalence) and the site-age effect
    # v_jk are all integrated out (GH quadrature / analytically), not sampled, so
    # they have no init values.

    return values
