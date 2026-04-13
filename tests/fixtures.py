"""A small, cheap instance of the real model, for testing the fit backends.

The observational scaffolding is the real one -- the same
:func:`msinf.model.hierarchical_model`, the same two ``numpyro.factor`` totals,
the same Gauss-Hermite marginalisation, the same truncated-Student-t and Beta
supports that a fit has to transform. Only the curve provider is replaced, by a
closed-form pair with no ODE in it, so a backend can be exercised in seconds
rather than in the twenty minutes a real fit costs.

That is the right substitution for testing :mod:`msinf.bridge` and
:mod:`msinf.fit`: what those modules touch is the *shape* of the model -- its
supports, its factor terms, its latent dimension -- and none of that comes from
the provider.
"""

import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist

from msinf.age_grid import GRIFFIN_AGE_GRID
from msinf.config import NumpyroConfig
from msinf.curves import CurveProvider
from msinf.dtypes import asfloat
from msinf.priors import build_xi_priors

_AGES = np.asarray(GRIFFIN_AGE_GRID)

N_SITE = 4
N_STUDY = 2
N_AGE_GROUPS = 3
AGE_BANDS = ((0.0, 2.0), (2.0, 10.0), (10.0, 60.0))


def toy_provider(deterministic: bool = False) -> CurveProvider:
    """A provider with no ODE: prevalence and incidence saturating in EIR.

    ``deterministic=True`` adds a ``numpyro.deterministic`` site, which is the
    thing a searched candidate may plausibly write and which decides whether
    :mod:`msinf.bridge` can constrain draws by bijection alone.
    """
    def parameters() -> dict:
        scale = numpyro.sample("toy_scale", dist.LogNormal(asfloat(0.0), asfloat(0.5)))
        shape = numpyro.sample("toy_shape", dist.Beta(asfloat(2.0), asfloat(2.0)))
        if deterministic:
            numpyro.deterministic("toy_ratio", scale / (shape + 1.0))
        return {"scale": scale, "shape": shape}

    def curves(params, log_eir, ft, ages, prop, mu):
        del mu
        eir = jnp.exp(log_eir)
        sat = eir / (eir + 10.0)
        age_shape = 1.0 - jnp.exp(-ages / (1.0 + 20.0 * params["shape"]))
        # Population fraction per person-day, the convention the binning and the
        # incidence likelihood actually impose (see TODO_discovery.md section 0).
        prev = 0.6 * sat * (1.0 - 0.5 * age_shape) * (1.0 - 0.3 * ft) * prop
        inc = (params["scale"] * sat * (1.0 - age_shape) * 2.0 / 365.0) * prop
        return prev, inc

    return CurveProvider(parameters=parameters, curves=curves)


def toy_model_kwargs(seed: int = 0) -> dict:
    """Model kwargs for the toy instance, in the shape ``build_model_kwargs`` returns.

    Built directly rather than through the CSV readers: what is under test is
    the fit backend, and routing through ``load_observed_data`` would make every
    backend test depend on the study table that is not on this machine.
    """
    rng = np.random.default_rng(seed)
    site_to_study = np.array([0, 0, 1, 1])[:N_SITE]

    # One incidence row and one prevalence row per (site, age group).
    site = np.repeat(np.arange(N_SITE), N_AGE_GROUPS)
    agek = np.tile(np.arange(N_AGE_GROUPS), N_SITE)
    age0 = np.array([AGE_BANDS[k][0] for k in agek])
    age1 = np.array([AGE_BANDS[k][1] for k in agek])
    n_obs = site.size

    prop = rng.dirichlet(np.ones(len(_AGES)), size=N_SITE)

    priors = build_xi_priors()
    return {
        "numer_inc": asfloat(rng.poisson(20.0, n_obs)),
        "denom_inc": asfloat(rng.uniform(50.0, 200.0, n_obs)),
        "study_inc": jnp.asarray(site_to_study[site], dtype=jnp.int32),
        "site_inc": jnp.asarray(site, dtype=jnp.int32),
        "agek_inc": jnp.asarray(agek, dtype=jnp.int32),
        "age0_inc": asfloat(age0),
        "age1_inc": asfloat(age1),
        "case_det_inc": jnp.asarray(rng.choice([1, 2, 3], n_obs), dtype=jnp.int32),
        "numer_prev_m": jnp.asarray(rng.integers(5, 40, n_obs), dtype=jnp.int32),
        "denom_prev_m": jnp.asarray(np.full(n_obs, 100), dtype=jnp.int32),
        "study_prev_m": jnp.asarray(site_to_study[site], dtype=jnp.int32),
        "site_prev_m": jnp.asarray(site, dtype=jnp.int32),
        "age0_prev_m": asfloat(age0),
        "age1_prev_m": asfloat(age1),
        "log_eir_hat": asfloat(rng.normal(2.0, 0.5, N_SITE)),
        "sigma_eir": asfloat(np.full(N_SITE, 0.5)),
        "ft_alpha": asfloat(np.full(N_STUDY, 2.0)),
        "ft_beta": asfloat(np.full(N_STUDY, 5.0)),
        "site_to_study": jnp.asarray(site_to_study, dtype=jnp.int32),
        "prop_site": asfloat(prop),
        "mu_site": asfloat(np.full((N_SITE, len(_AGES)), 1e-4)),
        "n_study": N_STUDY,
        "n_site": N_SITE,
        "n_age_groups": N_AGE_GROUPS,
        "obs_priors": {k: priors[k] for k in ("rW", "rP")},
        "parameters": None,   # filled in by toy_setup
        "curves": None,
    }


def toy_cfg(**overrides) -> NumpyroConfig:
    """A tiny fit budget. Every field the backends read has a value here."""
    base = {
        "data_csv": "unused.csv",
        "eir_priors_csv": "unused.csv",
        "num_warmup": 40,
        "num_samples": 40,
        "num_chains": 2,
        "chain_method": "sequential",
        "init_strategy": "median",
        "progress_bar": False,
        "seed": 0,
    }
    base.update(overrides)
    return NumpyroConfig(**base)


def toy_setup(deterministic: bool = False, seed: int = 0) -> tuple[dict, tuple[str, ...]]:
    """``(model_kwargs, active_core_params)`` for the toy instance."""
    provider = toy_provider(deterministic=deterministic)
    kwargs = toy_model_kwargs(seed=seed)
    kwargs["parameters"] = provider.parameters
    kwargs["curves"] = provider.curves
    return kwargs, ()


