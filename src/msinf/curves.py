"""The curve provider: the map from parameters to age curves.

Everything in :mod:`msinf.model` other than this is the fixed observational
model -- the study random effects, the incidence and prevalence likelihoods, the
site EIR priors, ``ft_study``, the age grid, the demography and the age-bin
averaging. What varies between candidate models is only what lives here.

A curve provider is a pair of callables, because ``vmap`` forces the split:
NumPyro sample sites cannot be created inside the ``vmap`` over sites, so the
sampling and the curve evaluation have to be two functions and not one.

    parameters()                                -> params
    curves(params, log_eir, ft, ages, prop, mu) -> (prev_m, inc)

``parameters`` runs once per model evaluation and owns every ``numpyro.sample``
call the candidate needs. ``curves`` is pure JAX, called under ``vmap`` over
sites, and returns two arrays shaped like ``ages`` on the fixed age grid:
microscopy prevalence and incidence. There is no PCR prevalence -- the 2014
table carries no ``type == 3`` rows, so it is not part of the contract.
``prop`` (the demographic weight per age class) and ``mu`` (the per-day death
rate per age class) are supplied as fixed data; a candidate that has no use for
either simply ignores them.
"""

from collections.abc import Callable
from typing import NamedTuple

import jax.numpy as jnp
from dmeq import default_parameters, solve

from .config import sample_param
from .dtypes import FloatT, asfloat
from .priors import CORE_PARAMS, DMEQ_OVERRIDE_PARAMS, PriorSpec


class CurveProvider(NamedTuple):
    """A candidate model: how to sample its parameters, and how to draw curves."""

    parameters: Callable[[], dict]
    curves: Callable[..., tuple[jnp.ndarray, jnp.ndarray]]


def griffin_parameters(
    priors: dict[str, PriorSpec],
    active_core_params: tuple[str, ...] = CORE_PARAMS,
    fixed_core_values: dict[str, float] | None = None,
) -> Callable[[], dict]:
    """The Griffin/dmeq parameter block, as a ``parameters()`` callable.

    Returns the full dmeq parameter dict with the sampled core parameters
    written over the defaults. The hardcoded override list and the ``ad0``
    years-to-days conversion live here rather than in the model, because both
    are Griffin-specific and mean nothing to a candidate that invented its own
    parameters.
    """
    fixed_core_values = dict(fixed_core_values or {})
    active_core_set = set(active_core_params)

    def parameters() -> dict:
        sampled = {}
        for name, spec in priors.items():
            # rW/rP belong to the observational model and are sampled by the
            # scaffolding, not by the curve provider.
            if name in _OBSERVATIONAL_PARAMS:
                continue
            if (name in CORE_PARAMS) and (name not in active_core_set):
                continue
            sampled[name] = sample_param(name, spec)
        for name, value in fixed_core_values.items():
            sampled[name] = asfloat(value)

        params = {k: asfloat(v) for k, v in default_parameters().items()}
        for k in DMEQ_OVERRIDE_PARAMS:
            if k in sampled:
                params[k] = sampled[k]
        # The Griffin parameterization keeps ad0 on a years scale in the
        # prior/latent space; the ODE model wants age in days.
        if "ad0" in params:
            params["ad0"] = params["ad0"] * 365.0
        return params

    return parameters


# Parameters of the fixed observational model rather than of the curve: the
# case-detection scalings for active (rW) and passive (rP) case detection.
_OBSERVATIONAL_PARAMS: tuple[str, ...] = ("rW", "rP")


def griffin_curves(params, log_eir, ft, ages, prop, mu):
    """Solve the Griffin equilibrium model for one site.

    ``prop`` is accepted and ignored: ``dmeq`` recomputes the same age
    proportions internally from ``mu``, and the model uses the ``prop`` passed
    in here for the age-bin averaging so that the demography is fixed
    scaffolding rather than something the solve returns.
    """
    del prop
    p = dict(params)
    p["EIR"] = jnp.exp(log_eir)
    p["ft"] = ft
    p["eta"] = mu
    out = solve(p, dtype=FloatT(), age_bins_years=ages)
    # Row 1 is dmeq's PCR prevalence, which nothing observes; it is not returned.
    return out[0, :], out[2, :]


def griffin_provider(
    priors: dict[str, PriorSpec],
    active_core_params: tuple[str, ...] = CORE_PARAMS,
    fixed_core_values: dict[str, float] | None = None,
) -> CurveProvider:
    """The incumbent model as a :class:`CurveProvider`."""
    return CurveProvider(
        parameters=griffin_parameters(priors, active_core_params, fixed_core_values),
        curves=griffin_curves,
    )
