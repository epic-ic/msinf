"""Age-bin aggregation for the model likelihood (JAX).

The numpy equivalents live in :mod:`msinf.age_grid`; these are the traced
versions used inside the model.
"""

import jax.numpy as jnp

from .dtypes import asfloat


def age_class_fraction_weights(age0: jnp.ndarray, age1: jnp.ndarray, ages: jnp.ndarray) -> jnp.ndarray:
    """Fraction of each model age class covered by each observed [age0, age1) bin.

    The terminal class is absorbing (``[last, inf)``) and gets zero weight for
    any finite bin, matching Griffin's strict ``age < age1`` membership test.
    Reduces to unit overlap on a uniform 1-year grid.
    """
    lo = ages[None, :]
    hi = jnp.append(ages[1:], jnp.inf)[None, :]
    width = hi - lo
    covered = jnp.clip(jnp.minimum(hi, age1[:, None]) - jnp.maximum(lo, age0[:, None]), 0.0, None)
    frac = jnp.where(jnp.isfinite(width), covered / width, asfloat(0.0))
    return jnp.clip(frac, 0.0, 1.0)


def weighted_bin_mean_batch(
    curves_by_site: jnp.ndarray,
    weights_by_site: jnp.ndarray,
    site_idx: jnp.ndarray,
    age0: jnp.ndarray,
    age1: jnp.ndarray,
    ages: jnp.ndarray,
) -> jnp.ndarray:
    """Per-person rate over each observed age bin.

    dmeq emits pop-fraction quantities (pos_M, pos_PCR, inc all sum-with-states
    to prop[a]), so convert to a per-person rate over the bin::

        mu_bin = sum_a w(a) * curve(a) / sum_a w(a) * prop(a)

    where ``w(a)`` is the fraction of class ``a`` covered by the observed bin.
    """
    overlap = age_class_fraction_weights(age0, age1, ages)
    curve_obs = curves_by_site[site_idx, :]
    prop_obs = weights_by_site[site_idx, :]
    denom = jnp.clip(jnp.sum(overlap * prop_obs, axis=1), 1e-12)
    return jnp.sum(overlap * curve_obs, axis=1) / denom
