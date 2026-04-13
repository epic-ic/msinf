"""Canonical age grid and grid-aware bin aggregation.

Griffin (2014/2016) solves the equilibrium model on a fine, non-uniform age
grid (``Griffin-2016-R-code/age.txt``): ~2-week resolution through infancy,
coarsening to 2-year steps in adulthood, 165 classes total. The immunity states
(IB/IC/ID) are built by a backward-Euler recursion over this grid, so the grid
resolution materially changes the equilibrium incidence/prevalence in early
childhood. dmeq's default uniform 100x1-year grid under-resolves infancy and
under-predicts incidence in the youngest (most informative) age bins.

This module provides:
  * ``GRIFFIN_AGE_GRID`` -- the canonical grid (years, lower edges of classes).
  * ``class_fraction_weights`` -- grid-aware bin aggregation weights, the
    fraction of each model age class covered by an observed [age0, age1) bin.

A dmeq class ``i`` spans ``[ages[i], ages[i+1])``; the terminal class is an
absorbing ``[ages[-1], inf)`` compartment that observed bins (age1 <= 100)
never overlap, matching Griffin's strict ``age[i] < age1`` membership test.
"""

import numpy as np

# `NpFloatT` is imported inside each function rather than here, so that reading
# the grid does not start a backend: :mod:`msinf.dtypes` imports jax, and this
# module is otherwise a tuple of numbers and some numpy. Anything that only
# wants to know where the age classes are -- an analysis environment, a script
# checking a mortality table is on the grid -- should not pay for jax to find
# out.

# Griffin-2016-R-code/age.txt (years). Lower edge of each model age class.
GRIFFIN_AGE_GRID: tuple[float, ...] = (
    0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24, 0.28, 0.32, 0.36, 0.4, 0.44, 0.48,
    0.52, 0.56, 0.6, 0.64, 0.68, 0.72, 0.76, 0.8, 0.84, 0.88, 0.92, 0.96, 1.0,
    1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.55, 1.6, 1.65, 1.7,
    1.75, 1.8, 1.85, 1.9, 1.95, 2.0, 2.06667, 2.13333, 2.2, 2.26667, 2.33333,
    2.4, 2.46667, 2.53333, 2.6, 2.66667, 2.73333, 2.8, 2.86666, 2.93334, 3.0,
    3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4.0, 4.12501, 4.25, 4.37501,
    4.5, 4.62501, 4.75, 4.87501, 5.0, 5.2, 5.4, 5.6, 5.8, 6.0, 6.25, 6.5, 6.75,
    7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0,
    10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5, 15.0, 15.5, 16.0,
    16.5, 17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 32.0, 34.0, 36.0, 38.0, 40.0, 42.0,
    44.0, 46.0, 48.0, 50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0, 64.0, 66.0,
    68.0, 70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0, 84.0, 86.0, 88.0, 90.0,
    92.0, 94.0, 96.0, 98.0, 100.0,
)


def age_grid_array(dtype=None) -> np.ndarray:
    """Return the canonical age grid as a numpy array of class lower edges.

    Defaults to the configured float precision (see :mod:`msinf.dtypes`); every
    caller used to override the old float64 default with float32 anyway.

    Passing an explicit ``dtype`` keeps this jax-free; the default asks
    :mod:`msinf.dtypes` what precision the run is at, which does not.
    """
    if dtype is None:
        from .dtypes import NpFloatT

        dtype = NpFloatT()
    return np.asarray(GRIFFIN_AGE_GRID, dtype=dtype)


def class_upper_edges(ages: np.ndarray) -> np.ndarray:
    """Upper edge of each age class; terminal class is absorbing ([last, inf))."""
    from .dtypes import NpFloatT

    ages = np.asarray(ages, dtype=NpFloatT())
    # np.append would promote to float64 via the default dtype of np.inf.
    return np.concatenate([ages[1:], np.array([np.inf], dtype=ages.dtype)])


def class_fraction_weights(age0: np.ndarray, age1: np.ndarray, ages: np.ndarray) -> np.ndarray:
    """Fraction of each model age class covered by each observed [age0, age1) bin.

    Returns an ``(n_obs, n_age)`` matrix ``w`` such that a per-person rate over
    the bin is ``sum_a w[:, a] * curve[a] / sum_a w[:, a] * prop[a]``. For a
    uniform 1-year grid this reduces exactly to the previous unit-overlap
    weighting; for a fine grid, partially covered classes get fractional weight
    and fully covered classes get weight 1, so narrow classes are not spuriously
    down-weighted relative to wide ones.
    """
    from .dtypes import NpFloatT

    lo = np.asarray(ages, dtype=NpFloatT())[None, :]
    hi = class_upper_edges(ages)[None, :]
    width = hi - lo
    a0 = np.asarray(age0, dtype=NpFloatT())[:, None]
    a1 = np.asarray(age1, dtype=NpFloatT())[:, None]
    covered = np.clip(np.minimum(hi, a1) - np.maximum(lo, a0), 0.0, None)
    # Terminal class has infinite width -> fraction 0 for any finite bin.
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(np.isfinite(width), covered / width, 0.0)
    return np.clip(frac, 0.0, 1.0)
