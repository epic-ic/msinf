"""Forward simulator for the fixed observational model.

NumPyro's ``Predictive`` cannot do this: the model has no observed sample site,
only two ``numpyro.factor`` totals, because the random effects are integrated
out rather than sampled. To simulate a replicate dataset they have to be drawn
back in.

Given a posterior draw of the hyperparameters and the per-observation
mechanistic means, one replicate is:

* ``u_ell ~ Normal(-sigma_c^2/2, sigma_c)`` -- study incidence effect, one per
  study, ``E[e^u] = 1``.
* ``v_jk ~ Gamma(1/alpha_c, alpha_c)`` -- site-age effect, mean 1, variance
  ``alpha_c``, one per incidence cell.
* incidence counts ``~ Poisson(r_i T_i mu_i e^{u} v_{jk})``.
* ``w_ell ~ Normal(0, sigma_p)`` -- study prevalence effect on the logit scale.
* ``rho_i ~ Beta(q_i theta, (1 - q_i) theta)`` with
  ``q_i = sigmoid(logit(p_i) + w_ell)`` -- the beta half of the beta-binomial,
  drawn per row.
* prevalence counts ``~ Binomial(n_i, rho_i)``.

The same function backs the posterior predictive check artifact and the
randomised PIT, so the two cannot disagree about which fit they describe.
"""

import jax
import jax.numpy as jnp
import numpy as np

from .dtypes import FloatT
from .likelihood import case_detection_scaling


def simulate_incidence(key, *, mu_inc, denom, case_det, cell_id, study_idx, n_cells, n_study,
                       rW, rP, sigma_c, alpha_c):
    """One replicate of the incidence counts, per observation row."""
    k_u, k_v, k_y = jax.random.split(key, 3)
    u = sigma_c * jax.random.normal(k_u, (n_study,), dtype=FloatT()) - 0.5 * sigma_c**2
    nb_shape = 1.0 / jnp.clip(alpha_c, 1e-8)
    v = jax.random.gamma(k_v, nb_shape, shape=(n_cells,), dtype=FloatT()) / nb_shape
    rate = (
        denom.astype(FloatT())
        * case_detection_scaling(case_det, rW, rP)
        * 365.0
        * jnp.clip(mu_inc, 1e-12)
        * jnp.exp(u[study_idx])
        * v[cell_id]
    )
    return jax.random.poisson(k_y, rate)


def simulate_prevalence(key, *, p, denom, study_idx, n_study, sigma_p, theta_prev):
    """One replicate of a prevalence stream's counts, per observation row."""
    k_w, k_rho, k_y = jax.random.split(key, 3)
    w = sigma_p * jax.random.normal(k_w, (n_study,), dtype=FloatT())
    q = jax.nn.sigmoid(jax.scipy.special.logit(jnp.clip(p, 1e-6, 1 - 1e-6)) + w[study_idx])
    rho = jax.random.beta(k_rho, q * theta_prev, (1.0 - q) * theta_prev)
    return jax.random.binomial(k_y, denom.astype(FloatT()), rho)


def randomised_pit(replicates: np.ndarray, observed: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Randomised PIT of each observation against its predictive replicates.

    Counts are discrete, so the plain PIT is not uniform even under a perfect
    model; randomising within the observation's probability atom restores
    uniformity. ``replicates`` is ``(n_rep, n_obs)``.
    """
    replicates = np.asarray(replicates)
    observed = np.asarray(observed)[None, :]
    below = np.mean(replicates < observed, axis=0)
    equal = np.mean(replicates == observed, axis=0)
    return below + rng.uniform(size=below.shape) * equal


def coverage(replicates: np.ndarray, observed: np.ndarray, level: float = 0.9) -> float:
    """Fraction of observations inside the central ``level`` predictive interval."""
    lo = np.quantile(replicates, (1.0 - level) / 2.0, axis=0)
    hi = np.quantile(replicates, 1.0 - (1.0 - level) / 2.0, axis=0)
    return float(np.mean((observed >= lo) & (observed <= hi)))
