"""The fixed observational likelihood, in per-unit pieces.

One implementation, used twice: :mod:`msinf.model` sums these into the two
``numpyro.factor`` totals the sampler sees, and :mod:`msinf.evaluate` mixes the
same per-unit terms over the Gauss-Hermite nodes to score held-out data. Keeping
them in one place is what stops the fitted likelihood and the reported
``elpd`` from describing different models.

Following Griffin (2014) the random effects are not sampled. The study effects
-- ``u`` for incidence (LogNormal, mean 1) and ``w`` for prevalence (logit
Normal) -- are integrated out by fixed Gauss-Hermite quadrature, and the
site-age effect ``v_jk`` (Gamma, mean 1) analytically, which turns the incidence
Poisson into a negative binomial.

**The unit of the incidence likelihood is the cell, not the row.** ``v_jk`` is
integrated out at the (site, age-group) level, so the negative-binomial marginal
is a joint density over the whole cell and does not factorise into rows. There
is no per-row incidence density to report; :func:`incidence_cell_loglik_nodes`
is per cell and holdout splits are required to keep cells whole
(see :mod:`msinf.splits`).
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.ops import segment_sum
from jax.scipy.special import betaln, gammaln, logsumexp

from .dtypes import FloatT, asfloat

# Number of fixed Gauss-Hermite quadrature nodes used to integrate out the
# study-level random effects (u for incidence, w for prevalence). Griffin uses
# *adaptive* GH; a fixed grid with enough nodes is an accurate, much simpler
# approximation for these 1-D per-study integrals. Raise if a study's effect is
# very sharply peaked relative to its prior.
N_GH_NODES = 32
# Probabilists' Gauss-Hermite nodes/weights for E_{Z~N(0,1)}[g(Z)] ~ sum w_i g(z_i):
# hermegauss integrates against exp(-z^2/2); divide weights by sqrt(2*pi) so they
# sum to 1 (a standard-normal expectation).
GH_NODES, _gh_w = np.polynomial.hermite_e.hermegauss(N_GH_NODES)
GH_WEIGHTS = _gh_w / np.sqrt(2.0 * np.pi)

# How far the beta-binomial's q is held off 0 and 1. See
# prevalence_row_loglik_nodes: this exists for the gradient, not the value.
Q_EPS = 1e-12


def study_effect_nodes(sigma_c, sigma_p) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Quadrature nodes for the two study effects, ``(u_nodes, w_nodes)``.

    ``u ~ Normal(-sigma_c^2/2, sigma_c)`` so that ``E[e^u] = 1``;
    ``w ~ Normal(0, sigma_p)`` on the logit scale.
    """
    gh_z = asfloat(GH_NODES)
    return sigma_c * gh_z - 0.5 * sigma_c**2, sigma_p * gh_z


def mix_over_nodes(node_loglik: jnp.ndarray) -> jnp.ndarray:
    """Integrate a ``(..., G)`` node log-likelihood over the study effect."""
    return logsumexp(jnp.log(asfloat(GH_WEIGHTS)) + node_loglik, axis=-1)


def case_detection_scaling(case_det, rW, rP):
    """Griffin coding: 2 -> rW, 6 -> rP, else 1.

    Many processed tables recode 6 -> 3, so both are supported.
    """
    return jnp.where(case_det == 2, rW, jnp.where((case_det == 6) | (case_det == 3), rP, 1.0))


def incidence_cell_loglik_nodes(
    *,
    mu_inc,
    numer,
    denom,
    case_det,
    cell_id,
    n_cells: int,
    rW,
    rP,
    alpha_c,
    u_nodes,
) -> jnp.ndarray:
    """``(n_cells, G)`` incidence log-likelihood, ``v_jk`` integrated out.

    ``mu_inc`` is the per-person-day mechanistic incidence already averaged over
    each observation's age bin; ``cell_id`` maps each observation row to its
    (site, age-group) cell. Cells with no observations contribute 0 at every
    node, so the array can be summed over freely.
    """
    # dmeq incidence is per person-day; the observational denom is person-years.
    mu_inc_per_year = 365.0 * jnp.clip(mu_inc, 1e-12)
    numer_f = numer.astype(FloatT())
    r_inc = case_detection_scaling(case_det, rW, rP)
    # base_inc_rate = r_i * T_i * mu_i: the conditional Poisson rate EXCLUDING
    # the study effect e^u and the site-age effect v_jk.
    base_inc_rate = denom.astype(FloatT()) * r_inc * mu_inc_per_year
    nb_shape = 1.0 / jnp.clip(alpha_c, 1e-8)   # Gamma/NB shape 1/alpha_c, since var(v_jk) = alpha_c

    cell_rate_sum = segment_sum(base_inc_rate, cell_id, num_segments=n_cells)
    cell_case_sum = segment_sum(numer_f, cell_id, num_segments=n_cells)
    cell_poisson_logconst = segment_sum(
        numer_f * jnp.log(base_inc_rate) - gammaln(numer_f + 1.0), cell_id, num_segments=n_cells
    )

    # At each node u the rate sum scales by e^u and the log-constant shifts by
    # u * (case sum).
    return (
        gammaln(nb_shape + cell_case_sum)[:, None]
        - (nb_shape + cell_case_sum)[:, None]
        * jnp.log(nb_shape + jnp.exp(u_nodes)[None, :] * cell_rate_sum[:, None])
        + cell_poisson_logconst[:, None]
        + u_nodes[None, :] * cell_case_sum[:, None]
        + nb_shape * jnp.log(nb_shape)
        - gammaln(nb_shape)
    )


def prevalence_row_loglik_nodes(*, p, numer, denom, theta_prev, w_nodes) -> jnp.ndarray:
    """``(n_rows, G)`` beta-binomial prevalence log-likelihood, per observation.

    ``p`` is the mechanistic prevalence averaged over each row's age bin. Unlike
    incidence this is genuinely per row: the only shared effect is the study
    effect ``w``, which is marginal, so a single row's density is well defined.

    ``q`` is held off 0 and 1 for the *gradient*, not for the value. At
    ``sigma_p >= 4`` the outer Gauss-Hermite nodes push ``logit(p) + w`` far
    enough that the sigmoid saturates to exactly 0 or 1 in f64; one of
    ``betaln``'s arguments is then 0, and ``betaln(a, 0) = inf`` with
    ``digamma(0) = NaN``. That is a genuine singularity of the function and not
    an artifact of how the derivative is taken -- forward and reverse mode both
    return NaN there, while the derivative approaching it is merely large
    (``-1e8`` at ``b = 1e-8``, ``-1e12`` at ``1e-12``).

    A sampler started near the median never reaches that region, which is why
    this stood for as long as it did. A variational fit draws from a Gaussian
    over the whole unconstrained space, reaches it about once in a few thousand
    draws, and Adam then carries the NaN into its moments and never recovers --
    one bad draw in ~3400 ends the run.

    The clip is inert everywhere a fit actually goes: over 400 draws with
    ``sigma_p ~ HalfNormal(0.5)`` and ``alpha_p`` from its half-t prior, the
    clipped and unclipped node log-likelihoods agree *bit for bit*. Where it is
    not inert the density was ``-inf`` already, and all that changes is that the
    gradient points away from the saturated region instead of being NaN.
    """
    logit_p = jax.scipy.special.logit(jnp.clip(p, 1e-6, 1 - 1e-6))
    q = jnp.clip(jax.nn.sigmoid(logit_p[:, None] + w_nodes[None, :]), Q_EPS, 1.0 - Q_EPS)
    a_bb = q * theta_prev
    b_bb = (1.0 - q) * theta_prev
    n_pos = numer.astype(FloatT())[:, None]
    n_tested = denom.astype(FloatT())[:, None]
    lchoose = gammaln(n_tested + 1.0) - gammaln(n_pos + 1.0) - gammaln(n_tested - n_pos + 1.0)
    return lchoose + betaln(a_bb + n_pos, n_tested - n_pos + b_bb) - betaln(a_bb, b_bb)


def sum_by_study(node_loglik: jnp.ndarray, study_idx, n_study: int) -> jnp.ndarray:
    """Aggregate per-unit ``(N, G)`` node log-likelihoods to ``(n_study, G)``."""
    return segment_sum(node_loglik, study_idx, num_segments=n_study)
