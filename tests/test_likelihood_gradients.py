"""The prevalence term has to be differentiable where a variational fit looks.

Pins the fix in :func:`msinf.likelihood.prevalence_row_loglik_nodes`. A sampler
started near the median never reaches the saturated tail, so this went unnoticed
until a Gaussian variational family started drawing from the whole unconstrained
space; the two tests below are the two halves of why the fix is safe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from msinf.dtypes import set_precision
from msinf.likelihood import Q_EPS, prevalence_row_loglik_nodes, study_effect_nodes

set_precision("f64")

NUMER = jnp.asarray([20.0])
DENOM = jnp.asarray([100.0])


def _total(p, sigma_p, alpha_p):
    _, w_nodes = study_effect_nodes(jnp.asarray(0.5), sigma_p)
    return jnp.sum(
        prevalence_row_loglik_nodes(
            p=p, numer=NUMER, denom=DENOM,
            theta_prev=1.0 / jnp.clip(alpha_p, 1e-8), w_nodes=w_nodes,
        )
    )


@pytest.mark.parametrize("sigma_p", [0.5, 2.0, 4.0, 6.0, 12.0])
def test_the_gradient_is_finite_however_far_out_the_study_effect_goes(sigma_p):
    """Where q saturates the density is tiny, but the gradient must still point.

    Without the clip this is NaN from sigma_p ~ 4 up -- and NaN in both forward
    and reverse mode, because betaln(a, 0) is genuinely singular rather than an
    artifact of how the derivative is taken. Adam carries one NaN into its
    moments and never recovers, so this is fatal rather than noisy.
    """
    p, alpha_p = jnp.asarray([0.2]), jnp.asarray(0.2)
    grads = jax.grad(_total, argnums=(0, 1, 2))(p, jnp.asarray(sigma_p), alpha_p)
    for g in grads:
        assert np.all(np.isfinite(np.asarray(g))), f"non-finite gradient at sigma_p={sigma_p}"

    # Forward mode too: if only reverse mode were fixed the clip would be
    # papering over an accumulation bug rather than a singularity.
    _, tangent = jax.jvp(
        lambda s: _total(p, s, alpha_p), (jnp.asarray(sigma_p),), (jnp.asarray(1.0),)
    )
    assert np.isfinite(float(tangent))


def test_the_clip_is_inert_where_the_sampler_actually_goes():
    """The clip must not move any number this repo has already reported.

    Drawn from the actual priors -- sigma_p ~ HalfNormal(0.5) and alpha_p from
    its half-t -- rather than from a convenient range, because the claim being
    tested is about the region the fit occupies.
    """
    rng = np.random.default_rng(0)
    numer = jnp.asarray(rng.integers(0, 100, 40).astype(float))
    denom = jnp.asarray(np.full(40, 100.0))

    def unclipped(p, theta_prev, w_nodes):
        logit_p = jax.scipy.special.logit(jnp.clip(p, 1e-6, 1 - 1e-6))
        q = jax.nn.sigmoid(logit_p[:, None] + w_nodes[None, :])
        a, b = q * theta_prev, (1.0 - q) * theta_prev
        n_pos, n_t = numer[:, None], denom[:, None]
        lch = (jax.scipy.special.gammaln(n_t + 1.0)
               - jax.scipy.special.gammaln(n_pos + 1.0)
               - jax.scipy.special.gammaln(n_t - n_pos + 1.0))
        return lch + jax.scipy.special.betaln(a + n_pos, n_t - n_pos + b) - jax.scipy.special.betaln(a, b)

    for _ in range(50):
        sigma_p = abs(rng.normal(0.0, 0.5))
        alpha_p = abs(rng.standard_t(6) * 0.25) + 1e-6
        p = jnp.asarray(rng.uniform(0.001, 0.999, 40))
        _, w_nodes = study_effect_nodes(jnp.asarray(0.5), jnp.asarray(sigma_p))
        theta = 1.0 / max(alpha_p, 1e-8)
        got = np.asarray(prevalence_row_loglik_nodes(
            p=p, numer=numer, denom=denom, theta_prev=theta, w_nodes=w_nodes))
        want = np.asarray(unclipped(p, theta, w_nodes))
        assert np.array_equal(got, want), "the clip moved a value inside the sampled region"


def test_q_eps_is_small_enough_to_be_invisible_and_large_enough_to_be_finite():
    assert 0 < Q_EPS <= 1e-9
