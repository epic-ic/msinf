"""How :func:`msinf.evaluate.score` combines chains and leaves units out.

Unit-level, on hand-built arrays. What is pinned here is the arithmetic: that
the conditional density is the ratio that removes one unit from a study whose
effect is shared, that stacking sends the weight to the chains that predict and
shares it among chains that predict alike, and that thinning keeps every chain.
"""

import numpy as np
import pytest

from msinf.dtypes import set_precision

set_precision("f64")

import jax.numpy as jnp  # noqa: E402
from scipy.special import logsumexp  # noqa: E402

from msinf.evaluate import (  # noqa: E402
    _conditional,
    _pointwise_elpd,
    _replicate_index,
    _score_draws,
    _stack_weights,
)
from msinf.fit import FitResult  # noqa: E402
from msinf.likelihood import GH_WEIGHTS, N_GH_NODES  # noqa: E402


def test_the_conditional_density_is_the_study_over_the_study_without_the_unit():
    rng = np.random.default_rng(0)
    nodes = rng.normal(-2.0, 1.0, size=(5, N_GH_NODES))
    study = np.array([0, 0, 0, 1, 2])
    cond, log_post = _conditional(jnp.asarray(nodes), jnp.asarray(study), 3)
    cond, log_post = np.asarray(cond), np.asarray(log_post)

    log_w = np.log(GH_WEIGHTS)
    joint = logsumexp(log_w + nodes[:3].sum(0))
    for i in range(3):
        without = logsumexp(log_w + np.delete(nodes[:3], i, axis=0).sum(0))
        assert cond[i] == pytest.approx(joint - without)
    # alone in its study, a unit's conditional is its marginal
    for i in (3, 4):
        assert cond[i] == pytest.approx(logsumexp(log_w + nodes[i]))
    # the conditionals of a study's units do not add up to its joint density...
    assert cond[:3].sum() != pytest.approx(joint)
    # ...and each study's node posterior is a distribution
    np.testing.assert_allclose(np.exp(log_post).sum(axis=1), 1.0)


def test_stacking_follows_the_chains_that_predict():
    rng = np.random.default_rng(1)
    good = rng.normal(-1.0, 0.3, size=40)
    lpd = np.stack([good, good - 3.0, good - 3.0])
    weights = _stack_weights(lpd)
    assert weights.sum() == pytest.approx(1.0)
    assert weights[0] > 0.95


def test_stacking_shares_weight_among_chains_that_predict_alike():
    rng = np.random.default_rng(2)
    mode = rng.normal(-1.0, 0.3, size=60)
    # four chains in one mode, two in another that is better on some units and
    # worse on others: the weight is shared within each mode, not handed to one chain
    other = mode + rng.normal(0.0, 0.8, size=60)
    lpd = np.stack([mode + rng.normal(0, 0.01, 60) for _ in range(4)] + [other, other])
    weights = _stack_weights(lpd)
    assert np.ptp(weights[:4]) < 0.05 and np.ptp(weights[4:]) < 0.05
    assert weights.min() > 0.0


def test_one_chain_carries_all_the_weight():
    assert _stack_weights(np.zeros((1, 5))).tolist() == [1.0]


def test_weighted_elpd_is_the_plain_mean_under_equal_weights():
    rng = np.random.default_rng(3)
    loglik = rng.normal(-2.0, 0.5, size=(30, 4))
    np.testing.assert_allclose(_pointwise_elpd(loglik, np.full(30, 1 / 30)), _pointwise_elpd(loglik))


def test_draws_are_thinned_within_each_chain():
    by_chain = {"a": np.arange(4 * 10, dtype=float).reshape(4, 10)}
    result = FitResult(backend="nuts", draws={"a": by_chain["a"].reshape(-1)}, seconds=0.0, draws_by_chain=by_chain)
    draws, chain = _score_draws(result, 12)
    assert np.bincount(chain).tolist() == [3, 3, 3, 3]
    # every kept draw is the one its chain label says it came from
    assert np.all(draws["a"] // 10 == chain)
    draws, chain = _score_draws(result, None)
    assert draws["a"].size == 40 and np.bincount(chain).tolist() == [10] * 4


def test_replicates_follow_the_weights():
    weights = np.repeat([0.5, 0.0, 0.5], 10) / 10
    idx = _replicate_index(weights, 20)
    assert len(idx) == 20 and not np.any((idx >= 10) & (idx < 20))
    np.testing.assert_array_equal(_replicate_index(np.full(10, 0.1), 5), [0, 2, 4, 6, 8])
