"""The per-row outputs of :func:`msinf.evaluate.evaluate`: PSIS-LOO and the fitted frame.

Unit-level, on hand-built arrays: the fit that feeds them is tested elsewhere,
and what is worth pinning here is the arithmetic and the alignment -- that a
row's prediction is in the row's own units and that an incidence row carries its
cell's score.
"""

import numpy as np
import pandas as pd

from msinf.evaluate import _fitted_frame, _pointwise_elpd, _psis_loo


def test_loo_sits_below_the_in_sample_elpd_and_reports_one_k_per_unit():
    rng = np.random.default_rng(0)
    loglik = rng.normal(-3.0, 0.4, size=(400, 6))
    loo, khat = _psis_loo(loglik)
    assert loo.shape == khat.shape == (6,)
    assert np.all(loo <= _pointwise_elpd(loglik) + 1e-12)
    assert np.all(khat < 0.7)


def test_loo_of_nothing_is_nothing():
    loo, khat = _psis_loo(np.zeros((10, 0)))
    assert loo.size == khat.size == 0


def test_the_fitted_frame_puts_each_row_beside_its_predictive_in_its_own_units():
    frame = pd.DataFrame(
        {
            "type": [1, 1, 2],
            "numer": [10.0, 20.0, 3.0],
            "denom": [10.0, 10.0, 6.0],
            "site_i": [0, 0, 1],
            "age_bracket": [1, 2, 1],
            "age0": [0.0, 1.0, 0.0],
            "age1": [1.0, 5.0, 5.0],
        }
    )
    reps = {"inc": np.array([[5.0, 10.0], [15.0, 30.0]]), "prev_m": np.array([[2.0], [4.0]])}
    pits = {"inc": np.array([0.2, 0.9]), "prev_m": np.array([0.5])}
    n_age_groups = 2
    # cells are site * n_age_groups + (age_bracket - 1): rows 0 and 1 are cells 0 and 1
    terms = {
        "loglik_inc": np.log(np.array([[0.5, 0.1, 1.0, 1.0], [0.5, 0.3, 1.0, 1.0]])),
        "loglik_prev_m": np.log(np.array([[0.2], [0.4]])),
    }
    loo = {"inc": np.array([-1.0, -2.0, np.nan, np.nan]), "prev_m": np.array([-3.0])}
    out = _fitted_frame(frame, reps, pits, terms, n_age_groups, loo)

    assert list(out["stream"]) == ["incidence", "incidence", "prevalence"]
    np.testing.assert_allclose(out["observed"], [1.0, 2.0, 0.5])
    # replicate counts over the row's denominator: 5/10 and 15/10 for the first row
    np.testing.assert_allclose(out["pred_q50"], [1.0, 2.0, 0.5])
    np.testing.assert_allclose(out["pit"], [0.2, 0.9, 0.5])
    np.testing.assert_allclose(out["elpd"], [np.log(0.5), np.log(0.2), np.log(0.3)])
    np.testing.assert_allclose(out["loo"], [-1.0, -2.0, -3.0])
    assert list(out["elpd_unit"]) == ["cell", "cell", "row"]
