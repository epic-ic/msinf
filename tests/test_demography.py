"""Demography is fixed scaffolding, so it has to be reproducible and to default
to exactly what runs before the WPP tables existed used."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from dmeq import default_parameters

from msinf.age_grid import GRIFFIN_AGE_GRID, age_grid_array
from msinf.data import encode_canonical_sites
from msinf.demography import DEFAULT_DEATH_RATE, resolve_demography

MORTALITY_CSV = Path(__file__).resolve().parent.parent / "site_mortality_griffin_grid.csv"


def frame(site_indices=(1, 2, 3)) -> pd.DataFrame:
    rows = [
        {"site_index": i, "study_index": 1, "country_name": "X", "site_name": f"s{i}",
             "type": 1, "numer": 1, "denom": 1, "age0": 0.0, "age1": 1.0, "age_bracket": 1, "case_detection": 1}
        for i in site_indices
    ]
    return encode_canonical_sites(pd.DataFrame(rows))[0]


def test_default_is_dmeqs_uniform_rate():
    df = frame()
    mu, prop = resolve_demography(df=df, mortality_csv=None, n_site=3)
    assert np.allclose(np.asarray(mu), float(default_parameters()["eta"]))
    assert np.allclose(np.asarray(mu), DEFAULT_DEATH_RATE)
    assert np.allclose(np.asarray(prop).sum(axis=1), 1.0)
    # Uniform mortality is the same everywhere, so every site shares one curve.
    assert np.allclose(np.asarray(prop)[0], np.asarray(prop)[1])


@pytest.mark.skipif(not MORTALITY_CSV.exists(), reason="WPP mortality table not present")
def test_wpp_table_is_on_the_model_grid_and_per_site():
    df = frame()
    mu, prop = resolve_demography(df=df, mortality_csv=MORTALITY_CSV, n_site=3)
    mu = np.asarray(mu)
    prop = np.asarray(prop)
    assert mu.shape == (3, len(GRIFFIN_AGE_GRID))
    assert np.allclose(prop.sum(axis=1), 1.0)
    # The terminal class is absorbing: death is the only way out of it.
    assert (mu[:, -1] > 0).all()
    # Age-varying rates, unlike the uniform default.
    assert mu[0].std() > 0


@pytest.mark.skipif(not MORTALITY_CSV.exists(), reason="WPP mortality table not present")
def test_recursion_agrees_with_the_continuous_age_structure():
    """dmeq's compartment recursion against the survival integral the demography
    scripts compute independently -- different derivations, same distribution."""
    df = frame(site_indices=(1,))
    _mu, prop = resolve_demography(df=df, mortality_csv=MORTALITY_CSV, n_site=1)
    tab = pd.read_csv(MORTALITY_CSV)
    ref = tab[tab["site_index"] == 1].sort_values("age_class")["prop"].to_numpy()
    ages = age_grid_array(dtype=np.float64)
    under_5 = ages < 5.0
    assert np.asarray(prop)[0][under_5].sum() == pytest.approx(ref[under_5].sum(), abs=1e-3)
