"""The split coherence rules, which are properties of the likelihood.

The one that matters: incidence cells cannot be scored in halves, because
``v_jk`` is integrated out at the cell level.
"""

import numpy as np
import pandas as pd
import pytest

from msinf.data import encode_canonical_sites
from msinf.splits import Split, apply_split, split_mask

BANDS = [(0.0, 1.0), (1.0, 5.0), (5.0, 15.0), (15.0, 100.0)]


def frame(n_studies: int = 3, sites_per_study: int = 2) -> pd.DataFrame:
    rows = []
    site_index = 0
    for study in range(1, n_studies + 1):
        for _ in range(sites_per_study):
            site_index += 1
            for k, (a0, a1) in enumerate(BANDS, start=1):
                for type_code in (1, 2):
                    rows.append(
                        {"type": type_code, "numer": 5, "denom": 50, "age0": a0, "age1": a1, "age_bracket": k,
                             "site_index": site_index, "study_index": study, "case_detection": 1,
                             "country_name": "X", "site_name": f"s{site_index}"}
                    )
    return encode_canonical_sites(pd.DataFrame(rows))[0]


def test_study_split_holds_out_whole_studies():
    df = frame()
    train, hold = apply_split(df, Split(kind="study", n_studies=1, seed=1))
    assert not set(hold["study_index"]) & set(train["study_index"])
    assert len(train) + len(hold) == len(df)


def test_age_band_split_keeps_every_study():
    df = frame()
    train, hold = apply_split(df, Split(kind="age_band", from_age=5.0))
    assert set(train["study_index"]) == set(df["study_index"])
    assert hold["age0"].min() >= 5.0
    assert train["age0"].max() < 5.0


def test_none_is_a_whole_data_fit():
    df = frame()
    train, hold = apply_split(df, Split(kind="none"))
    assert hold is None
    assert len(train) == len(df)
    assert not split_mask(df, None).any()


def test_a_boundary_inside_an_incidence_cell_is_refused():
    df = frame()
    # Second incidence row in the [5, 15) bracket, above a from_age=8 boundary,
    # so that cell would be split in half.
    extra = df[(df["type"] == 1) & (df["age_bracket"] == 3)].copy()
    extra["age0"] = 10.0
    df = encode_canonical_sites(pd.concat([df, extra], ignore_index=True))[0]
    with pytest.raises(ValueError, match="straddles"):
        apply_split(df, Split(kind="age_band", from_age=8.0))


def test_holding_out_every_study_is_refused():
    df = frame(n_studies=2)
    with pytest.raises(ValueError, match="nothing would be left to fit"):
        apply_split(df, Split(kind="study", n_studies=2, seed=0))


def test_empty_split_is_refused():
    df = frame()
    with pytest.raises(ValueError, match="holds out no rows"):
        apply_split(df, Split(kind="age_band", from_age=500.0))


def test_split_is_seed_reproducible_and_seed_dependent():
    df = frame(n_studies=6)
    a = split_mask(df, Split(kind="study", n_studies=2, seed=1))
    b = split_mask(df, Split(kind="study", n_studies=2, seed=1))
    c = split_mask(df, Split(kind="study", n_studies=2, seed=2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
