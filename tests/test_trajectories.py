"""Study grid output and the scales used to compare fitted curves with data."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from msinf.plotting import trajectories as plotting


def test_all_studies_write_separate_pngs(tmp_path, monkeypatch):
    # Two studies with different outcome coverage; preserve global site ordering.
    frame = pd.DataFrame({
        "country_name": ["A", "B", "B"], "site_name": ["one", "two", "two"],
        "study_index": [1, 2, 2], "site_index": [1, 2, 2],
        "type": [1, 1, 2], "age0": [0., 0., 0.], "age1": [2., 2., 2.],
        "numer": [2., 3., 4.], "denom": [10., 10., 10.],
        "case_detection": [2, 1, 0], "age_bracket": [1, 1, 1],
    })
    data_csv = tmp_path / "data.csv"
    frame.to_csv(data_csv, index=False)
    samples = tmp_path / "samples.npz"
    np.savez(samples, log_eir=np.zeros((1, 2, 2)), ft_study=np.ones((1, 2, 2)),
             rW=np.full((1, 2), .5), rP=np.full((1, 2), .25))
    solved_sites = []

    def solve_curves(*, log_eir, curve_row, **kwargs):
        solved_sites.append(log_eir.shape[1])
        shape = (*log_eir.shape, len(plotting.AGE_GRID))
        prop = np.full(shape, .01)
        rate = .4 if curve_row == plotting.PREV_ROW else 1 / 365
        return prop * rate, prop

    monkeypatch.setattr(plotting, "solve_curves", solve_curves)
    paths = plotting.plot_all_study_trajectories(samples, data_csv, outdir=tmp_path,
                                                effects="none", n_draws=2, ncols=1, dpi=30)
    assert set(paths) == {"incidence", "prevalence"}
    for path in paths.values():
        assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert solved_sites == [1, 1, 1, 1]


def test_joint_draws_include_detection_and_conditional_study_effects():
    tr = plotting.StudyTrajectories(
        study_index=1, sites=np.array([0]), site_labels=["site"], ages=np.array([0, 1]),
        prev=np.full((1, 1, 2), .5), inc=np.full((1, 1, 2), 4.),
        rW=np.array([.5]), rP=np.array([.25]), data=pd.DataFrame(),
        prop=np.full((1, 1, 2), .1), u=np.log([2., 3.]), w=np.log([1., 3.]),
    )
    incidence, prop = plotting._fitted_draws(tr, 0, "incidence", det=2)
    np.testing.assert_allclose(incidence[:, 0], [4., 6.])
    prevalence, _ = plotting._fitted_draws(tr, 0, "prevalence")
    np.testing.assert_allclose(prevalence[:, 0], [.5, .75])
    assert prop.shape == incidence.shape


def test_grid_rejects_invalid_options():
    with pytest.raises(ValueError, match="positive"):
        plotting.plot_all_study_trajectories("unused", "unused", ncols=0)
