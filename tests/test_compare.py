"""Round-tripping posterior draws through ``samples.npz``."""

import numpy as np
import pytest

from msinf.compare import BY_CHAIN_KEY, load_draws_npz


@pytest.mark.parametrize("by_chain", [True, False])
def test_npz_round_trip_keeps_vector_sites_the_right_shape(tmp_path, by_chain):
    """`(4, 800)` and `(800, 41)` are both 2-D and mean different things, so the
    file has to say which it is rather than the loader guessing from shapes."""
    path = tmp_path / "samples.npz"
    if by_chain:
        arrays = {"sigma_c": np.zeros((4, 50)), "log_eir": np.zeros((4, 50, 7))}
    else:
        arrays = {"sigma_c": np.zeros(200), "log_eir": np.zeros((200, 7))}
    np.savez(path, **arrays, **{BY_CHAIN_KEY: np.asarray(by_chain)},
             warmup__sigma_c=np.zeros((4, 10)))

    draws = load_draws_npz(path)
    assert set(draws) == {"sigma_c", "log_eir"}, "warmup draws leaked into the posterior"
    assert draws["sigma_c"].shape == (200,)
    assert draws["log_eir"].shape == (200, 7)


def test_a_file_without_the_marker_is_read_as_grouped_by_chain():
    """Every samples.npz written before the marker existed is an MCMC save."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "legacy.npz"
        np.savez(path, sigma_c=np.zeros((4, 50)), log_eir=np.zeros((4, 50, 7)))
        draws = load_draws_npz(path)
        assert draws["sigma_c"].shape == (200,)
        assert draws["log_eir"].shape == (200, 7)
