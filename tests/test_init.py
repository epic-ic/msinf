"""``griffin_jitter`` starts each chain near Griffin's values, and no two alike."""

import jax
import numpy as np

from msinf.dtypes import set_precision

set_precision("f64")

from fixtures import toy_cfg, toy_setup
from numpyro.infer.util import initialize_model

from msinf.model import hierarchical_model, resolve_init_strategy


def starts(init_strategy: str, chains: int = 3, **overrides) -> dict[str, np.ndarray]:
    """Each chain's constrained starting point, as ``initialize_model`` finds it."""
    model_kwargs, active = toy_setup()
    strategy = resolve_init_strategy(toy_cfg(init_strategy=init_strategy, **overrides), model_kwargs, active)
    keys = jax.random.split(jax.random.PRNGKey(0), chains)
    found = []
    for key in keys:
        info, _, postprocess, _ = initialize_model(
            key, hierarchical_model, init_strategy=strategy, model_kwargs=model_kwargs
        )
        found.append(postprocess(info.z))
    return {name: np.stack([np.asarray(f[name]) for f in found]) for name in found[0]}


def test_the_starts_differ_between_chains_and_sit_near_griffins():
    model_kwargs, _ = toy_setup()
    at = starts("griffin_jitter")
    # the scaffolding has Griffin values: every chain moved off them, and not far
    assert len({tuple(row) for row in at["log_eir"]}) == 3
    np.testing.assert_allclose(at["log_eir"].mean(0), np.asarray(model_kwargs["log_eir_hat"]), atol=1.5)
    np.testing.assert_allclose(np.median(at["rW"]), 0.723, rtol=0.6)
    # the toy's own parameters have none, and still get a start inside their support
    assert np.all(at["toy_scale"] > 0) and np.all((0 < at["toy_shape"]) & (at["toy_shape"] < 1))


def test_no_jitter_is_griffins_values_exactly():
    model_kwargs, _ = toy_setup()
    at = starts("griffin_jitter", init_jitter=0.0)
    np.testing.assert_allclose(at["log_eir"], np.broadcast_to(model_kwargs["log_eir_hat"], at["log_eir"].shape))
    np.testing.assert_allclose(at["rW"], 0.723)
