"""The heterogeneity quadrature: dmeq's own at 10 nodes, and no ripple at 40."""

import jax
import jax.numpy as jnp
import numpy as np

from msinf.dtypes import set_precision

set_precision("f64")

from dmeq import default_parameters, solve

from msinf.age_grid import GRIFFIN_AGE_GRID
from msinf.curves import DMEQ_HET_NODES, griffin_curves, het_quadrature

AGES = np.asarray(GRIFFIN_AGE_GRID)
MU = np.full(AGES.shape, 0.0001305)   # dmeq's default uniform death rate

# Core parameters of one posterior draw of the 2014 fit (ad0 in years). kc,
# the steepness of the clinical-immunity threshold, is 8.5 against dmeq's 2.37:
# a threshold that sharp is what ten nodes ~1.25 apart in log EIR alias.
POSTERIOR_DRAW = {
    "IB0": 7.4671, "IC0": 34.8878, "ID0": 2.2709, "PM": 0.7767, "ad0": 11.1369, "b0": 0.7777,
    "d1": 0.2582, "dm": 81.9652, "fd0": 0.0052, "gd": 5.9397, "kb": 1.3711, "kc": 8.5296,
    "kd": 0.7164, "phi0": 0.7718, "phi1": 0.0022, "ub": 1.5018, "uc": 8.4966, "ud": 12.4678,
}


def params(draw: dict | None = None) -> dict:
    p = {k: jnp.asarray(v) for k, v in default_parameters().items()}
    for k, v in (draw or {}).items():
        p[k] = jnp.asarray(v * (365.0 if k == "ad0" else 1.0))
    return p


def test_ten_nodes_is_dmeqs_own_grid():
    # dmeq hardcodes hermegauss(10) to seven significant figures
    z, w = het_quadrature(DMEQ_HET_NODES)
    assert abs(z.max() - 4.8594628) < 1e-6
    assert abs(w.sum() - 1.0) < 1e-12 and abs(w.max() - 3.446423e-01) < 1e-6
    # and at ten the provider leaves dmeq on it, so old runs reproduce exactly
    p = params()
    prev, inc = griffin_curves(p, jnp.log(50.0), 0.1, AGES, None, MU, het_nodes=DMEQ_HET_NODES)
    out = solve({**p, "EIR": jnp.exp(jnp.log(50.0)), "ft": 0.1, "eta": MU}, age_bins_years=AGES)
    np.testing.assert_array_equal(prev, out[0])
    np.testing.assert_array_equal(inc, out[2])


def ripples(het_nodes: int) -> int:
    """Turning points of 15-20y incidence along log EIR in [3, 6.5]."""
    p = params(POSTERIOR_DRAW)
    grid = jnp.linspace(3.0, 6.5, 57)
    curves = jax.jit(jax.vmap(lambda v: griffin_curves(p, v, 0.0, AGES, None, MU, het_nodes=het_nodes)[1]))
    rate = np.asarray(curves(grid))[:, (AGES >= 15) & (AGES < 20)].mean(1)
    return int((np.diff(np.sign(np.diff(rate))) != 0).sum())


def test_ten_nodes_ripple_and_forty_do_not():
    # One turning point is real: adult incidence is U-shaped in EIR. The rest
    # at ten nodes are the quadrature, and are what gave log_eir its comb of modes.
    assert ripples(DMEQ_HET_NODES) > 1
    assert ripples(40) == 1
