"""Single source of truth for float precision.

The repo previously resolved ``jnp.float64 if jax.config.jax_enable_x64 else
jnp.float32`` in several places while hardcoding ``jnp.float32`` for the data
arrays and the ODE solve, so enabling x64 promoted the parameters but not the
data or the solve. Everything float-valued now goes through this module.

Precision is a *config* option (``precision: f32|f64`` in ``conf/config.yaml``);
:func:`set_precision` must be called at process entry, before any array is
built. ``FloatT()`` is a function rather than a module-level constant precisely
so it cannot capture a stale value at import time -- it always reflects the
currently configured precision.

Integer index arrays (site/study/age-bracket indices, count data) are not
affected; this module is about float precision only.
"""

import jax
import jax.numpy as jnp
import numpy as np

PRECISIONS = ("f32", "f64")


def set_precision(precision: str) -> None:
    """Configure JAX float precision. Call once, at process entry.

    JAX only honours ``jax_enable_x64`` if it is set before arrays are created,
    so this must run before anything that builds arrays.
    """
    p = str(precision).strip().lower()
    if p not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}; got {precision!r}")
    jax.config.update("jax_enable_x64", p == "f64")


def x64_enabled() -> bool:
    return bool(jax.config.jax_enable_x64)


def FloatT():
    """The resolved JAX float dtype for the configured precision."""
    return jnp.float64 if x64_enabled() else jnp.float32


def NpFloatT():
    """The resolved NumPy float dtype for the configured precision."""
    return np.float64 if x64_enabled() else np.float32


def asfloat(x):
    """``jnp.asarray`` at the configured float precision."""
    return jnp.asarray(x, dtype=FloatT())


def as_np_float(x):
    """``np.asarray`` at the configured float precision."""
    return np.asarray(x, dtype=NpFloatT())
