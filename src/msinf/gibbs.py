"""A Gibbs step for the log EIR of sites whose posterior NUTS cannot cross.

Some sites' ``log_eir`` posterior has well-separated modes: the Dicko
Doneguebougou EIR has one near 40 and one near 350-550, with an ~12 nat trough
between them (at 40 heterogeneity nodes), and which one dominates depends on the
global parameters. A NUTS chain stays in whichever mode it starts in.

:class:`numpyro.infer.HMCGibbs` alternates a NUTS step on everything else with a
user update of the split-out ``log_eir_gibbs`` site (see
:func:`msinf.model.sample_log_eir`). The update here is exact for each site in
turn, holding the others fixed:

1. Evaluate the full model log density on a grid over the site's log EIR.
2. Propose from the piecewise-constant density that grid defines: a cell with
   probability proportional to its density, then a uniform point within it.
3. Accept or reject with Metropolis-Hastings against the true conditional.

The proposal is close to the conditional itself, so acceptance is near one and
a chain can land in either mode on every iteration, whatever the trough depth.
Because proposals never leave the grid, the conditional is in effect truncated
to it; the grid spans the prior mean +/- :data:`GRID_HALF_WIDTH_SD` prior sds,
where the prior mass outside is ~6e-7.

Cost is ``n_grid + 2`` full model evaluations per Gibbs site per iteration
(forward only, no gradient).
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.handlers import seed, trace
from numpyro.infer.util import log_density

from .dtypes import asfloat

#: The sample site an HMCGibbs kernel hands to :func:`make_eir_gibbs_fn`.
GIBBS_SITE = "log_eir_gibbs"

#: Grid half-width, in prior standard deviations around the prior mean.
GRID_HALF_WIDTH_SD = 5.0


def eir_gibbs_grids(log_eir_hat, sigma_eir, sites, n_grid: int) -> np.ndarray:
    """``(len(sites), n_grid)`` evenly spaced log EIR grids, one per Gibbs site."""
    if n_grid < 2:
        raise ValueError(f"gibbs_eir_grid must be at least 2; got {n_grid}")
    idx = np.asarray(tuple(sites), dtype=int)
    mu = np.asarray(log_eir_hat, dtype=float)[idx]
    sd = np.broadcast_to(np.asarray(sigma_eir, dtype=float), np.shape(log_eir_hat))[idx]
    unit = np.linspace(-GRID_HALF_WIDTH_SD, GRID_HALF_WIDTH_SD, n_grid)
    return mu[:, None] + sd[:, None] * unit[None, :]


def _latent_sites(model, model_kwargs: dict) -> frozenset[str]:
    """Names of the model's unobserved sample sites.

    What reaches a ``gibbs_fn`` as ``hmc_sites`` is HMC's postprocessed state,
    which carries deterministic sites too. ``substitute`` overrides those, so a
    stale ``log_eir`` deterministic would pin the curves at the old value and
    the conditional would come out flat; only latent sample sites go back in.
    """
    tr = trace(seed(model, 0)).get_trace(**model_kwargs)
    return frozenset(n for n, s in tr.items() if s["type"] == "sample" and not s["is_observed"])


def make_eir_gibbs_fn(model, model_kwargs: dict, grids) -> Callable:
    """A ``gibbs_fn`` for :class:`numpyro.infer.HMCGibbs` updating :data:`GIBBS_SITE`.

    ``grids`` is :func:`eir_gibbs_grids`'s output, row ``j`` for element ``j``
    of ``log_eir_gibbs``. ``model_kwargs`` are the sampler's (no holdout).
    """
    # Resolved on first call, which happens under jit: only the site names are
    # kept, so the trace's arithmetic is dead code XLA drops. Eagerly, at kernel
    # build time, the same trace would run the ODE op by op.
    latent: frozenset[str] | None = None
    grids = asfloat(np.asarray(grids))
    widths = grids[:, 1] - grids[:, 0]

    def logp(hmc_sites, z):
        return log_density(model, (), model_kwargs, {**hmc_sites, GIBBS_SITE: z})[0]

    def update(key, z, j: int, hmc_sites):
        grid, h = grids[j], widths[j]
        k_cell, k_pos, k_acc = jax.random.split(key, 3)

        lp_grid = jax.lax.map(lambda v: logp(hmc_sites, z.at[j].set(v)), grid)
        lp_grid = jnp.where(jnp.isfinite(lp_grid), lp_grid, -jnp.inf)
        # Log density of the piecewise-constant proposal on each cell.
        log_q = jax.nn.log_softmax(lp_grid) - jnp.log(h)

        cell = jax.random.categorical(k_cell, lp_grid)
        proposal = grid[cell] + h * (jax.random.uniform(k_pos, dtype=grid.dtype) - 0.5)

        current = z[j]
        cur_cell = jnp.clip(jnp.round((current - grid[0]) / h).astype(jnp.int32), 0, grid.size - 1)
        inside = jnp.abs(current - grid[cur_cell]) <= h / 2

        log_ratio = (
            logp(hmc_sites, z.at[j].set(proposal)) - logp(hmc_sites, z)
            + log_q[cur_cell] - log_q[cell]
        )
        # A start off the grid has no proposal density to balance against, and
        # once on the grid the chain never leaves it: move in unconditionally.
        log_ratio = jnp.where(inside, log_ratio, jnp.inf)
        # A NaN ratio (a failed solve at the proposal) compares False: rejected.
        accept = jnp.log(jax.random.uniform(k_acc, dtype=grid.dtype)) < log_ratio
        return jnp.where(accept, z.at[j].set(proposal), z)

    def gibbs_fn(rng_key, gibbs_sites, hmc_sites):
        nonlocal latent
        if latent is None:
            latent = _latent_sites(model, model_kwargs) - {GIBBS_SITE}
        hmc_sites = {k: v for k, v in hmc_sites.items() if k in latent}
        z = gibbs_sites[GIBBS_SITE]
        # One site at a time: sites in one study share its integrated-out
        # effect, so their conditionals are not independent.
        for j, key in enumerate(jax.random.split(rng_key, grids.shape[0])):
            z = update(key, z, j, hmc_sites)
        return {GIBBS_SITE: z}

    return gibbs_fn
