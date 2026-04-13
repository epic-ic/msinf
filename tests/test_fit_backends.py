"""Every backend has to hand back the same thing, so scoring stays one code path.

The point of :mod:`msinf.fit` is that :func:`msinf.evaluate.evaluate` does not
know which backend ran. That only holds if the draws come back under the same
names with the same shapes, so most of what is asserted here is sameness rather
than correctness -- correctness is the bake-off's job, against real data.

Fitted against `tests.fixtures`, which is the real observational scaffolding
(both ``numpyro.factor`` totals, the Gauss-Hermite marginalisation, every
awkward support) with a closed-form curve provider in place of the ODE.
"""

import numpy as np
import pytest

from msinf.dtypes import set_precision

set_precision("f64")

from fixtures import toy_cfg, toy_setup

from msinf.bridge import build_unconstrained
from msinf.evaluate import fit_diagnostics
from msinf.fit import fit

# Every test here runs real fits, which JIT-compile the model; see pyproject's
# pytest addopts for how to include them.
pytestmark = pytest.mark.slow

# Budgets small enough to run in CI, large enough that every code path executes.
ARMS = {
    "nuts": {"fit_backend": "mcmc"},
    "sa": {"fit_backend": "mcmc", "sampler": "sa"},
    "pathfinder": {"fit_backend": "pathfinder", "vi_draws": 120},
    "fullrank": {"fit_backend": "fullrank", "vi_steps": 120, "vi_draws": 120},
    "nuts+pathfinder": {"fit_backend": "mcmc", "init_strategy": "pathfinder",
                        "pathfinder_draws": 16},
}


@pytest.fixture(scope="module")
def setup():
    return toy_setup()


@pytest.fixture(scope="module")
def fits(setup):
    model_kwargs, active = setup
    return {name: fit(toy_cfg(**over), model_kwargs, active) for name, over in ARMS.items()}


def test_every_backend_returns_the_same_sites_with_the_same_shapes(fits):
    reference = fits["nuts"]
    for name, result in fits.items():
        assert set(result.draws) == set(reference.draws), f"{name} returned different sites"
        for site, value in result.draws.items():
            assert value.shape[1:] == reference.draws[site].shape[1:], f"{name}/{site}"
            assert np.all(np.isfinite(value)), f"{name}/{site} has non-finite draws"


def test_supports_are_respected_after_constraining(fits):
    """The bijections have to actually land inside each site's support.

    Cheap, and it is the one thing that would silently poison every downstream
    number if the unconstrained bridge picked the wrong transform.
    """
    for name, result in fits.items():
        for site in ("sigma_c", "sigma_p", "alpha_c", "alpha_p", "toy_scale"):
            assert np.all(result.draws[site] > 0), f"{name}/{site} left its support"
        for site in ("ft_study", "toy_shape", "rW", "rP"):
            v = result.draws[site]
            assert np.all((v > 0) & (v < 1)), f"{name}/{site} left the unit interval"


def test_mcmc_backends_carry_chains_and_variational_ones_do_not(fits):
    for name, result in fits.items():
        if result.backend in ("nuts", "sa"):
            assert result.draws_by_chain is not None, f"{name} lost its chain structure"
            assert result.khat is None and result.elbo is None
        else:
            assert result.draws_by_chain is None
            assert result.khat is not None and result.elbo is not None


def test_diagnostics_name_their_backend_and_always_gate(fits):
    """`fit_ok` and `fit_seconds` are the two keys a caller can rely on blind."""
    for name, result in fits.items():
        diagnostics = fit_diagnostics(result)
        assert "fit_ok" in diagnostics and "fit_seconds" in diagnostics
        chain_keys = {"max_r_hat", "min_ess", "min_ess_tail"}
        variational_keys = {"khat", "elbo"}
        if result.draws_by_chain is not None:
            assert chain_keys <= set(diagnostics), f"{name} is missing chain diagnostics"
            assert not (variational_keys & set(diagnostics)), \
                f"{name} reported a variational diagnostic for a chain"
        else:
            assert variational_keys <= set(diagnostics)
            assert not (chain_keys & set(diagnostics)), \
                f"{name} reported r_hat/ESS for a fit with no chains"


def test_pathfinder_init_is_still_an_mcmc_fit_but_reports_its_approximation(fits):
    """Seeding from Pathfinder must not turn the fit into a variational one."""
    seeded = fits["nuts+pathfinder"]
    assert seeded.backend == "nuts"
    assert seeded.draws_by_chain is not None
    assert seeded.pathfinder_khat is not None and seeded.pathfinder_seconds is not None
    diagnostics = fit_diagnostics(seeded)
    # The approximation's cost is inside fit_seconds and outside mcmc_seconds.
    assert diagnostics["fit_seconds"] > diagnostics["mcmc_seconds"]
    assert {"pathfinder_khat", "pathfinder_elbo", "max_r_hat"} <= set(diagnostics)


def test_pathfinder_seeds_each_chain_from_a_different_point(setup):
    """Four chains from one point cannot disagree about where they started.

    Half of what r_hat checks is start dispersion, so a shared start would make
    the diagnostic that decides this experiment quietly weaker.
    """
    model_kwargs, active = setup
    result = fit(toy_cfg(init_strategy="pathfinder", pathfinder_draws=16, num_chains=2),
                 model_kwargs, active)
    starts = result.extra["pathfinder_init"]
    for site, value in starts.items():
        assert value.shape[0] == 2, f"{site} does not carry one start per chain"
    assert any(not np.allclose(v[0], v[1]) for v in starts.values()), \
        "every chain was seeded from the same point"


def test_psir_resamples_without_changing_the_shape(setup):
    model_kwargs, active = setup
    base = toy_cfg(fit_backend="fullrank", vi_steps=120, vi_draws=120)
    plain = fit(base, model_kwargs, active)
    resampled = fit(toy_cfg(fit_backend="fullrank", vi_steps=120, vi_draws=120, vi_psir=True),
                    model_kwargs, active)
    assert set(plain.draws) == set(resampled.draws)
    for site in plain.draws:
        assert plain.draws[site].shape == resampled.draws[site].shape
    assert resampled.extra["psir"] is True


def test_a_deterministic_site_survives_into_the_draws(setup):
    """A searched candidate may write one, and both backends must agree on it.

    See msinf.bridge: without the model-replaying path this site would be
    present under MCMC and missing under a variational fit, and anything that
    enumerates draws across backends would then be comparing different things.
    """
    model_kwargs, active = toy_setup(deterministic=True)
    for backend, over in (("mcmc", {}), ("fullrank", {"vi_steps": 60, "vi_draws": 60})):
        result = fit(toy_cfg(fit_backend=backend, **over), model_kwargs, active)
        assert "toy_ratio" in result.draws, f"{backend} dropped the deterministic site"
        expected = result.draws["toy_scale"] / (result.draws["toy_shape"] + 1.0)
        np.testing.assert_allclose(result.draws["toy_ratio"], expected, rtol=1e-6)


def test_the_bridge_takes_the_cheap_path_when_it_can(setup):
    """Replaying the model per draw costs an ODE solve per draw on the real model."""
    model_kwargs, active = setup
    assert build_unconstrained(toy_cfg(), model_kwargs, active).replays_model is False
    deterministic_kwargs, _ = toy_setup(deterministic=True)
    assert build_unconstrained(toy_cfg(), deterministic_kwargs, active).replays_model is True


def test_the_factor_terms_reach_the_log_density(setup):
    """The whole likelihood is two numpyro.factor totals; if they were dropped
    the potential would still be finite and every fit would silently sample the
    prior. Compared against the same model with the data zeroed out."""
    model_kwargs, active = setup
    unconstrained = build_unconstrained(toy_cfg(), model_kwargs, active)
    with_data = float(unconstrained.logdensity(unconstrained.z_init))

    empty = dict(model_kwargs)
    empty["numer_inc"] = np.zeros_like(np.asarray(model_kwargs["numer_inc"]))
    empty["numer_prev_m"] = np.zeros_like(np.asarray(model_kwargs["numer_prev_m"]))
    without = float(build_unconstrained(toy_cfg(), empty, active).logdensity(unconstrained.z_init))
    assert not np.isclose(with_data, without), "the observations do not reach the log density"
