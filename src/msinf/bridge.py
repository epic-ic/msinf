"""The unconstrained view of the model, which is what BlackJAX needs.

BlackJAX takes a ``logdensity_fn`` over an unrestricted pytree and nothing else:
no notion of a support, a prior family or a transform. NumPyro already owns all
of that, so this module borrows it rather than restating it.

Two things this repo worried about turn out not to need any work:

* **The constrained/unconstrained transform.** ``initialize_model`` builds
  ``potential_fn`` over the *unconstrained* parameters, deriving each site's
  bijection from its support. ``ft_study``'s Beta becomes a logit, ``alpha_c``'s
  truncated Student-t becomes an exp, the LogNormal core parameters become
  identity-on-the-log-scale. Nothing here has to know which is which.
* **The two ``numpyro.factor`` totals.** ``numpyro.factor`` is an observed
  sample site carrying a ``Unit`` distribution, so ``log_density`` picks it up
  and it lands in ``potential_fn`` like any likelihood term. What ``factor``
  really costs this repo is ``Predictive`` and ``infer.util.log_likelihood``,
  and :mod:`msinf.predictive` and :mod:`msinf.evaluate` already replace both.

Going the other way costs whatever the candidate makes it cost, and this module
makes the same choice NumPyro makes internally. Applying each site's bijection
directly is pure array work, but it cannot recompute ``numpyro.deterministic``
sites, because those are derived rather than latent and never appear in ``z``.
NumPyro handles that by setting ``replay_model`` on any deterministic site
(``infer/util.py:527``) and switching from ``transform_fn`` to ``constrain_fn``,
which re-traces the model -- one full 41-site ODE solve per draw. So:

* **no deterministic sites** -- bijections applied directly, essentially free.
* **any deterministic site** -- ``constrain_fn``, which is exactly what an MCMC
  fit of the same model already pays through its own ``postprocess_fn``.

The incumbent declares none and takes the cheap path. A searched candidate owns
its ``parameters()`` block and may well write a ``numpyro.deterministic`` to
record a derived quantity -- that is idiomatic NumPyro, it costs the candidate
nothing at fit time, and refusing it would turn a stylistic choice by the
proposer into a dead candidate. It also keeps the two backends returning the
same *keys*: whatever an MCMC draw carries, a Pathfinder or VI draw carries too,
which is what discovery's identifiability metric assumes when it enumerates a
draw.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from numpyro import handlers
from numpyro.distributions.transforms import biject_to
from numpyro.infer.util import constrain_fn, initialize_model

from .config import NumpyroConfig
from .model import hierarchical_model, model_kwargs_for_sampler, resolve_init_strategy


@dataclass(frozen=True)
class Unconstrained:
    """The model as BlackJAX sees it, plus the way back to named parameters."""

    #: ``z -> log p(z, y)``, where ``z`` is the unconstrained pytree. The sign is
    #: the one BlackJAX wants; NumPyro's ``potential_fn`` is its negative.
    logdensity: Callable[[dict], jnp.ndarray]
    #: one unconstrained point, from the configured init strategy
    z_init: dict[str, jnp.ndarray]
    #: ``z -> constrained draw``, keyed by sample-site name, including any
    #: ``numpyro.deterministic`` sites the candidate declared. ``vmap``-able
    #: either way; see the module docstring for what it costs when it has to
    #: replay the model.
    constrain: Callable[[dict], dict]
    #: sample-site name -> unconstrained shape
    shapes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    #: whether constraining replays the model -- true iff the candidate declared
    #: deterministic sites. Reported so a slow scoring pass has a named cause.
    replays_model: bool = False

    @property
    def dim(self) -> int:
        """Total unconstrained dimension -- what a full-rank covariance is over."""
        return int(sum(int(np.prod(s, dtype=int)) for s in self.shapes.values()))

    def constrain_batch(self, zs: dict) -> dict[str, np.ndarray]:
        """Constrain a stack of draws, ``(S, ...)`` in and ``(S, ...)`` out."""
        return {k: np.asarray(v) for k, v in jax.vmap(self.constrain)(zs).items()}


def _inverse_transforms(model_trace: dict) -> dict:
    """The unconstrained -> constrained bijection for every latent sample site.

    Observed sites are skipped, which is what excludes the two ``factor``
    totals: they are observed ``Unit`` sites and carry no latent value.
    """
    transforms = {}
    for name, site in model_trace.items():
        if site["type"] != "sample" or site.get("is_observed", False):
            continue
        transforms[name] = biject_to(site["fn"].support)
    return transforms


def _deterministic_sites(model_trace: dict) -> tuple[str, ...]:
    return tuple(n for n, s in model_trace.items() if s["type"] == "deterministic")


def build_unconstrained(
    cfg: NumpyroConfig,
    model_kwargs: dict,
    active_core_params: tuple[str, ...],
    *,
    seed: int | None = None,
) -> Unconstrained:
    """The unconstrained bridge for one configured fit.

    ``model_kwargs`` is what :func:`msinf.model.build_model_kwargs` returned; the
    holdout arrays are stripped here exactly as the sampler strips them, so the
    log density is over the training rows and nothing else.

    The init strategy is the configured one -- resolved through the same
    :func:`msinf.model.resolve_init_strategy` the MCMC kernel uses, so a
    Pathfinder run starts where NUTS would have started and the comparison
    between them is about the algorithm rather than about the starting point.
    """
    sampler_kwargs = model_kwargs_for_sampler(model_kwargs)
    rng = jax.random.PRNGKey(cfg.seed if seed is None else seed)

    init_strategy = resolve_init_strategy(cfg, model_kwargs, active_core_params, allow_pathfinder=False)
    kwargs = {} if init_strategy is None else {"init_strategy": init_strategy}
    info = initialize_model(rng, hierarchical_model, model_kwargs=sampler_kwargs, **kwargs)

    z_init = info.param_info.z
    deterministic = _deterministic_sites(info.model_trace)

    if deterministic:
        # The candidate derived something worth recording. Recomputing it means
        # replaying the model, which is what NumPyro's own postprocess_fn does
        # for such a model -- so an MCMC fit of this same candidate pays the
        # identical cost and the two backends stay comparable.
        seeded = handlers.seed(hierarchical_model, rng_seed=cfg.seed if seed is None else seed)
        constrain = lambda z: constrain_fn(
            seeded, (), sampler_kwargs, z, return_deterministic=True
        )
    else:
        transforms = _inverse_transforms(info.model_trace)
        missing = sorted(set(z_init) - set(transforms))
        if missing:
            raise ValueError(f"no bijection resolved for latent sites {missing}")
        constrain = lambda z: {name: transforms[name](value) for name, value in z.items()}

    return Unconstrained(
        logdensity=lambda z: -info.potential_fn(z),
        z_init=z_init,
        constrain=constrain,
        shapes={k: tuple(np.shape(v)) for k, v in z_init.items()},
        replays_model=bool(deterministic),
    )
