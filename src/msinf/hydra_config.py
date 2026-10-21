"""Resolving a Hydra config into the plain dataclasses the package works with.

Only the Hydra entry points import this. :func:`msinf.evaluate.evaluate` and
everything under it take a :class:`msinf.config.NumpyroConfig` and a
:class:`msinf.splits.Split` and know nothing about Hydra, working directories or
output dirs, so they stay callable from inside somebody else's child process.
"""

from pathlib import Path

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from .config import NumpyroConfig
from .splits import Split


def parse_core_csv(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    txt = str(raw).strip()
    if txt == "":
        return None
    if txt.lower() == "none":
        return ()
    parts = tuple(p.strip() for p in txt.split(",") if p.strip())
    return tuple(dict.fromkeys(parts))


def _abs(value) -> str | None:
    if value is None:
        return None
    txt = str(value).strip()
    return str(Path(to_absolute_path(txt))) if txt else None


def _abs_required(value) -> str:
    path = _abs(value)
    if path is None:
        raise ValueError("a required input path is empty")
    return path


def numpyro_config(cfg: DictConfig, *, progress_bar: bool = True) -> NumpyroConfig:
    """The resolved run configuration, with every path made absolute."""
    return NumpyroConfig(
        data_csv=_abs_required(cfg.data.data_csv),
        eir_priors_csv=_abs_required(cfg.data.eir_priors_csv),
        num_warmup=int(cfg.mcmc.num_warmup),
        num_samples=int(cfg.mcmc.num_samples),
        num_chains=int(cfg.mcmc.num_chains),
        chain_method=str(cfg.mcmc.chain_method),
        max_tree_depth=int(cfg.mcmc.max_tree_depth),
        sampler=str(cfg.sampler.name),
        sa_adapt_state_size=(
            None if cfg.sampler.sa_adapt_state_size is None else int(cfg.sampler.sa_adapt_state_size)
        ),
        sa_dense_mass=bool(cfg.sampler.sa_dense_mass),
        active_core_params=parse_core_csv(cfg.core.active_core_csv),
        inactive_core_params=parse_core_csv(cfg.core.inactive_core_csv),
        reference_draws_csv=_abs(cfg.data.reference_draws_csv),
        ft_priors_csv=_abs(cfg.data.ft_priors_csv),
        mortality_csv=_abs(cfg.data.get("mortality_csv", None)),
        init_strategy=str(cfg.mcmc.init_strategy),
        init_jitter=float(cfg.mcmc.get("init_jitter", 0.5)),
        het_nodes=int(cfg.curves.get("het_nodes", 40)),
        prior_family=str(cfg.priors.name),
        collect_warmup=bool(cfg.run.collect_warmup),
        seed=int(cfg.run.seed),
        progress_bar=progress_bar,
        fit_backend=str(cfg.fit.backend),
        pathfinder_maxiter=int(cfg.fit.pathfinder.maxiter),
        pathfinder_maxcor=int(cfg.fit.pathfinder.maxcor),
        pathfinder_draws=int(cfg.fit.pathfinder.draws),
        vi_steps=int(cfg.fit.vi.steps),
        vi_learning_rate=float(cfg.fit.vi.learning_rate),
        vi_elbo_samples=int(cfg.fit.vi.elbo_samples),
        vi_draws=int(cfg.fit.vi.draws),
        vi_psir=bool(cfg.fit.vi.psir),
    )


def split_from_cfg(cfg: DictConfig) -> Split:
    """The configured holdout; ``kind: none`` (the default) is a whole-data fit."""
    holdout = cfg.holdout
    return Split(
        kind=str(holdout.kind),
        seed=int(holdout.seed),
        n_studies=int(holdout.n_studies),
        from_age=float(holdout.from_age),
    )
