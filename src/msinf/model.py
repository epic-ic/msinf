"""The hierarchical model: fixed observational scaffolding plus a curve provider.

The observational model is fixed and stays fixed -- the study random effects
(``u`` incidence, ``w`` prevalence, Gauss-Hermite integrated) and the site-age
effect ``v_jk`` (analytically integrated, negative binomial), the incidence
Poisson and the prevalence beta-binomial, the site EIR priors and ``ft_study``.
What varies between candidate models is only the map from parameters to age
curves, which lives behind the ``parameters``/``curves`` pair of
:mod:`msinf.curves`.

The model reads no configuration. Everything it needs -- including the resolved
prior specs -- arrives through :func:`build_model_kwargs`.
"""

from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.distributions import biject_to
from numpyro.infer.initialization import init_to_median, init_to_value

from .age_grid import GRIFFIN_AGE_GRID
from .binning import weighted_bin_mean_batch
from .config import (
    NumpyroConfig,
    griffin_refit_init_values,
    resolve_active_core_params,
    resolve_priors,
    sample_param,
)
from .curves import griffin_provider
from .data import (
    aggregate_eir_priors,
    encode_canonical_sites,
    load_ft_prior_table,
    load_observed_data,
    load_reference_means,
    site_to_study_from_df,
)
from .demography import resolve_demography
from .dtypes import as_np_float, asfloat
from .likelihood import (
    incidence_cell_loglik_nodes,
    mix_over_nodes,
    prevalence_row_loglik_nodes,
    study_effect_nodes,
    sum_by_study,
)
from .priors import CORE_PARAMS
from .splits import Split, apply_split

# Data stream codes in the observation table. The 2014 table carries no PCR
# prevalence (type 3), so the model fits microscopy prevalence and incidence and
# a curve provider returns a pair rather than a triple.
TYPE_INCIDENCE = 1
TYPE_PREV_MICROSCOPY = 2


def site_curves(*, params, curves, log_eir, ft_site, ages, prop, mu):
    """Evaluate the curve provider once per site, under ``vmap``.

    Returns ``(prev_m, inc)``, each ``(n_site, n_age)`` on the fixed age grid.
    Numpyro sample sites cannot be created inside this ``vmap``, which is why
    the provider is a pair of callables and not one.
    """
    return jax.vmap(
        lambda leir, ft, prop_i, mu_i: curves(params, leir, ft, ages, prop_i, mu_i)
    )(log_eir, ft_site, prop, mu)


def bin_streams(*, site_curve_pair, prop, ages, streams):
    """Average each site curve over each observation's age bin.

    ``streams`` maps a stream name to ``(curve, site_idx, age0, age1)`` where
    ``curve`` is one of ``'prev_m' | 'inc'``.
    """
    prev_m_site, inc_site = site_curve_pair
    by_name = {"prev_m": prev_m_site, "inc": inc_site}
    return {
        name: weighted_bin_mean_batch(by_name[curve], prop, site_idx, age0, age1, ages)
        for name, (curve, site_idx, age0, age1) in streams.items()
    }


def hierarchical_model(numer_inc, denom_inc, study_inc, site_inc, agek_inc, age0_inc, age1_inc, case_det_inc,
                       numer_prev_m, denom_prev_m, study_prev_m, site_prev_m, age0_prev_m, age1_prev_m,
                       log_eir_hat, sigma_eir, ft_alpha, ft_beta, site_to_study, prop_site, mu_site,
                       n_study: int, n_site: int, n_age_groups: int, obs_priors: dict,
                       parameters, curves):
    """``parameters``/``curves`` are a :class:`msinf.curves.CurveProvider`'s pair,
    supplied by :func:`build_model_kwargs`."""
    params = parameters()

    # ---- Fixed observational parameters ------------------------------------
    rW = sample_param("rW", obs_priors["rW"])
    rP = sample_param("rP", obs_priors["rP"])

    with numpyro.plate("site", n_site):
        log_eir = numpyro.sample("log_eir", dist.Normal(log_eir_hat, sigma_eir))

    sigma_c = numpyro.sample("sigma_c", dist.HalfNormal(asfloat(0.5)))
    sigma_p = numpyro.sample("sigma_p", dist.HalfNormal(asfloat(0.5)))
    # alpha_c/alpha_p are the VARIANCES of the site-age effects v_jk (Gamma, mean 1):
    # var(v) = alpha, so sd(v) = sqrt(alpha). Griffin (2014) used a very loose
    # scale-5 half-t(6) here (median alpha ~ 3.6 => sd(v) ~ 1.9), which lets v_jk
    # absorb the entire incidence/prevalence level and removes the level signal that
    # should inform EIR. We keep Griffin's heavy-tailed half-t(6) form but tighten the
    # scale to 0.25, consistent with the sigma_c/sigma_p HalfNormal(0.5) priors: since
    # alpha is on the variance scale and sigma on the sd scale, scale = 0.5**2 = 0.25
    # makes sd(v) = sqrt(alpha) track sigma (median ~0.42, p99 ~0.96).
    _alpha_scale = asfloat(0.25)
    alpha_c = numpyro.sample(
        "alpha_c",
        dist.TruncatedDistribution(
            dist.StudentT(df=asfloat(6.0), loc=asfloat(0.0), scale=_alpha_scale),
            low=asfloat(0.0),
        ),
    )
    alpha_p = numpyro.sample(
        "alpha_p",
        dist.TruncatedDistribution(
            dist.StudentT(df=asfloat(6.0), loc=asfloat(0.0), scale=_alpha_scale),
            low=asfloat(0.0),
        ),
    )
    # Beta-binomial concentration theta = 1/alpha_p (larger theta => less overdispersion).
    theta_prev = 1.0 / jnp.clip(alpha_p, 1e-8)

    with numpyro.plate("study", n_study):
        ft_study = numpyro.sample("ft_study", dist.Beta(ft_alpha, ft_beta))

    ft_site = ft_study[site_to_study]   # treatment rate broadcast from study ell(j) to site j
    age_grid = asfloat(GRIFFIN_AGE_GRID)

    site_pair = site_curves(
        params=params, curves=curves, log_eir=log_eir, ft_site=ft_site, ages=age_grid, prop=prop_site, mu=mu_site
    )
    binned = bin_streams(
        site_curve_pair=site_pair,
        prop=prop_site,
        ages=age_grid,
        streams={
            "inc": ("inc", site_inc, age0_inc, age1_inc),
            "prev_m": ("prev_m", site_prev_m, age0_prev_m, age1_prev_m),
        },
    )
    mu_inc, p_prev_m = binned["inc"], binned["prev_m"]

    u_nodes, w_nodes = study_effect_nodes(sigma_c, sigma_p)

    # cell = (site, age-group); each cell maps to one study via its site.
    n_cells = n_site * n_age_groups
    cell_to_study = site_to_study[jnp.arange(n_cells) // n_age_groups]
    cell_id = site_inc * n_age_groups + agek_inc

    nb_cell_node = incidence_cell_loglik_nodes(
        mu_inc=mu_inc,
        numer=numer_inc,
        denom=denom_inc,
        case_det=case_det_inc,
        cell_id=cell_id,
        n_cells=n_cells,
        rW=rW,
        rP=rP,
        alpha_c=alpha_c,
        u_nodes=u_nodes,
    )
    numpyro.factor("y_inc", jnp.sum(mix_over_nodes(sum_by_study(nb_cell_node, cell_to_study, n_study))))

    study_node_prev = sum_by_study(
        prevalence_row_loglik_nodes(
            p=p_prev_m, numer=numer_prev_m, denom=denom_prev_m, theta_prev=theta_prev, w_nodes=w_nodes
        ),
        study_prev_m,
        n_study,
    )
    numpyro.factor("y_prev", jnp.sum(mix_over_nodes(study_node_prev)))


def _stream_arrays(df: pd.DataFrame, site_i: np.ndarray, study_i: np.ndarray, agek: np.ndarray) -> dict:
    """Split an already index-encoded frame into the two observation streams."""
    types = df["type"].to_numpy()
    inc = types == TYPE_INCIDENCE
    prev_m = types == TYPE_PREV_MICROSCOPY
    return {
        "numer_inc": asfloat(df.loc[inc, "numer"].to_numpy()),
        "denom_inc": asfloat(df.loc[inc, "denom"].to_numpy()),
        "study_inc": jnp.asarray(study_i[inc], dtype=jnp.int32),
        "site_inc": jnp.asarray(site_i[inc], dtype=jnp.int32),
        "agek_inc": jnp.asarray(agek[inc], dtype=jnp.int32),
        "age0_inc": asfloat(df.loc[inc, "age0"].to_numpy()),
        "age1_inc": asfloat(df.loc[inc, "age1"].to_numpy()),
        "case_det_inc": jnp.asarray(df.loc[inc, "case_detection"].to_numpy(), dtype=jnp.int32),
        "numer_prev_m": jnp.asarray(df.loc[prev_m, "numer"].to_numpy(), dtype=jnp.int32),
        "denom_prev_m": jnp.asarray(df.loc[prev_m, "denom"].to_numpy(), dtype=jnp.int32),
        "study_prev_m": jnp.asarray(study_i[prev_m], dtype=jnp.int32),
        "site_prev_m": jnp.asarray(site_i[prev_m], dtype=jnp.int32),
        "age0_prev_m": asfloat(df.loc[prev_m, "age0"].to_numpy()),
        "age1_prev_m": asfloat(df.loc[prev_m, "age1"].to_numpy()),
    }


def build_model_kwargs(
    cfg: NumpyroConfig,
    provider=None,
    split: Split | None = None,
) -> tuple[dict, tuple[str, ...], dict[str, float]]:
    """Assemble the model's data arguments from the configured input CSVs.

    Returns ``(model_kwargs, active_core_params, fixed_core_values)``. Shared by
    the sampler, the likelihood diagnostic and the held-out evaluation, so all
    of them evaluate the model on identically constructed inputs.

    ``provider`` is the :class:`msinf.curves.CurveProvider` to fit; ``None``
    means the incumbent Griffin/dmeq path, resolved from ``cfg``.

    ``split`` names a holdout. The frame is split *before* index encoding and
    both halves are then encoded against the same union of site and study
    levels, so dropping a study cannot silently renumber the site indices out
    from under the held-out arrays. The held-out sites keep their ``log_eir``
    sample site -- with no training rows it simply sits on its prior, which is
    what scoring a held-out study means. The held-out arrays are returned on
    ``model_kwargs['holdout']`` for :mod:`msinf.evaluate` and are not read by
    the model.
    """
    # Resolved here rather than inside the model, so a bad family fails before
    # the sampler traces the model (where the error surfaces from inside pmap).
    # Only rW/rP reach the model; the rest go to the default provider.
    priors = resolve_priors(cfg.prior_family)

    # CORE_PARAMS, the active/fixed core split, the reference draws and
    # init-by-value are all keyed on Griffin's parameter names. A candidate
    # invents its own, so none of them apply and none of them are resolved.
    active_core_params: tuple[str, ...] = ()
    fixed_core_values: dict[str, float] = {}
    if provider is None:
        active_core_params = resolve_active_core_params(cfg)
        inactive_core_params = tuple([p for p in CORE_PARAMS if p not in set(active_core_params)])
        if inactive_core_params:
            if cfg.reference_draws_csv is None:
                raise ValueError(
                    "Inactive core parameters requested, but reference_draws_csv is not set. "
                    "Provide a reference posterior draws CSV to derive fixed marginal means."
                )
            ref_means = load_reference_means(Path(cfg.reference_draws_csv))
            fixed_core_values = {p: ref_means[p] for p in inactive_core_params}
        provider = griffin_provider(priors, active_core_params, fixed_core_values)
    elif str(cfg.init_strategy).strip().lower() == "griffin":
        raise ValueError(
            "init_strategy='griffin' initialises by Griffin's parameter names, which a custom "
            "curve provider does not have. Use init_strategy='median', or 'griffin_jitter' to "
            "start the sites it shares with Griffin from Griffin's values."
        )

    df = load_observed_data(Path(cfg.data_csv))
    # age_bracket is 1-indexed and can be arbitrary; don't hard-code 4 bands.
    if (df["age_bracket"].to_numpy(dtype=int) < 1).any():
        raise ValueError("age_bracket must be 1-indexed")
    known_types = {TYPE_INCIDENCE, TYPE_PREV_MICROSCOPY}
    unknown = sorted(set(df["type"].to_numpy().tolist()) - known_types)
    if unknown:
        raise ValueError(
            f"observation table has unsupported type codes {unknown}; this model fits "
            f"incidence ({TYPE_INCIDENCE}) and microscopy prevalence ({TYPE_PREV_MICROSCOPY}) only. "
            "PCR prevalence (3) was dropped because the 2014 table carries none; adding it back "
            "means a third curve from the provider and a third stream through the likelihood."
        )
    df, _ = encode_canonical_sites(df)

    df_train, df_hold = apply_split(df, split)

    # Both halves are encoded against the union: site_i already spans it (it is
    # derived from the full frame above) and the study map is taken from the
    # full frame too.
    site_to_study, study_map, _ = site_to_study_from_df(df)
    n_site = int(df["site_i"].max() + 1)
    n_study = len(study_map)
    n_age_groups = int(df["age_bracket"].to_numpy(dtype=int).max())

    def encode(frame: pd.DataFrame) -> dict:
        return _stream_arrays(
            frame,
            frame["site_i"].to_numpy(dtype=int),
            frame["study_index"].map(study_map).to_numpy(dtype=int),
            frame["age_bracket"].to_numpy(dtype=int) - 1,
        )

    ft_prior_df = load_ft_prior_table(df=df, ft_csv=Path(cfg.ft_priors_csv) if cfg.ft_priors_csv else None)
    ft_alpha = as_np_float(ft_prior_df["ft_alpha"].to_numpy())
    ft_beta = as_np_float(ft_prior_df["ft_beta"].to_numpy())
    if len(ft_alpha) != n_study:
        raise ValueError(f"FT prior rows ({len(ft_alpha)}) do not match n_study ({n_study})")

    # Resolved over the union, not over the training frame: a held-out study's
    # sites still need their EIR prior rows even though they contribute no data.
    site_eir_mu, site_eir_sigma = aggregate_eir_priors(
        df=df, eir_priors_csv=Path(cfg.eir_priors_csv), n_site=n_site
    )
    mu_site, prop_site = resolve_demography(df=df, mortality_csv=cfg.mortality_csv, n_site=n_site)

    model_kwargs = dict(
        **encode(df_train),
        log_eir_hat=asfloat(site_eir_mu),
        sigma_eir=asfloat(site_eir_sigma),
        ft_alpha=asfloat(ft_alpha),
        ft_beta=asfloat(ft_beta),
        site_to_study=jnp.asarray(site_to_study, dtype=jnp.int32),
        prop_site=asfloat(prop_site),
        mu_site=asfloat(mu_site),
        n_study=n_study,
        n_site=n_site,
        n_age_groups=n_age_groups,
        obs_priors={k: priors[k] for k in ("rW", "rP")},
        parameters=provider.parameters,
        curves=provider.curves,
    )
    if df_hold is not None:
        model_kwargs["holdout"] = encode(df_hold)
    return model_kwargs, active_core_params, fixed_core_values


def model_kwargs_for_sampler(model_kwargs: dict) -> dict:
    """Strip the entries the model itself does not take (the holdout arrays)."""
    return {k: v for k, v in model_kwargs.items() if k != "holdout"}


#: Ways to start a fit. ``pathfinder`` is not a NumPyro initializer -- it runs a
#: BlackJAX Pathfinder approximation first and seeds each chain from a draw of
#: it, so it is resolved by :mod:`msinf.fit` and not here.
INIT_STRATEGIES = ("griffin", "griffin_jitter", "median", "default", "pathfinder")


def init_to_jittered_value(site=None, values=None, scale=0.5, init_strategy=init_to_median):
    """``init_to_value``, displaced per chain by ``scale`` on the unconstrained scale.

    ``init_to_value`` starts every chain at one point, and chains that cannot
    disagree about where they started cannot show a posterior with two modes:
    under the study split the Dicko Doneguebougou EIR has two, about 50 and
    about 320, that fit the training rows equally well. Each chain draws its
    displacement from its own init key, so the starts differ between chains and
    are reproducible within one.

    A site with no value, with a value of the wrong shape, or whose value falls
    outside its support (a candidate model reusing a Griffin name for something
    else) falls back to ``init_strategy``.
    """
    if site is None:
        return partial(init_to_jittered_value, values=values or {}, scale=scale, init_strategy=init_strategy)
    if (
        site["type"] == "sample"
        and not site["is_observed"]
        and site["name"] in values
        and not site["fn"].support.is_discrete
    ):
        value = jnp.asarray(values[site["name"]])
        shape = tuple(site["kwargs"].get("sample_shape", ())) + tuple(site["fn"].shape())
        if value.shape == shape:
            key_jitter, key_fallback = jax.random.split(site["kwargs"]["rng_key"])
            transform = biject_to(site["fn"].support)
            start = transform.inv(value)
            moved = transform(start + scale * jax.random.normal(key_jitter, shape, dtype=start.dtype))
            fallback = init_strategy(dict(site, kwargs={**site["kwargs"], "rng_key": key_fallback}))
            return jnp.where(jnp.isfinite(start) & jnp.isfinite(moved), moved, fallback)
    return init_strategy(site)


def resolve_init_strategy(
    cfg: NumpyroConfig,
    model_kwargs: dict,
    active_core_params: tuple[str, ...],
    *,
    allow_pathfinder: bool = True,
):
    """The NumPyro init strategy named by ``cfg.init_strategy``, or ``None``.

    * ``griffin`` -- Griffin-refit fixed starts by parameter name. Only
      meaningful on the Hydra path, where the parameters *are* Griffin's.
    * ``griffin_jitter`` -- the same starts, displaced per chain by
      ``cfg.init_jitter`` (:func:`init_to_jittered_value`), for every sample
      site that has one; ``init_to_median`` for the rest. Works for a candidate
      model too: the observational scaffolding and any Griffin name it keeps
      start where the incumbent does.
    * ``median`` -- ``init_to_median``. The name-free option: a candidate model
      invents its own parameters, so there are no known-good values to init to.
    * ``default`` -- NumPyro's own initializer, returned as ``None``.
    * ``pathfinder`` -- resolved to ``init_to_median`` here, because that is
      where the Pathfinder run itself starts. :func:`msinf.fit.fit_mcmc` then
      seeds the chains from Pathfinder draws via ``init_params``.

    ``allow_pathfinder=False`` is for callers that *are* the Pathfinder run and
    must not recurse into it; they get the strategy it starts from.
    """
    init_mode = str(cfg.init_strategy).strip().lower()
    if init_mode not in INIT_STRATEGIES:
        raise ValueError(f"init_strategy must be one of {set(INIT_STRATEGIES)}")
    del allow_pathfinder  # both paths resolve 'pathfinder' to where it starts

    if init_mode in {"griffin", "griffin_jitter"}:
        values = griffin_refit_init_values(
            # A candidate provider has no active core set; offer every Griffin
            # name and let the sites it actually samples pick theirs out.
            active_core_params=active_core_params if init_mode == "griffin" else CORE_PARAMS,
            log_eir_hat=np.asarray(model_kwargs["log_eir_hat"]),
            ft_alpha=np.asarray(model_kwargs["ft_alpha"]),
            ft_beta=np.asarray(model_kwargs["ft_beta"]),
            n_study=model_kwargs["n_study"],
            n_site=model_kwargs["n_site"],
            n_age_groups=model_kwargs["n_age_groups"],
        )
        if init_mode == "griffin":
            return init_to_value(values=values)
        return init_to_jittered_value(values=values, scale=cfg.init_jitter)
    if init_mode in {"median", "pathfinder"}:
        return init_to_median
    return None
