"""Per-site demography: death rates and the equilibrium age distribution.

The death rates are a property of the data being fitted -- which country each
study site is in, and when it was surveyed -- and not of the simulator, so they
live here. ``dmeq`` owns the recursion (:func:`dmeq.age_proportions`); this
module owns the tables it is fed.

``scripts/demography/`` builds ``site_mortality_griffin_grid.csv`` from WPP
life tables: one row per (``site_index``, age class of the Griffin grid) with a
per-day death rate ``mu_per_day``. When no table is configured the model falls
back to ``dmeq``'s uniform default rate, which is what every run before this
module did, so the fallback reproduces those numbers exactly.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .age_grid import age_grid_array

# `dmeq` and `.dtypes` are imported inside the functions that need them, which
# is only the two that solve the recursion. Reading a mortality table is pandas
# and nothing else, and importing this module to do it should not start a jax
# backend: `discovery`'s analysis bench resolves the same WPP table the fit
# resolves and then weights with its own numpy copy of the recursion.


def __getattr__(name: str) -> float:
    """`DEFAULT_DEATH_RATE`, resolved on first use rather than at import.

    Uniform per-day death rate dmeq defaults to: an exponential age distribution
    with a mean around 21 years, used when no mortality table is configured.
    Asking dmeq for it is what would otherwise make importing this module import
    jax, so it is deferred like the rest.
    """
    if name == "DEFAULT_DEATH_RATE":
        from dmeq import default_parameters

        return float(default_parameters()["eta"])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_REQUIRED_COLUMNS = {"site_index", "age_class", "age_lower", "mu_per_day"}


def load_site_death_rates(
    *, df: pd.DataFrame, mortality_csv: Path, n_site: int, ages: np.ndarray | None = None
) -> np.ndarray:
    """Per-canonical-site, per-age-class death rate, in ``(n_site, n_age)``.

    ``df`` must already carry the canonical ``site_i`` column (see
    :func:`msinf.data.encode_canonical_sites`) alongside the raw ``site_index``
    the table is keyed on. Where a canonical site aggregates several raw site
    indices -- repeat surveys of the same village in different years -- the
    rates are averaged, matching how :func:`msinf.data.aggregate_eir_priors`
    aggregates the EIR priors.
    """
    ages = age_grid_array(dtype=np.float64) if ages is None else np.asarray(ages, dtype=np.float64)
    mortality_csv = Path(mortality_csv)
    if not mortality_csv.exists():
        raise FileNotFoundError(
            f"Mortality table CSV not found at {mortality_csv}. Expected columns "
            f"{sorted(_REQUIRED_COLUMNS)}; build one with scripts/demography/to_griffin_grid.py."
        )
    tab = pd.read_csv(mortality_csv)
    missing = _REQUIRED_COLUMNS - set(tab.columns)
    if missing:
        raise ValueError(f"Mortality table missing columns: {sorted(missing)}")

    n_age = int(ages.shape[0])
    rates: dict[int, np.ndarray] = {}
    for raw_id, grp in tab.groupby("site_index"):
        grp = grp.sort_values("age_class")
        if len(grp) != n_age:
            raise ValueError(
                f"Mortality table for site_index={raw_id} has {len(grp)} age classes, expected {n_age}"
            )
        if not np.allclose(grp["age_lower"].to_numpy(dtype=float), ages, atol=1e-4):
            raise ValueError(
                f"Mortality table for site_index={raw_id} is not on the model age grid"
            )
        rates[int(raw_id)] = grp["mu_per_day"].to_numpy(dtype=float)

    mu = np.zeros((n_site, n_age), dtype=np.float64)
    site_meta = df[["site_i", "site_index"]].drop_duplicates().sort_values(["site_i", "site_index"])
    seen = np.zeros(n_site, dtype=bool)
    for si, grp in site_meta.groupby("site_i"):
        raw_ids = [int(r) for r in grp["site_index"].tolist()]
        absent = [r for r in raw_ids if r not in rates]
        if absent:
            raise ValueError(f"Mortality table missing rows for raw site_index: {absent}")
        mu[int(si)] = np.mean([rates[r] for r in raw_ids], axis=0)
        seen[int(si)] = True

    if not seen.all():
        raise ValueError(f"No mortality rows resolved for canonical sites: {np.flatnonzero(~seen).tolist()}")
    if not np.all(mu[:, -1] > 0.0):
        raise ValueError(
            "the terminal age class must have a strictly positive death rate: it is "
            "absorbing, so death is the only way out of it"
        )
    return mu


def uniform_death_rates(n_site: int, n_age: int, rate: float | None = None) -> np.ndarray:
    """``(n_site, n_age)`` of a single uniform death rate -- the dmeq default.

    ``rate=None`` resolves :data:`DEFAULT_DEATH_RATE`, which asks dmeq; a default
    argument would have asked it at import time.
    """
    if rate is None:
        rate = __getattr__("DEFAULT_DEATH_RATE")
    return np.full((n_site, n_age), float(rate), dtype=np.float64)


def site_age_proportions(mu: np.ndarray, ages: np.ndarray | None = None) -> np.ndarray:
    """Equilibrium age distribution per site, ``(n_site, n_age)``, summing to 1.

    This is the demographic weight the age-bin averaging uses. Taking it from
    :func:`dmeq.age_proportions` here rather than reading it back off a row of
    ``solve``'s output makes it fixed scaffolding: a candidate curve family
    cannot move the likelihood by reweighting the age bins.
    """
    from dmeq import age_proportions

    from .dtypes import as_np_float

    ages = age_grid_array(dtype=np.float64) if ages is None else np.asarray(ages, dtype=np.float64)
    mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
    # dtype=None: dmeq solves the recursion at whatever precision JAX is
    # configured for, which is the precision the rest of the run uses.
    props = [np.asarray(age_proportions(ages, row, dtype=None)) for row in mu]
    return as_np_float(np.stack(props, axis=0))


def resolve_demography(
    *, df: pd.DataFrame, mortality_csv: Path | str | None, n_site: int, ages: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """``(mu, prop)`` for every canonical site, at the configured precision.

    With ``mortality_csv=None`` every site gets dmeq's uniform default rate and
    the resulting age distribution is the exponential one every run before the
    WPP tables existed used.
    """
    from .dtypes import as_np_float

    ages = age_grid_array(dtype=np.float64) if ages is None else np.asarray(ages, dtype=np.float64)
    if mortality_csv is None:
        mu = uniform_death_rates(n_site, int(ages.shape[0]))
    else:
        mu = load_site_death_rates(df=df, mortality_csv=Path(mortality_csv), n_site=n_site, ages=ages)
    return as_np_float(mu), site_age_proportions(mu, ages=ages)
