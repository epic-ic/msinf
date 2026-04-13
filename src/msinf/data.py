"""Observed-data, EIR-prior and treatment-prior table loading.

Split out of the model module (which had grown to ~775 lines); these helpers do
no inference, only IO and index encoding.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .priors import CORE_PARAMS, EIR_LOG_SD_BY_CATEGORY, REFERENCE_PARAM_MAP

# Filename the sampler writes and every diagnostic/plotting path looks for.
# Kept in one place so the artifact layout can be renamed in a single edit.
SAMPLES_FILENAME = "numpyro_jamie_samples.npz"

# Refit-2020 study-level treatment priors from malariaModelFit (ft_1..ft_14).
DEFAULT_FT_PRIOR_BY_STUDY: dict[int, tuple[float, float]] = {
    1: (8.4375, 16.5625),
    2: (11.25, 13.75),
    3: (16.875, 8.125),
    4: (11.25, 13.75),
    5: (6.5625, 18.4375),
    6: (7.5, 17.5),
    7: (7.5, 17.5),
    8: (2.8125, 22.1875),
    9: (1.875, 23.125),
    10: (1.875, 23.125),
    11: (4.6875, 20.3125),
    12: (10.3125, 14.6875),
    13: (11.25, 13.75),
    14: (11.25, 13.75),
}


def encode_indices(df: pd.DataFrame, col: str) -> np.ndarray:
    vals = pd.Categorical(df[col]).codes
    if (vals < 0).any():
        raise ValueError(f"column {col} has missing values; cannot encode indices")
    return vals


def encode_canonical_sites(df: pd.DataFrame) -> tuple[pd.DataFrame, list | None]:
    """Add a canonical ``site_i`` column shared by all rows for one physical site.

    Incidence and prevalence rows for the same site can carry different
    ``site_index`` values, so where country/site names are present they define
    the canonical grouping. Returns the annotated frame and the canonical
    category labels (``None`` when falling back to raw ``site_index``).
    """
    df = df.copy()
    if {"country_name", "site_name"}.issubset(df.columns):
        canon = pd.Categorical(df["country_name"].astype(str) + "||" + df["site_name"].astype(str))
        df["site_i"] = canon.codes.astype(int)
        canon_categories = list(canon.categories)
    else:
        df["site_i"] = encode_indices(df, "site_index")
        canon_categories = None
    return df, canon_categories


def site_to_study_from_df(df: pd.DataFrame) -> tuple[np.ndarray, dict, list]:
    """Map each canonical site to its study. Requires ``site_i`` from
    :func:`encode_canonical_sites`."""
    cross = df[["site_i", "study_index"]].drop_duplicates().sort_values(["site_i", "study_index"])
    if cross.groupby("site_i").size().max() != 1:
        raise ValueError("Each canonical site must map to exactly one study_index")
    study_levels = sorted(pd.unique(df["study_index"]))
    study_map = {s: i for i, s in enumerate(study_levels)}
    site_to_study = cross.set_index("site_i").sort_index()["study_index"].map(study_map).to_numpy(dtype=np.int32)
    return site_to_study, study_map, study_levels


def site_label_map(df: pd.DataFrame, canon_categories: list | None) -> dict[int, object]:
    if canon_categories is not None:
        return {i: lab for i, lab in enumerate(canon_categories)}
    return {i: s for i, s in enumerate(sorted(pd.unique(df["site_index"])))}


def site_to_study_index(df: pd.DataFrame) -> np.ndarray:
    """Site->study mapping keyed on the raw ``site_index`` column."""
    cross = (
        df[["site_index", "study_index"]]
        .drop_duplicates()
        .sort_values(["site_index", "study_index"])
    )
    if cross.groupby("site_index").size().max() != 1:
        raise ValueError("Each site_index must map to exactly one study_index")
    site_levels = sorted(pd.unique(df["site_index"]))
    cross = cross.set_index("site_index").loc[site_levels]
    study_levels = sorted(pd.unique(df["study_index"]))
    study_map = {s: i for i, s in enumerate(study_levels)}
    return np.asarray([study_map[s] for s in cross["study_index"].to_list()], dtype=np.int32)


def load_observed_data(data_csv: Path) -> pd.DataFrame:
    if not data_csv.exists():
        raise FileNotFoundError(
            f"Observed dataset CSV not found at {data_csv}. Provide a local path (not tracked in git)."
        )
    return pd.read_csv(data_csv)


def load_eir_prior_table(df: pd.DataFrame, eir_csv: Path) -> pd.DataFrame:
    if not eir_csv.exists():
        raise FileNotFoundError(
            f"EIR prior table CSV not found at {eir_csv}. Provide a local path with columns site_index,eir_estimate,eir_category."
        )
    tab = pd.read_csv(eir_csv)
    req = {"site_index", "eir_estimate", "eir_category"}
    miss = req - set(tab.columns)
    if miss:
        raise ValueError(f"EIR prior table missing columns: {sorted(miss)}")
    site_levels = pd.Index(sorted(pd.unique(df["site_index"])), name="site_index")
    tab = tab.copy()
    tab["site_index"] = tab["site_index"].astype(site_levels.dtype)
    tab = tab.set_index("site_index").reindex(site_levels)
    if tab.isna().any().any():
        missing_sites = tab[tab.isna().any(axis=1)].index.tolist()
        raise ValueError(f"EIR prior table has missing rows/values for sites: {missing_sites}")
    return tab.reset_index()


def eirs_for_sites(df: pd.DataFrame, eir_prior_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    site_levels = sorted(pd.unique(df["site_index"]))
    eir_prior_df = eir_prior_df.set_index("site_index").loc[site_levels]

    eir_est = eir_prior_df["eir_estimate"].to_numpy(dtype=float)
    eir_cat = eir_prior_df["eir_category"].to_numpy(dtype=int)
    if np.any(eir_est <= 0):
        raise ValueError("eir_estimate must be > 0 for all sites")
    if not np.all(np.isin(eir_cat, [1, 2, 3, 4, 5])):
        raise ValueError("eir_category must be integers in {1,2,3,4,5}")

    return np.log(eir_est), eir_cat


def aggregate_eir_priors(
    *, df: pd.DataFrame, eir_priors_csv: Path, n_site: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-canonical-site log-EIR prior mean and sd.

    Aggregates the raw per-``site_index`` rows of the EIR prior table up to the
    canonical ``site_i`` grouping. Shared by the sampler and the plotting code,
    which previously each had their own copy of this loop.
    """
    if not eir_priors_csv.exists():
        raise FileNotFoundError(f"EIR prior table CSV not found at {eir_priors_csv}.")
    eir_tab = pd.read_csv(eir_priors_csv)
    req = {"site_index", "eir_estimate", "eir_category"}
    miss = req - set(eir_tab.columns)
    if miss:
        raise ValueError(f"EIR prior table missing columns: {sorted(miss)}")

    log_eir_mu = np.zeros(n_site, dtype=float)
    log_eir_sigma = np.zeros(n_site, dtype=float)
    eir_lookup = eir_tab.set_index("site_index")
    site_meta = df[["site_i", "site_index"]].drop_duplicates().sort_values(["site_i", "site_index"])
    for si, grp in site_meta.groupby("site_i"):
        raw_ids = grp["site_index"].tolist()
        missing_ids = [r for r in raw_ids if r not in eir_lookup.index]
        if missing_ids:
            raise ValueError(f"EIR prior table missing rows for raw site_index: {missing_ids}")
        mu_vals = np.log(eir_lookup.loc[raw_ids, "eir_estimate"].to_numpy(dtype=float))
        cat_vals = eir_lookup.loc[raw_ids, "eir_category"].to_numpy(dtype=int)
        sig_vals = np.asarray([EIR_LOG_SD_BY_CATEGORY[int(c)] for c in cat_vals], dtype=float)
        log_eir_mu[int(si)] = float(np.mean(mu_vals))
        log_eir_sigma[int(si)] = float(np.mean(sig_vals))
    return log_eir_mu, log_eir_sigma


def load_ft_prior_table(df: pd.DataFrame, ft_csv: Path | None) -> pd.DataFrame:
    study_levels = pd.Index(sorted(pd.unique(df["study_index"])), name="study_index")

    if ft_csv is None:
        rows = []
        for s in study_levels:
            if int(s) not in DEFAULT_FT_PRIOR_BY_STUDY:
                raise ValueError(
                    f"No default FT prior found for study_index={s}. Provide --ft-priors-csv."
                )
            a, b = DEFAULT_FT_PRIOR_BY_STUDY[int(s)]
            rows.append({"study_index": s, "ft_alpha": a, "ft_beta": b})
        return pd.DataFrame(rows)

    if not ft_csv.exists():
        raise FileNotFoundError(
            f"FT prior table CSV not found at {ft_csv}. Expected columns study_index,ft_alpha,ft_beta."
        )

    tab = pd.read_csv(ft_csv)
    req = {"study_index", "ft_alpha", "ft_beta"}
    miss = req - set(tab.columns)
    if miss:
        raise ValueError(f"FT prior table missing columns: {sorted(miss)}")

    tab = tab.copy()
    tab["study_index"] = tab["study_index"].astype(study_levels.dtype)
    tab = tab.set_index("study_index").reindex(study_levels)
    if tab.isna().any().any():
        missing_studies = tab[tab.isna().any(axis=1)].index.tolist()
        raise ValueError(f"FT prior table has missing rows/values for studies: {missing_studies}")
    if (tab["ft_alpha"] <= 0).any() or (tab["ft_beta"] <= 0).any():
        raise ValueError("ft_alpha and ft_beta must be > 0")
    return tab.reset_index()


def load_reference_draws(reference_csv: Path | None) -> pd.DataFrame | None:
    """Read a reference posterior draws CSV in either wide or long format.

    Long format is ``draw,parameter,value``; wide format is one column per
    parameter. Returns ``None`` when the path is absent.
    """
    if reference_csv is None or not Path(reference_csv).exists():
        return None
    df = pd.read_csv(reference_csv)
    if {"draw", "parameter", "value"}.issubset(df.columns):
        df = (
            df.pivot_table(index="draw", columns="parameter", values="value", aggfunc="first")
            .reset_index(drop=True)
        )
    return df


def load_reference_means(reference_csv: Path) -> dict[str, float]:
    if not reference_csv.exists():
        raise FileNotFoundError(f"reference_draws_csv not found at {reference_csv}")
    wide = load_reference_draws(reference_csv)
    assert wide is not None  # existence checked above

    means: dict[str, float] = {}
    for p in CORE_PARAMS:
        ref_col = REFERENCE_PARAM_MAP[p]
        if ref_col not in wide.columns:
            raise ValueError(
                f"reference_draws_csv missing column '{ref_col}' needed for core parameter '{p}'"
            )
        col = pd.to_numeric(wide[ref_col], errors="coerce")
        if col.notna().sum() == 0:
            raise ValueError(f"reference column '{ref_col}' has no finite values")
        means[p] = float(col.mean(skipna=True))
    return means


def latest_samples_npz(artifacts_root: Path) -> Path:
    cands = sorted(Path(artifacts_root).glob(f"**/{SAMPLES_FILENAME}"), key=lambda p: p.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"No {SAMPLES_FILENAME} under {artifacts_root}")
    return cands[-1]
