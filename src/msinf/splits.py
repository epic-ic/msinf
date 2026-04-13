"""Held-out splits of the observation table.

Holding out belongs here rather than to whoever is calling: the splits are
defined by the likelihood -- which rows can be scored apart from which -- so
this package owns them, exercises them under Hydra, and anything importing
:mod:`msinf.evaluate` gets the same code rather than a copy of it.

Two splits, reported separately and never averaged:

* ``study`` -- a random subset of whole studies. Clean under the marginalised
  study effects: a held-out study's predictive integrates over ``u`` and ``w``
  without needing their realised values. Blunt, though, because its ``log_eir``
  then sits on its prior.
* ``age_band`` -- the upper age bands, within every study. Conditional by
  construction: ``log_eir`` stays informed by the study's training rows, so
  what is tested is age structure -- which is what an immunity model actually
  claims to explain.

There is deliberately no ``truncate_age`` here. The <=20y variant of the study
table is its own CSV selected by ``data.data_csv``; a function would be a second
way to draw the same boundary and the two could disagree.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

KINDS = ("none", "study", "age_band")

# Data stream code for incidence rows, whose likelihood unit is the cell.
_TYPE_INCIDENCE = 1


@dataclass(frozen=True)
class Split:
    """Which rows to hold out. ``kind='none'`` is the whole-data fit."""

    kind: str = "none"
    seed: int = 42
    n_studies: int = 3      # kind=study
    from_age: float = 5.0   # kind=age_band

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"holdout kind must be one of {KINDS}; got {self.kind!r}")
        if self.kind == "study" and self.n_studies < 1:
            raise ValueError("holdout.n_studies must be at least 1")

    @property
    def is_empty(self) -> bool:
        return self.kind == "none"

    def label(self) -> str:
        if self.kind == "none":
            return "none"
        if self.kind == "study":
            return f"study_n{self.n_studies}_seed{self.seed}"
        return f"age_band_from{self.from_age:g}"


def held_out_studies(df: pd.DataFrame, split: Split) -> list:
    """The ``study_index`` values held out, in the order they were drawn."""
    if split.kind != "study":
        return []
    studies = sorted(pd.unique(df["study_index"]))
    if split.n_studies >= len(studies):
        raise ValueError(
            f"holdout.n_studies={split.n_studies} would hold out every study "
            f"({len(studies)} available); nothing would be left to fit"
        )
    rng = np.random.default_rng(split.seed)
    chosen = rng.choice(len(studies), size=split.n_studies, replace=False)
    return [studies[int(i)] for i in chosen]


def split_studies(df: pd.DataFrame, n_studies: int, seed: int) -> np.ndarray:
    """Boolean holdout mask over ``df``'s rows: a random subset of whole studies."""
    chosen = held_out_studies(df, Split(kind="study", n_studies=n_studies, seed=seed))
    return df["study_index"].isin(chosen).to_numpy()


def split_age_bands(df: pd.DataFrame, from_age: float) -> np.ndarray:
    """Boolean holdout mask over ``df``'s rows: every band starting at or above
    ``from_age``, in every study."""
    return df["age0"].to_numpy(dtype=float) >= float(from_age)


def split_mask(df: pd.DataFrame, split: Split | None) -> np.ndarray:
    """The holdout mask for a split, after the coherence checks."""
    if split is None or split.is_empty:
        return np.zeros(len(df), dtype=bool)
    if split.kind == "study":
        mask = split_studies(df, split.n_studies, split.seed)
    else:
        mask = split_age_bands(df, split.from_age)
    check_coherence(df, mask, split)
    return mask


def check_coherence(df: pd.DataFrame, hold: np.ndarray, split: Split) -> None:
    """Reject a split the likelihood cannot score.

    These are properties of the likelihood rather than of the mask, which is why
    they live beside the splits and not beside the caller that named one.

    * No incidence cell (site x age-group) may straddle the boundary. ``v_jk``
      is integrated out at the cell level, so the negative-binomial marginal is
      a joint density over the whole cell; a boundary inside one makes
      ``elpd_holdout`` meaningless rather than merely noisy.
    * A held-out study must contribute no training rows.
    * Something must be held out, and something must be left to fit.
    """
    hold = np.asarray(hold, dtype=bool)
    if hold.all():
        raise ValueError(f"split {split.label()} holds out every row; nothing left to fit")
    if not hold.any():
        raise ValueError(f"split {split.label()} holds out no rows")

    if "site_i" not in df.columns:
        raise ValueError("check_coherence needs the canonical site_i column (see encode_canonical_sites)")

    inc = df["type"].to_numpy() == _TYPE_INCIDENCE
    if inc.any():
        cells = pd.DataFrame(
            {
                "site_i": df.loc[inc, "site_i"].to_numpy(),
                "agek": df.loc[inc, "age_bracket"].to_numpy(dtype=int),
                "hold": hold[inc],
            }
        )
        straddling = cells.groupby(["site_i", "agek"])["hold"].nunique()
        bad = straddling[straddling > 1]
        if len(bad):
            raise ValueError(
                f"split {split.label()} straddles {len(bad)} incidence cell(s) "
                f"(site_i, age_bracket): {list(bad.index)[:8]}. The site-age effect v_jk is "
                "integrated out at the cell level, so a cell cannot be scored in halves; "
                "choose a boundary that falls between age brackets."
            )

    if split.kind == "study":
        held = set(df.loc[hold, "study_index"].unique())
        leaked = sorted(held & set(df.loc[~hold, "study_index"].unique()))
        if leaked:
            raise ValueError(f"held-out studies still contribute training rows: {leaked}")


def apply_split(df: pd.DataFrame, split: Split | None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """``(train, holdout)`` frames; ``holdout`` is ``None`` for ``kind='none'``.

    Both frames keep the canonical ``site_i`` the *full* frame defines, so the
    caller can encode them against the same site and study levels. Splitting
    after index encoding is the bug this exists to avoid: dropping a study
    renumbers everything, and the two frames' arrays stop lining up.
    """
    mask = split_mask(df, split)
    if not mask.any():
        return df, None
    return df.loc[~mask].copy(), df.loc[mask].copy()
