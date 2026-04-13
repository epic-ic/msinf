from pathlib import Path

import pandas as pd


def canonicalize_study_and_eir(
    study_csv: Path,
    eir_csv: Path,
    out_study_csv: Path,
    out_eir_csv: Path,
) -> tuple[Path, Path]:
    """Canonicalize site indexing so incidence/prevalence rows share site latents.

    Rules:
    - canonical site key: country_name||site_name
    - canonical site_index: 1-based categorical code of site key
    - canonical EIR priors: aggregate raw site priors mapped to canonical sites
      using mean(eir_estimate), round(mean(eir_category)).
    """
    df = pd.read_csv(study_csv)
    eir = pd.read_csv(eir_csv)

    required_df = {
        "country_name",
        "site_name",
        "site_index",
        "study_index",
        "numer",
        "denom",
        "type",
        "age0",
        "age1",
        "case_detection",
        "age_bracket",
    }
    missing_df = required_df - set(df.columns)
    if missing_df:
        raise ValueError(f"study data missing columns: {sorted(missing_df)}")

    required_eir = {"site_index", "eir_estimate", "eir_category"}
    missing_eir = required_eir - set(eir.columns)
    if missing_eir:
        raise ValueError(f"eir prior data missing columns: {sorted(missing_eir)}")

    site_key = df["country_name"].astype(str) + "||" + df["site_name"].astype(str)
    cat = pd.Categorical(site_key)

    out_df = df.copy()
    out_df["site_index"] = cat.codes + 1

    raw_to_can = (
        pd.DataFrame(
            {
                "raw_site_index": df["site_index"].to_numpy(),
                "site_index": out_df["site_index"].to_numpy(),
            }
        )
        .drop_duplicates()
        .sort_values(["site_index", "raw_site_index"])
    )

    joined = raw_to_can.merge(
        eir.rename(columns={"site_index": "raw_site_index"}),
        on="raw_site_index",
        how="left",
        validate="many_to_one",
    )
    if joined[["eir_estimate", "eir_category"]].isna().any().any():
        missing = joined[joined[["eir_estimate", "eir_category"]].isna().any(axis=1)][
            "raw_site_index"
        ].unique()
        raise ValueError(f"EIR priors missing for raw site_index values: {sorted(map(int, missing))}")

    out_eir = (
        joined.groupby("site_index", as_index=False)
        .agg(
            eir_estimate=("eir_estimate", "mean"),
            eir_category=("eir_category", lambda x: round(float(pd.Series(x).mean()))),
        )
        .sort_values("site_index")
    )

    out_study_csv.parent.mkdir(parents=True, exist_ok=True)
    out_eir_csv.parent.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_study_csv, index=False)
    out_eir.to_csv(out_eir_csv, index=False)
    return out_study_csv, out_eir_csv
