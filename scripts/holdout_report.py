"""Aggregate the metrics of a holdout multirun into a between-fold spread.

The gate for building a search on top of this metric: fit the incumbent across
the seed sweep under both splits and compare the between-fold spread to the
spread between two deliberately different curve families. **If a fold costs more
than a candidate does, the metric cannot rank candidates** and there is no point
searching against it.

    python scripts/run_holdout.py -m +experiment=holdout_study holdout.seed=1,2,3,4,5
    python scripts/holdout_report.py multirun/<date>/<time>

The two splits are reported separately and never averaged: they answer different
questions, and the study split is expected to be the blunter of the two, because
a held-out study's log_eir sits on its prior (sd 0.3-0.8 on the log scale, see
priors.EIR_LOG_SD_BY_CATEGORY).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Metrics worth a spread: totals and the per-unit versions that are comparable
# across folds of different sizes.
REPORTED = (
    "elpd_holdout",
    "elpd_holdout_incidence",
    "elpd_holdout_prevalence",
    "elpd_holdout_incidence_per_cell",
    "elpd_holdout_prevalence_per_row",
)


def collect(root: Path) -> dict[str, list[dict]]:
    """Every ``metrics.json`` under ``root``, grouped by split label."""
    by_split: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(root.glob("**/metrics.json")):
        payload = json.loads(path.read_text())
        info = payload.get("info", {})
        label = str(info.get("split", "unknown"))
        # Fold seed is part of the label for the study split and irrelevant for
        # the age split; group on the kind so folds land together.
        kind = label.split("_")[0] if label != "unknown" else label
        by_split[kind].append({"path": str(path), **payload.get("metrics", {}), "_label": label})
    return by_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="Multirun (or run) directory to scan for metrics.json")
    args = ap.parse_args()

    by_split = collect(Path(args.root).resolve())
    if not by_split:
        raise SystemExit(f"no metrics.json found under {args.root}")

    for kind, runs in sorted(by_split.items()):
        print(f"\n=== split: {kind}  ({len(runs)} fold(s)) ===")
        for metric in REPORTED:
            vals = np.asarray([r[metric] for r in runs if metric in r], dtype=float)
            if vals.size == 0:
                continue
            spread = float(vals.max() - vals.min()) if vals.size > 1 else 0.0
            sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            print(f"  {metric:42s} mean={vals.mean():12.3f}  sd={sd:10.3f}  range={spread:10.3f}")
        for r in runs:
            got = r.get("elpd_holdout")
            print(f"    {r['_label']:28s} elpd_holdout={got if got is None else round(got, 3)}  {r['path']}")

    print(
        "\nRead this against the gap between two candidate curve families fitted under the "
        "same split and budget. A between-fold range of the same size as that gap means the "
        "metric cannot rank them."
    )


if __name__ == "__main__":
    main()
