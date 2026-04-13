"""Extract UN WPP 2024 qx and population-by-single-age for the (ISO3, year) pairs
needed by the study sites, in the 101-row layout peeps expects (ages 0..100+).
"""
import collections
import csv
import gzip

SITES = "site_years.csv"
LT = "wpp_lt.csv.gz"
POP = "wpp_pop.csv.gz"
OUT = "wpp_demography.csv"


def wanted_pairs():
    with open(SITES) as fh:
        rows = list(csv.DictReader(fh))
    return {(r["iso3"], r["wpp_year"]) for r in rows}


def scan(path, want, keep, sex_both=False):
    """One pass over a gzipped WPP CSV, collecting rows for the wanted pairs."""
    out = collections.defaultdict(dict)
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            key = (r["ISO3_code"], r["Time"])
            if key not in want:
                continue
            if sex_both and r["Sex"] != "Total":
                continue
            if r["Variant"] != "Medium":
                continue
            out[key][int(r["AgeGrpStart"])] = keep(r)
    return out


def main():
    want = wanted_pairs()
    print(f"{len(want)} (iso3, year) pairs")

    qx = scan(LT, want, lambda r: float(r["qx"]), sex_both=True)
    pop = scan(POP, want, lambda r: float(r["PopTotal"]))

    rows = []
    for key in sorted(want):
        iso3, year = key
        q, p = qx.get(key), pop.get(key)
        if not q or not p:
            raise SystemExit(f"missing WPP data for {key}: qx={bool(q)} pop={bool(p)}")
        ages = sorted(q)
        assert ages == list(range(101)), f"{key}: unexpected age range {ages[:3]}..{ages[-3:]}"
        total = sum(p[a] for a in ages)
        for a in ages:
            rows.append({
                "iso3": iso3, "year": year,
                "age_lower": a, "age_upper": a + 1 if a < 100 else 200,
                "qx": q[a], "p": p[a] / total,
            })

    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT}: {len(rows)} rows")

    for key in sorted(want):
        q = qx[key]
        s = sum(pop[key].values())
        print(f"  {key[0]} {key[1]}  q0={q[0]:.5f}  q1={q[1]:.5f}  "
              f"u5 pop share={sum(pop[key][a] for a in range(5)) / s:.4f}")


if __name__ == "__main__":
    main()
