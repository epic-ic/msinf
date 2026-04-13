"""Map the fitted annual mortality onto the Griffin age grid, per site.

fit_mortality.py fits a piecewise-constant hazard mu(a) = mu_fitted[floor(a)]
over each year of age, under the same integrate-survival-across-the-class
convention used here. Then for each Griffin class k = [g_k, g_{k+1}):

    mu_griffin[k] = (1/w_k) * int_{g_k}^{g_k+1} mu(a) da        (exposure-weighted)
    prop[k]       ∝ int_{g_k}^{g_k+1} S(a) da,  S(a) = exp(-int_0^a mu)

prop is the stationary (constant-birth, constant-population) age structure that
the Griffin/dmeq equilibrium uses as pi(a). The terminal class [100, inf) is
absorbing, with mass S(100) / mu(100).
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "src"))
from msinf.age_grid import GRIFFIN_AGE_GRID

ANNUAL = "mortality_annual.csv"
SITES = "site_years.csv"
OUT_SITE = "site_mortality_griffin_grid.csv"
OUT_CY = "country_year_mortality_griffin_grid.csv"

GRID = np.asarray(GRIFFIN_AGE_GRID, dtype=float)          # 165 lower edges
N_CLASS = GRID.size
EDGES = np.append(GRID, np.inf)                            # class k = [EDGES[k], EDGES[k+1])


def load_annual():
    """(iso3, year) -> per-year hazard array over ages 0..100."""
    by_key = {}
    with open(ANNUAL) as fh:
        for r in csv.DictReader(fh):
            key = (r["iso3"], int(r["year"]))
            by_key.setdefault(key, {})[int(r["age_lower"])] = float(r["mu_fitted"])
    out = {}
    for key, d in by_key.items():
        ages = sorted(d)
        assert ages == list(range(101)), f"{key}: bad age range"
        out[key] = np.array([d[a] for a in ages])
    return out


def to_griffin(mu_year):
    """Piecewise-constant annual hazard -> (mu per class, stationary prop per class)."""
    # Cumulative hazard at every integer age boundary 0..101.
    cum_int = np.concatenate([[0.0], np.cumsum(mu_year)])

    def cum_hazard(a):
        """int_0^a mu, for scalar/array a in [0, 101]."""
        a = np.asarray(a, dtype=float)
        i = np.clip(np.floor(a).astype(int), 0, 100)
        return cum_int[i] + (a - i) * mu_year[i]

    def integ_mu(lo, hi):
        return cum_hazard(hi) - cum_hazard(lo)

    def integ_surv(lo, hi):
        """int_lo^hi exp(-H(a)) da, exact for piecewise-constant hazard."""
        total = 0.0
        a = lo
        while a < hi - 1e-12:
            i = min(int(np.floor(a)), 100)
            b = min(hi, i + 1.0)
            m = mu_year[i]
            s0 = np.exp(-cum_hazard(a))
            # int_a^b exp(-m*(t-a)) dt
            total += s0 * ((b - a) if m <= 0 else (1.0 - np.exp(-m * (b - a))) / m)
            a = b
        return total

    mu_cls = np.empty(N_CLASS)
    mass = np.empty(N_CLASS)
    for k in range(N_CLASS):
        lo, hi = EDGES[k], EDGES[k + 1]
        if np.isinf(hi):
            # Absorbing terminal class: hazard is the age-100 rate.
            mu_cls[k] = mu_year[100]
            mass[k] = np.exp(-cum_hazard(lo)) / mu_year[100]
        else:
            mu_cls[k] = integ_mu(lo, hi) / (hi - lo)
            mass[k] = integ_surv(lo, hi)
    return mu_cls, mass / mass.sum()


def main():
    hazards = load_annual()
    grids = {key: to_griffin(mu) for key, mu in hazards.items()}

    # Per country-year table.
    with open(OUT_CY, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso3", "year", "age_class", "age_lower", "age_upper",
                    "mu_per_year", "mu_per_day", "prop"])
        for (iso3, year), (mu, prop) in sorted(grids.items()):
            for k in range(N_CLASS):
                w.writerow([iso3, year, k, f"{EDGES[k]:g}",
                            "Inf" if np.isinf(EDGES[k + 1]) else f"{EDGES[k + 1]:g}",
                            f"{mu[k]:.10g}", f"{mu[k] / 365.0:.10g}", f"{prop[k]:.10g}"])

    # Per site table (sites share a country-year row where they coincide).
    with open(SITES) as fh:
        sites = list(csv.DictReader(fh))
    with open(OUT_SITE, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site_index", "site_name", "country_name", "iso3", "wpp_year",
                    "age_class", "age_lower", "age_upper",
                    "mu_per_year", "mu_per_day", "prop"])
        for s in sites:
            key = (s["iso3"], int(s["wpp_year"]))
            mu, prop = grids[key]
            for k in range(N_CLASS):
                w.writerow([s["site_index"], s["site_name"], s["country_name"],
                            s["iso3"], s["wpp_year"], k, f"{EDGES[k]:g}",
                            "Inf" if np.isinf(EDGES[k + 1]) else f"{EDGES[k + 1]:g}",
                            f"{mu[k]:.10g}", f"{mu[k] / 365.0:.10g}", f"{prop[k]:.10g}"])

    print(f"wrote {OUT_CY} ({len(grids) * N_CLASS} rows) and "
          f"{OUT_SITE} ({len(sites) * N_CLASS} rows)")
    check(grids)


def check(grids):
    """Re-aggregate Griffin-grid props back to single years and compare.

    Two references: ``p_fitted`` (the equilibrium fit_mortality.py solved for) and
    ``p_wpp`` (the actual UN WPP target). Both the fit and this mapping integrate
    survival across the class, so the p_fitted column should be at solver
    tolerance -- it is a consistency check on the mapping. The p_wpp column is the
    end-to-end error, and is nonzero only where the fit hit a bound.
    """
    ann, wpp = {}, {}
    with open(ANNUAL) as fh:
        for r in csv.DictReader(fh):
            key = (r["iso3"], int(r["year"]))
            ann.setdefault(key, {})[int(r["age_lower"])] = float(r["p_fitted"])
            wpp.setdefault(key, {})[int(r["age_lower"])] = float(r["p_wpp"])

    # The Griffin grid only resolves single years of age below 30 (classes are 2
    # years wide above it), so re-aggregation is only comparable over ages 0..29.
    n_cmp = 30
    print("\n  ages 0-29, max relative error in single-year age proportion")
    print("  country-year   vs p_fitted   vs p_WPP    u5 share (grid / WPP)   E[life]")
    for key, (mu, prop) in sorted(grids.items()):
        regrouped = np.zeros(101)
        for k in range(N_CLASS):
            regrouped[min(int(np.floor(EDGES[k])), 100)] += prop[k]
        t_fit = np.array([ann[key][a] for a in range(101)])
        t_wpp = np.array([wpp[key][a] for a in range(101)])
        r_fit = np.abs(regrouped[:n_cmp] / t_fit[:n_cmp] - 1).max()
        r_wpp = np.abs(regrouped[:n_cmp] / t_wpp[:n_cmp] - 1).max()
        mids = np.where(np.isinf(EDGES[1:]), EDGES[:-1] + 1.0,
                        (EDGES[:-1] + EDGES[1:]) / 2.0)
        print(f"  {key[0]} {key[1]}       {r_fit:7.2%}      {r_wpp:7.2%}"
              f"        {float(prop[mids < 5].sum()):.4f} / {t_wpp[:5].sum():.4f}"
              f"          {float((mids * prop).sum()):6.2f}")


if __name__ == "__main__":
    main()
