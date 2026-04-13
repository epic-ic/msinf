"""Fit constant-population-adjusted mortality rates, per (ISO3, year).

Replaces the peeps (R) step. Same idea -- adjust the WPP death rates so that the
equilibrium age distribution of a fixed-size population reproduces the observed
WPP age distribution -- but with two changes.

**Continuous-time equilibrium.** peeps' equilibrium is a discrete annual Markov
chain, whose stationary distribution places a band's whole mass at its lower
edge: ``pi_a ∝ l_a``. The Griffin/dmeq convention instead integrates survival
across the class. Fitting under one convention and consuming under the other
leaves a 1-3% residual in single-year age proportions below 30 (worst where the
hazard is large, i.e. age 0 in the high-mortality country-years). So fit the
convention we actually use: with a piecewise-constant hazard ``mu_a`` over each
year of age,

    mass_a = int_a^{a+1} S(x) dx = S_a * (1 - exp(-mu_a)) / mu_a,   S_{a+1} = S_a e^-mu_a

and the open-ended band [100, inf) is absorbing with mass ``S_100 / mu_100``.

**Sequential solve instead of a 101-parameter optimisation.** Both conventions
are triangular -- the target proportions pin the hazards one age at a time -- so
neither needs L-BFGS-B. Under peeps' convention the inversion is outright closed
form, ``q_a = 1 - p_{a+1}/p_a``. Under ours, band ``a`` is a monotone scalar
solve once ``S_a`` is known, given one global constant:

    S_a * (1 - exp(-mu_a)) / mu_a = e0 * p_a

``e0 = sum_a mass_a`` is the fitted life expectancy at birth. Normalisation is
automatic (the masses sum to ``e0`` whenever the per-band equations hold), so e0
is a genuinely free scalar rather than something the data determines -- see
``--e0`` below. Hazards are clipped to peeps' bounds, ``mu_wpp/2`` to
``-log(0.1)``.

Input  ``wpp_demography.csv``  (from extract_wpp.py): iso3, year, age_lower,
       age_upper, qx, p -- 101 rows per country-year, p summing to 1.
Output ``mortality_annual.csv``: adds mu_wpp, mu_fitted, p_fitted.
"""
import argparse
import collections
import csv
import sys

import numpy as np

IN = "wpp_demography.csv"
OUT = "mortality_annual.csv"

N_AGE = 101
MU_MAX = -np.log(0.1)          # peeps' upper bound of 0.9 on the annual death probability


def wpp_hazard(qx):
    """WPP qx -> annual hazard. The open-ended band's qx = 1 is a closing
    convention rather than a rate, so carry the age-99 hazard into it."""
    mu = -np.log1p(-np.minimum(qx, 0.9))
    mu[100] = mu[99]
    return mu


def masses(mu):
    """Un-normalised continuous stationary mass per annual band (person-years)."""
    surv = np.exp(-np.concatenate([[0.0], np.cumsum(mu[:100])]))
    return np.concatenate([surv[:100] * -np.expm1(-mu[:100]) / mu[:100],
                           surv[100:] / mu[100]])


def life_expectancy(mu):
    return float(masses(mu).sum())


def solve(p, mu_lo, e0):
    """Sequential solve for the hazards, given the free scalar e0.

    Each band is ``g(mu) = (1 - e^-mu)/mu = e0 * p_a / S_a`` with g decreasing
    from 1 to 0, so it inverts by bisection. ``t >= 1`` means the band needs more
    mass than survival to its start can supply even at zero mortality -- possible
    where WPP's age distribution rises with age -- and pins it to the lower bound.
    """
    mu = np.empty(N_AGE)
    surv = 1.0
    for a in range(100):
        t = e0 * p[a] / surv if surv > 0 else np.inf
        if t >= 1.0:
            mu[a] = mu_lo[a]
        else:
            lo, hi = 1e-10, 60.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if -np.expm1(-mid) / mid > t:
                    lo = mid
                else:
                    hi = mid
            mu[a] = np.clip(0.5 * (lo + hi), mu_lo[a], MU_MAX)
        surv *= np.exp(-mu[a])
    mu[100] = np.clip(surv / (e0 * p[100]) if p[100] > 0 else mu[99], mu_lo[100], MU_MAX)
    return mu


def sse(p, mu):
    m = masses(mu)
    return float(np.sum((m / m.sum() - p) ** 2))


def fit(p, mu_lo, e0=None):
    """Fit one country-year. ``e0=None`` chooses it by least squares."""
    ceiling = 1.0 / p[0]           # band 0 needs e0 * p_0 < 1 to be feasible at all
    if e0 is not None:
        if e0 >= ceiling:
            raise ValueError(
                f"e0={e0:.1f} is infeasible: a stationary population cannot put "
                f"{e0 * p[0]:.2f}x of its person-years in the first year of age. "
                f"The ceiling here is 1/p_0 = {ceiling:.1f}.")
        return solve(p, mu_lo, e0), e0

    # Golden-section on the one free scalar.
    lo, hi = 5.0, 0.999 * ceiling
    gr = (np.sqrt(5) - 1) / 2
    b, c = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fb, fc = sse(p, solve(p, mu_lo, b)), sse(p, solve(p, mu_lo, c))
    for _ in range(80):
        if fb < fc:
            hi, c, fc = c, b, fb
            b = hi - gr * (hi - lo)
            fb = sse(p, solve(p, mu_lo, b))
        else:
            lo, b, fb = b, c, fc
            c = lo + gr * (hi - lo)
            fc = sse(p, solve(p, mu_lo, c))
    e0 = 0.5 * (lo + hi)
    return solve(p, mu_lo, e0), e0


def load(path):
    by_key = collections.defaultdict(dict)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            by_key[(r["iso3"], int(r["year"]))][int(r["age_lower"])] = (
                int(r["age_upper"]), float(r["qx"]), float(r["p"]))
    out = {}
    for key, d in by_key.items():
        ages = sorted(d)
        assert ages == list(range(N_AGE)), f"{key}: unexpected age range"
        arr = np.array([d[a] for a in ages], dtype=float)
        assert abs(arr[:, 2].sum() - 1) < 1e-8, f"{key}: p does not sum to 1"
        out[key] = arr
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e0", default="fit",
                    help="life expectancy at birth: 'fit' (least squares, the default), "
                         "'wpp' (the WPP life table's own value), or a number of years. "
                         "'wpp' is infeasible for growing populations -- see the README.")
    ap.add_argument("--input", default=IN)
    ap.add_argument("--output", default=OUT)
    args = ap.parse_args()

    data = load(args.input)
    rows, failures = [], []
    print(f"{'country-year':14} {'e0':>6} {'SSE':>10} {'max|dp|':>10} "
          f"{'u5 share (fit / WPP)':>24}")
    for key, arr in sorted(data.items()):
        age_upper, qx, p = arr[:, 0], arr[:, 1], arr[:, 2]
        mu_wpp = wpp_hazard(qx)

        if args.e0 == "fit":
            target_e0 = None
        elif args.e0 == "wpp":
            target_e0 = life_expectancy(mu_wpp)
        else:
            target_e0 = float(args.e0)

        try:
            mu, e0 = fit(p, mu_wpp / 2, target_e0)
        except ValueError as exc:
            failures.append(f"{key[0]} {key[1]}: {exc}")
            print(f"{key[0]} {key[1]}      INFEASIBLE")
            continue

        m = masses(mu)
        p_fit = m / m.sum()
        print(f"{key[0]} {key[1]}      {e0:6.1f} {sse(p, mu):10.2e} "
              f"{np.abs(p_fit - p).max():10.2e} "
              f"{p_fit[:5].sum():14.4f} / {p[:5].sum():.4f}")
        for a in range(N_AGE):
            rows.append({"iso3": key[0], "year": key[1], "age_lower": a,
                         "age_upper": int(age_upper[a]), "qx_wpp": f"{qx[a]:.10g}",
                         "mu_wpp": f"{mu_wpp[a]:.10g}", "mu_fitted": f"{mu[a]:.10g}",
                         "p_wpp": f"{p[a]:.10g}", "p_fitted": f"{p_fit[a]:.10g}"})

    if failures:
        print(f"\n{len(failures)} of {len(data)} country-years have no solution at "
              f"--e0={args.e0}:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        raise SystemExit(1)

    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.output}: {len(rows)} rows")


if __name__ == "__main__":
    main()
