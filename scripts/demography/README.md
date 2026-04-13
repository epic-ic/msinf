# Site-specific mortality rates on the Griffin age grid

Per-site demography for the 41 sites in
`artifacts/numpyro_jamie_100x4/study_data_2014_age_le20.csv`, fitted from UN WPP
2024 and mapped onto the 165-class Griffin age grid
(`msinf.age_grid.GRIFFIN_AGE_GRID`).

## Outputs (`artifacts/demography/`)

| File | Contents |
| --- | --- |
| `site_years.csv` | 41 sites → ISO3 + WPP reference year, with the Battle study period it came from |
| `mortality_annual.csv` | the fit on its native single-year grid: `qx_wpp`, `mu_wpp`, `mu_fitted`, `p_wpp`, `p_fitted` for 16 country-years |
| `country_year_mortality_griffin_grid.csv` | 16 country-years × 165 classes: `mu_per_year`, `mu_per_day`, `prop` |
| `site_mortality_griffin_grid.csv` | 41 sites × 165 classes, same columns keyed by `site_index` (the join-ready table) |

`prop` is the stationary age distribution per class, i.e. the `pi(a)` weight the
dmeq equilibrium uses; it sums to 1 across the 165 classes. `mu_per_day` is the
same hazard on the model's day timescale.

`peeps_mortality_annual.csv` is the superseded [`mrc-ide/peeps`](https://github.com/mrc-ide/peeps)
(v0.0.4) baseline, kept for comparison. Nothing in the pipeline reads it.

## Pipeline

```bash
# 0. Battle et al. 2016 incidence/prevalence database (study periods).
#    https://datadryad.org/dataset/doi:10.5061/dryad.j0kv0 -> ./battle.csv

# 1. Sites -> (ISO3, WPP year).
python scripts/demography/build_site_years.py

# 2. UN WPP 2024 qx + population by single age. Needs the two source files in cwd:
#    WPP2024_Life_Table_Complete_Medium_Both_1950-2023.csv.gz  -> wpp_lt.csv.gz
#    WPP2024_PopulationBySingleAgeSex_Medium_1950-2023.csv.gz  -> wpp_pop.csv.gz
python scripts/demography/extract_wpp.py

# 3. Fit the constant-population-adjusted hazards. ~3 s.
python scripts/demography/fit_mortality.py

# 4. Map onto the Griffin grid.
python scripts/demography/to_griffin_grid.py
```

Steps 2–4 read and write CSVs in the working directory.

## Choices worth knowing about

**Study years come from Battle et al. 2016**, the `GRIFFIN == "Yes"` subset —
i.e. exactly the records Griffin 2014 fitted. Each site's WPP reference year is
the calendar year containing the study-period midpoint (WPP year *Y* references
1 July, so this is the nearest WPP vintage). Years span 1992–2004.

Two sites needed disambiguation:

- **Diallo**, one `reference` in the study data but two Battle records:
  `Dakar_south` → Diallo 1998, 6/1994–5/1995; `Dakar_central` → Diallo 2000,
  3/1996–2/1997.
- **Dicko**, two survey years (1999, 2000) over the same two villages. The study
  data carries these as duplicate `site_index` blocks (11/12 and 31/32); the
  lower block is assigned 1999.

The other duplicated `site_index` blocks (Bonnet, Henry, Mwangi, Saute,
Thompson, Henning, Lusingu, Schellenberg) are the same site and year, so they
share demography.

**The fit uses the continuous-time equilibrium, not peeps'.** peeps adjusts the
WPP death probabilities so that the stationary distribution of a discrete annual
Markov chain matches the observed age distribution. That chain places a band's
whole mass at its lower edge (`pi_a ∝ l_a`), while the Griffin/dmeq convention
integrates survival across the class — the two differ by `(1-e^-mu)/mu`, which is
~`mu/2` and so bites wherever the hazard is large. Fitting under peeps'
convention and consuming under ours left a 1–3% residual in single-year age
proportions below 30, worst at age 0 in the highest-mortality country-years
(Mali ~2.8%). `fit_mortality.py` fits the convention we consume, and the residual
drops to solver tolerance — see the table below.

**No optimiser is involved.** Both conventions are triangular: the target
proportions pin the hazards one age at a time. peeps' 101-parameter L-BFGS-B is
solving something with the closed form `q_a = 1 - p_{a+1}/p_a`. The continuous
version is a monotone scalar solve per band given one global constant, `e0`. The
whole fit is ~3 s single-threaded, against ~2 min on 16 cores for peeps, and
drops the R and Docker dependency.

**`e0` is a free parameter, chosen by least squares.** `e0 = sum_a mass_a` is the
fitted life expectancy at birth. Normalisation is automatic — the band masses sum
to `e0` whenever the per-band equations hold — so the age distribution does not
determine it; `--e0` selects the policy. The fitted values (21.9–29.2) are within
~1 year of what peeps' discrete fit already implies (`1/p_0`, 23.2–30.0), so this
is not a new degree of freedom, just an explicit one.

**Those values are far below the real WPP life expectancies** (46–59 for these
country-years), and `--e0 wpp` is *infeasible* for all 16 — it fails rather than
silently returning something. A stationary model must hold `e0 * p_0` of its
person-years in the first year of age, so `e0 < 1/p_0` is a hard ceiling, and
WPP's `ex[0]` exceeds it by ~2.1x everywhere. The cause is the constant-population
assumption both peeps and this script make: these populations were growing
~2.7%/yr, which makes their age structure youth-heavy for reasons unrelated to
mortality, and a stationary fit charges all of it to the hazard. Lifting that
would mean a stable-population model, `n(a) ∝ e^{-ra} S(a)` with a growth rate
`r` — which would let the fit use the real life table — and is a larger change
than this one.

**Residual against WPP** (`to_griffin_grid.py` prints this on every run):
single-year age proportions below 30, re-aggregated from the Griffin grid.

| | peeps | this fit |
| --- | --- | --- |
| 12 of 16 country-years | 0.95–2.30% | < 1e-4 % |
| MLI 1999 / 2000 | 2.78% / 2.71% | 0.05% |
| BFA 2004, KEN 2000 | 1.78% / 1.38% | 0.11% / 0.12% |
| GHA 2002 | 1.85% | 1.31% |

The four nonzero rows are the country-years where WPP's own `p` rises with age
somewhere (GHA at ages 21 and 31, BFA at 55/56, KEN at 49/50, MLI at 50/51). No
stationary model can reproduce that at any hazard, so those bands sit on the
lower bound `mu_wpp/2`. peeps hits the same infeasibility; it was just hidden
under a larger baseline error.
