# msinf

Reproduction-focused comparison repo for Griffin-style malaria inference.

## Inputs (not tracked in git)

Place local inputs at:

- `artifacts/inputs/study_data_2014.csv`
- `artifacts/inputs/eir_site_priors.csv`

Required EIR columns: `site_index,eir_estimate,eir_category`

## Run inference (Hydra)

Note: the runner sets NumPyro/XLA host device count to match `mcmc.num_chains`.

```bash
uv venv .venv          # any Python >= 3.11
source .venv/bin/activate
uv sync
python scripts/run_inference.py

# SA sampler variant (instead of HMC/NUTS)
python scripts/run_inference.py sampler=sa

# Override output location (otherwise defaults to Hydra run dir)
python scripts/run_inference.py run.outdir=[out dir]

# Float precision for the whole pipeline (default f32)
python scripts/run_inference.py run.precision=f64

# Active core subset via comma-delimited string
python scripts/run_inference.py 'core.active_core_csv=IC0,kc,phi0,phi1,uc'

# EIR-only core block
python scripts/run_inference.py core.active_core_csv=none

# Sweep parameter sets
python scripts/run_inference.py -m 'core.active_core_csv="IC0,kc,phi0,phi1,uc","ID0,d1,fd0,gd,kd,ud","IC0,kc,phi0,phi1,uc,ID0,d1,fd0,gd,kd,ud"'

# Per-site WPP mortality instead of dmeq's uniform death rate (default: uniform)
python scripts/run_inference.py data.mortality_csv=site_mortality_griffin_grid.csv
```

## Fit backends

A posterior can come from three backends, selected by `fit.backend`
(`mcmc` by default, running whichever kernel `sampler` selects):

```sh
# NUTS (default), or SA
python scripts/run_inference.py sampler=sa

# Full-rank Gaussian VI, no MCMC at all
python scripts/run_inference.py fit.backend=fullrank

# A Pathfinder approximation, used as the posterior
python scripts/run_inference.py fit.backend=pathfinder

# Still a NUTS fit -- only the starting point changes
python scripts/run_inference.py mcmc.init_strategy=pathfinder
```

Diagnostics are backend-specific (`r_hat`/ESS for chains, `khat`/ELBO for
variational fits); `fit_seconds` and `fit_ok` are always present. For a
variational fit read **`khat`**: under 0.7 the approximation is usable, and
`fit.vi.psir=true` resamples against the PSIS weights. See `msinf/fit.py`.

## Held-out evaluation

Two splits, reported separately and never averaged (`msinf/splits.py`):

- `study` — whole studies held out. Clean under the marginalised study effects,
  but blunt: a held-out study's `log_eir` sits on its prior.
- `age_band` — the upper age bands within every study. Conditional by
  construction, so what is tested is age structure.

```bash
python scripts/run_holdout.py +experiment=holdout_study
python scripts/run_holdout.py +experiment=holdout_age

# Between-fold noise: the same mechanism the core-block sweep uses.
# The study split has many folds; the age split is deterministic, so sweep the
# sampler seed there instead.
python scripts/run_holdout.py -m +experiment=holdout_study holdout.seed=1,2,3,4,5
python scripts/run_holdout.py -m +experiment=holdout_age run.seed=1,2,3,4,5
python scripts/holdout_report.py multirun/<date>/<time>
```

Each run writes `metrics.json` plus posterior-predictive and randomised-PIT
figures into its output directory. **Gate on the between-fold spread before
trusting `elpd_holdout` to rank anything:** if a fold costs more than a
candidate curve family does, the metric cannot rank candidates.

`elpd_holdout` scores prevalence per row and incidence per (site × age-group)
cell; see `msinf/likelihood.py`.

## Fitting a different curve family

The observational model is fixed. What varies is the map from parameters to age
curves, which is a pair of callables (`msinf/curves.py`) because NumPyro sample
sites cannot be created inside the `vmap` over sites:

```python
parameters()                                -> params
curves(params, log_eir, ft, ages, prop, mu) -> (prev_m, inc)
```

`msinf.evaluate.evaluate(cfg, provider=..., split=...)` fits and scores one,
returning metrics and figures as bytes. It uses no Hydra, assumes no working
directory and writes nothing, so it is safe to call inside a child process;
`scripts/run_holdout.py` is the same function's other caller.

## Diagnostics bundle

Generates every diagnostic figure plus a markdown report for one run directory:

```bash
python scripts/generate_diagnostics_bundle.py --run-dir outputs/<date>/<time>
```

Outputs land in `<run-dir>/diagnostics_bundle/` (override with `--outdir`), with
`diagnostics_report.md` as the entry point. The observed dataset and EIR prior
table default to the ones the run was fitted to; override with `--data-csv` /
`--eir-priors-csv`.

## Likelihood diagnostic

Compares the data log-likelihood at the Griffin/prior init against posterior
draws, to check whether the likelihood is rewarding predictive bias:

```bash
python scripts/likelihood_diag.py --run-dir outputs/<date>/<time>
```

Works with any run directory; pass `--precision f64` to match a run fitted at
double precision.

## Age trajectories for all studies

```python
from msinf.plotting import plot_all_study_trajectories

pngs = plot_all_study_trajectories(
    "outputs/<date>/<time>/numpyro_jamie_samples.npz",
    "artifacts/inputs/study_data_2014.csv",
    outdir="artifacts/figures/trajectories",
    effects="conditional",
    n_draws=100,
    max_age=20,
    ncols=4,
)
```

Returns paths under `pngs["incidence"]` and `pngs["prevalence"]`. Each PNG
contains a panel for every site with observations for that outcome. Blue
diamonds show observations, green curves and shading show the posterior median
and pointwise 95% credible interval, and orange diamonds show fitted
population-weighted age-group averages. Conditional study effects and incidence
case-detection scaling are included; observation overdispersion is excluded
from these bands. Prevalence is microscopy prevalence. These curves use dmeq's
default demography, so use a fit with `mortality_csv: null`.
