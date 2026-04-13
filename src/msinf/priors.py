from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import beta, lognorm

PriorFamily = Literal["beta", "lognormal"]

# Core biological/immunity parameters the model can sample.
CORE_PARAMS: tuple[str, ...] = (
    "IB0",
    "IC0",
    "ID0",
    "PM",
    "ad0",
    "b0",
    "cU",
    "d1",
    "dm",
    "fd0",
    "gd",
    "kb",
    "kc",
    "kd",
    "phi0",
    "phi1",
    "ub",
    "uc",
    "ud",
)

# Core parameter name -> column name in the reference posterior draws CSV.
REFERENCE_PARAM_MAP: dict[str, str] = {
    "IB0": "ib0",
    "IC0": "ic0",
    "ID0": "id0",
    "b0": "b0",
    "cU": "cu",
    "d1": "d1",
    "fd0": "fd0",
    "gd": "gammad",
    "kb": "kb",
    "kc": "kc",
    "kd": "kd",
    "phi0": "phi0",
    "phi1": "phi1",
    "ub": "ub",
    "uc": "uc",
    "ud": "ud",
    "PM": "pcm",
    "dm": "rm",
    "ad0": "ad",
}

# Core/immunity parameters the model overrides on the dmeq defaults, in the order
# the model applies them (see msinf.model.hierarchical_model). The plotting code
# solves with this same list so overlays reproduce the curves the likelihood saw.
DMEQ_OVERRIDE_PARAMS: tuple[str, ...] = (
    "b0", "phi0", "phi1", "d1", "PM", "fd0", "ID0", "kd", "ud", "ad0",
    "gd", "IB0", "kb", "ub", "IC0", "kc", "uc", "dm",
)


@dataclass(frozen=True)
class PriorSpec:
    name: str
    family: PriorFamily
    p2_5: float
    median: float
    p97_5: float
    param1: float
    param2: float


def _fit_beta_from_quantiles(median: float, q025: float, q975: float) -> tuple[float, float]:
    m = np.clip(median, 1e-6, 1 - 1e-6)
    v0 = max(1e-4, ((q975 - q025) / 3.92) ** 2)
    k0 = max(2.0, m * (1 - m) / v0 - 1)
    x0 = np.log([m * k0, (1 - m) * k0])

    def residuals(z: np.ndarray) -> np.ndarray:
        a, b = np.exp(z)
        return np.array([
            beta.ppf(0.025, a, b) - q025,
            beta.ppf(0.5, a, b) - median,
            beta.ppf(0.975, a, b) - q975,
        ])

    res = least_squares(residuals, x0, max_nfev=2000)
    a, b = np.exp(res.x)
    return float(a), float(b)


def _fit_lognormal_from_quantiles(median: float, q025: float, q975: float) -> tuple[float, float]:
    z = 1.959963984540054
    mu = np.log(median)
    sigma = (np.log(q975) - np.log(q025)) / (2 * z)
    return float(mu), float(sigma)


def _build_specs(specs: list[tuple[str, PriorFamily, float, float]]) -> dict[str, PriorSpec]:
    out: dict[str, PriorSpec] = {}
    for name, family, p1, p2 in specs:
        if family == "beta":
            q025 = float(beta.ppf(0.025, p1, p2))
            med = float(beta.ppf(0.5, p1, p2))
            q975 = float(beta.ppf(0.975, p1, p2))
        else:
            q025 = float(lognorm.ppf(0.025, p2, scale=np.exp(p1)))
            med = float(np.exp(p1))
            q975 = float(lognorm.ppf(0.975, p2, scale=np.exp(p1)))
        out[name] = PriorSpec(name, family, q025, med, q975, float(p1), float(p2))
    return out


def build_xi_priors() -> dict[str, PriorSpec]:
    """Priors from Xi's subsequent Griffin-model refit."""
    # Parameterization from griffin-model-copy-copy.cpp, Eta::loglik.
    # For lognormal family: param1=meanlog, param2=sdlog.
    # For beta family: param1=alpha, param2=beta.
    xi_specs: list[tuple[str, PriorFamily, float, float]] = [
        ("d1", "beta", 10.1, 30.6),
        ("fd0", "beta", 1.0, 1.0),
        ("b0", "beta", 1.3, 1.3),
        ("phi0", "beta", 8.3, 2.1),
        ("phi1", "beta", 1.1, 2.0),
        ("PM", "beta", 1.0, 1.0),
        ("rW", "beta", 8.9, 3.0),
        ("rP", "beta", 1.0, 1.0),
        ("ID0", "lognormal", 2.38, 1.08),
        ("kd", "lognormal", 0.50, 0.32),
        ("ud", "lognormal", 1.40, 0.80),
        ("ad0", "lognormal", 2.02, 0.43),
        ("gd", "lognormal", 0.22, 0.64),
        ("IB0", "lognormal", 3.07, 0.93),
        ("kb", "lognormal", 0.50, 0.32),
        ("ub", "lognormal", 1.40, 0.80),
        ("IC0", "lognormal", 3.76, 1.02),
        ("kc", "lognormal", 0.50, 0.32),
        ("uc", "lognormal", 1.40, 0.80),
        ("dm", "lognormal", 5.26, 0.33),
    ]

    return _build_specs(xi_specs)


def build_griffin_priors() -> dict[str, PriorSpec]:
    """Priors transcribed from Griffin et al. (2014) Supplementary Table 1.

    The SI reports quantiles rather than distribution parameters for most rows;
    the same beta/lognormal families used by this module are fitted to those
    reported 2.5%, 50%, and 97.5% quantiles.
    """
    q: list[tuple[str, PriorFamily, float, float, float]] = [
        ("d1", "beta", .13, .25, .39), ("fd0", "beta", .025, .50, .975),
        ("b0", "beta", .05, .50, .95),
        ("phi0", "beta", .52, .82, .97), ("phi1", "beta", .02, .39, .85),
        ("PM", "beta", .025, .50, .975), ("rW", "beta", .48, .76, .94),
        ("rP", "beta", .025, .50, .975),
        ("ID0", "lognormal", 1.3, 21, 89), ("kd", "lognormal", .88, 2.01, 3.06),
        ("ud", "lognormal", .85, 5.85, 19.5), ("ad0", "lognormal", 3.3, 9.9, 17.3),
        ("gd", "lognormal", .36, 1.9, 4.3), ("IB0", "lognormal", 3.5, 43, 132),
        ("kb", "lognormal", .88, 2.01, 3.06), ("ub", "lognormal", .85, 5.85, 19.5),
        ("IC0", "lognormal", 5.8, 95, 316), ("kc", "lognormal", .88, 2.01, 3.06),
        ("uc", "lognormal", .85, 5.85, 19.5), ("dm", "lognormal", 102, 205, 362),
    ]
    out = {}
    for name, family, lo, med, hi in q:
        p1, p2 = (_fit_beta_from_quantiles(med, lo, hi) if family == "beta"
                  else _fit_lognormal_from_quantiles(med, lo, hi))
        out[name] = PriorSpec(name, family, lo, med, hi, p1, p2)
    return out


# Backwards-compatible active-prior entry point: inference currently uses Xi.
build_best_guess_priors = build_xi_priors


EIR_LOG_SD_BY_CATEGORY: dict[int, float] = {
    1: 0.3,
    2: 0.4,
    3: 0.5,
    4: 0.6,
    5: 0.8,
}
