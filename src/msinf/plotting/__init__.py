"""All plotting for the repo: site overlays, marginals, immunity curves, age grids."""

from ._common import placeholder_png
from .age_grid_bootstrap import (
    GridSpec,
    compute_incidence_age_grid_bootstrap,
    compute_incidence_age_grid_numpyro_vs_reference,
    compute_incidence_age_grid_prior_predictive,
    run_from_artifacts_root,
)
from .immunity import (
    ImmunityPlotUnavailable,
    missing_immunity_params,
    plot_immunity_functions_overlay,
)
from .incidence import (
    plot_site_incidence_overlay,
    plot_site_incidence_overlay_prior_predictive,
    run_site_overlay_from_latest,
)
from .marginals import plot_core_marginals_overlay, plot_logeir_marginals_overlay
from .prevalence import (
    plot_site_prevalence_overlay,
    plot_site_prevalence_overlay_prior_predictive,
)
from .trajectories import (
    StudyTrajectories,
    plot_all_study_trajectories,
    plot_study_trajectories,
    study_trajectories,
)

__all__ = [
    "GridSpec",
    "ImmunityPlotUnavailable",
    "StudyTrajectories",
    "compute_incidence_age_grid_bootstrap",
    "compute_incidence_age_grid_numpyro_vs_reference",
    "compute_incidence_age_grid_prior_predictive",
    "missing_immunity_params",
    "placeholder_png",
    "plot_all_study_trajectories",
    "plot_core_marginals_overlay",
    "plot_immunity_functions_overlay",
    "plot_logeir_marginals_overlay",
    "plot_site_incidence_overlay",
    "plot_site_incidence_overlay_prior_predictive",
    "plot_site_prevalence_overlay",
    "plot_site_prevalence_overlay_prior_predictive",
    "plot_study_trajectories",
    "run_from_artifacts_root",
    "run_site_overlay_from_latest",
    "study_trajectories",
]
