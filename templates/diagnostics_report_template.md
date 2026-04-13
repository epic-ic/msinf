# MCMC Diagnostics Bundle

Generated: $generated_at

- Chains: **$num_chains**
- Warmup draws per chain: **$num_warmup**
- Posterior draws per chain: **$num_samples**

Core diagnostics summary table (R-hat, ESS)

$summary_md_table

$warmup_trace_section

![Core parameter trace plots showing chain mixing over posterior iterations for active core parameters.]($trace_core)

![Per-site prior to posterior update for log(EIR).]($eir_panel)

![Core marginals comparing prior and NumPyro posterior, parameter-wise.]($core_prior_vs_numpyro)

![Core marginals comparing NumPyro posterior and Griffin reference posterior draws.]($core_numpyro_vs_ref)

![Immunity probability functions over exposure count with sampled trajectories and median overlays for prior, NumPyro posterior, and Griffin reference posterior.]($immunity_overlay)

![Incidence-age grid with one representative median draw from each posterior overlaid in each EIR×ft panel.]($inc_grid_median_draw_compare)

![Incidence-age grid comparing posterior median curves with bootstrap uncertainty bands for NumPyro and Griffin.]($inc_grid_bootstrap_compare)

![Site-faceted prior-predictive microscopy prevalence overlay of observed prevalence versus prior-predictive obs-scale prevalence.]($site_overlay_prev_prior)

![Site-faceted prior-predictive incidence-rate overlay of observed incidence versus prior-predictive obs-scale rates.]($site_overlay_prior)

![Site-faceted microscopy prevalence overlay of observed prevalence versus predicted observed-scale prevalence.]($site_overlay_prev)

![Site-faceted incidence-age overlay of observed incidence versus predicted observed-scale rates.]($site_overlay)

![Prior-predictive observed versus predicted microscopy prevalence calibration across site-age points.]($calibration_prev_prior)

![Prior-predictive observed versus predicted incidence calibration across site-age points.]($calibration_prior)

![Overall observed versus predicted microscopy prevalence calibration across site-age points.]($calibration_prev)

![Overall observed versus predicted incidence calibration across site-age points.]($calibration)
