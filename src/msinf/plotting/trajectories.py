"""Per-study age trajectories of prevalence and incidence, with uncertainty bands.

The site overlays in :mod:`.prevalence` and :mod:`.incidence` summarise the
model at each observation's age bin. This draws the whole curve instead: dmeq is
solved on the full age grid for a sample of posterior draws, and every site in
one study gets a prevalence panel and an incidence panel with the data points
(and their sampling CIs) on top.

Three kinds of band, chosen by ``effects``
-------------------------------------------
The study effects ``u`` (incidence) and ``w`` (prevalence), the site-age effect
``v_jk`` and the beta-binomial overdispersion are integrated out of the fit, so
none of them are in the samples file. A band that leaves them out describes the
mechanistic curve only, and a study whose level the study effect absorbed will
sit well outside it. So:

* ``none`` -- parameter uncertainty only: the curve with ``u = 0``, ``w = 0``.
* ``population`` -- what a *new* study at these sites would see: ``u`` and ``w``
  drawn from their fitted population distributions, plus ``v_jk`` and the
  beta overdispersion.
* ``conditional`` -- what *this* study's fitted level is: ``u`` and ``w`` drawn
  from their posterior given the study's data, plus ``v_jk`` and the beta
  overdispersion from their priors. That posterior is the Gauss-Hermite
  quadrature the likelihood already integrates over, reweighted by the study's
  node log-likelihood -- the same terms, from :mod:`msinf.likelihood`, so it
  cannot disagree with what the fit integrated.

The predictive bands are for the *rate* an infinitely large sample would
measure, not for counts: a curve has no denominator. Sampling noise belongs to
the data, and is drawn there as each point's CI.

The bands are pointwise. ``v_jk`` and the overdispersion are drawn
independently at each age, which gives the right marginal at each age and the
wrong joint across ages, so a band is to be read vertically and not as a set of
plausible curves.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats
from scipy.special import expit, logit

from ..age_grid import class_fraction_weights
from ..data import encode_canonical_sites, load_observed_data, site_to_study_from_df
from ..likelihood import (
    GH_WEIGHTS,
    incidence_cell_loglik_nodes,
    prevalence_row_loglik_nodes,
    study_effect_nodes,
)
from ..priors import DMEQ_OVERRIDE_PARAMS
from ._common import AGE_GRID, select_draw_indices, solve_curves

# dmeq solve(..) rows: 0 -> microscopy prevalence, 2 -> incidence (per person-day).
PREV_ROW = 0
INC_ROW = 2
TYPE_INCIDENCE = 1
TYPE_PREV_MICROSCOPY = 2

EFFECTS = ("none", "population", "conditional")

_DET_LABEL = {1: "Daily ACD", 2: "Weekly ACD (×rW)", 3: "PCD (×rP)"}
_DET_COLOR = {1: "#444444", 2: "#d62728", 3: "#9467bd"}
_PREV_COLOR = "#1f77b4"


@dataclass
class StudyTrajectories:
    """Per-age curves for every site in one study, one row per posterior draw.

    ``prev``/``inc`` are ``(D, n_sites, n_age)``: microscopy prevalence as a
    proportion and incidence per person-year, before case-detection scaling.
    ``prev_pred``/``inc_pred`` are the same with the random effects drawn back in,
    ``(D * n_rep, n_sites, n_age)``, and ``None`` when ``effects='none'``.
    """

    study_index: object
    sites: np.ndarray
    site_labels: list
    ages: np.ndarray
    prev: np.ndarray
    inc: np.ndarray
    rW: np.ndarray
    rP: np.ndarray
    data: pd.DataFrame
    effects: str = "none"
    prev_pred: np.ndarray | None = None
    inc_pred: np.ndarray | None = None
    #: rW/rP repeated to line up with the rows of ``inc_pred``
    rW_pred: np.ndarray | None = None
    rP_pred: np.ndarray | None = None
    #: the study effects behind the predictive draws, ``(D * n_rep,)``
    u: np.ndarray | None = None
    w: np.ndarray | None = None
    prop: np.ndarray | None = None


def load_posterior(samples_npz: Path) -> dict[str, np.ndarray]:
    """Posterior draws with chains flattened to ``(S, ...)``; warmup dropped."""
    post = np.load(samples_npz)
    out = {}
    for k in post.files:
        if k.startswith("warmup__") or k == "__grouped_by_chain__":
            continue
        arr = post[k]
        out[k] = arr.reshape((-1,) + arr.shape[2:])
    return out


def bin_means(curves: np.ndarray, prop: np.ndarray, age0: np.ndarray, age1: np.ndarray) -> np.ndarray:
    """Per-person rate over each row's age bin, ``(D, n_rows)``.

    ``curves`` and ``prop`` are each row's site curve, ``(D, n_rows, n_age)``,
    population-weighted as dmeq emits them. The same reduction as
    :func:`msinf.binning.weighted_bin_mean_batch`.
    """
    overlap = class_fraction_weights(age0, age1, AGE_GRID)[None, :, :]
    den = np.clip((overlap * prop).sum(-1), 1e-12, None)
    return (overlap * curves).sum(-1) / den


def study_effect_node_posterior(
    *,
    mu_inc: np.ndarray,
    numer_inc: np.ndarray,
    denom_inc: np.ndarray,
    case_det: np.ndarray,
    cell_id: np.ndarray,
    n_cells: int,
    p_prev: np.ndarray,
    numer_prev: np.ndarray,
    denom_prev: np.ndarray,
    rW: float,
    rP: float,
    sigma_c: float,
    sigma_p: float,
    alpha_c: float,
    alpha_p: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The posterior over the quadrature nodes of one study's ``u`` and ``w``.

    Returns ``(u_nodes, u_prob, w_nodes, w_prob)``. The likelihood integrates
    each effect as ``sum_g GH_WEIGHTS[g] * L(node_g)``; normalising those terms
    gives the effect's posterior on the same grid. A stream with no rows in the
    study leaves its effect at the prior. ``mu_inc`` is per person-day, as the
    likelihood takes it.
    """
    u_nodes, w_nodes = (np.asarray(x) for x in study_effect_nodes(sigma_c, sigma_p))
    log_gh = np.log(GH_WEIGHTS)

    u_log = log_gh.copy()
    if len(mu_inc):
        u_log = u_log + np.asarray(incidence_cell_loglik_nodes(
            mu_inc=mu_inc, numer=numer_inc, denom=denom_inc, case_det=case_det,
            cell_id=cell_id, n_cells=n_cells, rW=rW, rP=rP, alpha_c=alpha_c, u_nodes=u_nodes,
        )).sum(0)
    w_log = log_gh.copy()
    if len(p_prev):
        w_log = w_log + np.asarray(prevalence_row_loglik_nodes(
            p=p_prev, numer=numer_prev, denom=denom_prev,
            theta_prev=1.0 / max(alpha_p, 1e-8), w_nodes=w_nodes,
        )).sum(0)

    def normalise(log_p):
        p = np.exp(log_p - log_p.max())
        return p / p.sum()

    return u_nodes, normalise(u_log), w_nodes, normalise(w_log)


def _draw_on_grid(rng, nodes: np.ndarray, prob: np.ndarray, size: int) -> np.ndarray:
    """Sample a node, then jitter uniformly within its cell so draws are continuous."""
    idx = rng.choice(len(nodes), size=size, p=prob)
    # GH nodes are scaled copies of GH_NODES, so the cell half-widths scale too.
    half = np.diff(nodes) / 2
    lo = np.concatenate([[half[0]], half])
    hi = np.concatenate([half, [half[-1]]])
    return nodes[idx] + rng.uniform(-lo[idx], hi[idx])


def _predictive(
    *, rng, prev, inc, post, idx, effects, n_rep, data, sites, prop, n_age_groups,
) -> dict[str, np.ndarray]:
    """Draw ``u``, ``w``, ``v_jk`` and the beta overdispersion into the curves."""
    D, n_sites, n_age = prev.shape
    sigma_c, sigma_p = post["sigma_c"][idx], post["sigma_p"][idx]
    alpha_c, alpha_p = post["alpha_c"][idx], post["alpha_p"][idx]
    rW, rP = post["rW"][idx], post["rP"][idx]

    if effects == "conditional":
        local = {s: i for i, s in enumerate(sites)}
        inc_rows = data[data["type"] == TYPE_INCIDENCE]
        prev_rows = data[data["type"] == TYPE_PREV_MICROSCOPY]
        inc_site = inc_rows["site_i"].map(local).to_numpy(int)
        prev_site = prev_rows["site_i"].map(local).to_numpy(int)
        cell_id = inc_site * n_age_groups + (inc_rows["age_bracket"].to_numpy(int) - 1)
        # Raw dmeq curves (population-weighted) for each row's site, binned as the model bins them.
        raw_inc = inc * prop / 365.0
        raw_prev = prev * prop
        mu_inc = bin_means(raw_inc[:, inc_site], prop[:, inc_site],
                           inc_rows["age0"].to_numpy(float), inc_rows["age1"].to_numpy(float))
        p_prev = bin_means(raw_prev[:, prev_site], prop[:, prev_site],
                           prev_rows["age0"].to_numpy(float), prev_rows["age1"].to_numpy(float))

    u = np.empty((D, n_rep))
    w = np.empty((D, n_rep))
    for d in range(D):
        if effects == "conditional":
            u_nodes, u_prob, w_nodes, w_prob = study_effect_node_posterior(
                mu_inc=mu_inc[d], numer_inc=inc_rows["numer"].to_numpy(float),
                denom_inc=inc_rows["denom"].to_numpy(float),
                case_det=inc_rows["case_detection"].to_numpy(int), cell_id=cell_id,
                n_cells=n_sites * n_age_groups,
                p_prev=p_prev[d], numer_prev=prev_rows["numer"].to_numpy(float),
                denom_prev=prev_rows["denom"].to_numpy(float),
                rW=float(rW[d]), rP=float(rP[d]), sigma_c=float(sigma_c[d]), sigma_p=float(sigma_p[d]),
                alpha_c=float(alpha_c[d]), alpha_p=float(alpha_p[d]),
            )
            u[d] = _draw_on_grid(rng, u_nodes, u_prob, n_rep)
            w[d] = _draw_on_grid(rng, w_nodes, w_prob, n_rep)
        else:
            u[d] = sigma_c[d] * rng.standard_normal(n_rep) - 0.5 * sigma_c[d] ** 2
            w[d] = sigma_p[d] * rng.standard_normal(n_rep)

    # Site-age effect v ~ Gamma(1/alpha_c, alpha_c), mean 1, drawn per age point.
    shape = (1.0 / np.clip(alpha_c, 1e-8, None))[:, None, None, None]
    v = rng.gamma(shape, 1.0 / shape, size=(D, n_rep, n_sites, n_age))
    inc_pred = inc[:, None] * np.exp(u)[:, :, None, None] * v

    # Beta half of the beta-binomial around q = sigmoid(logit(p) + w).
    p = np.clip(prev, 1e-6, 1 - 1e-6)
    q = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p))[:, None] + w[:, :, None, None])))
    theta = (1.0 / np.clip(alpha_p, 1e-8, None))[:, None, None, None]
    q = np.clip(q, 1e-12, 1 - 1e-12)
    prev_pred = rng.beta(q * theta, (1 - q) * theta)

    flat = lambda x: x.reshape((D * n_rep,) + x.shape[2:])
    return {
        "prev_pred": flat(prev_pred), "inc_pred": flat(inc_pred),
        "rW_pred": np.repeat(rW, n_rep), "rP_pred": np.repeat(rP, n_rep),
        "u": u.reshape(-1), "w": w.reshape(-1),
    }


def study_trajectories(
    samples_npz: Path,
    data_csv: Path,
    study_index,
    *,
    n_draws: int = 100,
    effects: str = "none",
    n_rep: int = 20,
    seed: int = 0,
) -> StudyTrajectories:
    """Solve dmeq for every site in ``study_index`` over ``n_draws`` posterior draws.

    ``study_index`` is the raw ``study_index`` value in ``data_csv``. ``effects``
    is one of :data:`EFFECTS` (see the module docstring); for the two predictive
    modes each posterior draw is replicated ``n_rep`` times.

    The solve uses dmeq's default demography, so for a run fitted with
    ``mortality_csv`` set these are not exactly the curves the likelihood saw.
    """
    if effects not in EFFECTS:
        raise ValueError(f"effects must be one of {EFFECTS}; got {effects!r}")
    post = load_posterior(Path(samples_npz))
    df = load_observed_data(Path(data_csv))
    df, site_labels = encode_canonical_sites(df)
    site_to_study, study_map, _ = site_to_study_from_df(df)
    return _study_trajectories(
        post, df, site_labels, site_to_study, study_map, study_index,
        n_draws=n_draws, effects=effects, n_rep=n_rep, seed=seed,
    )


def _study_trajectories(
    post, df, site_labels, site_to_study, study_map, study_index,
    *, n_draws, effects, n_rep, seed,
):
    if study_index not in study_map:
        raise ValueError(f"study_index {study_index!r} not in data; have {sorted(study_map)}")
    sites = np.flatnonzero(site_to_study == study_map[study_index])

    idx = select_draw_indices(n_total=len(post["log_eir"]), n_eval=n_draws, seed=seed)
    core = {k: post[k][idx] for k in DMEQ_OVERRIDE_PARAMS if k in post}
    kw = {
        "log_eir": post["log_eir"][idx][:, sites], "ft_study": post["ft_study"][idx],
        "site_to_study": site_to_study[sites], "core_params": core,
    }
    prev_raw, prop = solve_curves(curve_row=PREV_ROW, **kw)
    inc_raw, _ = solve_curves(curve_row=INC_ROW, **kw)
    # Every row is population-weighted; per person is per age-class proportion.
    prop = np.clip(prop, 1e-12, None)
    prev = prev_raw / prop
    inc = inc_raw / prop * 365.0

    data = df[df["study_index"] == study_index]
    out = StudyTrajectories(
        study_index=study_index, sites=sites,
        site_labels=[site_labels[s] if site_labels else s for s in sites],
        ages=AGE_GRID, prev=prev, inc=inc,
        rW=post["rW"][idx], rP=post["rP"][idx], data=data, effects=effects, prop=prop,
    )
    if effects != "none":
        pred = _predictive(
            rng=np.random.default_rng(seed), prev=prev, inc=inc, post=post, idx=idx,
            effects=effects, n_rep=n_rep, data=data, sites=sites, prop=prop,
            # The model's cell grid is sized from the whole table, not the study.
            n_age_groups=int(df["age_bracket"].max()),
        )
        for k, v in pred.items():
            setattr(out, k, v)
    return out


def _band(ax, x, draws, color, label):
    q = np.quantile(draws, [0.025, 0.25, 0.5, 0.75, 0.975], axis=0)
    ax.fill_between(x, q[0], q[4], color=color, alpha=0.15, lw=0)
    ax.fill_between(x, q[1], q[3], color=color, alpha=0.30, lw=0)
    ax.plot(x, q[2], color=color, lw=1.6, label=f"{label} (median, 50/95%)")


def _pred_band(ax, x, draws, color, label):
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    ax.plot(x, lo, color=color, lw=1.0, ls="--", label=label)
    ax.plot(x, hi, color=color, lw=1.0, ls="--")


def _detection_scale(det: int, rW: np.ndarray, rP: np.ndarray):
    return {1: 1.0, 2: rW, 3: rP, 6: rP}.get(det, 1.0)


def plot_study_trajectories(
    tr: StudyTrajectories,
    *,
    max_age: float | None = None,
    log_inc: bool = False,
    out_png: Path | None = None,
):
    """One row per site: prevalence left, incidence right, data with 95% CIs on top.

    Shaded bands are parameter uncertainty (median, 50%, 95%). With predictive
    draws the dashed lines are their 95% band. Prevalence CIs are Jeffreys
    intervals, incidence CIs exact Poisson intervals with ``denom`` in
    person-years.
    """
    # Class midpoints; the last class is the absorbing [100, inf) compartment.
    x = tr.ages[:-1] + 0.5 * np.diff(tr.ages)
    n = len(tr.sites)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.2 * n), squeeze=False)
    data = tr.data
    max_age = max_age or float(data["age1"].max()) * 1.1
    pred_label = {"population": "new-study 95%", "conditional": "this-study 95%"}.get(tr.effects)

    for r, (s, lab) in enumerate(zip(tr.sites, tr.site_labels)):
        ax_p, ax_i = axes[r]
        sd = data[data["site_i"] == s]

        _band(ax_p, x, tr.prev[:, r, :-1], _PREV_COLOR, "model")
        if tr.prev_pred is not None:
            _pred_band(ax_p, x, tr.prev_pred[:, r, :-1], _PREV_COLOR, pred_label)
        pv = sd[sd["type"] == TYPE_PREV_MICROSCOPY]
        if len(pv):
            k, m = pv["numer"].to_numpy(float), pv["denom"].to_numpy(float)
            p = k / m
            lo = stats.beta.ppf(0.025, k + 0.5, m - k + 0.5)
            hi = stats.beta.ppf(0.975, k + 0.5, m - k + 0.5)
            mid = 0.5 * (pv["age0"] + pv["age1"]).to_numpy()
            ax_p.errorbar(mid, p, yerr=[p - lo, hi - p],
                          xerr=[mid - pv["age0"].to_numpy(), pv["age1"].to_numpy() - mid],
                          fmt="o", color="k", ms=4, capsize=2, lw=0.8, label="data (95% CI)")
        ax_p.set(xlim=(0, max_age), ylim=(0, 1), ylabel="microscopy prevalence", title=f"{lab}\nprevalence")

        iv = sd[sd["type"] == TYPE_INCIDENCE]
        for det in sorted(iv["case_detection"].astype(int).unique()) or [1]:
            color = _DET_COLOR.get(det, "#444444")
            scale = np.reshape(_detection_scale(det, tr.rW, tr.rP), (-1, 1))
            _band(ax_i, x, tr.inc[:, r, :-1] * scale, color, _DET_LABEL.get(det, f"case_detection={det}"))
            if tr.inc_pred is not None:
                scale_pred = np.reshape(_detection_scale(det, tr.rW_pred, tr.rP_pred), (-1, 1))
                _pred_band(ax_i, x, tr.inc_pred[:, r, :-1] * scale_pred, color, pred_label)
            dd = iv[iv["case_detection"] == det]
            if len(dd):
                k, t = dd["numer"].to_numpy(float), dd["denom"].to_numpy(float)
                rate = k / t
                lo = np.nan_to_num(stats.chi2.ppf(0.025, 2 * k) / 2 / t)
                hi = stats.chi2.ppf(0.975, 2 * (k + 1)) / 2 / t
                mid = 0.5 * (dd["age0"] + dd["age1"]).to_numpy()
                ax_i.errorbar(mid, rate, yerr=[rate - lo, hi - rate],
                              xerr=[mid - dd["age0"].to_numpy(), dd["age1"].to_numpy() - mid],
                              fmt="o", color=color, mec="k", ms=4, capsize=2, lw=0.8)
        ax_i.set(xlim=(0, max_age), ylabel="clinical incidence (per person-year)", title="incidence")
        if log_inc:
            ax_i.set_yscale("log")
        for ax in (ax_p, ax_i):
            ax.set_xlabel("age (years)")
            ax.legend(fontsize=7, frameon=False)

    fig.suptitle(f"study {tr.study_index}", y=1.0)
    fig.tight_layout()
    if out_png is not None:
        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    return fig


def _fitted_draws(tr, site, outcome, det=1):
    """Joint parameter/study-effect draws, excluding observation overdispersion."""
    curves = tr.inc[:, site] if outcome == "incidence" else tr.prev[:, site]
    prop = tr.prop[:, site]
    if outcome == "incidence":
        curves = curves * np.reshape(_detection_scale(det, tr.rW, tr.rP), (-1, 1))
    effect = tr.u if outcome == "incidence" else tr.w
    if effect is not None:
        repeats = len(effect) // len(curves)
        curves = np.repeat(curves, repeats, axis=0)
        prop = np.repeat(prop, repeats, axis=0)
        if outcome == "incidence":
            curves = curves * np.exp(effect[:, None])
        else:
            curves = expit(logit(np.clip(curves, 1e-12, 1 - 1e-12)) + effect[:, None])
    return curves, prop


def _plot_all_grid(trajectories, outcome, out_png, *, ncols, max_age, dpi, log_inc):
    type_id = TYPE_INCIDENCE if outcome == "incidence" else TYPE_PREV_MICROSCOPY
    panels = [
        (tr, j, label, tr.data[(tr.data["site_i"] == site) & (tr.data["type"] == type_id)])
        for tr in trajectories
        for j, (site, label) in enumerate(zip(tr.sites, tr.site_labels))
        if ((tr.data["site_i"] == site) & (tr.data["type"] == type_id)).any()
    ]
    if not panels:
        raise ValueError(f"No {outcome} observations in the supplied data")
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 2.25 * nrows), squeeze=False)
    detection_codes = set()
    try:
        for ax, (tr, site, label, rows) in zip(axes.flat, panels):
            x = tr.ages[:-1] + 0.5 * np.diff(tr.ages)
            groups = rows.groupby("case_detection", sort=True) if outcome == "incidence" else [(1, rows)]
            visible_max = 0.0
            for det, group in groups:
                det = int(det)
                detection_codes.add(det)
                draws, prop = _fitted_draws(tr, site, outcome, det)
                # Distinguish detection methods only where a site has several.
                linestyle = {1: "-", 2: "--", 3: ":", 6: "-."}.get(det, "-")
                lo, median, hi = np.quantile(draws[:, :-1], [0.025, 0.5, 0.975], axis=0)
                visible_max = max(visible_max, float(hi[x <= max_age].max(initial=0)))
                ax.fill_between(x, lo, hi, color="#b8df9c", alpha=0.65, lw=0)
                ax.plot(x, median, color="#16823b", lw=1.2, ls=linestyle if outcome == "incidence" else "-")
                age0, age1 = group["age0"].to_numpy(float), group["age1"].to_numpy(float)
                mid = (age0 + age1) / 2
                observed = (group["numer"] / group["denom"]).to_numpy()
                visible_max = max(visible_max, float(observed[mid <= max_age].max(initial=0)))
                ax.plot(mid, observed, "D", color="#2449d8", ms=3, zorder=4)
                # Use the same population-weighted age-bin reduction as the likelihood.
                weights = class_fraction_weights(age0, age1, tr.ages)
                binned = (draws * prop) @ weights.T / np.clip(prop @ weights.T, 1e-12, None)
                ax.plot(mid, np.median(binned, axis=0), "d", ms=4, mfc="white", mec="#e58a24", mew=0.9, zorder=5)
            title = str(label).replace("||", " · ").replace("_", " ")
            ax.set_title(f"{title} (study {tr.study_index})", fontsize=7, loc="left", wrap=True)
            ax.set_xlim(0, max_age)
            if outcome == "prevalence":
                ax.set_ylim(0, 1)
            elif log_inc:
                ax.set_yscale("log")
            else:
                # Autoscale using only the visible ages, not the adult tail.
                ax.set_ylim(0, max(visible_max * 1.08, 1e-6))
            ax.tick_params(labelsize=7)
        for ax in list(axes.flat)[len(panels):]:
            ax.set_visible(False)
        handles = [
            Line2D([], [], marker="D", ls="", color="#2449d8", ms=4, label="observed data"),
            Line2D([], [], color="#16823b", label="fitted model (posterior median)"),
            Patch(facecolor="#b8df9c", alpha=0.65, label="95% pointwise credible interval"),
            Line2D([], [], marker="d", ls="", mfc="white", mec="#e58a24", label="fitted age-group average"),
        ]
        if outcome == "incidence":
            for det in sorted(detection_codes):
                handles.append(Line2D([], [], color="#16823b", ls={1: "-", 2: "--", 3: ":", 6: "-."}.get(det, "-"),
                                      label=_DET_LABEL.get(det, f"case detection {det}")))
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
        mode = trajectories[0].effects
        uncertainty = {"conditional": "parameters and conditional study effects",
                       "population": "parameters and new-study effects", "none": "parameters"}[mode]
        fig.suptitle(f"{'Clinical incidence' if outcome == 'incidence' else 'Microscopy prevalence'} by age: {uncertainty}", fontsize=11)
        fig.supxlabel("Age (years)", y=0.075, fontsize=10)
        fig.supylabel("Incidence per person-year" if outcome == "incidence" else "Prevalence (proportion)", fontsize=10)
        fig.tight_layout(rect=(0.02, 0.09, 1, 0.965))
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(fig)
    return out_png


def plot_all_study_trajectories(
    samples_npz: Path,
    data_csv: Path,
    *,
    outdir: Path = Path("artifacts/figures"),
    n_draws: int = 100,
    effects: str = "conditional",
    n_rep: int = 20,
    seed: int = 0,
    ncols: int = 4,
    max_age: float = 20,
    dpi: int = 200,
    log_inc: bool = False,
    studies=None,
) -> dict[str, Path]:
    """Save separate incidence/prevalence PNG grids for all observed studies.

    Returns ``{"incidence": Path, "prevalence": Path}``. Each panel is a site
    with observations for that outcome. Curves include detection scaling and,
    by default, conditional study effects. Shading is the joint posterior's
    pointwise 95% credible interval, excluding site-age/observation noise.
    Orange diamonds are population-weighted fitted age-bin medians. Only
    microscopy prevalence (type 2) is shown. Default demography is used, as in
    :func:`study_trajectories`; pass the full data table used by the fit.

    ``studies`` limits the panels to those raw ``study_index`` values -- the
    training studies of a holdout fit, say -- while sites are still encoded
    against the whole table, as the fit encoded them.
    """
    if effects not in EFFECTS:
        raise ValueError(f"effects must be one of {EFFECTS}; got {effects!r}")
    if min(n_draws, n_rep, ncols, dpi) < 1 or max_age <= 0:
        raise ValueError("n_draws, n_rep, ncols, dpi and max_age must be positive")
    post = load_posterior(Path(samples_npz))
    df, labels = encode_canonical_sites(load_observed_data(Path(data_csv)))
    site_to_study, study_map, levels = site_to_study_from_df(df)
    trajectories = [
        _study_trajectories(
            post, df, labels, site_to_study, study_map, level,
            n_draws=n_draws, effects=effects, n_rep=n_rep, seed=seed,
        )
        for level in levels
        if studies is None or level in set(studies)
    ]
    return {
        outcome: _plot_all_grid(
            trajectories, outcome, Path(outdir) / f"all_studies_{outcome}.png",
            ncols=ncols, max_age=max_age, dpi=dpi, log_inc=log_inc,
        )
        for outcome in ("incidence", "prevalence")
    }
