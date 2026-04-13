# Equilibrium solution (dmeq) and full NumPyro model formulation

This note documents the deterministic equilibrium model (`dmeq`) and the hierarchical observation model used in `msinf/src/msinf/model.py`, using Griffin et al notation from the 2014 SI.

## 1) Deterministic equilibrium solution (`dmeq`)

Let age (years) be indexed by bin midpoint \(a\), with bin widths \(\Delta a\) in days. Let
\[
r(a)=\begin{cases}
1/\Delta a, & \text{non-terminal bin}\\
0, & \text{terminal bin}
\end{cases},
\qquad
r_e(a)=r(a)+\eta.
\]

Population age mass (stable demography) is computed recursively:
\[
\pi(a_1)=\frac{\eta}{r(a_1)+\eta},
\qquad
\pi(a_i)=\frac{r(a_{i-1})\,\pi(a_{i-1})}{r(a_i)+\eta}.
\]

Relative biting by age:
\[
\psi(a)=1-\rho\exp\!\left(-\frac{a}{a_0}\right).
\]

Heterogeneity in exposure uses Gauss-Hermite nodes \(z_h\), weights \(w_h\):
\[
\zeta_h=\exp\!\left(-\frac{s^2}{2}+\sqrt{s^2}\,z_h\right),
\qquad
\sum_h w_h=1.
\]

For each heterogeneity node, effective inoculation pressure is
\[
\varepsilon_h(a)=\frac{\mathrm{EIR}}{365}\,\zeta_h\,\psi(a).
\]

### 1.1 Immunity recursions

`dmeq` uses the Griffin recursion form for immunity variables:
\[
I_x(a_i)=\frac{\lambda_x(a_i)/(1+u_x\lambda_x(a_i)) + r_e(a_i) I_x(a_{i-1})}{1/d_x + r_e(a_i)}.
\]

Applied as:
- Pre-erythrocytic immunity \(I_B\): \(\lambda_B=\varepsilon_h,\; u_x=u_b,\; d_x=d_b\)
- Clinical immunity \(I_C\): \(\lambda_C=\Lambda,\; u_x=u_c,\; d_x=d_c\)
- Detection immunity \(I_D\): \(\lambda_D=\Lambda,\; u_x=u_d,\; d_x=d_d\)

where force of infection is
\[
b(a)=b_0\left[b_1+\frac{1-b_1}{1+(I_B/IB_0)^{k_b}}\right],
\qquad
\Lambda(a)=b(a)\,\varepsilon_h(a).
\]

Detection probability modifier:
\[
f_D(a)=1-\frac{1-f_{D0}}{1+(a/a_{D0})^{\gamma_D}},
\]
\[
q(a)=d_1+\frac{1-d_1}{1+(I_D/ID_0)^{k_D} f_D(a)}.
\]

Clinical disease probability:
\[
\phi(a)=\phi_0\left[\phi_1+\frac{1-\phi_1}{1+\left((I_C+I_{CM})/IC_0\right)^{k_C}}\right].
\]

Maternal immunity \(I_{CM}\) follows Griffin-style decay from age-20 acquired immunity, scaled by \(P_M\) and \(d_m\).

### 1.2 State equilibrium equations

States are \((T,D,P,A,U,S)\) for treated clinical, untreated clinical, prophylaxis, asymptomatic patent, subpatent, susceptible.

Define
\[
\beta_T=r_T+r_e,\;\beta_D=r_D+r_e,\;\beta_P=r_P+r_e,\;\beta_A=\phi\Lambda+r_A+r_e,\;\beta_U=\Lambda+r_U+r_e.
\]
\[
a_T=\frac{f_t\phi\Lambda}{\beta_T},\quad
 a_D=\frac{(1-f_t)\phi\Lambda}{\beta_D},\quad
 a_P=\frac{r_T a_T}{\beta_P}.
\]

With carry-over terms from previous age bin \((b_T,b_D,b_P,r_AA_-,r_UU_-)\),
\[
Y=\frac{\pi-(b_T+b_D+b_P)}{1+a_T+a_D+a_P},
\]
\[
T=a_TY+b_T,\quad D=a_DY+b_D,\quad P=a_PY+b_P,
\]
\[
A=\frac{r_AA_- + (1-\phi)Y\Lambda + r_DD}{\beta_A+(1-\phi)\Lambda},
\qquad
U=\frac{r_UU_-+r_AA}{\beta_U},
\]
\[
S=Y-A-U.
\]

Outputs per age:
\[
\text{Prev}_{\mathrm{mic}}=T+D+A\,q,
\]
\[
\text{Prev}_{\mathrm{PCR}}=T+D+A\,q^{a_A}+U\,q^{a_U},
\]
\[
\mu_{\mathrm{inc}}=(S+U+A)\,\Lambda\,\phi.
\]

Final site curves are heterogeneity averages \(\sum_h w_h(\cdot)_h\).

---

## 2) Full NumPyro model (`hierarchical_model`)

Indices (Griffin-consistent):
- studies \(\ell=1,\dots,L\)
- sites \(j=1,\dots,J\)
- age groups \(k=1,\dots,K\)
- observations \(i\) within each data stream

### 2.1 Priors for latent epidemiology

For each site:
\[
\log \mathrm{EIR}_j \sim \mathcal N(\widehat{\log \mathrm{EIR}}_j,\sigma_{\mathrm{EIR},j}^2).
\]

For each study:
\[
f_{t,\ell} \sim \mathrm{Beta}(\alpha^{(f_t)}_\ell,\beta^{(f_t)}_\ell).
\]

Core biological parameters (subset of \(\{IB_0,IC_0,ID_0,P_M,a_{D0},b_0,c_U,d_1,d_m,f_{D0},\gamma_D,k_b,k_c,k_d,\phi_0,\phi_1,u_b,u_c,u_d\}\)) are sampled from the specified priors in `priors.py`; inactive ones are fixed to reference means.

### 2.2 Random effects and hyperpriors

Incidence multiplicative study effect (as in Griffin):
\[
u_\ell \sim \mathrm{LogNormal}\!\left(-\frac{\sigma_c^2}{2},\sigma_c\right),
\qquad
\sigma_c\sim \mathrm{HalfNormal}(5).
\]

Incidence site-age multiplicative effect:
\[
v_{jk}\sim \mathrm{Gamma}\!\left(\frac{1}{\alpha_c},\frac{1}{\alpha_c}\right),
\qquad
\alpha_c\sim t_6(0,5)\;\text{truncated to }(0,\infty).
\]
(shape-rate form, so \(\mathbb E[v_{jk}]=1\), \(\mathrm{Var}(v_{jk})=\alpha_c\)).

Prevalence additive logit study effect:
\[
w_\ell\sim \mathcal N(0,\sigma_p^2),
\qquad
\sigma_p\sim \mathrm{HalfNormal}(5),
\qquad
\theta=1/\alpha_p,
\]
\[
\alpha_p\sim t_6(0,5)\;\text{truncated to }(0,\infty).
\]

### 2.3 Mechanistic mean curves from equilibrium solver

For each site \(j\), with \(f_{t,j}=f_{t,\ell(j)}\), run the equilibrium solver:
\[
(\mu^{\mathrm{mic}}_j(a),\mu^{\mathrm{pcr}}_j(a),\mu^{\mathrm{inc}}_j(a),\pi_j(a))
=\mathrm{dmeq}(\mathrm{EIR}_j,f_{t,j},\vartheta).
\]

For each observed age interval \([a_0,a_1)\), map curve to bin-average using age-overlap and demographic weighting \(\pi_j(a)\):
\[
\bar\mu = \frac{\sum_a \omega(a;[a_0,a_1))\,\pi_j(a)\,\mu(a)}{\sum_a \omega(a;[a_0,a_1))\,\pi_j(a)}.
\]

### 2.4 Observation models

#### Incidence

Let case-detection multiplier be
\[
r_i=\begin{cases}
r_W,&\text{case\_detection}=2\\
r_P,&\text{case\_detection}\in\{6,3\}\\
1,&\text{otherwise}
\end{cases}
\]
(with \(r_W,r_P\) among sampled parameters).

Then
\[
y^{\mathrm{inc}}_i \sim \mathrm{Poisson}(\lambda_i),
\]
\[
\lambda_i = T_i\,r_i\,\nu_{\ell(i)}\,v_{j(i)k(i)}\,\bar\mu^{\mathrm{inc}}_{j(i),k(i)}.
\]

#### Microscopy prevalence

\[
q^{\mathrm{mic}}_i = \operatorname{logit}^{-1}\!\left(\operatorname{logit}(\bar\mu^{\mathrm{mic}}_{j(i),k(i)}) + w_{\ell(i)}\right),
\]
\[
y^{\mathrm{mic}}_i \sim \mathrm{BetaBinomial}(n_i,\;q^{\mathrm{mic}}_i\theta,\;(1-q^{\mathrm{mic}}_i)\theta).
\]

#### PCR prevalence

\[
q^{\mathrm{pcr}}_i = \operatorname{logit}^{-1}\!\left(\operatorname{logit}(\bar\mu^{\mathrm{pcr}}_{j(i),k(i)}) + w_{\ell(i)}\right),
\]
\[
y^{\mathrm{pcr}}_i \sim \mathrm{BetaBinomial}(n_i,\;q^{\mathrm{pcr}}_i\theta,\;(1-q^{\mathrm{pcr}}_i)\theta).
\]

---

## 3) Symbol/code correspondence (key)

- \(f_t\) ↔ `ft` / `ft_study`
- \(\phi_0,\phi_1\) ↔ `phi0`, `phi1`
- \(IB_0,IC_0,ID_0\) ↔ `IB0`, `IC0`, `ID0`
- \(k_b,k_c,k_d\) ↔ `kb`, `kc`, `kd`
- \(u_b,u_c,u_d\) ↔ `ub`, `uc`, `ud`
- \(a_{D0},f_{D0},\gamma_D\) ↔ `ad0`, `fd0`, `gd`
- \(P_M,d_m\) ↔ `PM`, `dm`
- \(\nu_\ell\) (incidence study effect) ↔ `u_study`
- \(v_{jk}\) ↔ `v_site_age`
- \(w_\ell\) ↔ `w_study`

This notation follows Griffin SI conventions while matching current NumPyro implementation details.
