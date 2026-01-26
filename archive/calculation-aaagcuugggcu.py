# ================================= calculation.py ===============================
"""
Key summary metrics for the manuscript.

1. MAE of the hierarchical quadratic model.
2. ΔELPD (quadratic – linear) + SE; LOO preferred, WAIC fallback.
3. Median world-GDP growth 2025→2040 (95 % HDI) pooled across UN scenarios.
4. Top-three pooled economies in 2040.
5. Share of forecast variance attributable to between-scenario heterogeneity.
6. Marginal R²: % of cross-country log-GDP variance explained by population.
"""

import numpy as np
import pandas as pd
import arviz as az
from  scipy.special import logsumexp          # stable Akaike weights

# -------------------------------------------------------------------- paths ----
ROOT = "C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP"
PATH_POST_QUAD   = f"{ROOT}/hierarchical_model_with_quadratic.nc"
PATH_POST_LINEAR = f"{ROOT}/hierarchical_model_linear.nc"
PATH_META        = f"{ROOT}/gdp_predictions_meta.csv"
PATH_SCENARIO    = f"{ROOT}/gdp_predictions_scenarios.csv"
PATH_MERGED      = f"{ROOT}/merged.csv"

# ------------------------------------------------------------------- 1.  MAE ----
idata_quad = az.from_netcdf(PATH_POST_QUAD)

if "posterior_predictive" in idata_quad.groups():
    mu_hat = (
        idata_quad.posterior_predictive["Log_GDP_obs"]
        .mean(("chain", "draw"))
        .values
        .ravel()
    )
    y_obs = idata_quad.observed_data["Log_GDP_obs"].values.ravel()
else:
    a_hat = idata_quad.posterior["alpha_country"].mean(("chain", "draw")).values
    b_hat = idata_quad.posterior["beta_country"].mean(("chain", "draw")).values
    g_hat = idata_quad.posterior["gamma_country"].mean(("chain", "draw")).values

    df_raw = (
        pd.read_csv(PATH_MERGED, index_col=0)
          .dropna(subset=["Region","Country Name","Population","GDP"])
    )
    df_raw["country_code"] = df_raw["Country Name"].astype("category").cat.codes
    df_raw["Log_Population_c"] = df_raw["Log_Population"] - df_raw["Log_Population"].mean()
    df_raw["Log_Population_Sq"] = df_raw["Log_Population_c"]**2

    idx = df_raw["country_code"].values
    x, x2 = df_raw["Log_Population_c"].values, df_raw["Log_Population_Sq"].values
    mu_hat = a_hat[idx] + b_hat[idx]*x + g_hat[idx]*x2
    y_obs  = np.log10(df_raw["GDP"].values)

print(f"[1] MAE (log₁₀ units): {np.abs(y_obs - mu_hat).mean():.3f}")

# ----------------------------------------------------------- 2.  ΔELPD block ----
idata_lin = az.from_netcdf(PATH_POST_LINEAR)

def get_ic(idata):
    """Return LOO if available; otherwise WAIC."""
    try:
        ic = az.loo(idata, var_name="Log_GDP_obs");       ic.ic_type = "loo";  return ic
    except Exception:
        ic = az.waic(idata, var_name="Log_GDP_obs");      ic.ic_type = "waic"; return ic

ic_quad, ic_lin = get_ic(idata_quad), get_ic(idata_lin)

def to_elpd(ic):
    if hasattr(ic, "elpd_loo"):  return float(ic.elpd_loo),  float(ic.se)
    if hasattr(ic, "elpd_waic"): return float(ic.elpd_waic), float(ic.se)
    if hasattr(ic, "loo"):       return float(ic.loo),       float(ic.se)
    if hasattr(ic, "waic"):      return -0.5*float(ic.waic), 0.5*float(ic.se)
    raise AttributeError("No ELPD field found")

elpd_q, se_q = to_elpd(ic_quad)
elpd_l, se_l = to_elpd(ic_lin)
d_elpd = elpd_q - elpd_l
se_diff = np.sqrt(se_q**2 + se_l**2)

# numerically stable Akaike weights
log_w   = np.array([elpd_q, elpd_l])
w_quad  = np.exp(log_w[0] - logsumexp(log_w))
w_lin   = 1.0 - w_quad

print("\n[2] Hierarchical model comparison")
print(f"   quadratic ({ic_quad.ic_type.upper()}): ELPD = {elpd_q:8.1f}   SE = {se_q:6.1f}")
print(f"   linear    ({ic_lin.ic_type.upper()}): ELPD = {elpd_l:8.1f}   SE = {se_l:6.1f}")
print(f"   ΔELPD                      : {d_elpd:8.1f}   (SE = {se_diff:6.1f})")
print(f"   Akaike weights →  quadratic: {w_quad:.2f} | linear: {w_lin:.2f}")

# --------------------------------------------- 3. 2025→2040 world growth ----
df_scen = pd.read_csv(PATH_SCENARIO)
yr_a, yr_b = 2025, 2040

# (i) world totals per scenario
totals = (
    df_scen[df_scen["Year"].isin([yr_a, yr_b])]
    .groupby(["Scenario", "Year"])[["Pred_Median", "Pred_Lower", "Pred_Upper"]]
    .sum()
    .unstack("Year")
)

growth_pct = 100 * (
    totals["Pred_Median", yr_b] - totals["Pred_Median", yr_a]
) / totals["Pred_Median", yr_a]

# (ii) within-scenario SE(log10) for each year
se_a = (np.log10(totals["Pred_Upper", yr_a] / totals["Pred_Lower", yr_a])
        / (2 * 1.96)).clip(lower=1e-6)
se_b = (np.log10(totals["Pred_Upper", yr_b] / totals["Pred_Lower", yr_b])
        / (2 * 1.96)).clip(lower=1e-6)

df_ic = pd.DataFrame({
    "growth_pct": growth_pct,
    "se_log": np.sqrt(se_a**2 + se_b**2)
}).replace([np.inf, -np.inf], np.nan).dropna()

# ---------------------------------------------------------------------------
# Fallback if no usable SE: treat variant spread as the only uncertainty
# ---------------------------------------------------------------------------
if df_ic.empty:
    growth_vals = growth_pct.dropna().values        # shape (n_variants,)
    if growth_vals.size < 2:
        raise RuntimeError("Need ≥2 variants to estimate world-growth spread.")
    g_med        = np.median(growth_vals)
    hdi_lo, hdi_hi = np.quantile(growth_vals, [0.025, 0.975])
    heter_pct    = 100.0          # all variance is between scenarios
else:
    # (iii) DerSimonian–Laird random-effects pooling
    log_g   = np.log10(df_ic["growth_pct"].values)
    se      = df_ic["se_log"].values
    w0      = 1 / se**2
    Q       = np.sum(w0*log_g**2) - (np.sum(w0*log_g)**2)/np.sum(w0)
    tau2    = max(0, (Q - (len(w0)-1)) /
                     (np.sum(w0) - np.sum(w0**2)/np.sum(w0)))
    w_re    = 1 / (se**2 + tau2)

    mu_re   = np.sum(w_re*log_g) / np.sum(w_re)
    var_re  = 1 / np.sum(w_re)

    draws   = np.random.normal(mu_re, np.sqrt(var_re + tau2), 40_000)
    g_med   = np.median(10**draws)
    hdi_lo, hdi_hi = np.quantile(10**draws, [0.025, 0.975])
    heter_pct = 100 * tau2 / (tau2 + var_re)

print(f"[3] World GDP growth {yr_a}-{yr_b}: {g_med:.0f}% "
      f"(95 % HDI {hdi_lo:.0f}%–{hdi_hi:.0f}%), "
      f"between-scenario heterogeneity = {heter_pct:.0f}%")

# --------------------------------------------- 5. Scenario heterogeneity ----
df_scen["log_median"] = np.log10(df_scen["Pred_Median"])

def heterogeneity_share(g):
    se = (np.log10(g["Pred_Upper"]) - np.log10(g["Pred_Lower"])) / (2*1.96)
    w  = 1 / se**2
    Q  = np.sum(w*g.log_median**2) - (np.sum(w*g.log_median)**2)/np.sum(w)
    tau2 = max(0, (Q - (len(w)-1)) / (np.sum(w) - np.sum(w**2)/np.sum(w)))
    within = se.mean()**2
    return tau2 / (tau2 + within)

share = (df_scen
         .groupby(["Country","Year"])
         .apply(heterogeneity_share)
         .mean())
print(f"\n[5] Between-scenario heterogeneity (country-year avg): {share*100:.0f}%")

# ---------- 6. Marginal R² (population terms) -------------------------------
a_hat = idata_quad.posterior["alpha_country"].mean(("chain","draw")).values
b_hat = idata_quad.posterior["beta_country" ].mean(("chain","draw")).values
g_hat = idata_quad.posterior["gamma_country"].mean(("chain","draw")).values

# posterior の順序どおりに国→index を作成
posterior_countries = idata_quad.posterior.coords["Country"].values.tolist()
idx_map = pd.Series(range(len(posterior_countries)), index=posterior_countries)

df_raw = pd.read_csv(PATH_MERGED, index_col=0)

# マッピング成功国のみ残す
df_raw = df_raw[df_raw["Country Name"].isin(idx_map.index)].copy()
df_raw["idx"] = df_raw["Country Name"].map(idx_map)

# 中心化は学習時と同じ global mean
x  = df_raw["Log_Population"] - df_raw["Log_Population"].mean()
x2 = x**2

mu_hat = (a_hat[df_raw["idx"].values] +
          b_hat[df_raw["idx"].values] * x +
          g_hat[df_raw["idx"].values] * x2)

y_obs = df_raw["Log_GDP"]

# ---------- 欠損・非有限値を同時に除去 ------------------------------------
mask = np.isfinite(mu_hat) & np.isfinite(y_obs)
mu_hat, y_obs = mu_hat[mask], y_obs[mask]

# ---------- bounded marginal R² --------------------------------------------
r2_marg = 1.0 - np.var(y_obs - mu_hat, ddof=1) / np.var(y_obs, ddof=1)
print(f"[6] Population terms explain ≈ {100*r2_marg:.1f}% of cross-country log-GDP variance.")


