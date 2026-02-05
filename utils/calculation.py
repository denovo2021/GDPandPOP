# calculation.py
"""
Utility script to compute key summary statistics for the manuscript:

1. Mean absolute error (MAE) of the hierarchical quadratic model.
2. ΔELPD (with SE) versus a linear hierarchical alternative.
3. World-GDP growth between 2025 and 2040 (median and 95 % HDI).
4. Top-three economies in 2040 (pooled medians).
5. Share of total forecast variance explained by between-scenario heterogeneity.

Required input files
--------------------
hierarchical_model_with_quadratic_ver2.nc    full posterior (quadratic model)
hierarchical_model_linear.nc                 full posterior (linear model w/o γ)
gdp_predictions_meta.csv                     pooled (meta-analytic) forecasts
gdp_predictions_scenarios.csv                per-scenario forecasts (df_scen)

Adjust the paths below as needed.
"""

import numpy as np
import pandas as pd
import arviz as az

# ---------- file paths ------------------------------------------------------
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PATH_MODEL_HIERARCHICAL_QUAD, PATH_MODEL_HIERARCHICAL, PATH_GDP_PREDICTIONS_META,
    PATH_GDP_PREDICTIONS_SCENARIOS, PATH_MERGED
)

PATH_POST_QUAD = str(PATH_MODEL_HIERARCHICAL_QUAD)
PATH_POST_LINEAR = str(PATH_MODEL_HIERARCHICAL)
PATH_META = str(PATH_GDP_PREDICTIONS_META)
PATH_SCENARIO = str(PATH_GDP_PREDICTIONS_SCENARIOS)

# ---------- 1. MAE on the log scale ----------------------------------------
idata_quad = az.from_netcdf(PATH_POST_QUAD)

if "posterior_predictive" in idata_quad.groups():
    # Posterior-predictive draws are present → use their mean as fitted values
    y_hat = (
        idata_quad.posterior_predictive["Log_GDP_obs"]
        .mean(dim=["chain", "draw"])
        .values
        .ravel()
    )
    y_obs = idata_quad.observed_data["Log_GDP_obs"].values.ravel()
else:
    # No posterior-predictive group → approximate fitted values
    # 1) Extract posterior means of country-level coefficients
    alpha_hat = idata_quad.posterior["alpha_country"].mean(dim=["chain", "draw"]).values
    beta_hat  = idata_quad.posterior["beta_country" ].mean(dim=["chain", "draw"]).values
    gamma_hat = idata_quad.posterior["gamma_country"].mean(dim=["chain", "draw"]).values

    # 2) Reload the original data and rebuild centered predictors
    df_raw = (
        pd.read_csv(PATH_MERGED, header=0, index_col=0)
        .dropna(subset=["Region", "Country Name", "Population", "GDP"])
    )

    df_raw["region_code"] = df_raw["Region"].astype("category").cat.codes
    df_raw["country_code"] = df_raw["Country Name"].astype("category").cat.codes

    df_raw["Log_Population_c"] = (
        df_raw["Log_Population"] - df_raw["Log_Population"].mean()
    )
    df_raw["Log_Population_Sq"] = df_raw["Log_Population_c"] ** 2

    # 3) Compute fitted values μ̂ = α + β·x + γ·x²
    idx = df_raw["country_code"].values
    x   = df_raw["Log_Population_c"].values
    x2  = df_raw["Log_Population_Sq"].values

    y_hat = alpha_hat[idx] + beta_hat[idx] * x + gamma_hat[idx] * x2
    y_obs = np.log10(df_raw["GDP"].values)  # observed log-GDP

# Final MAE on the log10 scale
mae_log = float(np.abs(y_obs - y_hat).mean())
print(f"[1] MAE (log₁₀ units): {mae_log:.3f}")

# ---------- 2. ΔELPD vs. linear model -------------------------------------
idata_lin = az.from_netcdf(PATH_POST_LINEAR)

def try_ic(func, idata, label):
    """Attempt to compute an information criterion; return None on failure."""
    try:
        return func(idata)
    except Exception as err:
        print(f"  · {label}: {func.__name__.upper()} failed ({err}).")
        return None

# prefer LOO; fall back to WAIC
ic_quad = try_ic(az.loo,  idata_quad, "quadratic") or \
          try_ic(az.waic, idata_quad, "quadratic")
ic_lin  = try_ic(az.loo,  idata_lin,  "linear")    or \
          try_ic(az.waic, idata_lin,  "linear")

if ic_quad is not None and ic_lin is not None:
    cmp = az.compare({"quad": ic_quad, "linear": ic_lin})
    d_ic  = float(cmp.loc["quad", "elpd_diff"])
    se_ic = float(cmp.loc["quad", "se"])
    metric = "ΔELPD" if ic_quad.ic_type == "loo" else "ΔWAIC"
    print(f"[2] {metric} = {d_ic:.0f}  (SE = {se_ic:.0f})")
else:
    print("[2] Information-criterion comparison skipped: "
          "log_likelihood group missing in one or both files.")

# ---------- 3. Global GDP growth 2025 → 2040 -------------------------------
meta = pd.read_csv(PATH_META)
yr1, yr2 = 2025, 2040
g1 = meta.loc[meta["Year"] == yr1, "Pooled Median"].sum()
g2 = meta.loc[meta["Year"] == yr2, "Pooled Median"].sum()
growth_pct = 100 * (g2 - g1) / g1

hdi1 = meta.loc[meta["Year"] == yr1, ["Pooled Lower", "Pooled Upper"]].sum()
hdi2 = meta.loc[meta["Year"] == yr2, ["Pooled Lower", "Pooled Upper"]].sum()
grow_lo = 100 * (hdi2["Pooled Lower"] - hdi1["Pooled Upper"]) / hdi1["Pooled Upper"]
grow_hi = 100 * (hdi2["Pooled Upper"] - hdi1["Pooled Lower"]) / hdi1["Pooled Lower"]

print(f"[3] World GDP growth {yr1}-{yr2}: {growth_pct:.0f}% "
      f"(95 % HDI {grow_lo:.0f}%–{grow_hi:.0f}%)")

# ---------- 4. Top-three economies in 2040 ---------------------------------
top3 = (
    meta[meta["Year"] == yr2]
    .sort_values("Pooled Median", ascending=False)
    .head(3)
    .loc[:, ["Country Name", "Pooled Median"]]
)
print("\n[4] Top-3 pooled GDP in 2040")
print(top3.to_string(index=False, formatters={"Pooled Median": "{:,.1f} T".format}))

# ---------- 5. Scenario heterogeneity share --------------------------------
df_scen = pd.read_csv(PATH_SCENARIO)
df_scen["log_median"] = np.log10(df_scen["Pred Median"])

def heterogeneity_share(group):
    se = (np.log10(group["Pred Upper"]) -
          np.log10(group["Pred Lower"])) / (2 * 1.96)
    w  = 1 / se**2
    Q  = np.sum(w * group.log_median**2) - (np.sum(w * group.log_median)**2) / np.sum(w)
    k  = len(se)
    tau2 = max(0, (Q - (k - 1)) /
                  (np.sum(w) - np.sum(w**2) / np.sum(w)))
    within = se.mean()**2
    return tau2 / (tau2 + within)

share = df_scen.groupby(["Country Name", "Year"]).apply(heterogeneity_share).mean()
print(f"\n[5] Between-scenario heterogeneity: {share*100:.0f}% of total variance")
