# make_table1_v5.py
import arviz as az
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PATH_MODEL_HIERARCHICAL_AGE

NC_PATH = PATH_MODEL_HIERARCHICAL_AGE

print(f"Loading {NC_PATH} ...")
idata = az.from_netcdf(NC_PATH)

# 抽出したいパラメータ
var_names = [
    "beta0",           # Global Population Elasticity
    "tau0",            # Global Autonomous Drift
    "delta_washare",   # Working-Age Share Effect
    "delta_olddep",    # Old-Age Dependency Effect
    "sigma",           # Residual Scale
    "theta_sd"         # Spline Scale
]

# Summaryの計算
summary = az.summary(idata, var_names=var_names, hdi_prob=0.95, kind="stats")

# フォーマット整形 (Mean, HDI lower, HDI upper)
df_out = pd.DataFrame(index=var_names)
df_out["Parameter"] = [
    "Population Elasticity (beta0)",
    "Global Time Drift (tau0 / decade)",
    "Working-Age Share Effect (delta_WA)",
    "Old-Age Dependency Effect (delta_OD)",
    "Residual Scale (sigma)",
    "Spline Heterogeneity (theta_sd)"
]
df_out["Mean"] = summary["mean"].values
df_out["95% HDI"] = summary.apply(lambda x: f"{x['hdi_2.5%']:.3f} – {x['hdi_97.5%']:.3f}", axis=1)

# Region-specific Time Drift (sigma_tau_region) の要約も追加
tau_r_summ = az.summary(idata, var_names=["sigma_tau_region"], hdi_prob=0.95)
tau_r_mean = tau_r_summ["mean"].mean()
tau_r_range = f"{tau_r_summ['mean'].min():.3f} – {tau_r_summ['mean'].max():.3f}"

print("\n=== Table 1 Draft ===")
print(df_out[["Parameter", "Mean", "95% HDI"]])
print(f"\nRegional Drift Heterogeneity (sigma_tau): Mean {tau_r_mean:.3f} (Range: {tau_r_range})")

# CSV保存
df_out.to_csv(ROOT / "Table1_model_parameters.csv", index=False)
print(f"\nSaved to {ROOT / 'Table1_model_parameters.csv'}")