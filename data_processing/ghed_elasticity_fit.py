# ghed_elasticity_fit.py
# Fit a region-intercept, log–log elasticity model for government health expenditure per capita.

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path

# ---- paths -------------------------------------------------------------------
PROJ = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
IN_CSV = PROJ / "merged_health.csv"      # created by ingest_health_data.py
OUT_NC = PROJ / "ghed_elasticity.nc"     # forecast code loads from project root

# ---- load and prep -----------------------------------------------------------
df = pd.read_csv(IN_CSV)

# keep rows with GHEDpc and GDP/Population; GHED is 2000+ by construction
df = df.dropna(subset=["GDP", "Population", "ghed_pc_usd", "Region", "Year"])
df = df[(df["Year"] >= 2000) &
        (df["GDP"] > 0) &
        (df["Population"] > 0) &
        (df["ghed_pc_usd"] > 0)].copy()

df["Year"]  = df["Year"].astype(int)
df["GDPpc"] = df["GDP"] / df["Population"]

# logs
df["log_ghed_pc"] = np.log(df["ghed_pc_usd"])
df["log_gdppc"]   = np.log(df["GDPpc"])

# region coords
df["reg_id"] = df["Region"].astype("category").cat.codes
regions = df["Region"].astype("category").cat.categories
coords = {"Region": regions}

# (optional) decadal time drift; uncomment if you wish to include it
# df["t_decade"] = (df["Year"] - 2000) / 10.0

with pm.Model(coords=coords) as fs_model:
    # region-specific intercept
    alpha_r = pm.Normal("alpha_r", mu=0.0, sigma=2.0, dims="Region")
    # income elasticity (literature median around 1, allow variation)
    rho     = pm.Normal("rho", mu=1.0, sigma=0.5)
    # residual scale
    sigma   = pm.HalfStudentT("sigma", nu=3, sigma=0.3)

    mu = alpha_r[df["reg_id"].values] + rho * df["log_gdppc"].values
    # If you include drift:
    # gamma = pm.Normal("gamma", mu=0.0, sigma=0.05)  # per decade, small prior
    # mu = mu + gamma * df["t_decade"].values

    pm.Normal("log_ghed_obs", mu=mu, sigma=sigma, observed=df["log_ghed_pc"].values)

    idata_fs = pm.sample(
        draws=5000, tune=2000, chains=4, target_accept=0.95,
        nuts_sampler="nutpie", random_seed=42,
        return_inferencedata=True
    )

# save where the forecast script expects it
az.to_netcdf(idata_fs, OUT_NC)
print(f"✓ GHED elasticity fit saved → {OUT_NC}")

# tiny summary for sanity
summ = az.summary(idata_fs, var_names=["rho","sigma"], hdi_prob=0.95)
print(summ[["mean","hdi_2.5%","hdi_97.5%"]])
