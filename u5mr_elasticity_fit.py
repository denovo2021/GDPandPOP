# u5mr_elasticity_fit.py
# Fit a region-intercept, log–log U5MR elasticity model with optional time drift.
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path

# ---- paths -------------------------------------------------------------------
PROJ = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
IN_CSV  = PROJ / "merged_health.csv"          # created by ingest_health_data.py
OUT_NC  = PROJ / "u5mr_elasticity.nc"         # THIS is what the forecast code loads

# ---- load panel ---------------------------------------------------------------
df = pd.read_csv(IN_CSV)

# keep rows with GDP per capita and U5MR
df = df.dropna(subset=["GDP", "Population", "u5mr_per_1000", "Region"])
df = df[(df["GDP"] > 0) & (df["Population"] > 0) & (df["u5mr_per_1000"] > 0)].copy()
df["Year"] = df["Year"].astype(int)
df["GDPpc"] = df["GDP"] / df["Population"]

# log transform
df["log_u5mr"]  = np.log(df["u5mr_per_1000"])
df["log_gdppc"] = np.log(df["GDPpc"])

# region index / coords
df["reg_id"] = df["Region"].astype("category").cat.codes
regions = df["Region"].astype("category").cat.categories
coords = {"Region": regions}

# (optional) decade-scaled time drift around year 2000
df["t_decade"] = (df["Year"] - 2000) / 10.0

with pm.Model(coords=coords) as u5_model:
    # region-specific intercepts
    alpha_r = pm.Normal("alpha_r", mu=0.0, sigma=2.0, dims="Region")
    # income elasticity (negative expected)
    beta    = pm.Normal("beta",  mu=-0.8, sigma=0.5)
    # time drift per decade (negative expected; broaden if needed)
    gamma   = pm.Normal("gamma", mu=-0.05, sigma=0.05)

    sigma   = pm.HalfStudentT("sigma", nu=3, sigma=0.3)
    nu_raw  = pm.Gamma("nu_raw", alpha=2.0, beta=0.2)
    nu      = pm.Deterministic("nu", pm.math.clip(nu_raw + 1.0, 2.0, 30.0))

    mu = (alpha_r[df["reg_id"].values]
          + beta  * df["log_gdppc"].values
          + gamma * df["t_decade"].values)

    pm.StudentT("log_u5mr_obs", nu=nu, mu=mu, sigma=sigma,
                observed=df["log_u5mr"].values)

    idata_u5 = pm.sample(
        draws=5000, tune=2000, chains=4, target_accept=0.95,
        nuts_sampler="nutpie", random_seed=42,
        return_inferencedata=True
    )

# ---- save & quick summary -----------------------------------------------------
az.to_netcdf(idata_u5, OUT_NC)
print(f"✓ U5MR elasticity fit saved → {OUT_NC}")

# optional: a brief console summary for sanity
summ = az.summary(idata_u5, var_names=["beta","gamma","sigma","nu"], hdi_prob=0.95)
print(summ[["mean","hdi_2.5%","hdi_97.5%"]])
