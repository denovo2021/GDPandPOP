# u5mr_hindcast.py  — train (≤2010), test (2011–2023), report MAE & 95% coverage
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import sys
from pathlib import Path
from sklearn.metrics import mean_absolute_error

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PATH_MERGED_HEALTH

# ---------------- paths ----------------
IN_CSV = PATH_MERGED_HEALTH  # created by ingest_health_data.py

# ---------------- load & prep ----------
df = (pd.read_csv(IN_CSV)
        .dropna(subset=["GDP","Population","u5mr_per_1000","Region","Year"]))
df = df[(df["GDP"]>0) & (df["Population"]>0) & (df["u5mr_per_1000"]>0)].copy()
df["Year"]  = df["Year"].astype(int)
df["GDPpc"] = df["GDP"] / df["Population"]
df["log_u5mr"]  = np.log(df["u5mr_per_1000"])
df["log_gdppc"] = np.log(df["GDPpc"])
df["t_decade"]  = (df["Year"] - 2000)/10.0  # match production elasticity

# split
train = df[df["Year"] <= 2010].copy()
test  = df[(df["Year"] > 2010) & (df["Year"] <= 2023)].copy()

# align regions to TRAIN (avoid unseen categories in test)
regions = train["Region"].astype("category").cat.categories
reg_id_map = {r:i for i,r in enumerate(regions)}
train["reg_id"] = train["Region"].map(reg_id_map)
test ["reg_id"] = test ["Region"].map(reg_id_map)
# drop test rows whose region was never seen in train (usually none)
test = test.dropna(subset=["reg_id"]).copy()
train["reg_id"] = train["reg_id"].astype(int)
test ["reg_id"] = test ["reg_id"].astype(int)

coords = {"Region": regions}

with pm.Model(coords=coords) as m:
    # data containers (so we can swap to test later)
    reg_idx = pm.Data("reg_idx", train["reg_id"].values, dims="obs")
    x_gdp   = pm.Data("x_gdp",   train["log_gdppc"].values, dims="obs")
    t_dec   = pm.Data("t_dec",   train["t_decade"].values, dims="obs")
    y_obs   = pm.Data("y_obs",   train["log_u5mr"].values, dims="obs")

    # priors: region intercepts, income elasticity, time drift
    a_r  = pm.Normal("a_r", 0.0, 2.0, dims="Region")
    b    = pm.Normal("b",  -0.8, 0.5)     # income elasticity (<0 expected)
    gma  = pm.Normal("gamma", -0.05, 0.05)# per-decade drift (<0 expected)
    sig  = pm.HalfStudentT("sig", 3, 0.3)
    nuw  = pm.Gamma("nu_raw", 2.0, 0.2)
    nu   = pm.Deterministic("nu", pm.math.clip(nuw + 1.0, 2.0, 30.0))

    mu = a_r[reg_idx] + b * x_gdp + gma * t_dec
    y  = pm.StudentT("y", nu=nu, mu=mu, sigma=sig, observed=y_obs)

    idata = pm.sample(3000, tune=1500, chains=4, target_accept=0.95,
                      nuts_sampler="nutpie", random_seed=42,
                      return_inferencedata=True)

# ---------- test predictions ----------
with m:
    n_test = len(test)
    pm.set_data(
        {
            "reg_idx": test["reg_id"].values.astype(int),
            "x_gdp":   test["log_gdppc"].values.astype(float),
            "t_dec":   test["t_decade"].values.astype(float),
            # keep the observed RV shape 1-D; dummy placeholder is fine for prediction
            "y_obs":   np.empty(n_test, dtype=float),
        },
        coords={"obs": np.arange(n_test)}  # resize the 'obs' dimension to test length
    )

    # use classic posterior_predictive to avoid the 'predictions' group ambiguity
    ppc = pm.sample_posterior_predictive(
        idata, var_names=["y"], random_seed=123
    )

# -------- posterior-predictive summaries on test --------
# Use classic posterior_predictive (we removed predictions=True)
y_da = ppc.posterior_predictive["y"]   # dims: ('chain','draw', <obs-dim>)

# Find the observation dimension name (e.g., 'obs' or 'y_dim_0')
obs_dim = [d for d in y_da.dims if d not in ("chain", "draw")][0]

# Stack samples → (S, n_test) robustly, regardless of the obs-dim name/order
y_samp = (y_da
          .stack(s=("chain", "draw"))     # dims now ('s', <obs-dim>) or (<obs-dim>, 's')
          .transpose("s", obs_dim)        # enforce ('s', <obs-dim>)
          .to_numpy())                    # shape: (S, n_test)

# Median & central 95% intervals from the sample matrix
y_med = np.median(y_samp, axis=0)                     # (n_test,)
y_lo  = np.quantile(y_samp, 0.025, axis=0)            # (n_test,)
y_hi  = np.quantile(y_samp, 0.975, axis=0)            # (n_test,)

# True values
y_true = test["log_u5mr"].to_numpy()

# Metrics
mae_log  = mean_absolute_error(y_true, y_med)
coverage = np.mean((y_true >= y_lo) & (y_true <= y_hi))

print(f"[hindcast] Test MAE (log)        = {mae_log:.3f}")
print(f"[hindcast] Test 95% coverage     = {coverage*100:.1f}%")
print(f"[hindcast] dims = {y_da.dims}, obs_dim='{obs_dim}', shapes: true={y_true.shape}, med={y_med.shape}")


