# hierarchical_model_with_rcs_age.py
# ---------------------------------------------------------------------------
# Hierarchical RCS GDP model + Age structure (ΔWAshare, ΔOldDep) + Country random time slope
# Trains on 1960–2023 using merged_age.csv (historical panel with WAshare & OldDep)
# and age_base_anchors.csv (per-country base-year age anchors).
# Outputs: hierarchical_model_with_rcs_age.nc (InferenceData)
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pathlib import Path

# --------------------------- paths & config -----------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PATH_MERGED_AGE, PATH_MERGED, PATH_KNOTS, DIR_OUTPUT, DIR_DATA
)

PATH_MERGED_HERE = PATH_MERGED_AGE  # from add_age_to_merged.py
PATH_MERGED_RAW = PATH_MERGED  # for global centering mean
PATH_AGE_ANCHORS = DIR_DATA / "age_base_anchors.csv"  # from build_age_covariates_from_owid.py
PATH_KNOTS_HERE = PATH_KNOTS  # reuse your main knots file
PATH_OUT = DIR_OUTPUT / "hierarchical_model_with_rcs_age.nc"

# RCS knots (quantiles on centered x)
KNOT_QUANTS = [0.05, 0.35, 0.65, 0.95]

# --------------------------- helpers ------------------------------------------
def rcs_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Restricted cubic spline (natural cubic spline) with linear tails.
    Returns a matrix with (len(x), K-2) basis columns for K knots.
    """
    k = np.asarray(knots)
    K = k.size
    if K < 3:
        return np.zeros((x.size, 0))
    def d(u, j):
        return np.maximum(u - k[j], 0.0) ** 3
    cols = []
    for j in range(1, K - 1):
        term = (
            d(x, j)
            - d(x, K - 1) * (k[K - 1] - k[j]) / (k[K - 1] - k[0])
            + d(x, 0)     * (k[j]     - k[0]) / (k[K - 1] - k[0])
        )
        cols.append(term)
    return np.column_stack(cols)

# --------------------------- 0) global mean for centering ----------------------
df_for_mu = pd.read_csv(PATH_MERGED_RAW, index_col=0)
if "Log_Population" not in df_for_mu.columns:
    df_for_mu["Log_Population"] = np.log10(df_for_mu["Population"])
MU_GLOBAL_LOGPOP = df_for_mu["Log_Population"].mean()

# --------------------------- 1) load panel + anchors ---------------------------
df = pd.read_csv(PATH_MERGED)
# required columns: ISO3, Country Name, Region, Year, GDP, Population, WAshare, OldDep
df = df.dropna(subset=["ISO3","Country Name","Region","Year","GDP","Population","WAshare","OldDep"]).copy()
df["Year"] = df["Year"].astype(int)

# age base anchors (per ISO3): Year_base, WAshare_base, OldDep_base
age_anchor = pd.read_csv(PATH_AGE_ANCHORS)
age_anchor["Year_base"] = age_anchor["Year_base"].astype(int)

# merge anchors
df = df.merge(age_anchor[["ISO3","WAshare_base","OldDep_base","Year_base"]], on="ISO3", how="left")
df = df.dropna(subset=["WAshare_base","OldDep_base"])

# --------------------------- 2) construct covariates ---------------------------
# core predictors
if "Log_Population" not in df.columns:
    df["Log_Population"] = np.log10(df["Population"])
if "Log_GDP" not in df.columns:
    df["Log_GDP"] = np.log10(df["GDP"])

# centered log-population
x = (df["Log_Population"].values - MU_GLOBAL_LOGPOP)

# RCS basis (compute & save knots if missing)
try:
    knots = np.load(PATH_KNOTS)
except Exception:
    knots = np.quantile(x, KNOT_QUANTS)
    np.save(PATH_KNOTS, knots)
Z = rcs_design(x, knots)   # shape: (N, m)
m = Z.shape[1]

# age deltas (anchor-and-delta design for training too)
df["dWA"] = df["WAshare"].values - df["WAshare_base"].values
df["dOD"] = df["OldDep"].values  - df["OldDep_base"].values
# decade-scaled time (convergence/TFP drift)
df["t_dec"] = (df["Year"].values - 2000) / 10.0

# codes & coords
df["region_code"]  = df["Region"].astype("category").cat.codes
df["country_code"] = df["Country Name"].astype("category").cat.codes
regions  = df["Region"].astype("category").cat.categories
countries= df["Country Name"].astype("category").cat.categories
coords = {"Region": regions, "Country": countries, "Spline": np.arange(m)}

# index arrays
ri = df["region_code"].values.astype(int)
ci = df["country_code"].values.astype(int)

# observed target
y_obs = df["Log_GDP"].values

# --------------------------- 3) model ------------------------------------------
with pm.Model(coords=coords) as mdl:

    # ----- region-level means for α, β, and region-level spline weights θ -----
    alpha_region = pm.Normal("alpha_region", mu=0.0, sigma=2.0, dims="Region")
    beta_region  = pm.Normal("beta_region",  mu=0.0, sigma=0.5, dims="Region")
    # region-level spline weights: independent Normal with small sd (linear tails outside RCS)
    theta_region = pm.Normal("theta_region", mu=0.0, sigma=0.2, dims=("Region","Spline"))

    # ----- country-level random effects around region means -----
    sigma_alpha_country = pm.HalfNormal("sigma_alpha_country", sigma=0.5, dims="Region")
    sigma_beta_country  = pm.HalfNormal("sigma_beta_country",  sigma=0.5, dims="Region")

    alpha_country = pm.Normal(
        "alpha_country",
        mu=alpha_region[ri.max()] if False else alpha_region, # to keep dims clear; we index later
        sigma=sigma_alpha_country,
        dims="Region"   # temporary dims to satisfy shapes
    )
    # reshape alpha_country to Country via indexing of region per country:
    # Instead, more straightforward: define per Country directly with region means:
    # (PyMC dims need consistency; so we define a per-Country variable by indexing region means)
with pm.Model(coords=coords) as mdl:
    # region means
    alpha_region = pm.Normal("alpha_region", 0.0, 2.0, dims="Region")
    beta_region  = pm.Normal("beta_region",  0.0, 0.5, dims="Region")
    theta_region = pm.Normal("theta_region", 0.0, 0.2, dims=("Region","Spline"))

    sigma_alpha_country = pm.HalfNormal("sigma_alpha_country", 0.5, dims="Region")
    sigma_beta_country  = pm.HalfNormal("sigma_beta_country",  0.5, dims="Region")

    # map each Country to its Region index (per-country constant)
    # build arrays of region index per Country:
    country_to_region = (
        df[["country_code","region_code"]]
        .drop_duplicates()
        .sort_values("country_code")
        .set_index("country_code")["region_code"].values.astype(int)
    )

    # α_i, β_i ~ N(α_r, σ_r), N(β_r, σ_r) with region of country i
    alpha_country = pm.Normal(
        "alpha_country",
        mu = alpha_region[country_to_region],
        sigma = sigma_alpha_country[country_to_region],
        dims="Country"
    )
    beta_country = pm.Normal(
        "beta_country",
        mu = beta_region[country_to_region],
        sigma = sigma_beta_country[country_to_region],
        dims="Country"
    )

    # ----- new global fixed effects for age structure (deltas) -----
    delta_washare = pm.Normal("delta_washare", mu=0.0, sigma=0.5)
    delta_olddep  = pm.Normal("delta_olddep",  mu=0.0, sigma=0.5)

    # ----- new country random time slope (TFP/convergence drift) -----
    tau_sd      = pm.HalfNormal("tau_sd", 0.2)
    tau_country = pm.Normal("tau_country", mu=0.0, sigma=tau_sd, dims="Country")

    # ----- residual scale & heavy tails -----
    sigma = pm.HalfStudentT("sigma", nu=3, sigma=0.3)
    nu_raw = pm.Gamma("nu_raw", alpha=2.0, beta=0.2)
    nu     = pm.Deterministic("nu", pm.math.clip(nu_raw + 1.0, 2.0, 30.0))

    # ----- linear predictor -----
    # Build spline contribution θ_r' s(x) for each row
    # Z: (N, m), theta_region[ri,:] : (N, m) via ri indexing
    theta_rows = theta_region[ri, :]                       # (N, m)
    spline = (theta_rows * Z).sum(axis=1)                  # (N,)

    mu = (
        alpha_country[ci]                                  # α_i
        + beta_country[ci] * x                             # β_i x
        + spline                                           # θ_r · s(x)
        + delta_washare * df["dWA"].values                 # ΔWAshare
        + delta_olddep  * df["dOD"].values                 # ΔOldDep
        + tau_country[ci] * df["t_dec"].values            # τ_i * time
    )

    pm.StudentT("Log_GDP_obs", nu=nu, mu=mu, sigma=sigma, observed=y_obs)

    # ----- sample -----
    idata_age = pm.sample(
        draws=5000, tune=2000, chains=4, target_accept=0.99,
        nuts_sampler="nutpie", random_seed=42, return_inferencedata=True,
        idata_kwargs=dict(log_likelihood=True)
    )

# --------------------------- 4) save & brief summary ---------------------------
az.to_netcdf(idata_age, PATH_OUT)
print(f"✓ saved → {PATH_OUT}")

# quick checks on new terms
summ = az.summary(idata_age, var_names=["delta_washare","delta_olddep","tau_sd","sigma","nu"], hdi_prob=0.95)
print(summ[["mean","hdi_2.5%","hdi_97.5%"]])
