# Layer 2: Partial Pooling by Region with Restricted Cubic Splines
# ---------------------------------------------------------------------
# Model: log(GDP) = alpha_r + beta_r * x_c + sum_j(theta_rj * rcs_j(x_c)) + epsilon
# where r = region index, x_c = centered log(Population)
#
# Hierarchical structure (partial pooling):
#   alpha_r ~ Normal(mu_alpha, sigma_alpha)   <- regions share common distribution
#   beta_r  ~ Normal(mu_beta, sigma_beta)     <- allows shrinkage toward global mean
#   theta_rj ~ Normal(0, sigma_theta)         <- shrinkage prior for splines
#
# This allows regions with less data to borrow strength from other regions.
# ---------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
import pytensor.tensor as pt

from config import PATH_MERGED, PATH_MODEL_SIMPLE, PATH_MODEL_REGIONAL, DIR_CACHE, DIR_FIGURES

az.style.use("arviz-whitegrid"); az.rcParams["stats.ci_prob"] = 0.95
sns.set()
print(f"Running on PyMC v{pm.__version__}")

# ------------------------------- 1) Data --------------------------------
df = (pd.read_csv(PATH_MERGED, header=0, index_col=0)
        .dropna(subset=["Region", "Population", "GDP"]))

# log10 variables (use existing columns if already present)
if "Log_Population" not in df.columns:
    df["Log_Population"] = np.log10(df["Population"])
if "Log_GDP" not in df.columns:
    df["Log_GDP"] = np.log10(df["GDP"])

# centered predictor
df["Log_Population_c"] = df["Log_Population"] - df["Log_Population"].mean()
x = df["Log_Population_c"].values

# region codes / coords
df["region_code"] = df["Region"].astype("category").cat.codes
regions = df["Region"].astype("category").cat.categories
coords  = {"Region": regions}

# -------------------- 2) Restricted Cubic Spline basis -----------------
# Natural cubic spline: linear tails outside [k0, k_last]
def rcs_design(x_in: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted cubic spline basis with linear tails.
       Returns matrix of shape (N, K-2)."""
    k = np.asarray(knots); K = k.size
    def d(u, j):  # truncated cubic
        return np.maximum(u - k[j], 0.0) ** 3
    cols = []
    for j in range(1, K-1):
        term = (d(x_in, j)
                - d(x_in, K-1) * (k[K-1] - k[j]) / (k[K-1] - k[0])
                + d(x_in, 0)   * (k[j]   - k[0]) / (k[K-1] - k[0]))
        cols.append(term)
    return np.column_stack(cols) if cols else np.zeros((x_in.size, 0))

# choose 4 knots (5th, 35th, 65th, 95th percentiles) -> (K-2)=2 spline cols
knots = np.quantile(x, [0.05, 0.35, 0.65, 0.95])
Z = rcs_design(x, knots)             # shape (N, m)
m = Z.shape[1]
# Save knots for reuse in forecasting
np.save(DIR_CACHE / "rcs_knots_region.npy", knots)

# ------------------ 3) Prior info from simple model (optional) ----------
# If you trained the simple (global) model, use its alpha, beta as weak centers.
try:
    idata_simple = az.from_netcdf(PATH_MODEL_SIMPLE)
    summ_simple  = az.summary(idata_simple, var_names=["alpha", "beta"], hdi_prob=0.95)
    alpha_mean, alpha_sd = float(summ_simple.loc["alpha", "mean"]), float(summ_simple.loc["alpha", "sd"])
    beta_mean,  beta_sd  = float(summ_simple.loc["beta",  "mean"]), float(summ_simple.loc["beta",  "sd"])
except Exception:
    # Fallback centers if file not present
    alpha_mean, alpha_sd = float(df["Log_GDP"].mean()), 2.0
    beta_mean,  beta_sd  = 0.0, 0.5

# ------------------------------- 4) Model --------------------------------
# Partial pooling: region parameters drawn from common hyperprior distributions
# This creates shrinkage toward the global mean for regions with less data

with pm.Model(coords=coords) as regional_model_with_rcs:
    # ===== Hyperpriors (population-level parameters) =====
    # These define the distribution from which region-level parameters are drawn

    # Intercept hyperpriors
    mu_alpha = pm.Normal("mu_alpha", mu=alpha_mean, sigma=2.0)
    sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1.0)

    # Slope (elasticity) hyperpriors
    mu_beta = pm.Normal("mu_beta", mu=beta_mean, sigma=0.5)
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.3)

    # ===== Region-level parameters (partial pooling) =====
    # Each region's parameters are drawn from the hyperprior distribution
    alpha_region = pm.Normal("alpha_region", mu=mu_alpha, sigma=sigma_alpha, dims="Region")
    beta_region = pm.Normal("beta_region", mu=mu_beta, sigma=sigma_beta, dims="Region")

    # Region-level spline coefficients with shrinkage
    if m > 0:
        # Fixed: Removed mutable=True as it caused TypeError and is not needed for static dimension
        regional_model_with_rcs.add_coord("Spline", np.arange(m))
        
        # Hierarchical shrinkage for spline coefficients
        sigma_theta = pm.HalfNormal("sigma_theta", sigma=0.1)
        theta_region = pm.Normal("theta_region", mu=0.0, sigma=sigma_theta,
                                 dims=("Region", "Spline"))
        # Gather per-row theta by region, then row-wise dot with Z
        theta_row = theta_region[df["region_code"].values]  # (N, m)
        spline_term = pm.math.sum(theta_row * Z, axis=1)    # (N,)
    else:
        spline_term = 0.0

    # ===== Observation model =====
    # Student-t likelihood for robustness to outliers
    sigma = pm.HalfStudentT("sigma", nu=3, sigma=0.3)
    nu_raw = pm.Gamma("nu_raw", alpha=2.0, beta=0.2)  # mean ~ 10
    nu = pm.Deterministic("nu", pt.clip(nu_raw + 1.0, 2.0, 30.0))

    # Linear predictor
    mu = (alpha_region[df["region_code"].values]
          + beta_region[df["region_code"].values] * x
          + spline_term)

    # Likelihood
    Log_GDP_obs = pm.StudentT("Log_GDP_obs", nu=nu, mu=mu, sigma=sigma,
                              observed=df["Log_GDP"].values)

    # Bayesian R-squared
    fitted_var = pt.var(mu)
    R2 = pm.Deterministic("R2", fitted_var / (fitted_var + sigma**2))

# ------------------------------ sampling -----------------------------
# Optimized settings for faster sampling:
#   - draws: 4,000 is sufficient for stable posterior estimates
#   - tune: 2,000 is usually enough for adaptation
#   - target_accept: 0.95 balances accuracy and speed (0.99 is overly conservative)
#   - Total effective samples: 4 chains x 4,000 = 16,000 (plenty for inference)

if __name__ == '__main__':
    with regional_model_with_rcs:
        idata_regional_rcs = pm.sample(
            draws=4_000, tune=2_000, chains=4, cores=12,
            target_accept=0.95, nuts_sampler="numpyro",
            return_inferencedata=True
        )

# ----------------------------- 5) Save / Load ---------------------------
    az.to_netcdf(idata_regional_rcs, PATH_MODEL_REGIONAL)

# Load saved results (for analysis without re-running sampling)
idata_regional_rcs = az.from_netcdf(PATH_MODEL_REGIONAL)

# ----------------------------- 6) Summary / Plots -----------------------
# Include hyperparameters in summary
var_list = ["mu_alpha", "sigma_alpha", "mu_beta", "sigma_beta",
            "alpha_region", "beta_region", "sigma", "nu", "R2"]
if m > 0:
    var_list.extend(["sigma_theta", "theta_region"])

az.plot_posterior(idata_regional_rcs, var_names=var_list)
plt.savefig(DIR_FIGURES / "regional_model_rcs_posterior.png", dpi=300)

print(az.summary(idata_regional_rcs, var_names=var_list, hdi_prob=0.95))

# Posterior predictive (kept separate; use as needed)
with regional_model_with_rcs:
    ppc = pm.sample_posterior_predictive(
        trace=idata_regional_rcs,
        var_names=["Log_GDP_obs"],
        return_inferencedata=True
    )

# Observed vs fitted by region (posterior-predictive mean)
plt.figure(figsize=(10, 6))
palette = sns.color_palette("tab10", n_colors=len(regions))

fitted = ppc.posterior_predictive["Log_GDP_obs"].mean(dim=["chain", "draw"]).values
for i, reg in enumerate(regions):
    mask = (df["Region"] == reg)
    plt.scatter(df.loc[mask, "Log_Population_c"], df.loc[mask, "Log_GDP"],
                alpha=0.5, color=palette[i], label=f"Observed: {reg}")
    plt.scatter(df.loc[mask, "Log_Population_c"], fitted[mask],
                alpha=0.5, color=palette[i], marker="x", label=f"Fitted: {reg}")

# Region-level RCS lines using posterior means
x_line = np.linspace(df["Log_Population_c"].min(), df["Log_Population_c"].max(), 100)
Z_line = rcs_design(x_line, knots)  # (100, m)
for i, reg in enumerate(regions):
    a = idata_regional_rcs.posterior["alpha_region"].sel(Region=reg).mean(dim=("chain","draw")).values
    b = idata_regional_rcs.posterior["beta_region" ].sel(Region=reg).mean(dim=("chain","draw")).values
    if m > 0:
        th = idata_regional_rcs.posterior["theta_region"].sel(Region=reg).mean(dim=("chain","draw")).values  # (m,)
        y_line = a + b * x_line + (Z_line @ th)
    else:
        y_line = a + b * x_line
    plt.plot(x_line, y_line, color=palette[i], linestyle="--", label=f"RCS: {reg}")

plt.xlabel("Centered log10 population").
plt.ylabel("log10 GDP")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig(DIR_FIGURES / "regional_model_rcs.png", dpi=600)
# plt.show()