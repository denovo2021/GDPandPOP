import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns

from config import PATH_MERGED, PATH_MODEL_SIMPLE_QUAD, PATH_MODEL_REGIONAL_QUAD, DIR_FIGURES

az.style.use("arviz-whitegrid")
az.rcParams["stats.ci_prob"] = 0.95
sns.set()

print(f"Running on PyMC v{pm.__version__}")

# 1) Load and preprocess data
df = pd.read_csv(PATH_MERGED, header=0, index_col=0)
df = df.dropna(subset=["Region", "Population", "GDP"])  # drop rows missing essential data

# Create 'region_code' category
df["region_code"] = df["Region"].astype("category").cat.codes
regions = df["Region"].astype("category").cat.categories
n_regions = len(regions)
coords = {"Region": regions}

# Create quadratic term (centering)
df["Log_Population_c"]  = df["Log_Population"] - df["Log_Population"].mean()
df["Log_Population_Sq"] = df["Log_Population_c"] ** 2

# ---------------------------------------------------------------------------
# Load simple_model_with_quadratic & extract alpha/beta/gamma means/SDs
# ---------------------------------------------------------------------------
idata_simple_quad = az.from_netcdf(PATH_MODEL_SIMPLE_QUAD)

summary_simple_quad = az.summary(
    idata_simple_quad,
    var_names=["alpha", "beta", "gamma", "sigma"],
    hdi_prob=0.95
)
print(summary_simple_quad)

alpha_mean = summary_simple_quad.loc["alpha", "mean"]
alpha_sd   = summary_simple_quad.loc["alpha", "sd"]

beta_mean  = summary_simple_quad.loc["beta", "mean"]
beta_sd    = summary_simple_quad.loc["beta", "sd"]

gamma_mean = summary_simple_quad.loc["gamma", "mean"]
gamma_sd   = summary_simple_quad.loc["gamma", "sd"]

# Define the PyMC model
regional_model_with_quadratic = pm.Model(coords=coords)

with regional_model_with_quadratic:
    alpha_region = pm.Normal(
        "alpha_region",
        mu=alpha_mean,
        sigma=alpha_sd,
        dims=["Region"]
    )
    beta_region = pm.Normal(
        "beta_region",
        mu=beta_mean,
        sigma=beta_sd,
        dims=["Region"]
    )
    gamma_region = pm.Normal(
        "gamma_region",
        mu=gamma_mean,
        sigma=gamma_sd,
        dims=["Region"]
    )

    sigma = pm.HalfNormal("sigma", sigma=1.0)

    mu = (
        alpha_region[df["region_code"].values]
        + beta_region[df["region_code"].values] * df["Log_Population_c"].values
        + gamma_region[df["region_code"].values] * df["Log_Population_Sq"].values
    )

    # heavy‑tailed likelihood
    nu = pm.Exponential("nu", 1 / 30)
    Log_GDP_obs = pm.StudentT(
        "Log_GDP_obs",
        nu=nu,
        mu=mu,
        sigma=sigma,
        observed=df["Log_GDP"].values
    )

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
idata_regional_quad = pm.sample(
    draws=10000,
    tune=5000,
    chains=4,
    cores=12,
    target_accept=0.99,
    n_init=2500,
    nuts_sampler="nutpie",
    model=regional_model_with_quadratic,
    return_inferencedata=True
)

# ---------------------------------------------------------------------------
# Save & Load InferenceData
# ---------------------------------------------------------------------------
az.to_netcdf(
    idata_regional_quad,
    "PATH_MODEL_REGIONAL_QUAD"
)

idata_regional_quad = az.from_netcdf(
    "PATH_MODEL_REGIONAL_QUAD"
)

# ---------------------------------------------------------------------------
# Posterior Plots & Summary
# ---------------------------------------------------------------------------
az.plot_posterior(
    idata_regional_quad,
    var_names=["alpha_region", "beta_region", "gamma_region", "sigma"]
)
plt.show()

summary_table = az.summary(
    idata_regional_quad,
    var_names=["alpha_region", "beta_region", "gamma_region", "sigma"],
    hdi_prob=0.95
)
print(summary_table)

az.plot_trace(
    idata_regional_quad,
    var_names=["alpha_region", "beta_region", "gamma_region", "sigma"]
)
plt.show()

# ---------------------------------------------------------------------------
# Posterior Predictive
# ---------------------------------------------------------------------------
with regional_model_with_quadratic:
    ppc = pm.sample_posterior_predictive(
        idata_regional_quad, 
        var_names=["Log_GDP_obs"]
    )

print(ppc.keys())

# ---------------------------------------------------------------------------
# Plot Observed vs. Fitted by Region (Quadratic Lines)
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
palette = sns.color_palette("tab10", n_colors=n_regions)

# Observed data by region
for i, reg in enumerate(regions):
    mask = df["Region"] == reg
    plt.scatter(
        df.loc[mask, "Log_Population"],
        df.loc[mask, "Log_GDP"],
        alpha=0.5,
        label=f"Observed: {reg}",
        color=palette[i]
    )

# Posterior predictive mean by region
if "Log_GDP_obs" in ppc:
    pred_means = ppc["Log_GDP_obs"].mean(axis=0)
    for i, reg in enumerate(regions):
        mask = df["Region"] == reg
        plt.scatter(
            df.loc[mask, "Log_Population"],
            pred_means[mask],
            color=palette[i],
            alpha=0.5,
            marker='x',
            label=f"Fitted: {reg}"
        )
else:
    print("Variable 'Log_GDP_obs' not found in posterior predictive samples.")

# Add region-level quadratic regression lines using posterior means
x_bar = df["Log_Population"].mean()  
x_line = np.linspace(df["Log_Population"].min(), df["Log_Population"].max(), 100)
x_line_c = x_line - x_bar
for i, reg in enumerate(regions):
    alpha_i = idata_regional_quad.posterior["alpha_region"].sel(Region=reg).mean().values
    beta_i  = idata_regional_quad.posterior["beta_region"].sel(Region=reg).mean().values
    gamma_i = idata_regional_quad.posterior["gamma_region"].sel(Region=reg).mean().values

    y_line = alpha_i + beta_i*x_line_c + gamma_i*(x_line_c**2)
    plt.plot(
        x_line, 
        y_line, 
        color=palette[i],
        linestyle='--',
        label=f"Quadratic line: {reg}"
    )

plt.xlabel("Log Population")
plt.ylabel("Log GDP")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(
    DIR_FIGURES / "regional_model_with_quadratic.png", 
    dpi=600
)
# plt.show()