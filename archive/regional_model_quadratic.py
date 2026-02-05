# regional_model.py

# Import libraries
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pylab as plt
import matplotlib.lines
import matplotlib.collections
import seaborn as sns
sns.set()
import arviz as az
az.style.use("arviz-whitegrid")
az.rcParams["stats.ci_prob"] = 0.95
import pymc as pm

from config import PATH_MERGED, DIR_OUTPUT, DIR_FIGURES

print(f"Running on PyMC v{pm.__version__}")

# Import data
df = pd.read_csv(PATH_MERGED, header = 0, index_col = 0)

# Handle NaN values
df = df.dropna(subset=["Region", "Population", "GDP"])

# Assign region codes
df["region_code"] = df["Region"].astype("category").cat.codes
regions = df["Region"].astype("category").cat.categories
n_regions = len(regions)

coords = {"Region": regions}

# Load simple_model results
idata_simple = az.from_netcdf(DIR_OUTPUT / "simple_model.nc")

# Summarize to extract posterior means/HDIs
summary_simple = az.summary(
    idata_simple, 
    var_names=["alpha", "beta", "sigma"], 
    hdi_prob=0.95
)

alpha_mean = summary_simple.loc["alpha", "mean"]
alpha_sd = summary_simple.loc["alpha", "sd"]

beta_mean = summary_simple.loc["beta", "mean"]
beta_sd = summary_simple.loc["beta", "sd"]

with pm.Model(coords=coords) as regional_model:
    # Hyperpriors for alpha
    alpha_mu = pm.Normal("alpha_mu", mu=alpha_mean, sigma=alpha_sd)
    alpha_sigma = pm.HalfNormal("alpha_sigma", sigma=1.0)
    alpha_region = pm.Normal("alpha_region", 
                             mu=alpha_mu, 
                             sigma=alpha_sigma, 
                             dims=["Region"])
    
    # Hyperpriors for beta
    beta_mu = pm.Normal("beta_mu", mu=beta_mean, sigma=beta_sd)
    beta_sigma = pm.HalfNormal("beta_sigma", sigma=1.0)
    beta_region = pm.Normal("beta_region",
                            mu=beta_mu,
                            sigma=beta_sigma,
                            dims=["Region"])
    
    # Prior on residual scale (you can also inform this from sigma_mean, sigma_sd if desired)
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Expected value
    mu = alpha_region[df["region_code"].values] + beta_region[df["region_code"].values] * df["Log_Population"].values
    
    # Likelihood
    Log_GDP_obs = pm.Normal("Log_GDP_obs", mu=mu, sigma=sigma, observed=df["Log_GDP"].values)

# Then sample as usual
if __name__ == "__main__":
    with regional_model:
        idata_regional = pm.sample(
            draws=10000,
            tune=5000,
            chains=4,
            cores=12,
            target_accept=0.99,
            n_init=2500,
            nuts_sampler = "nutpie",
            return_inferencedata=True
        )
   
# Save
az.to_netcdf(idata_regional, DIR_OUTPUT / "regional_model.nc")

# Load
idata_regional = az.from_netcdf(DIR_OUTPUT / "regional_model.nc")

# Posterior
az.plot_posterior(idata_regional, var_names=["alpha_region", "beta_region", "sigma"])
az.summary(idata_regional, var_names=["alpha_region", "beta_region", "sigma"])

# Trace
az.plot_trace(idata_regional, var_names=["alpha_region", "beta_region", "sigma"])

# Posterior predictive
with regional_model:
    ppc = pm.sample_posterior_predictive(idata_regional, var_names=["Log_GDP_obs"])

print(ppc.keys())

# Plot observed vs fitted values, colored by region, and add regression lines
plt.figure(figsize=(10, 6))
palette = sns.color_palette("tab10", n_colors=n_regions)

# Plot observed data by region
for i, reg in enumerate(regions):
    mask = df["Region"] == reg
    plt.scatter(df.loc[mask, "Log_Population"], df.loc[mask, "Log_GDP"],
                alpha=0.5, label=f"Observed: {reg}", color=palette[i])

# Plot fitted values by region
if "Log_GDP_obs" in ppc:
    pred_means = ppc["Log_GDP_obs"].mean(axis=0)
    for i, reg in enumerate(regions):
        mask = df["Region"] == reg
        plt.scatter(df.loc[mask, "Log_Population"], pred_means[mask],
                    color=palette[i], alpha=0.5, marker='x', label=f"Fitted: {reg}")
else:
    print("Variable 'Log_GDP_obs' not found in posterior predictive samples.")

# Add regression lines by region using posterior means
x_line = np.linspace(df["Log_Population"].min(), df["Log_Population"].max(), 100)
for i, reg in enumerate(regions):
    alpha_i = idata_regional.posterior["alpha_region"].sel(Region=reg).mean().values
    beta_i = idata_regional.posterior["beta_region"].sel(Region=reg).mean().values
    y_line = alpha_i + beta_i * x_line
    plt.plot(x_line, y_line, color=palette[i], linestyle='--', label=f"Regression line: {reg}")

plt.xlabel("Log Population")
plt.ylabel("Log GDP")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(DIR_FIGURES / "regional_model.png", dpi=600)
# plt.show()