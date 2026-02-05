# import libraries
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
import pytensor.tensor as pt

from config import PATH_MERGED, PATH_MODEL_SIMPLE, DIR_OUTPUT

print(f"Running on PyMC v{pm.__version__}")

# import data
df = pd.read_csv(PATH_MERGED, header = 0, index_col = 0)

# Handle NaN values in Population and GDP
df = df.dropna(subset = ["Region", "Population", "GDP"])

# Center the predictor for better interpretability and sampling efficiency
# After centering, alpha represents E[log(GDP)] at mean log(Population)
log_pop_mean = df["Log_Population"].mean()
log_pop_centered = df["Log_Population"].values - log_pop_mean

# Observed outcome
log_gdp_obs = df["Log_GDP"].values

# Simple Elasticity Model
# Model: log(GDP) = alpha + beta * (log(Population) - mean(log(Population))) + epsilon
# Interpretation:
#   - alpha: Expected log(GDP) for a country with average log(Population)
#   - beta: Elasticity - a 1% increase in Population is associated with beta% change in GDP
#   - sigma: Residual standard deviation on log scale

with pm.Model() as simple_model:
    # Priors
    # alpha: log(GDP) at mean population; weakly informative centered on observed mean
    alpha = pm.Normal("alpha", mu=df["Log_GDP"].mean(), sigma=5)

    # beta: elasticity of GDP w.r.t. Population
    # Economic theory suggests ~1 (constant returns to scale)
    # Weakly informative prior centered at 1
    beta = pm.Normal("beta", mu=1, sigma=1)

    # sigma: residual SD on log scale
    sigma = pm.HalfNormal("sigma", sigma=2)

    # Expected value of log GDP (linear predictor)
    mu = alpha + beta * log_pop_centered

    # Likelihood
    Log_GDP_obs = pm.Normal("Log_GDP_obs", mu=mu, sigma=sigma, observed=log_gdp_obs)

    # Bayesian R-squared (variance explained)
    # R² = Var(mu) / (Var(mu) + sigma²)
    fitted_var = pt.var(mu)
    R2 = pm.Deterministic("R2", fitted_var / (fitted_var + sigma**2))

if __name__ == '__main__':
    with simple_model:
        idata = pm.sample(draws = 10000,
        cores = 12,
        tune = 5000,
        chains = 4,
        n_init = 2500,
        target_accept = 0.99,
        nuts_sampler = "nutpie",
        return_inferencedata = True)

# save the analysis
az.to_netcdf(idata, DIR_OUTPUT / "simple_model.nc")

# load the analysis
idata = az.from_netcdf(DIR_OUTPUT / "simple_model.nc")

# Plot the posterior distributions
az.plot_posterior(idata, var_names=["alpha", "beta", "sigma", "R2"])
# Summarize the results
print(az.summary(idata, var_names=["alpha", "beta", "sigma", "R2"]))

# Plot the trace
with simple_model:
    az.plot_trace(idata)

# Generate posterior predictive samples
# Posterior predictive
with simple_model:
    ppc = pm.sample_posterior_predictive(idata, var_names=["Log_GDP_obs"])

# 5. Plot observed vs. predicted values
plt.figure()
plt.scatter(df["Log_Population"], df["Log_GDP"], alpha=0.5, label="Observed")

# Plot the mean of 'Log_GDP_obs' if it exists in posterior_predictive
if "Log_GDP_obs" in ppc.posterior_predictive:
    # The shape is typically (chain, draw, ...), so average across chain and draw
    fitted_vals = (
        ppc.posterior_predictive["Log_GDP_obs"]
        .mean(dim=["chain", "draw"])
        .values
    )
    plt.scatter(df["Log_Population"], fitted_vals, alpha=0.5, label="Fitted")
else:
    print("Variable 'Log_GDP_obs' not found in posterior predictive samples.")

plt.xlabel("Log Population")
plt.ylabel("Log GDP")
plt.legend()
plt.savefig(DIR_OUTPUT / "simple_model.png", dpi=600)
# plt.show()