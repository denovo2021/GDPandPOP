# import libraries
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

print(f"Running on PyMC v{pm.__version__}")

# import data
df = pd.read_csv("C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/merged.csv", header=0, index_col=0)

# Handle NaN values in Population and GDP
df = df.dropna(subset=["Region", "Population", "GDP"])

# Create a quadratic term
df["Log_Population_c"] = df["Log_Population"] - df["Log_Population"].mean()
df["Log_Population_Sq"] = df["Log_Population_c"] ** 2

# simple model with a quadratic term
with pm.Model() as simple_model_with_quadratic:
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)      # linear effect
    gamma = pm.Normal("gamma", mu=0, sigma=10)    # quadratic effect
    sigma = pm.HalfNormal("sigma", sigma=1)

    # Linear predictor: alpha + beta*Log_Pop + gamma*(Log_Pop^2)
    mu = (
        alpha
        + beta * df["Log_Population_c"].values
        + gamma * df["Log_Population_Sq"].values
    )

    # Likelihood
    v = pm.Exponential("v", 1/30)
    Log_GDP_obs = pm.StudentT(
        "Log_GDP_obs",
        nu=v,
        mu=mu,
        sigma=sigma,
        observed=df["Log_GDP"].values
    )

if __name__ == '__main__':
    with simple_model_with_quadratic:
        idata_simple_with_quadratic = pm.sample(
            draws=10000,
            cores=12,
            tune=5000,
            chains=4,
            n_init=2500,
            target_accept=0.99,
            nuts_sampler="nutpie",
            return_inferencedata=True
        )

# save the analysis
az.to_netcdf(idata_simple_with_quadratic, "C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/simple_model_with_quadratic.nc")

# load the analysis
idata_simple_with_quadratic = az.from_netcdf("C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/simple_model_with_quadratic.nc")

# Plot the posterior distributions (now including gamma)
az.plot_posterior(idata_simple_with_quadratic, var_names=["alpha", "beta", "gamma", "sigma"])
# Summarize the results
summary_table = az.summary(
    idata_simple_with_quadratic,
    var_names=["alpha", "beta", "gamma", "sigma"],
    hdi_prob=0.95
)
print(summary_table)

# Plot the trace (including gamma)
with simple_model_with_quadratic:
    az.plot_trace(idata_simple_with_quadratic, var_names=["alpha", "beta", "gamma", "sigma"])

# Posterior predictive
with simple_model_with_quadratic:
    ppc = pm.sample_posterior_predictive(idata_simple_with_quadratic, var_names=["Log_GDP_obs"])

# Plot observed vs. predicted values
plt.figure()
plt.scatter(df["Log_Population"], df["Log_GDP"], alpha=0.5, label="Observed")

if "Log_GDP_obs" in ppc.posterior_predictive:
    # Average across chain and draw dimensions
    fitted_vals = (
        ppc.posterior_predictive["Log_GDP_obs"]
        .mean(dim=["chain", "draw"])
        .values
    )
    plt.scatter(
        df["Log_Population"],
        fitted_vals,
        alpha=0.5,
        label="Fitted"
    )
else:
    print("Variable 'Log_GDP_obs' not found in posterior predictive samples.")

plt.xlabel("Log Population")
plt.ylabel("Log GDP")
plt.legend()
plt.savefig("C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/simple_model_with_quadratic.png", dpi=600)
# plt.show()
