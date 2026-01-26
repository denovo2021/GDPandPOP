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
    alpha = pm.Normal("alpha", mu=df["Log_GDP"].mean(), sigma=2.0)
    beta = pm.Normal("beta", mu=0, sigma=0.5)      # linear effect
    gamma = pm.Normal("gamma", mu=0, sigma=0.1)    # quadratic effect
    sigma = pm.HalfStudentT("sigma", nu=3, sigma=0.3)

    # Linear predictor: alpha + beta*Log_Pop + gamma*(Log_Pop^2)
    mu = (
        alpha
        + beta * df["Log_Population_c"].values
        + gamma * df["Log_Population_Sq"].values
    )

    # Likelihood
    nu_raw = pm.Gamma("nu_raw", alpha=2.0, beta=0.2)     # mean ≈ 10
    nu = pm.Deterministic("nu", pm.math.clip(nu_raw + 1, 2.0, 30.0))
    
    Log_GDP_obs = pm.StudentT(
        "Log_GDP_obs",
        nu=nu,
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
plt.scatter(df["Log_Population_c"], df["Log_GDP"], alpha=0.5, label="Observed")

if "Log_GDP_obs" in ppc.posterior_predictive:
    # Average across chain and draw dimensions
    fitted_vals = (
        ppc.posterior_predictive["Log_GDP_obs"]
        .mean(dim=["chain", "draw"])
        .values
    )
    plt.scatter(
        df["Log_Population_c"],
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

# ------------------------------------------------------------------
# Global association curve with posterior uncertainty
# ------------------------------------------------------------------
from arviz.stats import hdi

# 1. Extract posterior draws and flatten chain × draw
post = idata_simple_with_quadratic.posterior
alpha_s = post["alpha"].stack(s=("chain", "draw")).values
beta_s  = post["beta"].stack(s=("chain", "draw")).values
gamma_s = post["gamma"].stack(s=("chain", "draw")).values

# 2. Grid of centered log-population
x_grid = np.linspace(df["Log_Population_c"].min(),
                     df["Log_Population_c"].max(),
                     200)

# 3. Predicted log-GDP for every draw and grid point
y_pred = (
    alpha_s[:, None]
    + beta_s[:, None]  * x_grid[None, :]
    + gamma_s[:, None] * x_grid[None, :]**2
)

# 4. Posterior median and 95 % HDI along the grid
y_med = np.median(y_pred, axis=0)
y_hdi = hdi(y_pred, hdi_prob=0.95)   # shape (200, 2)

# 5. Plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["Log_Population_c"], df["Log_GDP"],
           alpha=0.05, s=10, label="Observed")
ax.plot(x_grid, y_med, color="C1", lw=2, label="Posterior median")
ax.fill_between(x_grid, y_hdi[:, 0], y_hdi[:, 1],
                color="C1", alpha=0.25, label="95 % HDI")
ax.set_xlabel("Centered log10 population")
ax.set_ylabel("log10 GDP")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(
    "C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/fig_global_association.png",
    dpi=600
)
