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
# from fastprogress.fastprogress import force_console_behavior
# master_bar, progress_bar = force_console_behavior()
print(f"Running on PyMC v{pm.__version__}")

# import data
df = pd.read_csv("C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/merged.csv", header = 0, index_col = 0)

# Handle NaN values in Population and GDP
df = df.dropna(subset = ["Region", "Population", "GDP"])
df["Log_Population_c"]  = df["Log_Population"] - df["Log_Population"].mean()

# simple model
with pm.Model() as simple_model:
    # Priors for the intercept and slope
    alpha = pm.Normal("alpha", mu = 0, sigma = 10)
    beta = pm.Normal("beta", mu = 0, sigma = 10)
    sigma = pm.HalfNormal("sigma", sigma = 0.3)

    # Expected value of log GDP
    mu = alpha + beta * df["Log_Population_c"].values

    # Likelihood (sampling distribution) of observations
    Log_GDP_obs = pm.Normal("Log_GDP_obs", mu = mu, sigma = sigma, observed = df["Log_GDP"].values)

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
az.to_netcdf(idata, "C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/simple_model.nc")

# load the analysis
idata = az.from_netcdf("C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/simple_model.nc")

# Plot the posterior distributions
az.plot_posterior(idata, var_names = ["alpha", "beta", "sigma"])
# Summarize the results
az.summary(idata, var_names = ["alpha", "beta", "sigma"])

# Plot the trace
with simple_model:
    az.plot_trace(idata)

# Generate posterior predictive samples
# Posterior predictive
with simple_model:
    ppc = pm.sample_posterior_predictive(idata, var_names=["Log_GDP_obs"])

# Plot observed vs. predicted values
plt.figure()
plt.scatter(df["Log_Population_c"], df["Log_GDP"], alpha=0.5, label="Observed")

# Plot the mean of 'Log_GDP_obs' if it exists in posterior_predictive
if "Log_GDP_obs" in ppc.posterior_predictive:
    # The shape is typically (chain, draw, ...), so average across chain and draw
    fitted_vals = (
        ppc.posterior_predictive["Log_GDP_obs"]
        .mean(dim=["chain", "draw"])
        .values
    )
    plt.scatter(df["Log_Population_c"], fitted_vals, alpha=0.5, label="Fitted")
else:
    print("Variable 'Log_GDP_obs' not found in posterior predictive samples.")

plt.xlabel("Log Population")
plt.ylabel("Log GDP")
plt.legend()
plt.savefig("C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/simple_model.png", dpi=600)
# plt.show()

with simple_model:
    pp_idata = pm.sample_posterior_predictive(
        trace=idata,                          # PyMC ≤5.1 uses keyword 'trace'
        var_names=["Log_GDP_obs"],
        return_inferencedata=True
    )
idata.extend(pp_idata)

# ---------------------------------------------------------------------------
# Bayesian R² (helper works for any ArviZ version)
# ---------------------------------------------------------------------------
def bayes_r2_twoarray(y: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """Return draw-wise Bayesian R² given observed y and predicted y_hat."""
    tss = np.var(y, ddof=0)
    rss = np.var(y_hat - y, axis=1, ddof=0)
    return 1.0 - rss / tss

# --- (a) conditional / overall R² ------------------------------------------
y = idata.observed_data["Log_GDP_obs"].values
y_hat_full = (idata.posterior_predictive["Log_GDP_obs"]
              .stack(sample=("chain", "draw"))
              .transpose("sample", ...)       # (draws, obs)
              .values)
r2_full = bayes_r2_twoarray(y, y_hat_full).mean()

# --- (b) marginal (population effect only) R² ------------------------------
alpha_d = (idata.posterior["alpha"]
           .stack(sample=("chain", "draw"))
           .values)[:, None]                  # (draws, 1)
beta_d  = (idata.posterior["beta"]
           .stack(sample=("chain", "draw"))
           .values)[:, None]                  # (draws, 1)
X = df["Log_Population_c"].values            # (obs,)
y_hat_fix = alpha_d + beta_d * X             # (draws, obs)
r2_marg = bayes_r2_twoarray(y, y_hat_fix).mean()

print(f"Bayesian R² (overall)  : {r2_full:.3f}")
print(f"Marginal R² (population): {r2_marg:.3f}")

# ---------------------------------------------------------------------------
# Save InferenceData
# ---------------------------------------------------------------------------
OUT_NC = r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/simple_model_with_R2.nc"
az.to_netcdf(idata, OUT_NC)
print(f"InferenceData written → {OUT_NC}")