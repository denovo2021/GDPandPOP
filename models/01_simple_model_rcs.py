# 01_simple_model_rcs.py
# Layer 1: Pooled model with Restricted Cubic Splines (no hierarchy)
# -------------------------------------------------------------------
# Model: log(GDP) = alpha + beta * x_c + sum_j(theta_j * rcs_j(x_c)) + epsilon
# where x_c = log(Population) - mean(log(Population))
#
# This is the baseline model treating all countries as exchangeable
# (single global intercept and slope, no grouping structure)
# -------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from matplotlib import pylab as plt
import seaborn as sns
sns.set()
import arviz as az
az.style.use("arviz-whitegrid")
az.rcParams["stats.ci_prob"] = 0.95
import pymc as pm
import pytensor.tensor as pt
from arviz.stats import hdi

from config import PATH_MERGED, PATH_MODEL_SIMPLE, PATH_FIG_SIMPLE_MODEL

print(f"Running on PyMC v{pm.__version__}")

# ----------------------------- data ---------------------------------
df = pd.read_csv(
    PATH_MERGED,
    header=0, index_col=0
).dropna(subset=["Region", "Population", "GDP"])

# log-scale variables
df["Log_Population"]   = np.log10(df["Population"])
df["Log_GDP"]          = np.log10(df["GDP"])
df["Log_Population_c"] = df["Log_Population"] - df["Log_Population"].mean()

# --------------------- restricted cubic spline ----------------------
# natural cubic spline basis: linear tails outside knot range
x = df["Log_Population_c"].values
knots = np.quantile(x, [0.05, 0.35, 0.65, 0.95])  # 4 knots -> (K-2)=2 spline bases

def rcs_design(x_in, k):
    """Harrell's restricted cubic spline with linear tails."""
    k = np.asarray(k)
    K = k.shape[0]
    def d(u, j):  # truncated cubic
        return np.maximum(u - k[j], 0.0) ** 3
    cols = []
    for j in range(1, K-1):
        term = (
            d(x_in, j)
            - d(x_in, K-1) * (k[K-1] - k[j]) / (k[K-1] - k[0])
            + d(x_in, 0)   * (k[j]   - k[0]) / (k[K-1] - k[0])
        )
        cols.append(term)
    return np.column_stack(cols)  # shape: (N, K-2)

Z = rcs_design(x, knots)
m = Z.shape[1]
for j in range(m):
    df[f"rcs{j+1}"] = Z[:, j]

# ------------------------------ model --------------------------------
# Priors rationale:
#   - alpha: Expected log(GDP) at mean log(Population); centered on data mean
#   - beta: Linear elasticity component; weakly informative around 1
#   - theta: RCS coefficients; shrinkage prior to prevent overfitting
#   - sigma: Residual SD; weakly informative
#   - nu: Degrees of freedom for StudentT; allows heavy tails if needed

with pm.Model() as simple_model_with_rcs:
    # Intercept: log(GDP) at mean log(Population)
    alpha = pm.Normal("alpha", mu=df["Log_GDP"].mean(), sigma=2.0)

    # Linear slope (elasticity baseline)
    beta = pm.Normal("beta", mu=1.0, sigma=0.5)

    # RCS coefficients with shrinkage prior
    theta = pm.Normal("theta", mu=0.0, sigma=0.10, shape=m)

    # Residual scale
    sigma = pm.HalfStudentT("sigma", nu=3, sigma=0.3)

    # Degrees of freedom for StudentT likelihood (robust to outliers)
    nu_raw = pm.Gamma("nu_raw", alpha=2.0, beta=0.2)  # mean ~ 10
    nu = pm.Deterministic("nu", pm.math.clip(nu_raw + 1, 2.0, 30.0))

    # Linear predictor: alpha + beta*x + sum_j(theta_j * rcs_j(x))
    mu = (
        alpha
        + beta * df["Log_Population_c"].values
        + pm.math.dot(df[[f"rcs{j+1}" for j in range(m)]].values, theta)
    )

    # Likelihood
    Log_GDP_obs = pm.StudentT("Log_GDP_obs", nu=nu, mu=mu, sigma=sigma,
                              observed=df["Log_GDP"].values)

    # Bayesian R-squared
    fitted_var = pt.var(mu)
    R2 = pm.Deterministic("R2", fitted_var / (fitted_var + sigma**2))

# ------------------------------ sampling -----------------------------
# Optimized settings: draws=4000, tune=2000, target_accept=0.95

if __name__ == '__main__':
    with simple_model_with_rcs:
        idata_simple_with_rcs = pm.sample(
            draws=4_000, tune=2_000, chains=4, cores=12,
            target_accept=0.95, nuts_sampler="nutpie",
            return_inferencedata=True
        )

# ------------------------------ save/load ----------------------------
    az.to_netcdf(
        idata_simple_with_rcs,
        PATH_MODEL_SIMPLE
    )

# Load saved results (for analysis without re-running sampling)
idata_simple_with_rcs = az.from_netcdf(PATH_MODEL_SIMPLE)

# ------------------------------ posterior plots ----------------------
az.plot_posterior(idata_simple_with_rcs, var_names=["alpha", "beta", "sigma", "nu", "theta", "R2"])
summary_table = az.summary(
    idata_simple_with_rcs,
    var_names=["alpha", "beta", "sigma", "nu", "theta", "R2"],
    hdi_prob=0.95
)
print(summary_table)

# ---------------- global association curve with uncertainty ----------
post    = idata_simple_with_rcs.posterior

# draws as the first axis
alpha_s = post["alpha"].stack(sample=("chain", "draw")).values            # (D,)
beta_s  = post["beta" ].stack(sample=("chain", "draw")).values            # (D,)

# ensure theta has shape (D, m), where m = number of RCS basis columns
theta_da = post["theta"]
theta_dim = [d for d in theta_da.dims if d not in ("chain", "draw")][0]   # e.g., "theta_dim"
theta_s = (theta_da
           .stack(sample=("chain", "draw"))
           .transpose("sample", theta_dim)                                 # (D, m)
           .values)

# prediction grid and its RCS basis
x_grid = np.linspace(df["Log_Population_c"].min(),
                     df["Log_Population_c"].max(), 200)
Zg = rcs_design(x_grid, knots)                                             # (G=200, m)

# sanity check (optional)
assert theta_s.shape[1] == Zg.shape[1], (theta_s.shape, Zg.shape)

# predicted log-GDP: (D,1) + (D,G) + (D,m)@(m,G) = (D,G)
y_pred = (alpha_s[:, None]
          + beta_s[:, None] * x_grid[None, :]
          + theta_s @ Zg.T)

# summarize and plot
y_med = np.median(y_pred, axis=0)
y_hdi = hdi(y_pred, hdi_prob=0.95)                                         # (G, 2)

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df["Log_Population_c"], df["Log_GDP"], alpha=0.05, s=10, label="Observed")
ax.plot(x_grid, y_med, lw=2, label="Posterior median")
ax.fill_between(x_grid, y_hdi[:, 0], y_hdi[:, 1], alpha=0.25, label="95 % HDI")
ax.set_xlabel("Centered log10 population")
ax.set_ylabel("log10 GDP")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PATH_FIG_SIMPLE_MODEL, dpi=600)
