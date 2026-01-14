# hierarchical_model_with_rcs.py
# Layer 3: Full Hierarchical Model with Restricted Cubic Splines
# ---------------------------------------------------------------------------
# Model: log(GDP) = alpha_c + beta_c * x_c + sum_j(theta_rj * rcs_j(x_c)) + epsilon
# where c = country index, r = region index, x_c = centered log(Population)
#
# 3-Level Hierarchical Structure:
#   Level 1 (Global hyperpriors):
#       mu_alpha ~ Normal(...)           <- global mean intercept
#       sigma_alpha_region ~ HalfNormal  <- between-region variance
#       mu_beta ~ Normal(...)            <- global mean elasticity
#       sigma_beta_region ~ HalfNormal   <- between-region variance
#
#   Level 2 (Region):
#       alpha_region ~ Normal(mu_alpha, sigma_alpha_region)
#       beta_region ~ Normal(mu_beta, sigma_beta_region)
#       theta_region ~ Normal(0, sigma_theta)  <- RCS coefficients
#
#   Level 3 (Country within Region):
#       alpha_country ~ Normal(alpha_region[r], sigma_alpha_country)
#       beta_country ~ Normal(beta_region[r], sigma_beta_country)
#
# This allows full shrinkage: countries → regions → global mean
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr

az.style.use("arviz-whitegrid"); az.rcParams["stats.ci_prob"] = 0.95
sns.set()
print(f"Running on PyMC v{pm.__version__}")

# ---------------------------------------------------------------------------
# 1) Load & preprocess
# ---------------------------------------------------------------------------
ROOT = r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP"

df = (pd.read_csv(f"{ROOT}/merged.csv", header=0, index_col=0)
        .dropna(subset=["Region", "Country Name", "Population", "GDP"]))

# log10 variables (create if missing)
if "Log_Population" not in df.columns:
    df["Log_Population"] = np.log10(df["Population"])
if "Log_GDP" not in df.columns:
    df["Log_GDP"] = np.log10(df["GDP"])

# centered predictor
df["Log_Population_c"] = df["Log_Population"] - df["Log_Population"].mean()
x = df["Log_Population_c"].values

# indices / coords
df["region_code"]  = df["Region"].astype("category").cat.codes
regions            = df["Region"].astype("category").cat.categories
df["country_code"] = df["Country Name"].astype("category").cat.codes
countries          = df["Country Name"].astype("category").cat.categories
obs_idx            = np.arange(len(df))

coords = {"Region": regions, "Country": countries, "obs_id": obs_idx}

# map country → region (for country-level priors)
c2r = (df[["country_code", "region_code"]]
         .drop_duplicates()
         .set_index("country_code")
         .sort_index())["region_code"].values  # shape: (n_countries,)

# ---------------------------------------------------------------------------
# 2) Restricted cubic spline (natural cubic with linear tails)
# ---------------------------------------------------------------------------
def rcs_design(x_in: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted cubic spline basis with linear tails.
       Returns (N, K-2) matrix."""
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

# choose 4 knots → (K-2)=2 spline columns; adjust if needed
knots = np.quantile(x, [0.05, 0.35, 0.65, 0.95])
Z = rcs_design(x, knots)              # (N, m)
m = Z.shape[1]
np.save(f"{ROOT}/rcs_knots_hier.npy", knots)  # save for forecasting

# ---------------------------------------------------------------------------
# 3) Prior centers (use empirical data means as weak centers)
# ---------------------------------------------------------------------------
alpha_prior_mean = float(df["Log_GDP"].mean())
beta_prior_mean = 1.0  # economic theory: elasticity ~ 1

# ---------------------------------------------------------------------------
# 4) Build model - Full 3-Level Hierarchy
# ---------------------------------------------------------------------------
with pm.Model(coords=coords) as hierarchical_model_with_rcs:

    # ===== Level 1: Global Hyperpriors =====
    # These define the population-level distributions

    # Intercept hyperpriors (global mean and between-region variance)
    mu_alpha = pm.Normal("mu_alpha", mu=alpha_prior_mean, sigma=2.0)
    sigma_alpha_region = pm.HalfNormal("sigma_alpha_region", sigma=1.0)

    # Slope (elasticity) hyperpriors
    mu_beta = pm.Normal("mu_beta", mu=beta_prior_mean, sigma=0.5)
    sigma_beta_region = pm.HalfNormal("sigma_beta_region", sigma=0.3)

    # ===== Level 2: Region-level parameters =====
    # Partial pooling: regions shrink toward global mean
    alpha_region = pm.Normal("alpha_region",
                             mu=mu_alpha,
                             sigma=sigma_alpha_region,
                             dims="Region")
    beta_region = pm.Normal("beta_region",
                            mu=mu_beta,
                            sigma=sigma_beta_region,
                            dims="Region")

    # ===== Level 3: Country-level parameters =====
    # Partial pooling: countries shrink toward their region mean
    sigma_alpha_country = pm.HalfNormal("sigma_alpha_country", sigma=0.5)
    sigma_beta_country = pm.HalfNormal("sigma_beta_country", sigma=0.3)

    alpha_country = pm.Normal("alpha_country",
                              mu=alpha_region[c2r],  # map country → region
                              sigma=sigma_alpha_country,
                              dims="Country")
    beta_country = pm.Normal("beta_country",
                             mu=beta_region[c2r],
                             sigma=sigma_beta_country,
                             dims="Country")

    # ===== RCS coefficients (region-level with shrinkage) =====
    if m > 0:
        hierarchical_model_with_rcs.add_coord("Spline", np.arange(m), mutable=True)
        sigma_theta = pm.HalfNormal("sigma_theta", sigma=0.1)
        theta_region = pm.Normal("theta_region", mu=0.0, sigma=sigma_theta,
                                 dims=("Region", "Spline"))
        # Gather per-row theta by region and compute row-wise dot with Z
        region_idx = pm.Data("region_idx", df["region_code"].values, mutable=False)
        Z_shared = pm.Data("Z", Z, mutable=False)  # (N, m)
        theta_row = theta_region[region_idx]        # (N, m)
        spline_term = pm.math.sum(theta_row * Z_shared, axis=1)  # (N,)
    else:
        spline_term = 0.0

    # ===== Observation model =====
    x_c = pm.Data("x_c", x, mutable=False)
    country_idx = pm.Data("country_idx", df["country_code"].values, mutable=False)

    # Linear predictor
    mu = (alpha_country[country_idx]
          + beta_country[country_idx] * x_c
          + spline_term)

    # Student-t likelihood for robustness
    sigma = pm.HalfStudentT("sigma", nu=3, sigma=0.3)
    nu_raw = pm.Gamma("nu_raw", alpha=2.0, beta=0.2)  # mean ≈ 10
    nu = pm.Deterministic("nu", pt.clip(nu_raw + 1.0, 2.0, 30.0))

    Log_GDP_obs = pm.StudentT("Log_GDP_obs", nu=nu, mu=mu, sigma=sigma,
                              observed=df["Log_GDP"].values, dims="obs_id")

    # Bayesian R-squared
    fitted_var = pt.var(mu)
    R2 = pm.Deterministic("R2", fitted_var / (fitted_var + sigma**2))

# ------------------------------ sampling -----------------------------
# Optimized settings: draws=4000, tune=2000, target_accept=0.95

if __name__ == '__main__':
    with hierarchical_model_with_rcs:
        idata_hier_rcs = pm.sample(
            draws=4_000, tune=2_000, chains=4, cores=12,
            target_accept=0.95, nuts_sampler="nutpie",
            return_inferencedata=True,
            idata_kwargs=dict(
                log_likelihood=True,
                coords={"obs_id": obs_idx},
                dims={"Log_GDP_obs": ["obs_id"]}
            )
        )

# ---------------------------------------------------------------------------
# 5) Posterior predictive, save, and quick summary
# ---------------------------------------------------------------------------
    with hierarchical_model_with_rcs:
        ppc = pm.sample_posterior_predictive(
            trace=idata_hier_rcs,
            var_names=["Log_GDP_obs"],
            return_inferencedata=True,
            predictions=True
        )
    idata_hier_rcs.extend(ppc)

    out_nc = f"{ROOT}/hierarchical_model_with_rcs.nc"
    az.to_netcdf(idata_hier_rcs, out_nc)
    print("Saved:", out_nc)

# Load saved results (for analysis without re-running sampling)
idata_hier_rcs = az.from_netcdf(f"{ROOT}/hierarchical_model_with_rcs.nc")

# Posterior summaries (all hierarchical parameters)
var_names = [
    # Global hyperpriors
    "mu_alpha", "sigma_alpha_region", "mu_beta", "sigma_beta_region",
    # Region-level
    "alpha_region", "beta_region",
    # Country-level variance
    "sigma_alpha_country", "sigma_beta_country",
    # Observation model
    "sigma", "nu", "R2"
]
if m > 0:
    var_names.extend(["sigma_theta", "theta_region"])

print(az.summary(idata_hier_rcs, var_names=var_names, hdi_prob=0.95))

# (Optional) simple PPC scatter by region
try:
    fitted = idata_hier_rcs.posterior_predictive["Log_GDP_obs"].mean(dim=["chain","draw"]).values
    palette = sns.color_palette("tab10", n_colors=len(regions))
    plt.figure(figsize=(10,6))
    for i, reg in enumerate(regions):
        mask = (df["Region"] == reg)
        plt.scatter(df.loc[mask, "Log_Population_c"], df.loc[mask, "Log_GDP"],
                    alpha=0.4, color=palette[i], label=f"Obs: {reg}")
        plt.scatter(df.loc[mask, "Log_Population_c"], fitted[mask],
                    alpha=0.4, color=palette[i], marker="x", label=f"Fit: {reg}")
    plt.xlabel("Centered log10 population"); plt.ylabel("log10 GDP")
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left"); plt.tight_layout()
    plt.savefig(f"{ROOT}/hierarchical_model_with_rcs_ppc.png", dpi=600)
except Exception as e:
    print("PPC plot skipped:", e)
