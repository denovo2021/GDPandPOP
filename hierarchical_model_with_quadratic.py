# hierarchical_model_with_quadratic.py

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Set ArviZ / plotting options
az.style.use("arviz-whitegrid")
az.rcParams["stats.ci_prob"] = 0.95
sns.set()

print(f"Running on PyMC v{pm.__version__}")

# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------
df = pd.read_csv(
    "C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/merged.csv",
    header=0,
    index_col=0
)

df = df.dropna(subset=["Region", "Country Name", "Population", "GDP"])

# Create region codes
df["region_code"] = df["Region"].astype("category").cat.codes
regions = df["Region"].astype("category").cat.categories
n_regions = len(regions)

# Create country codes
df["country_code"] = df["Country Name"].astype("category").cat.codes
countries = df["Country Name"].astype("category").cat.categories
n_countries = len(countries)

coords = {
    "Region": regions,
    "Country": countries
}

# Quadratic term (centering)
df["Log_Population_c"]  = df["Log_Population"] - df["Log_Population"].mean()
df["Log_Population_Sq"] = df["Log_Population_c"] ** 2

# Mapping from country → region (for random effects)
country_region_mapping = (
    df[["country_code", "region_code"]]
    .drop_duplicates()
    .set_index("country_code")
    .sort_index()
)

# ---------------------------------------------------------------------------
# Load the regional_model_with_quadratic & extract region-level means/SDs
# ---------------------------------------------------------------------------
idata_regional_quad = az.from_netcdf(
    "C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/regional_model_with_quadratic.nc"
)

summary_regional_quad = az.summary(
    idata_regional_quad,
    var_names=["alpha_region", "beta_region", "gamma_region", "sigma"],
    hdi_prob=0.95
)

print(summary_regional_quad)

# Create arrays for alpha_region, beta_region, gamma_region posterior means & sds
alpha_region_means = []
alpha_region_sds   = []
beta_region_means  = []
beta_region_sds    = []
gamma_region_means = []
gamma_region_sds   = []

for reg in regions:
    # Row labels look like alpha_region[East Asia & Pacific], etc.
    alpha_row = f"alpha_region[{reg}]"
    beta_row  = f"beta_region[{reg}]"
    gamma_row = f"gamma_region[{reg}]"

    alpha_region_means.append(summary_regional_quad.loc[alpha_row, "mean"])
    alpha_region_sds.append(summary_regional_quad.loc[alpha_row, "sd"])

    beta_region_means.append(summary_regional_quad.loc[beta_row, "mean"])
    beta_region_sds.append(summary_regional_quad.loc[beta_row, "sd"])

    gamma_region_means.append(summary_regional_quad.loc[gamma_row, "mean"])
    gamma_region_sds.append(summary_regional_quad.loc[gamma_row, "sd"])

alpha_region_means = np.array(alpha_region_means)
alpha_region_sds   = np.array(alpha_region_sds)
beta_region_means  = np.array(beta_region_means)
beta_region_sds    = np.array(beta_region_sds)
gamma_region_means = np.array(gamma_region_means)
gamma_region_sds   = np.array(gamma_region_sds)

# ---------------------------------------------------------------------------
# Define the hierarchical model with region-level & country-level parameters
# ---------------------------------------------------------------------------
hierarchical_model_with_quadratic = pm.Model(coords=coords)

with hierarchical_model_with_quadratic:
    # -----------------------------------------------------------------------
    # Region-level parameters (informed by the previous model's posteriors)
    # -----------------------------------------------------------------------
    alpha_region = pm.Normal(
        "alpha_region",
        mu=alpha_region_means,
        sigma=alpha_region_sds,
        dims="Region"
    )
    beta_region = pm.Normal(
        "beta_region",
        mu=beta_region_means,
        sigma=beta_region_sds,
        dims="Region"
    )
    gamma_region = pm.Normal(
        "gamma_region",
        mu=gamma_region_means,
        sigma=gamma_region_sds,
        dims="Region"
    )

    # -----------------------------------------------------------------------
    # Country-level random effects
    # We allow each country to vary around its region-level parameters.
    # -----------------------------------------------------------------------
    sigma_alpha_country = pm.HalfNormal(
        "sigma_alpha_country", 
        sigma=1, 
        dims="Region"
    )
    sigma_beta_country = pm.HalfNormal(
        "sigma_beta_country", 
        sigma=1, 
        dims="Region"
    )
    sigma_gamma_country = pm.HalfNormal(
        "sigma_gamma_country", 
        sigma=1,
        dims="Region"
    )

    alpha_country = pm.Normal(
        "alpha_country",
        mu=alpha_region[country_region_mapping["region_code"].values],
        sigma=sigma_alpha_country[country_region_mapping["region_code"].values],
        dims="Country"
    )
    beta_country = pm.Normal(
        "beta_country",
        mu=beta_region[country_region_mapping["region_code"].values],
        sigma=sigma_beta_country[country_region_mapping["region_code"].values],
        dims="Country"
    )
    gamma_country = pm.Normal(
        "gamma_country",
        mu=gamma_region[country_region_mapping["region_code"].values],
        sigma=sigma_gamma_country[country_region_mapping["region_code"].values],
        dims="Country"
    )

    # -----------------------------------------------------------------------
    # Residual standard deviation
    # -----------------------------------------------------------------------
    sigma = pm.HalfNormal("sigma", sigma=1)

    # -----------------------------------------------------------------------
    # Expected value (country-level)
    #    alpha_country + beta_country * LogPop + gamma_country * (LogPop^2)
    # -----------------------------------------------------------------------
    mu = (
        alpha_country[df["country_code"].values]
        + beta_country[df["country_code"].values]  * df["Log_Population_c"].values
        + gamma_country[df["country_code"].values] * df["Log_Population_Sq"].values
    )

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
idata_hierarchical_quad = pm.sample(
    draws=10000,
    tune=5000,
    chains=4,
    cores=12,
    target_accept=0.99,
    model=hierarchical_model_with_quadratic,
    nuts_sampler="nutpie",
    return_inferencedata=True
)

# ---------------------------------------------------------------------------
# Attach log-likelihood and (optionally) posterior predictive
# ---------------------------------------------------------------------------
with hierarchical_model_with_quadratic:
    idata_hierarchical_quad = pm.compute_log_likelihood(
        idata_hierarchical_quad,
        var_names=["Log_GDP_obs"]
    )

# ---------------------------------------------------------------------------
# Save InferenceData (quadratic model)
# ---------------------------------------------------------------------------
az.to_netcdf(
    idata_hierarchical_quad,
    "C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/hierarchical_model_with_quadratic.nc"
)

idata_hierarchical_quad = az.from_netcdf(
    "C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/hierarchical_model_with_quadratic.nc"
)

# ---------------------------------------------------------------------------
# Posterior Plots & Summary
# ---------------------------------------------------------------------------
# We include both region-level and country-level parameters if desired
var_names = [
    "alpha_region", "beta_region", "gamma_region",
    "sigma_alpha_country", "sigma_beta_country", "sigma_gamma_country",
    "alpha_country", "beta_country", "gamma_country",
    "sigma"
]

az.plot_posterior(idata_hierarchical_quad, var_names=var_names)
plt.show()

summary_table = az.summary(idata_hierarchical_quad, var_names=var_names, hdi_prob=0.95)
print(summary_table)

az.plot_trace(idata_hierarchical_quad, var_names=var_names)
plt.show()

# ---------------------------------------------------------------------------
# Posterior Predictive
# ---------------------------------------------------------------------------
with hierarchical_model_with_quadratic:
    ppc = pm.sample_posterior_predictive(
        idata_hierarchical_quad,
        var_names=["Log_GDP_obs"]
    )

print(ppc.keys())

# ---------------------------------------------------------------------------
# Plot Observed vs. Fitted Values
# ---------------------------------------------------------------------------
# Because each country is modeled separately, we can display the
# fitted values country by country or region by region.
# For demonstration, we'll color by region but use the posterior predictive
# mean at each country to show variety across countries.

plt.figure(figsize=(10, 6))
palette = sns.color_palette("tab10", n_colors=n_regions)

if "Log_GDP_obs" in ppc.posterior_predictive.data_vars:
    pred_means = (
        ppc.posterior_predictive["Log_GDP_obs"]
        .mean(dim=["chain", "draw"])
        .values
    )
else:
    print("Variable 'Log_GDP_obs' not found in posterior_predictive")
    pred_means = None

for i, reg in enumerate(regions):
    mask_reg = df["Region"] == reg
    
    # Observed
    plt.scatter(
        df.loc[mask_reg, "Log_Population"],
        df.loc[mask_reg, "Log_GDP"],
        alpha=0.5,
        color=palette[i],
        label=f"Observed: {reg}"
    )
    # Fitted
    if pred_means is not None:
        plt.scatter(
            df.loc[mask_reg, "Log_Population"],
            pred_means[mask_reg],
            alpha=0.5,
            color=palette[i],
            marker='x',
            label=f"Fitted: {reg}"
        )

plt.xlabel("Log Population")
plt.ylabel("Log GDP")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(
    "C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/hierarchical_model_with_quadratic.png",
    dpi=600
)
plt.show()

