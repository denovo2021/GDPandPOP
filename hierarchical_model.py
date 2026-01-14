# hierarchical_model.py

# Import libraries
import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
from matplotlib import pylab as plt
import matplotlib.lines
import matplotlib.collections
import seaborn as sns
sns.set()
from sklearn.metrics import r2_score

# Set plotting style and confidence interval
az.style.use("arviz-whitegrid")
az.rcParams["stats.ci_prob"] = 0.95

print(f"Running on PyMC v{pm.__version__}")

# Import data
df = pd.read_csv("C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/merged.csv", header=0, index_col=0)

# Drop rows with NaN values in essential columns
df = df.dropna(subset=["Region", "Country Name", "Population", "GDP"])

# Create region codes
df["region_code"] = df["Region"].astype("category").cat.codes
regions = df["Region"].astype("category").cat.categories

# Create country codes
df["country_code"] = df["Country Name"].astype("category").cat.codes
countries = df["Country Name"].astype("category").cat.categories

# Define model coordinates
coords = {"Region": regions, "Country": countries}

# Create mapping from country to region
country_region_mapping = df[['country_code', 'region_code']].drop_duplicates().set_index('country_code').sort_index()

# Make sure regional_model saved its results in regional_model.nc (or adjust path accordingly)
idata_regional = az.from_netcdf("C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/regional_model.nc")

# Summarize alpha_region, beta_region by index; these typically appear as alpha_region[0], alpha_region[1], etc.
summary_regional = az.summary(
    idata_regional, 
    var_names=["alpha_region", "beta_region", "sigma"], 
    hdi_prob=0.95
)

# We'll create numeric arrays matching len(regions) 
n_regions = len(regions)
alpha_region_means = np.zeros(n_regions)
alpha_region_sds   = np.zeros(n_regions)
beta_region_means  = np.zeros(n_regions)
beta_region_sds    = np.zeros(n_regions)

# Loop over the region names in 'regions'
for i, region_name in enumerate(regions):
    # Build row names in summary for alpha_region, beta_region
    # e.g. alpha_region[East Asia & Pacific]
    alpha_row = f"alpha_region[{region_name}]"
    beta_row  = f"beta_region[{region_name}]"

    alpha_region_means[i] = summary_regional.loc[alpha_row, "mean"]
    alpha_region_sds[i]   = summary_regional.loc[alpha_row, "sd"]

    beta_region_means[i]  = summary_regional.loc[beta_row,  "mean"]
    beta_region_sds[i]    = summary_regional.loc[beta_row,  "sd"]

# ---------------------------------------------------------------------------
# Define hierarchical model, using regional_model posteriors as new priors
# ---------------------------------------------------------------------------
with pm.Model(coords=coords) as hierarchical_model:
    # --- Region-level intercept & slope ---
    # We now center each region’s intercept on alpha_region_means[i], 
    # with a prior SD = alpha_region_sds[i], etc.
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

    # --- Country-level standard deviations (random effects per region) ---
    sigma_alpha_country = pm.HalfNormal("sigma_alpha_country", sigma=1, dims="Region")
    sigma_beta_country  = pm.HalfNormal("sigma_beta_country",  sigma=1, dims="Region")

    # --- Country-level intercepts & slopes ---
    alpha_country = pm.Normal(
        "alpha_country",
        mu=alpha_region[country_region_mapping['region_code'].values],
        sigma=sigma_alpha_country[country_region_mapping['region_code'].values],
        dims="Country"
    )
    beta_country = pm.Normal(
        "beta_country",
        mu=beta_region[country_region_mapping['region_code'].values],
        sigma=sigma_beta_country[country_region_mapping['region_code'].values],
        dims="Country"
    )

    # --- Residual scale ---
    sigma = pm.HalfNormal("sigma", sigma=1)

    # --- Expected value ---
    mu = alpha_country[df["country_code"].values] + beta_country[df["country_code"].values] * df["Log_Population"].values

    # --- Likelihood ---
    nu = pm.Exponential("nu", 1 / 30)
    Log_GDP_obs = pm.StudentT("Log_GDP_obs", nu=nu, mu=mu, sigma=sigma, observed=df["Log_GDP"].values)

# Run sampling
if __name__ == "__main__":
    with hierarchical_model:
        idata_hierarchical = pm.sample(draws=10000,
                          cores=12,
                          tune=5000,
                          chains=4,
                          n_init=5000,
                          target_accept=0.95,
                          nuts_sampler="nutpie",
                          return_inferencedata=True)

# ---------------------------------------------------------------------------
# Attach log-likelihood and (optionally) posterior predictive
# ---------------------------------------------------------------------------
with hierarchical_model:
    # Compute point-wise log-likelihood for Log_GDP_obs
    ll_lin = pm.compute_log_likelihood(
        idata_hierarchical             # draws already sampled
    )

# Add group if not already present
if "log_likelihood" not in idata_hierarchical.groups():
    idata_hierarchical.add_groups({"log_likelihood": ll_lin.log_likelihood})
    print("log_likelihood group added.")
else:
    print("log_likelihood group already present – skipping add_groups().")

# ---------------------------------------------------------------------------
# Save InferenceData (linear model)
# ---------------------------------------------------------------------------
az.to_netcdf(
    idata_hierarchical,
    "C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/hierarchical_model_linear.nc"
)
print("InferenceData written with log_likelihood group.")

# Load inference data
idata_hierarchical = az.from_netcdf("C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/hierarchical_model.nc")

# Posterior plots and summary
az.plot_posterior(idata_hierarchical, var_names=["alpha_region", "beta_region", "sigma"])
print(az.summary(idata_hierarchical, var_names=["alpha_region", "beta_region"]))
print(az.summary(idata_hierarchical, var_names=["alpha_country"]))

# Trace plots
az.plot_trace(idata_hierarchical, var_names=["alpha_region", "beta_region"])

# Posterior predictive checks
with hierarchical_model:
    ppc = pm.sample_posterior_predictive(idata_hierarchical, var_names=["Log_GDP_obs"])

# Plot observed vs fitted values
plt.figure(figsize=(10, 6))
palette = sns.color_palette("tab10", n_colors=len(regions))

# Observed and predicted data by region
pred_means = ppc.posterior_predictive["Log_GDP_obs"].mean(dim=["chain", "draw"]).values
# Observed and predicted data by region
for i, reg in enumerate(regions):
    mask = df["Region"] == reg
    mask_indices = np.where(mask)[0]  # ブール型からインデックスに変換
    plt.scatter(df.loc[mask, "Log_Population"], df.loc[mask, "Log_GDP"],
                alpha=0.5, label=f"Observed: {reg}", color=palette[i])
    plt.scatter(df.loc[mask, "Log_Population"], pred_means[mask_indices],
                color=palette[i], alpha=0.5, marker='x', label=f"Fitted: {reg}")

# Regression lines by region
x_line = np.linspace(df["Log_Population"].min(), df["Log_Population"].max(), 100)
for i, reg in enumerate(regions):
    alpha_i = idata_hierarchical.posterior["alpha_region"].sel(Region=reg).mean().values
    beta_i = idata_hierarchical.posterior["beta_region"].sel(Region=reg).mean().values
    plt.plot(x_line, alpha_i + beta_i * x_line, color=palette[i], linestyle='--', label=f"Regression line: {reg}")

plt.xlabel("Log Population")
plt.ylabel("Log GDP")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP/hierarchical_model.png", dpi=600)
# plt.show()

# Function to plot regression line with credible interval and R-squared
def plot_country_regression(idata, df, country_name):
    country_data = df[df["Country Name"] == country_name]

    # Posterior samples
    alpha_samples = idata_hierarchical.posterior["alpha_country"].sel(Country=country_name).values.flatten()
    beta_samples = idata_hierarchical.posterior["beta_country"].sel(Country=country_name).values.flatten()

    # Observed data
    x_obs = country_data["Log_Population"].values
    y_obs = country_data["Log_GDP"].values

    plt.scatter(x_obs, y_obs, color='steelblue', alpha=0.6, label="Observed")

    # Regression line with credible interval
    x_vals = np.linspace(x_obs.min(), x_obs.max(), 100)
    y_preds = np.array([a + b * x_vals for a, b in zip(alpha_samples, beta_samples)])

    mean_pred = y_preds.mean(axis=0)
    hdi_pred = az.hdi(y_preds, hdi_prob=0.95)

    # Calculate R-squared
    y_pred_obs = np.mean([a + b * x_obs for a, b in zip(alpha_samples, beta_samples)], axis=0)
    r_squared = r2_score(y_obs, y_pred_obs)

    # Plot regression line and credible interval
    plt.plot(x_vals, mean_pred, color='navy', lw=2.5, label=f'Regression Line\n$R^2$ = {r_squared:.3f}')
    plt.fill_between(x_vals, hdi_pred[:, 0], hdi_pred[:, 1], color='navy', alpha=0.2, label='95% CI')

    plt.xlabel("Log Population")
    plt.ylabel("Log GDP")
    plt.title(f"Observed vs Predicted Log GDP for {country_name}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
plot_country_regression(idata_hierarchical, df, "Germany")




