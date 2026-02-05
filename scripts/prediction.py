# Prediction_with_quadratic.py
"""
This script uses the hierarchical Bayesian model defined in 'hierarchical_model_with_quadratic.py' to predict future GDP
based on population forecasts (2024-2100) for various countries. Key steps:
 1. Loading historical and future population data.
 2. Loading the inference data from the hierarchical Bayesian model.
 3. Preparing and verifying the data.
 4. Generating GDP predictions (including a quadratic population term).
 5. Saving the prediction results.
 6. (Optional) Visualizing the predictions.
"""

# -----------------------------------
# 0. Import Necessary Libraries
# -----------------------------------
import pandas as pd
import numpy as np
import arviz as az
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns
import gc  # for garbage collection
from sklearn.preprocessing import StandardScaler
import joblib  # for saving and loading scaler objects
import warnings

# Set visualization styles
sns.set()
az.style.use("arviz-whitegrid")
az.rcParams["stats.ci_prob"] = 0.95

# Define file paths (use config)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PATH_MERGED, PATH_WPP_XLSX, PATH_WB_METADATA,
    PATH_MODEL_HIERARCHICAL_QUAD, PATH_GDP_PREDICTIONS, PATH_GDP_PREDICTIONS_META, DIR_OUTPUT
)

historical_data_path = str(PATH_MERGED)
future_population_path = str(PATH_WPP_XLSX)
metadata_path = str(PATH_WB_METADATA)

# IMPORTANT: Use your hierarchical model with a quadratic term
inference_data_path = str(PATH_MODEL_HIERARCHICAL_QUAD)

output_predictions_path = str(PATH_GDP_PREDICTIONS)
scaler_path = str(DIR_OUTPUT / "year_scaler.pkl")  # Path to save/load the scaler

# -----------------------------------
# 1. Load Historical Population and GDP Data
# -----------------------------------
df_hist = pd.read_csv(historical_data_path, header=0)
df_hist = df_hist.dropna(subset=["Region", "Population", "GDP", "Year"])
df_hist["Log_Population"] = np.log10(df_hist["Population"])
df_hist["Log_GDP"] = np.log10(df_hist["GDP"])
df_hist = df_hist[["Country Name", "Country Code", "Year", "Log_Population", "Log_GDP"]]

# ---------------------------------------------------------------------------
# 2. Load the selected UN‑projection sheets only
# ---------------------------------------------------------------------------
variant_sheets = [
    "Medium variant",
    "High variant",
    "Low variant",
    "Constant-fertility",
    "Instant-replacement zero migr",
    "Momentum",
    "Zero-migration",
    "No change",
    "No fertility below age 18",
    "Accelarated ABR decline recup",   # UN spelling
]

xl = pd.ExcelFile(future_population_path)

pop_scenarios = {}
for sheet in variant_sheets:
    if sheet not in xl.sheet_names:
        print(f"Warning: sheet '{sheet}' not found in workbook — skipped.")
        continue

    df_pred = xl.parse(sheet, header=16, index_col=0)
    df_pred = df_pred[["Year",
                       "ISO3 Alpha-code",
                       "Total Population, as of 1 July (thousands)"]]
    df_pred.columns = ["Year", "Country Code", "Population"]
    df_pred = df_pred.dropna()

    df_pred["Year"] = df_pred["Year"].astype(int)
    df_pred["Population"]     = df_pred["Population"] * 1_000
    df_pred["Log_Population"] = np.log10(df_pred["Population"].astype(float))

    pop_scenarios[sheet] = df_pred

# ---------------------------------------------------------------------------
# 3.  Merge each scenario with country names and region info
# ---------------------------------------------------------------------------
df_meta = pd.read_csv(metadata_path)

for sheet, df_pred in pop_scenarios.items():
    df_pred["Year"] = df_pred["Year"].astype(int)

    df_pred = df_pred.merge(
        df_hist[["Country Code", "Country Name"]].drop_duplicates(),
        on="Country Code", how="left"
    ).merge(df_meta, on="Country Code", how="left")

    df_pred = df_pred.dropna(subset=["Country Name", "Region"])
    pop_scenarios[sheet] = df_pred.reindex(
        columns=["Year", "Country Name", "Country Code",
                 "Region", "IncomeGroup", "Population", "Log_Population"]
    )

# -----------------------------------
# 4. Load Inference Data from the Hierarchical Bayesian Model
# -----------------------------------
print("\nLoading inference data...")
idata = az.from_netcdf(inference_data_path)
print("Inference data loaded successfully.")

# -----------------------------------
# 5. Quick Checks on the Posterior
# -----------------------------------
print("\nSummary of Posterior Distributions for Country-Level Parameters:")
# Note: We now have alpha_country, beta_country, gamma_country
print(az.summary(idata, var_names=["alpha_country", "beta_country", "gamma_country"]))

print("\nInference Data Coordinates:")
# print(idata.coords)

# -----------------------------------
# 6. Verify Country Mapping
# -----------------------------------
model_countries = idata.posterior.coords["Country"].values.tolist()
pred_countries = df_pred["Country Name"].unique().tolist()

common_countries = list(set(model_countries).intersection(set(pred_countries)))
print(f"\nNumber of common countries: {len(common_countries)}")

missing_in_model = list(set(pred_countries) - set(model_countries))
if missing_in_model:
    print("\nThe following countries are present in the prediction data but not in the model and will be excluded:")
    print(missing_in_model)
    df_pred = df_pred[df_pred["Country Name"].isin(common_countries)]
else:
    print("\nAll countries in the prediction data are present in the model.")

# ---------------------------------------------------------------------------
# 7.  Generate GDP predictions (loop over scenarios)
# ---------------------------------------------------------------------------
scenario_predictions = []

for sheet, df_pred in pop_scenarios.items():
    common = set(idata.posterior.coords["Country"].values) \
             & set(df_pred["Country Name"].unique())
    df_pred = df_pred[df_pred["Country Name"].isin(common)]

    for country in common:
        fut = df_pred[df_pred["Country Name"] == country]
        log_pop = fut["Log_Population"].values

        alpha_samples = idata.posterior["alpha_country"].sel(Country=country).values.flatten()
        beta_samples = idata.posterior["beta_country" ].sel(Country=country).values.flatten()
        gamma_samples = idata.posterior["gamma_country"].sel(Country=country).values.flatten()

        log_gdp = alpha_samples[:, None] + beta_samples[:, None]*log_pop + gamma_samples[:, None]*(log_pop**2)
        gdp_samples = 10**log_gdp

        med  = np.median(gdp_samples, axis=0)
        hdi  = az.hdi(gdp_samples, 0.95)

        scenario_predictions.append(pd.DataFrame({
            "Scenario":      sheet,
            "Country Name":  country,
            "Country Code":  fut["Country Code"].iloc[0],
            "Year":          fut["Year"].values,
            "Pred Median":   med,
            "Pred Lower":    hdi[:,0],
            "Pred Upper":    hdi[:,1]
        }))

df_scen = pd.concat(scenario_predictions, ignore_index=True)

# ---------------------------------------------------------------------------
# 8.  Random‑effects meta‑analysis across scenarios
# ---------------------------------------------------------------------------
# --- replace only the DerSimonian–Laird helper --------------------------------
def dersimonian_laird(group):
    # work entirely on log10 scale
    y  = np.log10(group["Pred Median"].values)
    se = (np.log10(group["Pred Upper"].values) -
          np.log10(group["Pred Lower"].values)) / (2 * 1.96)

    w   = 1 / se**2
    Q   = np.sum(w * y**2) - (np.sum(w * y)**2) / np.sum(w)
    k   = len(y)
    tau2 = max(0, (Q - (k - 1)) / (np.sum(w) - np.sum(w**2) / np.sum(w)))

    w_re = 1 / (se**2 + tau2)
    pooled      = np.sum(w_re * y) / np.sum(w_re)
    se_pooled   = np.sqrt(1 / np.sum(w_re))

    lower = pooled - 1.96 * se_pooled
    upper = pooled + 1.96 * se_pooled

    return pd.Series({
        "Pooled Median": 10**pooled,
        "Pooled Lower":  10**lower,
        "Pooled Upper":  10**upper
    })

meta_results = (
    df_scen.groupby(["Country Name", "Country Code", "Year"])
           .apply(dersimonian_laird)
           .reset_index()
)

meta_results.to_csv(PATH_GDP_PREDICTIONS_META, index=False)
print("Meta‑analytic GDP predictions saved.")

def show_meta_predictions(meta_df, years=(2025, 2030, 2035, 2040),  
                          n_top=10, sort_desc=True):
    """
    Display pooled‑GDP forecasts for the requested years.

    Parameters
    ----------
    meta_df : pd.DataFrame
        Output of the meta‑analysis step. Must contain at least the columns
        ['Country Name', 'Country Code', 'Year',
         'Pooled Median', 'Pooled Lower', 'Pooled Upper'].
    years : iterable
        Target calendar years (int) to display.
    n_top : int
        Number of countries to show per year (after sorting by Pooled Median).
    sort_desc : bool
        If True (default) show largest predicted GDP first.

    Returns
    -------
    dict(year -> pd.DataFrame)
        Dictionary of the filtered and sorted tables.
    """
    results = {}
    for yr in years:
        df_y = (
            meta_df[meta_df["Year"] == yr]
            .sort_values("Pooled Median", ascending=not sort_desc)
            .head(n_top)
            .reset_index(drop=True)
        )
        print(f"\n=== Meta‑analytic GDP (year {yr}) ===")
        display_cols = ["Country Name", "Pooled Median",
                        "Pooled Lower", "Pooled Upper"]
        print(df_y[display_cols].to_string(index=False, formatters={
            "Pooled Median": "{:,.0f}".format,
            "Pooled Lower":  "{:,.0f}".format,
            "Pooled Upper":  "{:,.0f}".format
        }))
        results[yr] = df_y
    return results


# ------------------------------------------------------------------
# Usage example after meta_results has been created:
# ------------------------------------------------------------------
meta_tables = show_meta_predictions(meta_results,
                                    years=[2027, 2030, 2035, 2040],
                                    n_top=20)  # top‑20 if desired

