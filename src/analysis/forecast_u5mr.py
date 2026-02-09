# forecast_u5mr.py
# U5MR (Under-5 Mortality Rate) projection based on GDP forecasts
# -------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from src.config import (
    PATH_MERGED, PATH_MERGED_AGE, PATH_POP_PREDICTIONS,
    DIR_OUTPUT, DIR_FIGURES
)

# --- Settings ---
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

# Input/Output
GDP_PRED    = DIR_OUTPUT / "gdp_predictions_scenarios_rcs_age.csv"
OUT_CSV     = DIR_OUTPUT / "u5mr_global_fan_v5.csv"
OUT_FIG     = DIR_FIGURES / "Figure3_Global_U5MR_Fan_v5.png"

def find_u5mr_data():
    """Auto-detect data frame containing U5MR column"""
    # Candidate 1: merged_age.csv
    print(f"Checking {PATH_MERGED_AGE.name}...")
    try:
        df = pd.read_csv(PATH_MERGED_AGE)
        if "U5MR" in df.columns:
            print("Found 'U5MR' column in merged_age.csv.")
            return df, "U5MR"
        # Check for similar column names
        candidates = [c for c in df.columns if "u5" in c.lower() or "mort" in c.lower()]
        if candidates:
            print(f"  Note: 'U5MR' not found, but found similar columns: {candidates}")
    except Exception as e:
        print(f"  Could not read {PATH_MERGED_AGE.name}: {e}")

    # Candidate 2: merged.csv
    print(f"Checking {PATH_MERGED.name}...")
    try:
        df = pd.read_csv(PATH_MERGED)
        if "U5MR" in df.columns:
            print("Found 'U5MR' column in merged.csv!")
            return df, "U5MR"

        # Common WDI code possibility
        wdi_code = "SH.DYN.MORT"
        if wdi_code in df.columns:
            print(f"Found '{wdi_code}' in merged.csv! Using this as U5MR.")
            return df, wdi_code

        candidates = [c for c in df.columns if "u5" in c.lower() or "mort" in c.lower()]
        if candidates:
             print(f"  Note: 'U5MR' not found, but found similar columns: {candidates}")

    except Exception as e:
        print(f"  Could not read {PATH_MERGED.name}: {e}")

    return None, None

def main():
    print("--- Starting U5MR Projection (Smart Mode) ---")

    # 1. Load training data for U5MR model
    df_hist, u5mr_col = find_u5mr_data()

    if df_hist is None:
        print("\n[Error] Could not find U5MR data in either csv file.")
        print("Please check the column name in your 'merged.csv'. It should be 'U5MR' or 'SH.DYN.MORT'.")
        return

    # Data preparation
    print(f"Fitting U5MR elasticity model using column: '{u5mr_col}'...")
    df_hist = df_hist.dropna(subset=[u5mr_col, "GDP", "Population", "Region", "Year"])
    df_hist = df_hist[df_hist[u5mr_col] > 0].copy()

    df_hist["log_u5mr"] = np.log(df_hist[u5mr_col])
    df_hist["log_gdppc"] = np.log(df_hist["GDP"] / df_hist["Population"])
    df_hist["time"] = (df_hist["Year"] - 2000) / 10.0

    # Model fitting
    model = smf.ols("log_u5mr ~ C(Region) + log_gdppc + time", data=df_hist).fit()
    print(f"Model R2: {model.rsquared:.3f}")
    print(f"Income Elasticity: {model.params['log_gdppc']:.3f}")

    # 2. Load future data
    print("Loading Future Data...")
    try:
        df_gdp = pd.read_csv(GDP_PRED)
        df_pop = pd.read_csv(PATH_POP_PREDICTIONS)
    except FileNotFoundError as e:
        print(f"[Error] Prediction files not found: {e}")
        print("Please run the GDP projection scripts first.")
        return

    # Merge
    df_future = pd.merge(df_gdp, df_pop, on=["ISO3", "Year", "Scenario"], how="inner")

    # Add Region info
    iso_region = df_hist[["ISO3", "Region"]].drop_duplicates().set_index("ISO3")["Region"].to_dict()
    df_future["Region"] = df_future["ISO3"].map(iso_region)
    df_future = df_future.dropna(subset=["Region", "Pred_Median", "Population"])

    # 3. Prediction
    print("Projecting U5MR...")
    df_future["log_gdppc"] = np.log(df_future["Pred_Median"] / df_future["Population"])
    df_future["time"] = (df_future["Year"] - 2000) / 10.0

    pred_log_u5mr = model.predict(df_future)
    df_future["Pred_U5MR"] = np.exp(pred_log_u5mr)

    # 4. Global aggregation
    print("Aggregating Global Weighted Average...")
    df_future["weighted_u5"] = df_future["Pred_U5MR"] * df_future["Population"]

    world_grp = df_future.groupby(["Year", "Scenario"]).agg(
        sum_w_u5 = ("weighted_u5", "sum"),
        sum_pop  = ("Population", "sum")
    ).reset_index()

    world_grp["Global_U5MR"] = world_grp["sum_w_u5"] / world_grp["sum_pop"]

    # 5. Fan chart data
    print("Calculating Quantiles...")
    wide = world_grp.pivot(index="Year", columns="Scenario", values="Global_U5MR")
    arr = wide.to_numpy()
    years = wide.index.values

    fan_df = pd.DataFrame({"Year": years})
    fan_df["Median"] = np.median(arr, axis=1)
    fan_df["p95_lo"] = np.quantile(arr, 0.025, axis=1)
    fan_df["p95_hi"] = np.quantile(arr, 0.975, axis=1)
    fan_df["p80_lo"] = np.quantile(arr, 0.10, axis=1)
    fan_df["p80_hi"] = np.quantile(arr, 0.90, axis=1)
    fan_df["p50_lo"] = np.quantile(arr, 0.25, axis=1)
    fan_df["p50_hi"] = np.quantile(arr, 0.75, axis=1)

    fan_df.to_csv(OUT_CSV, index=False)

    # 6. Plotting
    print("Plotting Figure 3...")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Color palette (red-ish for mortality)
    ax.fill_between(fan_df["Year"], fan_df["p95_lo"], fan_df["p95_hi"], color="#e377c2", alpha=0.15, label="95% Interval")
    ax.fill_between(fan_df["Year"], fan_df["p80_lo"], fan_df["p80_hi"], color="#e377c2", alpha=0.25, label="80% Interval")
    ax.fill_between(fan_df["Year"], fan_df["p50_lo"], fan_df["p50_hi"], color="#e377c2", alpha=0.35, label="50% Interval")
    ax.plot(fan_df["Year"], fan_df["Median"], lw=3, color="#d62728", label="Pooled Median")

    ax.set_title("Global Under-5 Mortality Rate Projection", fontsize=14, pad=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Deaths per 1,000 live births", fontsize=12)
    ax.set_xlim(2025, 2100)
    ax.set_ylim(bottom=0)

    ax.legend(loc="upper right", frameon=False, fontsize=10)

    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=600)
    print(f"Saved Figure 3 to {OUT_FIG}")
    plt.show()

if __name__ == "__main__":
    main()
