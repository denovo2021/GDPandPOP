# run_fig3_u5mr_v5.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import seaborn as sns

# --- Settings ---
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PATH_MERGED_AGE, PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE,
    PATH_POP_PREDICTIONS, DIR_OUTPUT, DIR_FIGURES
)

# Input Files
HIST_CSV = PATH_MERGED_AGE  # historical data (training)
GDP_PRED = PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE  # new GDP predictions
POP_PRED = PATH_POP_PREDICTIONS  # future population (for weighting)

OUT_CSV = DIR_OUTPUT / "u5mr_global_fan_v5.csv"
OUT_FIG = DIR_FIGURES / "Figure3_Global_U5MR_Fan_v5.png"

def main():
    print("--- Starting U5MR Projection ---")
    
    # 1. U5MRモデルの学習 (簡易OLS: log(U5MR) ~ Region + log(GDPpc) + Year)
    print("Fitting U5MR elasticity model...")
    df_hist = pd.read_csv(HIST_CSV)
    df_hist = df_hist.dropna(subset=["U5MR", "GDP", "Population", "Region", "Year"])
    df_hist = df_hist[df_hist["U5MR"] > 0].copy()
    
    df_hist["log_u5mr"] = np.log(df_hist["U5MR"])
    df_hist["log_gdppc"] = np.log(df_hist["GDP"] / df_hist["Population"])
    df_hist["time"] = (df_hist["Year"] - 2000) / 10.0
    
    # RegionをCategoryに
    model = smf.ols("log_u5mr ~ C(Region) + log_gdppc + time", data=df_hist).fit()
    print(f"Model R2: {model.rsquared:.3f}")
    print(f"Income Elasticity: {model.params['log_gdppc']:.3f}")
    print(f"Time Trend: {model.params['time']:.3f}")

    # 2. 将来データの準備 (GDP + Population)
    print("Loading Future Data...")
    df_gdp = pd.read_csv(GDP_PRED) # Pred_Median, Year, ISO3, Scenario
    df_pop = pd.read_csv(POP_PRED) # Population, Year, ISO3, Scenario
    
    # Merge
    # 注意: GDP_PREDはISO3/Scenario/Yearのキーを持つ
    df_future = pd.merge(df_gdp, df_pop, on=["ISO3", "Year", "Scenario"], how="inner")
    
    # Region情報の付与 (Historyから)
    iso_region = df_hist[["ISO3", "Region"]].drop_duplicates().set_index("ISO3")["Region"].to_dict()
    df_future["Region"] = df_future["ISO3"].map(iso_region)
    df_future = df_future.dropna(subset=["Region", "Pred_Median", "Population"])
    
    # 3. 予測計算
    print("Projecting U5MR...")
    df_future["log_gdppc"] = np.log(df_future["Pred_Median"] / df_future["Population"])
    df_future["time"] = (df_future["Year"] - 2000) / 10.0
    
    # OLSモデルで予測 (log scale)
    pred_log_u5mr = model.predict(df_future)
    df_future["Pred_U5MR"] = np.exp(pred_log_u5mr)
    
    # 4. 世界集計 (人口重み付け)
    # Weighted Average U5MR = sum(U5MR * Pop) / sum(Pop) per Scenario/Year
    print("Aggregating Global Weighted Average...")
    
    df_future["weighted_u5"] = df_future["Pred_U5MR"] * df_future["Population"]
    
    world_grp = df_future.groupby(["Year", "Scenario"]).agg(
        sum_w_u5 = ("weighted_u5", "sum"),
        sum_pop  = ("Population", "sum")
    ).reset_index()
    
    world_grp["Global_U5MR"] = world_grp["sum_w_u5"] / world_grp["sum_pop"]
    
    # 5. ファンチャート用データ作成 (Quantiles across scenarios)
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
    
    # 6. 作図
    print("Plotting Figure 3...")
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.fill_between(fan_df["Year"], fan_df["p95_lo"], fan_df["p95_hi"], color="#d62728", alpha=0.15, label="95% Interval")
    ax.fill_between(fan_df["Year"], fan_df["p80_lo"], fan_df["p80_hi"], color="#d62728", alpha=0.25, label="80% Interval")
    ax.fill_between(fan_df["Year"], fan_df["p50_lo"], fan_df["p50_hi"], color="#d62728", alpha=0.35, label="50% Interval")
    ax.plot(fan_df["Year"], fan_df["Median"], lw=3, color="#b30000", label="Pooled Median")
    
    ax.set_title("Global Under-5 Mortality Rate Projection", fontsize=14, pad=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Deaths per 1,000 live births", fontsize=12)
    ax.set_xlim(2025, 2100)
    
    # 凡例
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=600)
    print(f"✓ Saved Figure 3 to {OUT_FIG}")
    plt.show()

if __name__ == "__main__":
    main()