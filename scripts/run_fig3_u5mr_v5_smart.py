# run_fig3_u5mr_v5_smart.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import seaborn as sns
import sys

# --- Settings ---
sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

ROOT = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")

# Input Files
FILE_AGE    = ROOT / "merged_age.csv"
FILE_RAW    = ROOT / "merged.csv"
GDP_PRED    = ROOT / "gdp_predictions_scenarios_rcs_age.csv"
POP_PRED    = ROOT / "pop_predictions_scenarios.csv"

OUT_CSV     = ROOT / "u5mr_global_fan_v5.csv"
OUT_FIG     = ROOT / "Figure3_Global_U5MR_Fan_v5.png"

def find_u5mr_data():
    """U5MR列を含むデータフレームを自動探索する"""
    # 候補1: merged_age.csv
    print(f"Checking {FILE_AGE.name}...")
    try:
        df = pd.read_csv(FILE_AGE)
        if "U5MR" in df.columns:
            print("Found 'U5MR' column in merged_age.csv.")
            return df, "U5MR"
        # 列名が違うかも？検索
        candidates = [c for c in df.columns if "u5" in c.lower() or "mort" in c.lower()]
        if candidates:
            print(f"  Note: 'U5MR' not found, but found similar columns: {candidates}")
    except Exception as e:
        print(f"  Could not read {FILE_AGE.name}: {e}")

    # 候補2: merged.csv
    print(f"Checking {FILE_RAW.name}...")
    try:
        df = pd.read_csv(FILE_RAW)
        if "U5MR" in df.columns:
            print("Found 'U5MR' column in merged.csv!")
            return df, "U5MR"
        
        # よくあるWDIコードの可能性
        wdi_code = "SH.DYN.MORT"
        if wdi_code in df.columns:
            print(f"Found '{wdi_code}' in merged.csv! Using this as U5MR.")
            return df, wdi_code
            
        candidates = [c for c in df.columns if "u5" in c.lower() or "mort" in c.lower()]
        if candidates:
             print(f"  Note: 'U5MR' not found, but found similar columns: {candidates}")
             
    except Exception as e:
        print(f"  Could not read {FILE_RAW.name}: {e}")
        
    return None, None

def main():
    print("--- Starting U5MR Projection (Smart Mode) ---")
    
    # 1. U5MRモデルの学習データのロード
    df_hist, u5mr_col = find_u5mr_data()
    
    if df_hist is None:
        print("\n[Error] Could not find U5MR data in either csv file.")
        print("Please check the column name in your 'merged.csv'. It should be 'U5MR' or 'SH.DYN.MORT'.")
        return

    # データ整形
    print(f"Fitting U5MR elasticity model using column: '{u5mr_col}'...")
    df_hist = df_hist.dropna(subset=[u5mr_col, "GDP", "Population", "Region", "Year"])
    df_hist = df_hist[df_hist[u5mr_col] > 0].copy()
    
    df_hist["log_u5mr"] = np.log(df_hist[u5mr_col])
    df_hist["log_gdppc"] = np.log(df_hist["GDP"] / df_hist["Population"])
    df_hist["time"] = (df_hist["Year"] - 2000) / 10.0
    
    # モデル学習
    model = smf.ols("log_u5mr ~ C(Region) + log_gdppc + time", data=df_hist).fit()
    print(f"Model R2: {model.rsquared:.3f}")
    print(f"Income Elasticity: {model.params['log_gdppc']:.3f}")
    
    # 2. 将来データの準備
    print("Loading Future Data...")
    try:
        df_gdp = pd.read_csv(GDP_PRED)
        df_pop = pd.read_csv(POP_PRED)
    except FileNotFoundError as e:
        print(f"[Error] Prediction files not found: {e}")
        print("Please run the GDP projection scripts first.")
        return

    # Merge
    df_future = pd.merge(df_gdp, df_pop, on=["ISO3", "Year", "Scenario"], how="inner")
    
    # Region情報の付与
    iso_region = df_hist[["ISO3", "Region"]].drop_duplicates().set_index("ISO3")["Region"].to_dict()
    df_future["Region"] = df_future["ISO3"].map(iso_region)
    df_future = df_future.dropna(subset=["Region", "Pred_Median", "Population"])
    
    # 3. 予測計算
    print("Projecting U5MR...")
    df_future["log_gdppc"] = np.log(df_future["Pred_Median"] / df_future["Population"])
    df_future["time"] = (df_future["Year"] - 2000) / 10.0
    
    pred_log_u5mr = model.predict(df_future)
    df_future["Pred_U5MR"] = np.exp(pred_log_u5mr)
    
    # 4. 世界集計
    print("Aggregating Global Weighted Average...")
    df_future["weighted_u5"] = df_future["Pred_U5MR"] * df_future["Population"]
    
    world_grp = df_future.groupby(["Year", "Scenario"]).agg(
        sum_w_u5 = ("weighted_u5", "sum"),
        sum_pop  = ("Population", "sum")
    ).reset_index()
    
    world_grp["Global_U5MR"] = world_grp["sum_w_u5"] / world_grp["sum_pop"]
    
    # 5. ファンチャート用データ作成
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
    
    # カラーパレット (U5MRは赤系で警告色っぽく、かつ医学論文らしい落ち着いた色)
    # 95%: 薄い赤, 80%: 中くらいの赤, 50%: 濃いめの赤, Median: 線
    ax.fill_between(fan_df["Year"], fan_df["p95_lo"], fan_df["p95_hi"], color="#e377c2", alpha=0.15, label="95% Interval")
    ax.fill_between(fan_df["Year"], fan_df["p80_lo"], fan_df["p80_hi"], color="#e377c2", alpha=0.25, label="80% Interval")
    ax.fill_between(fan_df["Year"], fan_df["p50_lo"], fan_df["p50_hi"], color="#e377c2", alpha=0.35, label="50% Interval")
    ax.plot(fan_df["Year"], fan_df["Median"], lw=3, color="#d62728", label="Pooled Median")
    
    ax.set_title("Global Under-5 Mortality Rate Projection", fontsize=14, pad=15)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Deaths per 1,000 live births", fontsize=12)
    ax.set_xlim(2025, 2100)
    
    # Y軸を0始まりにするか、トレンドが見やすいようにするか
    # U5MRは0に近づくので0含めが良い
    ax.set_ylim(bottom=0)
    
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=600)
    print(f"✓ Saved Figure 3 to {OUT_FIG}")
    plt.show()

if __name__ == "__main__":
    main()