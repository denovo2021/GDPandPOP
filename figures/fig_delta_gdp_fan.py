# fig_delta_gdp_fan.py（補助図：baseline vs age の中央値差）
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
ROOT = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
base = pd.read_csv(ROOT/"fig_global_gdp_fan.csv") if (ROOT/"fig_global_gdp_fan.csv").exists() else None
age  = pd.read_csv(ROOT/"gdp_world_fan_rcs_age.csv")
# baselineのファンCSVを保存していない場合は、先にbaseline側のfan生成スクリプトを流してCSV化してください
if base is not None:
    m = pd.merge(base[["Year","Median"]].rename(columns={"Median":"Median_base"}),
                 age[["Year","Median"]].rename(columns={"Median":"Median_age"}),
                 on="Year", how="inner")
    fig, ax = plt.subplots(figsize=(6,3.6))
    ax.plot(m["Year"], (m["Median_age"]-m["Median_base"])/1e12 if m["Median_age"].max()>1e12 else (m["Median_age"]-m["Median_base"]),
            lw=2, label="Age-augmented − Baseline")
    ax.axhline(0, color="k", lw=1, alpha=.5)
    ax.set_xlabel("Year"); ax.set_ylabel("Δ Global GDP (USD trillions)")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(ROOT/"fig_delta_gdp_fan.png", dpi=600); print("✓ fig_delta_gdp_fan.png")
