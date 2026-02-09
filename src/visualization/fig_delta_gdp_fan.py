# fig_delta_gdp_fan.py (supplementary: baseline vs age median difference)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DIR_OUTPUT, DIR_FIGURES

base_path = DIR_OUTPUT / "fig_global_gdp_fan.csv"
age_path = DIR_OUTPUT / "gdp_world_fan_rcs_age.csv"
base = pd.read_csv(base_path) if base_path.exists() else None
age = pd.read_csv(age_path)
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
    fig.savefig(DIR_FIGURES / "fig_delta_gdp_fan.png", dpi=600); print("✓ fig_delta_gdp_fan.png")
