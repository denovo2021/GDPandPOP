# fig_global_gdp_fan_age.py
# Updated Figure 2 using age-augmented world fan (50/80/95%), constant ISO3 + log-pool
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

ROOT = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")

# ← 直近で作った「一定ISO3＋logプール」のCSVを使います
FAN_AGE = ROOT / "gdp_world_fan_rcs_age.csv"

# （任意）ベースラインのファンを重ねる場合は指定（無ければ None のままでOK）
FAN_BASE_OPT = None  # 例: ROOT/"fig_global_gdp_fan.csv"

OUT_PNG = ROOT / "fig_global_gdp_fan_age.png"

# 1) 読み込み
fan = pd.read_csv(FAN_AGE)
fan["Year"] = pd.to_numeric(fan["Year"], errors="coerce").astype("Int64")
fan = fan.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int)).sort_values("Year")

# 2) 数値化 + 表示単位（兆USD）
num_cols = [c for c in fan.columns if c != "Year"]
fan[num_cols] = fan[num_cols].apply(pd.to_numeric, errors="coerce")
DIV = 1e12  # CSVがUSDなら 1e12 で「兆USD」に変換
for c in num_cols:
    fan[c] = fan[c] / DIV

# 3) （任意）ベースラインを薄く重ねる
base = None
if FAN_BASE_OPT is not None and Path(FAN_BASE_OPT).exists():
    base = pd.read_csv(FAN_BASE_OPT)
    if {"Year","Median"}.issubset(base.columns):
        base["Year"] = pd.to_numeric(base["Year"], errors="coerce").astype("Int64")
        base = base.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int)).sort_values("Year")
        if base["Median"].max() > 1e6:  # USDのままなら兆USDに
            for c in [c for c in base.columns if c != "Year"]:
                base[c] = pd.to_numeric(base[c], errors="coerce") / DIV

# 4) プロット
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(fan["Year"], fan["Median"], lw=2.2, label="Median (age-augmented)")
ax.fill_between(fan["Year"], fan["p95_lo"], fan["p95_hi"], alpha=0.15, label="95%")
ax.fill_between(fan["Year"], fan["p80_lo"], fan["p80_hi"], alpha=0.25, label="80%")
ax.fill_between(fan["Year"], fan["p50_lo"], fan["p50_hi"], alpha=0.35, label="50%")

if base is not None:
    ax.plot(base["Year"], base["Median"], lw=1.5, color="k", alpha=0.35, linestyle="--", label="Median (baseline)")

ax.set_xlim(int(fan["Year"].min()), int(fan["Year"].max()))
ax.set_xlabel("Year")
ax.set_ylabel("Global GDP (USD trillions)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
ax.legend(frameon=False, ncol=2)
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=600)
print("✓", OUT_PNG)
