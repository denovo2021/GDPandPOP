# step3_plot_figure2.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path
import seaborn as sns

sns.set_context("paper", font_scale=1.2); sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

ROOT = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
CSV_PATH = ROOT / "gdp_world_fan_v5_full_uncertainty.csv"

df = pd.read_csv(CSV_PATH)
cols = ["Median", "p95_lo", "p95_hi", "p80_lo", "p80_hi", "p50_lo", "p50_hi"]
for c in cols: df[c] = df[c] / 1e12

fig, ax = plt.subplots(figsize=(8, 5))
ax.fill_between(df["Year"], df["p95_lo"], df["p95_hi"], color="#1f77b4", alpha=0.15, label="95% Interval")
ax.fill_between(df["Year"], df["p80_lo"], df["p80_hi"], color="#1f77b4", alpha=0.25, label="80% Interval")
ax.fill_between(df["Year"], df["p50_lo"], df["p50_hi"], color="#1f77b4", alpha=0.35, label="50% Interval")
ax.plot(df["Year"], df["Median"], lw=3, color="#0b559f", label="Pooled Median")

ax.set_title("Global GDP Projection (Full Uncertainty Pool)", fontsize=14, pad=15)
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Global GDP (USD Trillions, 2023 prices)", fontsize=12)
ax.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f"{x:,.0f}"))
ax.set_xlim(2025, 2100)
ax.legend(loc="upper left", frameon=False, fontsize=10)
plt.tight_layout()
SAVE_PATH = ROOT / "Figure2_Global_GDP_Fan_v5_Final.png"
fig.savefig(SAVE_PATH, dpi=600)
print(f"✓ Saved {SAVE_PATH}")
plt.show()