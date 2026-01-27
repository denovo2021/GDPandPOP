# fig_gdp_fan_chart.py
# Global GDP Fan Chart visualization
# -------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from config import DIR_OUTPUT, DIR_FIGURES

sns.set_context("paper", font_scale=1.2)
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'sans-serif'

CSV_PATH = DIR_OUTPUT / "gdp_world_fan_v5_full_uncertainty.csv"

# Load file
try:
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        raise ValueError("CSV is empty.")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Convert to Trillions
cols = ["Median", "p95_lo", "p95_hi", "p80_lo", "p80_hi", "p50_lo", "p50_hi"]
for c in cols:
    if c in df.columns:
        df[c] = df[c] / 1e12

fig, ax = plt.subplots(figsize=(8, 5))

# Fan Chart
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
SAVE_PATH = DIR_FIGURES / "Figure2_Global_GDP_Fan_v5_Final.png"
fig.savefig(SAVE_PATH, dpi=600)
print(f"Saved Figure 2 to {SAVE_PATH}")
plt.show()
