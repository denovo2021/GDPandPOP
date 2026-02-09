# --------------------------- plot_fanchart.py ---------------------------------
"""
Draw global-GDP fan chart (50 % / 80 % / 95 %) from
gdp_predictions_scenarios.csv.

Run after prediction_fixed.py has produced that CSV.
"""
# ------------------------------------------------------------------- imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ------------------------------------------------------------------- paths -----
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PATH_GDP_PREDICTIONS_RCS, DIR_FIGURES

SCEN_CSV = PATH_GDP_PREDICTIONS_RCS
OUT_FIG = DIR_FIGURES / "fig_global_fanchart_rcs.png"

# ------------------------------------------------ 1. load per-scenario totals --
df = pd.read_csv(SCEN_CSV, usecols=["Scenario", "Year", "Pred_Median"])
df = df[~df["Scenario"].isin(["High variant", "Low variant"])]   # drop 2 variants

# ------------------------------------------------ 2. build trimmed-fan dataframe
def trimmed_quantile(values: np.ndarray, q: float, trim: float = 0.1) -> float:
    """
    Return q-quantile after dropping trim% of the smallest *and* largest values.
    Works on a 1-D numpy array; NaNs and ±inf are ignored automatically.
    """
    vals = np.sort(values[np.isfinite(values)])
    k    = int(trim * len(vals))
    if len(vals) - 2 * k <= 0:          # not enough points after trimming
        return np.nan
    return np.quantile(vals[k : len(vals) - k], q)

# aggregate country medians → scenario totals
totals = (df.groupby(["Scenario", "Year"], as_index=False)["Pred_Median"]
            .sum()
            .rename(columns={"Pred_Median": "Global_Median"}))

years = np.sort(totals["Year"].unique())
bands = []

for y in years:
    arr = totals.loc[totals["Year"] == y, "Global_Median"].values
    bands.append({
        "Year":   y,
        "Median": trimmed_quantile(arr, 0.50, trim=0.0),   # no trim for median
        "p50_lo": trimmed_quantile(arr, 0.25, trim=0.05),
        "p50_hi": trimmed_quantile(arr, 0.75, trim=0.05),
        "p80_lo": trimmed_quantile(arr, 0.10, trim=0.05),
        "p80_hi": trimmed_quantile(arr, 0.90, trim=0.05),
        "p95_lo": trimmed_quantile(arr, 0.025, trim=0.05),
        "p95_hi": trimmed_quantile(arr, 0.975, trim=0.05),
    })

fan = pd.DataFrame(bands)

# convert to trillions
for col in fan.columns[1:]:
    fan[col] /= 1e12

# ---------- 3. plot: global GDP fan chart on a linear y-axis -------------------
import matplotlib.ticker as mtick

fig, ax = plt.subplots(figsize=(6, 4))

ax.plot(fan["Year"], fan["Median"], lw=2, label="median")

# lightest band first (95 % → 50 %)
ax.fill_between(fan["Year"], fan["p95_lo"], fan["p95_hi"],
                alpha=0.15, label="95 % interval")
ax.fill_between(fan["Year"], fan["p80_lo"], fan["p80_hi"],
                alpha=0.25, label="80 % interval")
ax.fill_between(fan["Year"], fan["p50_lo"], fan["p50_hi"],
                alpha=0.35, label="50 % interval")

ax.set_xlabel("Year")
ax.set_ylabel("Global GDP (USD trillions)")

# ------- linear-scale formatting ---------------
ax.yaxis.set_major_formatter(
    mtick.FuncFormatter(lambda v, _ : f"{v:,.0f}")
)

ax.set_xlim(fan["Year"].min(), fan["Year"].max())
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(OUT_FIG, dpi=600)
print(f"✓ Fan chart saved → {OUT_FIG}")


