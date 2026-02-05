# fig_global_gdp_fan.py  — robust, pivot + quantiles
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATH_GDP_PREDICTIONS_RCS

df = pd.read_csv(PATH_GDP_PREDICTIONS_RCS,
                 usecols=["Scenario", "Year", "Pred_Median"])

# 1) force numeric
df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
df["Pred_Median"] = pd.to_numeric(df["Pred_Median"], errors="coerce")
df = df.dropna(subset=["Year","Pred_Median"])

# 2) world total per scenario-year
world = df.groupby(["Year","Scenario"], as_index=False)["Pred_Median"].sum()

# 3) pivot to wide (Year × Scenario) and compute quantiles across scenarios
wide = world.pivot(index="Year", columns="Scenario", values="Pred_Median").sort_index()
arr  = wide.to_numpy()  # shape (n_years, n_scenarios)

fan = pd.DataFrame({
    "Year": wide.index.astype(int),
    "Median":   np.nanmedian(arr, axis=1),
    "p50_lo":   np.nanquantile(arr, 0.25, axis=1),
    "p50_hi":   np.nanquantile(arr, 0.75, axis=1),
    "p80_lo":   np.nanquantile(arr, 0.10, axis=1),
    "p80_hi":   np.nanquantile(arr, 0.90, axis=1),
    "p95_lo":   np.nanquantile(arr, 0.025, axis=1),
    "p95_hi":   np.nanquantile(arr, 0.975, axis=1),
})

# 4) trillions for display
for c in fan.columns[1:]:
    fan[c] = fan[c] / 1e12

# 5) plot
fig, ax = plt.subplots(figsize=(6.2, 4))
ax.plot(fan["Year"], fan["Median"], lw=2, label="Median")
ax.fill_between(fan["Year"], fan["p95_lo"], fan["p95_hi"], alpha=0.15, label="95%")
ax.fill_between(fan["Year"], fan["p80_lo"], fan["p80_hi"], alpha=0.25, label="80%")
ax.fill_between(fan["Year"], fan["p50_lo"], fan["p50_hi"], alpha=0.35, label="50%")
ax.set_xlim(int(fan["Year"].min()), int(fan["Year"].max()))
ax.set_xlabel("Year"); ax.set_ylabel("Global GDP (USD trillions)")
ax.yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:,.0f}"))
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(ROOT/"fig_global_gdp_fan.png", dpi=600)
print("✓ fig_global_gdp_fan.png")
