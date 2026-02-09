# fig_global_gdp_fan_age.py
# Updated Figure 2 using age-augmented world fan (50/80/95%)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import sys
from pathlib import Path

# ---------------- paths ----------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DIR_OUTPUT, DIR_FIGURES

FAN_AGE = DIR_OUTPUT / "gdp_world_fan_rcs_age_constISO3_logpool.csv"  # produced by prediction_rcs_age.py
# (optional) baseline fan CSV, if you want a faint overlay for comparison
FAN_BASE_OPT = DIR_OUTPUT / "fig_global_gdp_fan.csv"  # or your baseline world-fan CSV; set to None if absent

OUT_PNG = DIR_FIGURES / "fig_global_gdp_fan_age.png"

# ---------------- load fan (age) ----------------
fan = pd.read_csv(FAN_AGE)

# coerce numerics safely
fan["Year"] = pd.to_numeric(fan["Year"], errors="coerce").astype("Int64")
num_cols = [c for c in fan.columns if c != "Year"]
fan[num_cols] = fan[num_cols].apply(pd.to_numeric, errors="coerce")

# drop rows without year and sort
fan = fan.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int)).sort_values("Year")

# convert to trillions if still in USD units (heuristic: values are large)
# (if your CSV is already in trillions, this division will make them too small—set DIV=1.0 then)
DIV = 1e12
for c in num_cols:
    fan[c] = fan[c] / DIV

# ---------------- optional: load baseline for overlay ----------------
base = None
if FAN_BASE_OPT is not None and FAN_BASE_OPT.exists():
    base = pd.read_csv(FAN_BASE_OPT)
    if "Year" in base.columns and "Median" in base.columns:
        base["Year"] = pd.to_numeric(base["Year"], errors="coerce").astype("Int64")
        base = base.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int)).sort_values("Year")
        # if baseline not already in trillions, coerce (same heuristic)
        if base["Median"].max() > 1e6:  # likely in USD
            for c in [c for c in base.columns if c != "Year"]:
                base[c] = pd.to_numeric(base[c], errors="coerce") / DIV

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(6.4, 4.2))

# main age-augmented fan
ax.plot(fan["Year"], fan["Median"], lw=2.2, label="Median (age-augmented)")
ax.fill_between(fan["Year"], fan["p95_lo"], fan["p95_hi"], alpha=0.15, label="95%")
ax.fill_between(fan["Year"], fan["p80_lo"], fan["p80_hi"], alpha=0.25, label="80%")
ax.fill_between(fan["Year"], fan["p50_lo"], fan["p50_hi"], alpha=0.35, label="50%")

# (optional) faint baseline overlay
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
