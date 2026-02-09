# rebuild_world_fan_constant_set.py
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE, PATH_GDP_WORLD_FAN_CONSTANT

g = pd.read_csv(PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE,
                usecols=["Scenario", "Year", "Pred_Median"])

# 1) find a constant set of scenarios present in ALL years (2025–2100)
yr_min, yr_max = 2025, 2100
yrs = pd.Series(range(yr_min, yr_max+1))
by_year = (g[g["Year"].between(yr_min, yr_max)]
             .groupby("Year")["Scenario"].apply(set))
common = set.intersection(*by_year.tolist())
print("[fan] scenarios used in ALL years:", sorted(common))
if not common:
    raise RuntimeError("No common scenario set across years; check upstream files.")

# 2) restrict to the constant set and sum to world per scenario-year
g2 = g[g["Scenario"].isin(common)].copy()
world = (g2.groupby(["Year","Scenario"], as_index=False)["Pred_Median"].sum()
           .pivot(index="Year", columns="Scenario", values="Pred_Median")
           .reindex(index=range(yr_min, yr_max+1))
           .sort_index())

arr = world.to_numpy()  # (n_years, n_scenarios)

def q(a, q): a=np.asarray(a); a=a[np.isfinite(a)]; return np.quantile(a, q) if a.size else np.nan
fan = pd.DataFrame({
    "Year": world.index.astype(int),
    "Median":  np.nanmedian(arr, axis=1),
    "p50_lo":  np.nanquantile(arr, 0.25, axis=1),
    "p50_hi":  np.nanquantile(arr, 0.75, axis=1),
    "p80_lo":  np.nanquantile(arr, 0.10, axis=1),
    "p80_hi":  np.nanquantile(arr, 0.90, axis=1),
    "p95_lo":  np.nanquantile(arr, 0.025, axis=1),
    "p95_hi":  np.nanquantile(arr, 0.975, axis=1),
})

# 3) save fan and plot (optional)
fan.to_csv(PATH_GDP_WORLD_FAN_CONSTANT, index=False)
print("✓ world fan (constant scenarios) →", PATH_GDP_WORLD_FAN_CONSTANT)
