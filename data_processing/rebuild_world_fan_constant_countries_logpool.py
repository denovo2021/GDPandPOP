# rebuild_world_fan_constant_countries_logpool.py
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE, DIR_OUTPUT

GCSV = PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE  # country-level age-augmented forecasts
OUT = DIR_OUTPUT / "gdp_world_fan_rcs_age_constISO3_logpool.csv"

# 1) load and coerce
g = pd.read_csv(GCSV, usecols=["ISO3","Scenario","Year","Pred_Median"])
g["Year"] = pd.to_numeric(g["Year"], errors="coerce").astype("Int64")
g = g.dropna(subset=["ISO3","Year","Pred_Median"]).assign(Year=lambda d: d["Year"].astype(int))

yr_min, yr_max = 2025, 2100
g = g[g["Year"].between(yr_min, yr_max)]

# 2) for each scenario, find the ISO3 set present in *all* years (constant country set)
const_series = []
for scen, sub in g.groupby("Scenario"):
    by_year = sub.groupby("Year")["ISO3"].apply(set)
    common_iso3 = set.intersection(*by_year.tolist()) if len(by_year) else set()
    if not common_iso3:
        continue
    const_series.append((scen, common_iso3))
const_map = {s:iso3s for s, iso3s in const_series}

if not const_map:
    raise RuntimeError("No scenario has a constant ISO3 set across years; check upstream data.")

# 3) compute per-scenario world totals using only the constant ISO3 set,
#    then pool across scenarios *on the log scale* to get fan bands
world_wide = {}
for scen, iso3s in const_map.items():
    sub = g[(g["Scenario"]==scen) & (g["ISO3"].isin(iso3s))].copy()
    world = (sub.groupby("Year", as_index=False)["Pred_Median"].sum()
               .set_index("Year")
               .reindex(range(yr_min, yr_max+1)))
    world_wide[scen] = world["Pred_Median"].to_numpy()

# shape to (n_years, n_scenarios_const)
scens = sorted(world_wide.keys())
arr   = np.column_stack([world_wide[s] for s in scens])  # (n_years, n_scenarios)

# log-space pooling to damp outliers
log_arr = np.log10(np.clip(arr, 1.0, None))
fan = pd.DataFrame({
    "Year": np.arange(yr_min, yr_max+1, dtype=int),
    "Median":  10**np.nanmedian(log_arr, axis=1),
    "p50_lo":  10**np.nanquantile(log_arr, 0.25, axis=1),
    "p50_hi":  10**np.nanquantile(log_arr, 0.75, axis=1),
    "p80_lo":  10**np.nanquantile(log_arr, 0.10, axis=1),
    "p80_hi":  10**np.nanquantile(log_arr, 0.90, axis=1),
    "p95_lo":  10**np.nanquantile(log_arr, 0.025, axis=1),
    "p95_hi":  10**np.nanquantile(log_arr, 0.975, axis=1),
})
fan.to_csv(OUT, index=False)
print("✓ world fan (const ISO3 per scenario, log-pooled) →", OUT)

# diagnostics
# how many scenarios survived the const-ISO3 filter?
print("[fan] scenarios used:", len(scens), "|", scens[:8], "...")
# print change in #ISO3 per scenario (should be constant across years now)
for s in scens[:3]:
    print(f"[{s}] const ISO3 count:", len(const_map[s]))
