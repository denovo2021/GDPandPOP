# table_u5mr_regions.py — regional U5MR medians (95% CI) for 2035/2050/2100
# Population-weighted within region & scenario, then quantiles across scenarios.

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import DIR_OUTPUT, PATH_POP_PREDICTIONS

# -------- load forecasts (country-level) and population cache --------
u5 = pd.read_csv(DIR_OUTPUT / "u5mr_predictions_scenarios_rcs.csv",
                 usecols=["ISO3", "Region", "Year", "Scenario", "U5MR_median"])
pop = pd.read_csv(PATH_POP_PREDICTIONS,
                  usecols=["ISO3", "Year", "Scenario", "Population"])

# numeric coercion
for d, cols in [(u5, ["Year","U5MR_median"]), (pop, ["Year","Population"])]:
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")

u5  = u5.dropna(subset=["Region","Year","U5MR_median"])
pop = pop.dropna(subset=["Year","Population"])

# -------- merge & compute regional weighted U5MR per scenario-year --------
u5w = (u5.merge(pop, on=["ISO3","Year","Scenario"], how="left")
         .dropna(subset=["Population"]))

# weighted sum (no groupby.apply → no deprecation)
wx = (u5w.assign(wx = u5w["U5MR_median"] * u5w["Population"])
         .groupby(["Region","Scenario","Year"], as_index=False)["wx"].sum())
W  = (u5w.groupby(["Region","Scenario","Year"], as_index=False)["Population"].sum()
         .rename(columns={"Population":"W"}))
reg_year = (wx.merge(W, on=["Region","Scenario","Year"])
              .assign(U5MR_reg=lambda d: d["wx"]/d["W"])
              .loc[:, ["Region","Scenario","Year","U5MR_reg"]])

# -------- take quantiles across scenarios at target years --------
YEARS = [2035, 2050, 2100]

def q(a, p):
    a = np.asarray(a); a = a[np.isfinite(a)]
    return np.quantile(a, p) if a.size else np.nan

rows = []
for (region, year), g in reg_year[reg_year["Year"].isin(YEARS)].groupby(["Region","Year"]):
    s = g["U5MR_reg"]
    rows.append({
        "Region": region, "Year": int(year),
        "Median": q(s, 0.50),
        "Lower":  q(s, 0.025),
        "Upper":  q(s, 0.975),
        "Scenarios_count": int(s.notna().sum())
    })

tab = pd.DataFrame(rows)

# pivot to Region × (Median/Lower/Upper) by year, like your previous output
table_wide = (tab.pivot(index="Region", columns="Year", values=["Median","Lower","Upper"])
                .sort_index())

# optional sanity print
print(table_wide.head())
print("[info] scenario counts by Region×Year (should be ≤ number of scenarios present):")
print(tab.pivot(index="Region", columns="Year", values="Scenarios_count").fillna(0).astype(int).head())

# save
OUT = DIR_OUTPUT / "table_u5mr_regions_2035_2050_2100.csv"
table_wide.to_csv(OUT)
print(f"✓ {OUT}")
