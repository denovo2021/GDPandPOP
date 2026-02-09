# late_year_diagnostics.py
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE

g = pd.read_csv(PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE,
                usecols=["ISO3", "Scenario", "Year", "Pred_Median"])

for y in (2085, 2090, 2095, 2100):
    world = (g[g["Year"]==y]
             .groupby("Scenario", as_index=False)["Pred_Median"].sum()
             .sort_values("Pred_Median", ascending=False))
    print(f"\n[World totals at {y}]")
    print(world.head(10))
