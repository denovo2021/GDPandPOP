# late_year_diagnostics.py
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
g = pd.read_csv(ROOT/"gdp_predictions_scenarios_rcs_age.csv",
                usecols=["ISO3","Scenario","Year","Pred_Median"])

for y in (2085, 2090, 2095, 2100):
    world = (g[g["Year"]==y]
             .groupby("Scenario", as_index=False)["Pred_Median"].sum()
             .sort_values("Pred_Median", ascending=False))
    print(f"\n[World totals at {y}]")
    print(world.head(10))
