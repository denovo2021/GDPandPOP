# diagnose_world_fan_coverage.py
import pandas as pd
from pathlib import Path

ROOT = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
g = pd.read_csv(ROOT/"gdp_predictions_scenarios_rcs_age.csv",
                usecols=["Scenario","Year","Pred_Median"])

cov = (g.groupby("Year")["Scenario"].nunique()
         .rename("n_variants")
         .reset_index())
print(cov.tail(20))  # check late years
