# diagnose_world_fan_coverage.py
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE

g = pd.read_csv(PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE,
                usecols=["Scenario", "Year", "Pred_Median"])

cov = (g.groupby("Year")["Scenario"].nunique()
         .rename("n_variants")
         .reset_index())
print(cov.tail(20))  # check late years
