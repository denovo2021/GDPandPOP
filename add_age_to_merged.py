# add_age_to_merged.py
import numpy as np, pandas as pd
from pathlib import Path

PROJ = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
MERGED_IN  = PROJ/"merged.csv"                # your base panel
AGE_LONG   = PROJ/"age_predictions_scenarios.csv"
MERGED_OUT = PROJ/"merged_age.csv"            # new training panel with age

base = pd.read_csv(MERGED_IN, index_col=0).rename(columns={"Country Code":"ISO3"})
base["Year"] = pd.to_numeric(base["Year"], errors="coerce").astype("Int64")
base = base.dropna(subset=["ISO3","Year"]).assign(Year=lambda d: d["Year"].astype(int))

age = pd.read_csv(AGE_LONG)
age = age[age["Scenario"].str.contains("Estimate", case=False, na=False)].copy()
age = age[["ISO3","Year","WAshare","OldDep"]]

panel = base.merge(age, on=["ISO3","Year"], how="left")
panel.to_csv(MERGED_OUT, index=False)
print("✓ merged_age.csv written →", MERGED_OUT)
