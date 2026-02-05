# add_age_to_merged.py
import numpy as np, pandas as pd
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DIR_DATA, PATH_MERGED, PATH_MERGED_AGE, PATH_AGE_PREDICTIONS

MERGED_IN  = PATH_MERGED                      # your base panel
AGE_LONG   = PATH_AGE_PREDICTIONS             # age predictions scenarios
MERGED_OUT = PATH_MERGED_AGE                  # new training panel with age

base = pd.read_csv(MERGED_IN, index_col=0).rename(columns={"Country Code":"ISO3"})
base["Year"] = pd.to_numeric(base["Year"], errors="coerce").astype("Int64")
base = base.dropna(subset=["ISO3","Year"]).assign(Year=lambda d: d["Year"].astype(int))

age = pd.read_csv(AGE_LONG)
age = age[age["Scenario"].str.contains("Estimate", case=False, na=False)].copy()
age = age[["ISO3","Year","WAshare","OldDep"]]

panel = base.merge(age, on=["ISO3","Year"], how="left")
panel.to_csv(MERGED_OUT, index=False)
print("[OK] merged_age.csv written ->", MERGED_OUT)
