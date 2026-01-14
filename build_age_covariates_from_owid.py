# build_age_covariates_from_owid.py
import re
import numpy as np
import pandas as pd
from pathlib import Path

PROJ = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
IN   = PROJ/"BasicData/population-by-age-group-with-projections/population-by-age-group-with-projections.csv"
OUT  = PROJ/"age_predictions_scenarios.csv"
ANCH = PROJ/"age_base_anchors.csv"

def norm(s:str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

# load
df = pd.read_csv(IN)

# For clarity, rename the relevant columns
col_all_est = "Population - Sex: all - Age: all - Variant: estimates"
col_all_med = "Population - Sex: all - Age: all - Variant: medium"
col_65_est  = "Population - Sex: all - Age: 65+ - Variant: estimates"
col_65_med  = "Population - Sex: all - Age: 65+ - Variant: medium"
col_25_64_e = "Population - Sex: all - Age: 25-64 - Variant: estimates"
col_25_64_m = "Population - Sex: all - Age: 25-64 - Variant: medium"
col_0_24_e  = "Population - Sex: all - Age: 0-24 - Variant: estimates"
col_0_24_m  = "Population - Sex: all - Age: 0-24 - Variant: medium"
col_0_14_e  = "Population - Sex: all - Age: 0-14 - Variant: estimates"
col_0_14_m  = "Population - Sex: all - Age: 0-14 - Variant: medium"

# Coerce numerics
num_cols = [col_all_est, col_all_med, col_65_est, col_65_med,
            col_25_64_e, col_25_64_m, col_0_24_e, col_0_24_m,
            col_0_14_e, col_0_14_m]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# helper to compute WAshare and OldDep for either "estimates" or "medium"
def compute_variant(df_in, variant="estimates"):
    if variant == "estimates":
        total = df_in[col_all_est]
        age65 = df_in[col_65_est]
        a25_64 = df_in[col_25_64_e]
        a0_24  = df_in[col_0_24_e]
        a0_14  = df_in[col_0_14_e]
        scen   = "Estimates"
    else:
        total = df_in[col_all_med]
        age65 = df_in[col_65_med]
        a25_64 = df_in[col_25_64_m]
        a0_24  = df_in[col_0_24_m]
        a0_14  = df_in[col_0_14_m]
        scen   = "Medium variant"

    # 15–24 = (0–24) – (0–14)
    a15_24 = (a0_24 - a0_14)
    # 15–64 = (25–64) + (15–24)
    a15_64 = (a25_64 + a15_24)
    # Shares
    WAshare = (a15_64 / total).replace([np.inf,-np.inf], np.nan)
    OldDep  = (age65   / a15_64).replace([np.inf,-np.inf], np.nan)

    out = df_in[["Code","Year"]].copy().rename(columns={"Code":"ISO3"})
    out["Scenario"]      = scen
    out["scenario_norm"] = norm(scen)
    out["WAshare"] = WAshare
    out["OldDep"]  = OldDep
    return out.dropna(subset=["ISO3","Year","WAshare","OldDep"])

# build both variants present in OWID (estimates for history, medium for projections)
age_est = compute_variant(df, "estimates")
age_med = compute_variant(df, "medium")

age = pd.concat([age_est, age_med], ignore_index=True)
age["ISO3"] = age["ISO3"].str.upper()
age["Year"] = pd.to_numeric(age["Year"], errors="coerce").astype("Int64")
age = age.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))

# Save long file (history+projections)
age = age.drop_duplicates(["ISO3","Year","Scenario"])
age.to_csv(OUT, index=False)
print("✓ age_predictions_scenarios.csv written →", OUT)

# Create base anchors (use 2023 if available, otherwise latest ≤2023)
base_rows = []
for iso, g in age[age["Year"]<=2023].groupby("ISO3"):
    gb = g.sort_values("Year")
    if gb.empty: continue
    row = gb.iloc[-1]  # last available ≤2023
    base_rows.append({"ISO3": iso, "Year_base": int(row["Year"]),
                      "WAshare_base": float(row["WAshare"]),
                      "OldDep_base":  float(row["OldDep"])})
anchors = pd.DataFrame(base_rows)
anchors.to_csv(ANCH, index=False)
print("✓ age_base_anchors.csv written →", ANCH)
