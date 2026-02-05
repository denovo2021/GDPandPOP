# build_age_covariates_from_owid.py
# Updated to work with OWID "Population by age group to 2100" file format
import re
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DIR_DATA, PATH_AGE_PREDICTIONS, PATH_MERGED

# Input/Output paths
IN = DIR_DATA / "Population by age group to 2100 (based on UNWPP, 2017 medium scenario).csv"
OUT = PATH_AGE_PREDICTIONS
ANCH = DIR_DATA / "age_base_anchors.csv"

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

# Load OWID data
print(f"Loading OWID data from: {IN}")
df = pd.read_csv(IN)

# Column names in this file format
col_under15 = "Under 15 years old (UNWPP, 2017)"
col_working = "Working age (15-64 years old) (UNWPP, 2017)"
col_65plus = "65+ years old (UNWPP, 2017)"

# Coerce numerics
for c in [col_under15, col_working, col_65plus]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Build country name to ISO3 mapping from merged.csv
merged = pd.read_csv(PATH_MERGED, index_col=0)
name_to_iso3 = merged.drop_duplicates("Country Name").set_index("Country Name")["Country Code"].to_dict()

# Add some common name variations
name_variations = {
    "United States": "USA",
    "United Kingdom": "GBR",
    "South Korea": "KOR",
    "Russia": "RUS",
    "Iran": "IRN",
    "Syria": "SYR",
    "Venezuela": "VEN",
    "Bolivia": "BOL",
    "Tanzania": "TZA",
    "Vietnam": "VNM",
    "Laos": "LAO",
    "Democratic Republic of Congo": "COD",
    "Republic of Congo": "COG",
    "Ivory Coast": "CIV",
    "Cote d'Ivoire": "CIV",
    "Czech Republic": "CZE",
    "Czechia": "CZE",
    "Slovakia": "SVK",
    "Timor": "TLS",
    "East Timor": "TLS",
    "Timor-Leste": "TLS",
    "Brunei": "BRN",
    "Cape Verde": "CPV",
    "Micronesia (country)": "FSM",
    "Eswatini": "SWZ",
    "Swaziland": "SWZ",
    "North Macedonia": "MKD",
    "Macedonia": "MKD",
    "Palestine": "PSE",
    "Taiwan": "TWN",
    "Hong Kong": "HKG",
    "Macao": "MAC",
    "Macau": "MAC",
}
name_to_iso3.update(name_variations)

# Map Entity to ISO3
df["ISO3"] = df["Entity"].map(name_to_iso3)

# Filter to only countries with valid ISO3 (excludes regions like "Africa", "Asia", etc.)
df = df.dropna(subset=["ISO3"])
print(f"Matched {df['ISO3'].nunique()} countries with ISO3 codes")

# Compute total population and age shares
df["Total"] = df[col_under15] + df[col_working] + df[col_65plus]

# WAshare = working age (15-64) / total
# OldDep = 65+ / working age (15-64)
df["WAshare"] = (df[col_working] / df["Total"]).replace([np.inf, -np.inf], np.nan)
df["OldDep"] = (df[col_65plus] / df[col_working]).replace([np.inf, -np.inf], np.nan)

# This dataset only has medium variant projections, label accordingly
# Years <= 2017 are estimates, years > 2017 are projections
df["Scenario"] = df["Year"].apply(lambda y: "Estimates" if y <= 2017 else "Medium variant")
df["scenario_norm"] = df["Scenario"].apply(norm)

# Select output columns
age = df[["ISO3", "Year", "Scenario", "scenario_norm", "WAshare", "OldDep"]].copy()
age["Year"] = pd.to_numeric(age["Year"], errors="coerce").astype("Int64")
age = age.dropna(subset=["Year", "WAshare", "OldDep"]).assign(Year=lambda d: d["Year"].astype(int))

# Save long file (history+projections)
age = age.drop_duplicates(["ISO3", "Year", "Scenario"])
age.to_csv(OUT, index=False)
print(f"[OK] age_predictions_scenarios.csv written -> {OUT}")
print(f"  Rows: {len(age)}, Countries: {age['ISO3'].nunique()}, Years: {age['Year'].min()}-{age['Year'].max()}")

# Create base anchors (use 2017 if available, otherwise latest <=2017 for estimates)
base_rows = []
for iso, g in age[age["Year"] <= 2017].groupby("ISO3"):
    gb = g.sort_values("Year")
    if gb.empty:
        continue
    row = gb.iloc[-1]  # last available <=2017
    base_rows.append({
        "ISO3": iso,
        "Year_base": int(row["Year"]),
        "WAshare_base": float(row["WAshare"]),
        "OldDep_base": float(row["OldDep"])
    })
anchors = pd.DataFrame(base_rows)
anchors.to_csv(ANCH, index=False)
print(f"[OK] age_base_anchors.csv written -> {ANCH}")
