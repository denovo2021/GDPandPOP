# ingest_health_data.py
# ---------------------------------------------------------------------------
# Merge GHED (2000+), UHC SCI (mostly 2000+), and UNICEF U5MR (1960+) into
# the historical panel. Output: merged_health.csv (ISO3 + Year key) at project root.
# ---------------------------------------------------------------------------
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ---- paths -------------------------------------------------------------------
PROJ = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")          # project root
DATA = PROJ / "BasicData"                                                 # data dir

# ---- base panel (must contain Country Code/Name, Year, GDP, Population) ------
base = (pd.read_csv(PROJ/"merged.csv", index_col=0)
          .rename(columns={"Country Code": "ISO3"})
          .assign(Year=lambda d: d["Year"].astype(int)))

# ============================== 1) GHED (WHO) =================================
# File: BasicData/GHED_data.xlsx  (sheet 'Data')
ghed_xls = DATA / "GHED_data.xlsx"
ghed = (pd.read_excel(ghed_xls, sheet_name="Data")
          .rename(columns={
              "code": "ISO3", "year": "Year",
              "che_pc_usd": "ghed_pc_usd",           # per-capita USD
              "che_gdp":    "ghed_gdp_share"         # % of GDP (optional)
          }))
ghed = (ghed.loc[:, ["ISO3","Year","ghed_pc_usd","ghed_gdp_share"]]
            .dropna(subset=["ISO3","Year"]))
ghed["Year"] = ghed["Year"].astype(int)
ghed = (ghed.groupby(["ISO3","Year"], as_index=False)
            .agg({"ghed_pc_usd": "median", "ghed_gdp_share": "median"}))

# ============================== 2) UHC SCI (WHO) ==============================
# File: BasicData/UHC_SERVICE_COVERAGE_INDEX.csv (long)
uhc_raw = pd.read_csv(DATA/"UHC_SERVICE_COVERAGE_INDEX.csv")

# robust column mapping
cm = {c.lower(): c for c in uhc_raw.columns}
iso_col  = cm.get("spatialdimvaluecode", "SpatialDimValueCode")
year_col = cm.get("period", "Period")
val_col  = cm.get("factvaluenumeric", "FactValueNumeric")
loc_type = cm.get("location type", "Location type")

uhc = (uhc_raw.rename(columns={iso_col:"ISO3", year_col:"Year", val_col:"uhc_sci"})
               .loc[:, ["ISO3","Year","uhc_sci"] + ([loc_type] if loc_type in uhc_raw.columns else [])]
               .dropna(subset=["ISO3","Year","uhc_sci"]))
if loc_type in uhc.columns:
    uhc = uhc[uhc[loc_type].str.contains("Country", case=False, na=False)]
uhc["Year"] = uhc["Year"].astype(int)
uhc = uhc.groupby(["ISO3","Year"], as_index=False)["uhc_sci"].median()

# ============================== 3) U5MR (UNICEF/IGME) =========================
# File: BasicData/UNIGME-2024-Total-U5MR-IMR-and-NMR-database.xlsx (sheet 'Total U5MR')
u5_path = DATA / "UNIGME-2024-Total-U5MR-IMR-and-NMR-database.xlsx"
if u5_path.suffix.lower() in [".xlsx", ".xls"]:
    u5_raw = pd.read_excel(u5_path, sheet_name="Total U5MR", header=2)
else:
    u5_raw = pd.read_csv(u5_path, skiprows=2)

cm = {c.lower(): c for c in u5_raw.columns}
iso3_c = cm.get("country.iso", "Country.ISO")
year_c = cm.get("series.year", "Series.Year")
est_c  = cm.get("estimates", "Estimates")
sex_c  = cm.get("sex", "Sex")
ind_c  = cm.get("indicator", "Indicator")

u5 = (u5_raw.rename(columns={iso3_c:"ISO3", year_c:"Year", est_c:"u5mr_per_1000",
                             sex_c:"Sex", ind_c:"Indicator"})
             .dropna(subset=["ISO3","Year","u5mr_per_1000"]))

# keep Total sex and Under-five indicator if mixed
if "Sex" in u5.columns:
    u5 = u5[u5["Sex"].str.contains("Total", case=False, na=False)]
if "Indicator" in u5.columns:
    u5 = u5[u5["Indicator"].str.contains("Under-five|U5MR", case=False, na=False)]

# normalize Year like "2022-2023" → 2023 (end year)
def normalize_year_to_end(v):
    if pd.isna(v): return np.nan
    s = str(v).strip()
    s = (s.replace("\u2010","-").replace("\u2011","-").replace("\u2012","-")
           .replace("\u2013","-").replace("\u2014","-").replace("\u2212","-")
           .replace("/", "-").replace(" to ", "-").replace("–", "-").replace("—", "-"))
    s = re.sub(r"[^0-9\-]", "", s)
    years = re.findall(r"(?:19|20)\d{2}", s)
    if years: return int(years[-1])
    m = re.search(r"\d{4}", s)
    return int(m.group()) if m else np.nan

u5["Year"] = u5["Year"].apply(normalize_year_to_end).astype("Int64")
u5 = u5.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))
u5["u5mr_per_1000"] = pd.to_numeric(u5["u5mr_per_1000"], errors="coerce")
u5 = u5.dropna(subset=["u5mr_per_1000"])
u5 = u5.groupby(["ISO3","Year"], as_index=False)["u5mr_per_1000"].median()

# ============================== 4) Merge into historical panel =================
panel = (base
         .merge(ghed, on=["ISO3","Year"], how="left")
         .merge(uhc,  on=["ISO3","Year"], how="left")
         .merge(u5,   on=["ISO3","Year"], how="left"))

# ------------------------------ sanity checks ---------------------------------
def check_stats(df, col):
    # min, median, max, missing% on the full column
    s = pd.to_numeric(df[col], errors="coerce")
    return (float(np.nanmin(s)) if s.notna().any() else np.nan,
            float(np.nanmedian(s)) if s.notna().any() else np.nan,
            float(np.nanmax(s)) if s.notna().any() else np.nan,
            float(s.isna().mean()))

print("[check] GHED_pc_usd (min, med, max, missing%):", check_stats(panel, "ghed_pc_usd"))
print("[check] UHC_SCI     (min, med, max, missing%):", check_stats(panel, "uhc_sci"))
print("[check] U5MR/1000   (min, med, max, missing%):", check_stats(panel, "u5mr_per_1000"))

# ------------------------------ save ------------------------------------------
out_path = PROJ / "merged_health.csv"
panel.to_csv(out_path, index=False)
print("✓ merged_health.csv written →", out_path)
