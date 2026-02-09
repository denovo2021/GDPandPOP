# build_pop_predictions.py  — robust WPP → pop_predictions_scenarios.csv
import re
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# ---------------- paths ----------------
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PATH_WPP_XLSX, PATH_POP_PREDICTIONS, PATH_MERGED

WPP_XLSX = PATH_WPP_XLSX
OUT = PATH_POP_PREDICTIONS

# (optional) use the same centering mean as training
MERGED = PATH_MERGED
if MERGED.exists():
    df_mu = pd.read_csv(MERGED, index_col=0)
    if "Log_Population" not in df_mu.columns:
        df_mu["Log_Population"] = np.log10(df_mu["Population"])
    MU_GLOBAL = float(df_mu["Log_Population"].mean())
else:
    MU_GLOBAL = None

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

# Variant allowlist to match GDP scenarios (8 variants; High/Low excluded)
ALLOW = {
    "mediumvariant",
    "constantfertility",
    "instantreplacementzeromigr",
    "momentum",
    "zeromigration",
    "nochange",
    "nofertilitybelowage18",
    "acceleratedabrdecline",        # sometimes appears as this
    "acceleratedabrdeclinerecup",   # or this
}

# Sheet→canonical mapping (handles minor name differences)
CANON = {
    # left: normalized sheet, right: display name you use in GDP CSV
    "mediumvariant":                  "Medium variant",
    "constantfertility":              "Constant-fertility",
    "instantreplacementzeromigr":     "Instant-replacement zero migr",
    "momentum":                       "Momentum",
    "zeromigration":                  "Zero-migration",
    "nochange":                       "No change",
    "nofertilitybelowage18":          "No fertility below age 18",
    "acceleratedabrdecline":          "Accelerated ABR decline",
    "acceleratedabrdeclinerecup":     "Accelerated ABR decline recup",
}

xl = pd.ExcelFile(WPP_XLSX)
rows = []
for sheet in xl.sheet_names:
    s_norm = norm(sheet)
    if (("variant" in sheet.lower()) or ("fertility" in sheet.lower())) and s_norm in ALLOW:
        try:
            df = xl.parse(
                sheet, header=16,
                usecols=["Year","ISO3 Alpha-code","Total Population, as of 1 July (thousands)"]
            ).rename(columns={
                "ISO3 Alpha-code": "ISO3",
                "Total Population, as of 1 July (thousands)": "Pop_thou"
            })
        except Exception:
            df = xl.parse(sheet, header=16)
            col_iso = [c for c in df.columns if "ISO3" in c][0]
            col_pop = [c for c in df.columns if "Population" in c and "thousand" in c.lower()][0]
            df = df.rename(columns={col_iso: "ISO3", col_pop: "Pop_thou"})

        df = df.dropna(subset=["ISO3","Year","Pop_thou"]).copy()
        df["ISO3"] = df["ISO3"].str.upper()
        df["Year"] = df["Year"].astype(int)
        df["Population"] = pd.to_numeric(df["Pop_thou"], errors="coerce") * 1_000
        df = df.dropna(subset=["Population"])

        # Canonical scenario labels (to match GDP file)
        scen_canon = CANON.get(s_norm, sheet)
        df["Scenario"]      = scen_canon
        df["scenario_norm"] = norm(scen_canon)
        rows.append(df[["ISO3","Year","Scenario","scenario_norm","Population"]])

if not rows:
    raise RuntimeError("No WPP variant sheets matched the allowlist; check file/sheet names.")

pop = (pd.concat(rows, ignore_index=True)
         .drop_duplicates(["ISO3","Year","Scenario"], keep="last"))

# Optional: add logs and centered x using the training mean
if MU_GLOBAL is not None:
    pop["Log_Population"] = np.log10(pop["Population"])
    pop["Log_Pop_c"]      = pop["Log_Population"] - MU_GLOBAL

pop.to_csv(OUT, index=False)
print("✓ written →", OUT)

# quick sanity
print(pop.head())
print("[coverage] rows:", len(pop), "| years:", pop["Year"].min(), "→", pop["Year"].max())
print("[coverage] scenarios:", sorted(pop["Scenario"].unique()))
