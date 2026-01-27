# u5mr_forecast_from_gdp.py  — robust version (uses WPP if Population missing),
#                              supports optional time drift 'gamma' in elasticity model
import re
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path

ROOT = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
WPP_XLSX = ROOT / "BasicData/UN/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_FULL.xlsx"

# ---------- 0) helpers ----------
def norm(s: str) -> str:
    """normalize scenario/sheet names for robust matching"""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())

def read_wpp_population(wpp_xlsx: Path, wanted_norms: set[str]) -> pd.DataFrame:
    """Read WPP sheets and return long DF with ISO3, Year, Scenario, Population, scenario_norm."""
    xl = pd.ExcelFile(wpp_xlsx)
    pop_frames = []
    for sheet in xl.sheet_names:
        s_norm = norm(sheet)
        if ("variant" in sheet.lower()) or ("fertility" in sheet.lower()) or (s_norm in wanted_norms):
            try:
                df = xl.parse(
                    sheet, header=16,
                    usecols=["Year", "ISO3 Alpha-code",
                             "Total Population, as of 1 July (thousands)"]
                ).rename(columns={
                    "ISO3 Alpha-code": "ISO3",
                    "Total Population, as of 1 July (thousands)": "Population_thousands"
                })
            except Exception:
                df = xl.parse(sheet, header=16)
                col_iso = [c for c in df.columns if "ISO3" in c][0]
                col_pop = [c for c in df.columns if "Population" in c and "thousand" in c.lower()][0]
                df = df.rename(columns={col_iso:"ISO3", col_pop:"Population_thousands"})
            df = df.dropna(subset=["ISO3", "Year", "Population_thousands"]).copy()
            df["Year"] = df["Year"].astype(int)
            df["Population"] = pd.to_numeric(df["Population_thousands"], errors="coerce") * 1_000
            df = df.dropna(subset=["Population"])
            df["Scenario"] = sheet
            df["scenario_norm"] = s_norm
            pop_frames.append(df[["ISO3","Year","Scenario","scenario_norm","Population"]])
    if not pop_frames:
        raise RuntimeError("No usable WPP sheets were found. Check WPP_XLSX path/sheet names.")
    return pd.concat(pop_frames, ignore_index=True)

# ---------- 1) load scenario GDP (future) ----------
scen = pd.read_csv(ROOT / "gdp_predictions_scenarios_rcs.csv")  # needs: ISO3, Year, Scenario, Pred_Median
for c in ["ISO3","Year","Scenario","Pred_Median"]:
    if c not in scen.columns:
        raise RuntimeError(f"Missing column in gdp_predictions_scenarios_rcs.csv: {c}")

scen["Year"] = scen["Year"].astype(int)
scen["scenario_norm"] = scen["Scenario"].apply(norm)

# ---------- 2) ensure Population; then compute GDPpc ----------
if "Population" not in scen.columns:
    wanted_norms = set(scen["scenario_norm"].unique())
    pop_all = read_wpp_population(WPP_XLSX, wanted_norms)
    scen = scen.merge(pop_all[["ISO3","Year","scenario_norm","Population"]],
                      on=["ISO3","Year","scenario_norm"], how="left")

if scen["Population"].isna().any():
    miss = scen[scen["Population"].isna()][["ISO3","Year","Scenario"]].head(10)
    raise RuntimeError(f"Population missing for some rows. Example:\n{miss}")

scen["GDPpc"] = scen["Pred_Median"] / scen["Population"]

# ---------- 3) attach Region/Country (coalesce duplicates safely) ----------
meta = pd.read_csv(
    ROOT/"BasicData/API_SP.POP.TOTL_DS2_en_csv_v2_3401680/Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_3401680.csv"
)
reg_col_candidates = [c for c in meta.columns if "Region" in c]
if not reg_col_candidates:
    raise RuntimeError("Region column not found in metadata CSV.")
reg_col = reg_col_candidates[0]
name_col = "TableName" if "TableName" in meta.columns else (
    [c for c in meta.columns if "Table" in c and "Name" in c][0]
)
meta2 = meta.rename(columns={"Country Code":"ISO3", reg_col:"Region", name_col:"Country"})[["ISO3","Region","Country"]]

scen = scen.merge(meta2, on="ISO3", how="left", suffixes=("", "_meta"))
region_cols  = [c for c in scen.columns if c.lower().startswith("region")]
country_cols = [c for c in scen.columns if c.lower().startswith("country")]
scen["Region"]  = scen[region_cols].bfill(axis=1).iloc[:, 0]
scen["Country"] = scen[country_cols].bfill(axis=1).iloc[:, 0]
scen = scen.drop(columns=[c for c in region_cols + country_cols if c not in ["Region","Country"]])

# ---------- 4) filter to regions present in the U5MR elasticity model ----------
# NOTE: path fixed (saved earlier as ROOT/"u5mr_elasticity.nc")
idata_u5 = az.from_netcdf(ROOT / "u5mr_elasticity.nc")
regions = idata_u5.posterior.coords["Region"].values.tolist()
scen = scen[scen["Region"].isin(regions)].copy()

# ---------- 5) posterior draws & forecast (supports optional 'gamma') ----------
a_r = idata_u5.posterior["alpha_r"]                  # (chain, draw, Region)
b   = idata_u5.posterior["beta"]                     # (chain, draw)
has_gamma = "gamma" in idata_u5.posterior.data_vars  # time drift (decade scale)
gma = idata_u5.posterior["gamma"] if has_gamma else None

def forecast_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    reg = df_chunk["Region"].iloc[0]
    gdppc = df_chunk["GDPpc"].values
    log_gdppc = np.log(gdppc.clip(min=1e-6))
    ar = a_r.sel(Region=reg).values.reshape(-1, 1)   # (S,1)
    bS = b.values.reshape(-1, 1)                     # (S,1)

    if has_gamma:
        t_dec = (df_chunk["Year"].values - 2000) / 10.0
        gS = gma.values.reshape(-1, 1)               # (S,1)
        log_u5 = ar + bS * log_gdppc[None, :] + gS * t_dec[None, :]
    else:
        log_u5 = ar + bS * log_gdppc[None, :]

        # median/95% bands on the natural scale without az.hdi
    med = np.exp(np.median(log_u5, axis=0))
    lo  = np.exp(np.quantile(log_u5, 0.025, axis=0))
    hi  = np.exp(np.quantile(log_u5, 0.975, axis=0))
    out = df_chunk[["ISO3","Country","Region","Scenario","Year"]].copy()
    out["U5MR_median"] = med
    out["U5MR_lower"]  = lo
    out["U5MR_upper"]  = hi
    return out

res = []
for reg, g in scen.groupby("Region"):
    g2 = g.sort_values(["ISO3","Year","Scenario"])
    res.append(forecast_chunk(g2))
u5_fore = pd.concat(res, ignore_index=True)
u5_fore.to_csv(ROOT/"u5mr_predictions_scenarios_rcs.csv", index=False)
print("✓ u5mr_predictions_scenarios_rcs.csv written")

# ---------- 6) world fan (population-weighted; use Population from 'scen') ----------
# 'scen' already has Population aligned to the exact Scenario strings used in GDP file
pop_w = scen[["ISO3","Year","Scenario","Population"]].copy()

u5w = (u5_fore.merge(pop_w, on=["ISO3","Year","Scenario"], how="left")
               .dropna(subset=["Population"]))

def wavg(x, w):
    x, w = np.asarray(x), np.asarray(w)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    return np.average(x[m], weights=w[m]) if m.any() else np.nan

# scenario-wise, year-wise population-weighted U5MR
world = (u5w.groupby(["Scenario","Year"])
            .apply(lambda g: wavg(g["U5MR_median"], g["Population"]))
            .reset_index(name="U5MR_world"))

def qtile(a, q):
    a = np.asarray(a); a = a[np.isfinite(a)]
    return np.quantile(a, q) if a.size else np.nan

fan = (world.groupby("Year")["U5MR_world"]
             .apply(lambda s: pd.Series({
                 "Median": qtile(s,0.50), "p50_lo": qtile(s,0.25), "p50_hi": qtile(s,0.75),
                 "p80_lo": qtile(s,0.10), "p80_hi": qtile(s,0.90),
                 "p95_lo": qtile(s,0.025),"p95_hi": qtile(s,0.975),
             }))
             .reset_index())

fan.to_csv(ROOT/"u5mr_world_fan.csv", index=False)
print("✓ u5mr_world_fan.csv written")
# optional diagnostics
print("[world-fan] scenarios counted:", world["Scenario"].nunique())
print("[world-fan] years:", fan["Year"].min(), "→", fan["Year"].max())
print("[GDP scenarios in file]:", sorted(scen["Scenario"].unique()))
