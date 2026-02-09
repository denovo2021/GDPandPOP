# prediction_rcs_age.py
"""
Age-augmented GDP forecasting (anchor-and-delta, tapered tails)
---------------------------------------------------------------
This script produces 2024–2100 GDP projections under UN-variant scenarios, using
the age-augmented hierarchical RCS model:

  log10(GDP_it) = log10(GDP_i,base)
                + β_i * (x_it − x_i0)
                + θ_r(i)^T( s(x_it) − s(x_i0) )
                + δ_WA * (ΔWA_s) + δ_OD * (ΔOD_s)
                + τ_i   * (Δt_s)

Stability: post-2085 tapering in forecast
  • Population-driven deltas (βΔx + θ·Δs) taper from 1.0 → 0.60 (2085→2100)
  • Time drift (τ Δt) tapers from 1.0 → 0.00 (2085→2100)
  • Composition deltas (age) typically untapered

Inputs
  • Posterior: hierarchical_model_with_rcs_age_v2.nc (or _age.nc)
  • Scales:    scale_rcs_age.json  (μ and stds for Δ terms)
  • Scenarios: gdp_predictions_scenarios_rcs.csv  (ISO3, Year, Scenario)
  • Population: pop_predictions_scenarios.csv     (ISO3, Year, Scenario, Population)
  • Age covars: age_predictions_scenarios.csv, age_base_anchors.csv
  • Metadata:   ISO3 → Region, Country (TableName)
  • RCS knots:  rcs_knots_hier.npy

Outputs
  • gdp_predictions_scenarios_rcs_age.csv
  • gdp_world_fan_rcs_age.csv (simple pool; for publication use constant-ISO3 + log pool)
"""
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import arviz as az

# ── paths ────────────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    PATH_MERGED, PATH_GDP_PREDICTIONS_RCS, PATH_POP_PREDICTIONS, PATH_AGE_PREDICTIONS,
    PATH_WB_METADATA, PATH_KNOTS, PATH_SCALE_JSON, PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE,
    DIR_DATA, DIR_OUTPUT
)

IDATA_NC = DIR_OUTPUT / "hierarchical_model_with_rcs_age_v2.nc"  # or _age.nc
MERGED = PATH_MERGED
GSCEN = PATH_GDP_PREDICTIONS_RCS
POPCACHE = PATH_POP_PREDICTIONS
AGECSV = PATH_AGE_PREDICTIONS
AGEANCH = DIR_DATA / "age_base_anchors.csv"
META_CSV = PATH_WB_METADATA
KNOTS_NPY = PATH_KNOTS
SCALE_JSON = PATH_SCALE_JSON

OUT_CNTRY = PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE
OUT_WORLD = DIR_OUTPUT / "gdp_world_fan_rcs_age.csv"

# ── helpers ──────────────────────────────────────────────────────────────────
def norm(s: str) -> str: return re.sub(r"[^a-z0-9]", "", str(s).lower())

def rcs_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline (natural tails); returns (N, K-2)."""
    k = np.asarray(knots); K = k.size
    if K < 3: return np.zeros((x.size, 0))
    def d(u, j): return np.maximum(u - k[j], 0.0) ** 3
    cols=[]
    for j in range(1, K-1):
        cols.append(d(x,j)
                    - d(x,K-1)*(k[K-1]-k[j])/(k[K-1]-k[0])
                    + d(x,0)  *(k[j]   -k[0])/(k[K-1]-k[0]))
    return np.column_stack(cols)

# ═════════════════════════════════════════════════════════════════════════════
#  LOAD POSTERIOR, SCALES, GLOBAL μ, KNOTS
# ═════════════════════════════════════════════════════════════════════════════
idata = az.from_netcdf(IDATA_NC)
post  = idata.posterior

a_c   = post["alpha_country"]            # (chain, draw, Country)
b_c   = post["beta_country"]             # (chain, draw, Country)
theta = post["theta_region"]             # (chain, draw, Region, Spline)
delta_WA = post["delta_washare"]         # (chain, draw)
delta_OD = post["delta_olddep"]          # (chain, draw)
tau_c    = post["tau_country"]           # (chain, draw, Country)

post_countries = post.coords["Country"].values.tolist()
post_regions   = post.coords["Region"].values.tolist()

# load scales (μ, stds of deltas)
with open(SCALE_JSON, "r", encoding="utf-8") as f:
    SCALE = json.load(f)
MU_GLOBAL = SCALE["MU_GLOBAL"]
s_dWA = SCALE["s_dWA"]; s_dOD = SCALE["s_dOD"]; s_dt = SCALE["s_dt"]

# knots
knots = np.load(KNOTS_NPY)

# ═════════════════════════════════════════════════════════════════════════════
#  SCENARIO FRAME, METADATA, AGE COVARIATES, POPULATION, BASE ANCHORS
# ═════════════════════════════════════════════════════════════════════════════
# scenarios
scen = pd.read_csv(GSCEN, usecols=["ISO3","Year","Scenario"])
scen["Year"] = pd.to_numeric(scen["Year"], errors="coerce").astype("Int64")
scen = scen.dropna(subset=["ISO3","Year"]).assign(Year=lambda d: d["Year"].astype(int))
scen["scenario_norm"] = scen["Scenario"].apply(norm)

# ISO3 → Region / Country
meta = pd.read_csv(META_CSV)
reg_col  = [c for c in meta.columns if "Region" in c][0]
name_col = "TableName" if "TableName" in meta.columns else [c for c in meta.columns if "Table" in c and "Name" in c][0]
meta2 = meta.rename(columns={"Country Code":"ISO3", reg_col:"Region", name_col:"Country"})[["ISO3","Region","Country"]]
scen = scen.merge(meta2, on="ISO3", how="left", suffixes=("", "_m"))
for c in ["Region","Country"]:
    mc = c+"_m"
    if mc in scen.columns:
        scen[c] = scen[c].fillna(scen[mc])
scen = scen.drop(columns=[c for c in scen.columns if c.endswith("_m")])
scen = scen[scen["Country"].isin(post_countries)].copy()

# base anchors (GDP base level, x0)
base0 = (pd.read_csv(MERGED)
           .rename(columns={"Country Code":"ISO3"})
           .dropna(subset=["ISO3","Year","GDP","Population","Country Name"]))
base0["Year"] = pd.to_numeric(base0["Year"], errors="coerce").astype("Int64")
base0 = base0.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))
base0["Log_Population"] = np.log10(base0["Population"])
base0["logGDP_base"]    = np.log10(base0["GDP"])
b1 = (base0[base0["Year"]<=2023]
        .sort_values(["ISO3","Year"]).groupby("ISO3", as_index=False).tail(1))
b2 = (base0.loc[~base0["ISO3"].isin(b1["ISO3"])]
        .sort_values(["ISO3","Year"]).groupby("ISO3", as_index=False).tail(1))
base = pd.concat([b1,b2], ignore_index=True)
base = (base.rename(columns={"Country Name":"Country","Year":"Year_base","Log_Population":"LogPop_base"})
             [["ISO3","Country","Year_base","logGDP_base","LogPop_base"]])

# age covariates (OWID Medium variant for future), anchors for base
age_long = pd.read_csv(AGECSV)  # ISO3, Year, Scenario (Estimates/Medium), WAshare, OldDep
age_long["Year"] = pd.to_numeric(age_long["Year"], errors="coerce").astype("Int64")
age_long = age_long.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))
age_med  = age_long[age_long["Scenario"].str.contains("Medium", case=False, na=False)].copy()
age_med  = age_med.drop(columns=["Scenario","scenario_norm"])
scen = scen.merge(age_med, on=["ISO3","Year"], how="left")

age_anchor = pd.read_csv(AGEANCH)   # ISO3, Year_base, WAshare_base, OldDep_base
scen = scen.merge(age_anchor, on="ISO3", how="left")

# deltas & time (base-anchored)
scen["dWA"]   = scen["WAshare"] - scen["WAshare_base"]
scen["dOD"]   = scen["OldDep"]  - scen["OldDep_base"]
scen["t_dec"] = (scen["Year"] - 2000)/10.0

# population for x, x0
if not POPCACHE.exists():
    raise RuntimeError("Population cache not found. Build pop_predictions_scenarios.csv first.")
pop = pd.read_csv(POPCACHE)[["ISO3","Year","Scenario","Population"]]
pop["Year"] = pd.to_numeric(pop["Year"], errors="coerce").astype("Int64")
pop = pop.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))
scen = scen.merge(pop, on=["ISO3","Year","Scenario"], how="left")
if scen["Population"].isna().any():
    miss = scen[scen["Population"].isna()][["ISO3","Year","Scenario"]].head(10)
    raise RuntimeError(f"Population missing after merge. Example:\n{miss}")

scen["LogPop"] = np.log10(scen["Population"].astype(float).clip(lower=1.0))
scen = scen.merge(base, on=["ISO3","Country"], how="left", suffixes=("", "_b"))

# coalesce Year_base if colliding
if "Year_base_b" in scen.columns:
    if "Year_base" not in scen.columns:
        scen["Year_base"] = scen["Year_base_b"]
    else:
        scen["Year_base"] = scen["Year_base"].fillna(scen["Year_base_b"])

# Δt and standardization using training scales
scen["t_dec_base"] = (scen["Year_base"] - 2000)/10.0
scen["dt_dec"]     = scen["t_dec"] - scen["t_dec_base"]
scen["dWA_s"] = scen["dWA"]   / (s_dWA + 1e-8)
scen["dOD_s"] = scen["dOD"]   / (s_dOD + 1e-8)
scen["dt_s"]  = scen["dt_dec"]/ (s_dt  + 1e-8)

# centered x, x0 and spline deltas
scen = scen.dropna(subset=["logGDP_base","LogPop_base","dWA_s","dOD_s","t_dec","t_dec_base","dt_dec","Region","Country"])
scen["x"]  = scen["LogPop"]      - MU_GLOBAL
scen["x0"] = scen["LogPop_base"] - MU_GLOBAL
Z_it = rcs_design(scen["x"].to_numpy(),  knots)
Z_0  = rcs_design(scen["x0"].to_numpy(), knots)
spline_delta = Z_it - Z_0

# ═════════════════════════════════════════════════════════════════════════════
#  PER-COUNTRY POSTERIOR SIMULATION (with post-2085 taper)
# ═════════════════════════════════════════════════════════════════════════════
rows = []
for country, g in scen.groupby("Country"):
    if country not in post_countries:
        continue
    reg = g["Region"].iloc[0]
    if reg not in post_regions:
        continue

    yrs   = g["Year"].to_numpy()
    dx    = (g["x"] - g["x0"]).to_numpy()
    dWA_s = g["dWA_s"].to_numpy()
    dOD_s = g["dOD_s"].to_numpy()
    dt_s  = g["dt_s"].to_numpy()
    log_base = float(g["logGDP_base"].iloc[0])

    idx = g.index.to_numpy()
    Sdel = spline_delta[idx, :]

    # post-2085 taper (population vs time)
    g_years = np.clip((yrs - 2085)/(2100 - 2085), 0.0, 1.0)
    TAPER_MIN_POP  = 0.60
    TAPER_MIN_TIME = 0.00
    taper_pop  = 1.0 - (1.0 - TAPER_MIN_POP)  * g_years  # 1 → 0.60
    taper_time = 1.0 - (1.0 - TAPER_MIN_TIME) * g_years  # 1 → 0.00

    # posterior draws (flatten)
    a   = a_c.sel(Country=country).values.reshape(-1,1)
    b   = b_c.sel(Country=country).values.reshape(-1,1)
    th  = theta.sel(Region=reg).values.reshape(-1, theta.sizes["Spline"])
    dW  = delta_WA.values.reshape(-1,1)
    dO  = delta_OD.values.reshape(-1,1)
    tau = tau_c.sel(Country=country).values.reshape(-1,1)

    # components
    pop_delta  = (b @ dx[None,:]) + (th @ Sdel.T)       # (S,T)
    age_delta  = (dW @ dWA_s[None,:]) + (dO @ dOD_s[None,:])
    time_delta = (tau @ dt_s[None,:])

    # taper: population (→0.60), time (→0.00), age (untapered)
    delta_all = pop_delta * taper_pop[None,:] + time_delta * taper_time[None,:] + age_delta

    log_gdp = log_base + delta_all
    med = np.median(10.0**log_gdp, axis=0)
    lo  = np.quantile(10.0**log_gdp, 0.025, axis=0)
    hi  = np.quantile(10.0**log_gdp, 0.975, axis=0)

    out = g[["ISO3","Country","Region","Scenario","Year"]].copy()
    out["Pred_Median"] = med
    out["Pred_Lower"]  = lo
    out["Pred_Upper"]  = hi
    rows.append(out)

gdp_age = (pd.concat(rows, ignore_index=True)
             .sort_values(["ISO3","Year","Scenario"]))
gdp_age.to_csv(OUT_CNTRY, index=False)
print("✓ country-level forecasts →", OUT_CNTRY)

# ── world fan (simple pooling; for paper use constant-ISO3 + log-pool builder) ─
world = gdp_age.groupby(["Scenario","Year"], as_index=False)["Pred_Median"].sum()
wide  = world.pivot(index="Year", columns="Scenario", values="Pred_Median").sort_index()
arr   = wide.to_numpy()
yrs   = wide.index.to_numpy()

fan = pd.DataFrame({
    "Year": yrs.astype(int),
    "Median":  np.nanmedian(arr, axis=1),
    "p50_lo":  np.nanquantile(arr, 0.25, axis=1),
    "p50_hi":  np.nanquantile(arr, 0.75, axis=1),
    "p80_lo":  np.nanquantile(arr, 0.10, axis=1),
    "p80_hi":  np.nanquantile(arr, 0.90, axis=1),
    "p95_lo":  np.nanquantile(arr, 0.025, axis=1),
    "p95_hi":  np.nanquantile(arr, 0.975, axis=1),
})
fan.to_csv(OUT_WORLD, index=False)
print("✓ world fan →", OUT_WORLD)

# quick diagnostics
print("[scen] rows:", len(scen),
      "| missing Pop %:", scen["Population"].isna().mean(),
      "| years:", int(scen["Year"].min()), "→", int(scen["Year"].max()))
print("[posterior dims] alpha_country:", a_c.sizes, "| theta_region:", theta.sizes)
