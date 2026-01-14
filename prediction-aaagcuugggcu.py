# --------------------------- prediction_fixed.py ------------------------------
"""
Generate 2024–2100 GDP forecasts for ten WPP-2024 population variants using
the hierarchical Bayesian model with centred log-population.

Outputs
-------
gdp_predictions_scenarios.csv  : variant-specific medians + 95 % HDI
gdp_predictions_meta.csv       : DerSimonian–Laird pooled medians + HDI
"""
# ------------------------------------------------------------------- imports ---
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path

# ------------------------------------------------------------------- paths -----
ROOT      = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
HIST_CSV  = ROOT / "merged.csv"
WPP_XLSX  = ROOT / "BasicData/UN/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_FULL.xlsx"
META_CSV  = ROOT / "BasicData/API_SP.POP.TOTL_DS2_en_csv_v2_3401680/Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_3401680.csv"
IDATA_NC  = ROOT / "hierarchical_model_with_quadratic.nc"

SCEN_CSV  = ROOT / "gdp_predictions_scenarios.csv"
META_OUT  = ROOT / "gdp_predictions_meta.csv"

# ------------------------------ helper: safe base-10 exponentiation ------------
def safe_pow10(logx, clip=308):
    """Return 10**logx; clip exponent to avoid float overflow (~1e308)."""
    return np.power(10.0, np.minimum(logx, clip))

# ------------------------------ 1. historical mean for centring ----------------
df_hist = (
    pd.read_csv(HIST_CSV)
      .dropna(subset=["Population", "GDP", "Year", "Region", "Country Code"])
      .assign(Log_Population=lambda d: np.log10(d.Population))
)
mu_logpop = df_hist["Log_Population"].mean()

# ------------------------------ 2. read WPP variants ---------------------------
xl          = pd.ExcelFile(WPP_XLSX)
sheet_want = [
    s for s in xl.sheet_names
    if ("variant" in s.lower() or "fertility" in s.lower())
    and s not in {"High variant", "Low variant"}
]

pop_scen    = {}

for sheet in sheet_want:
    df = xl.parse(
        sheet, header=16,
        usecols=["Year", "ISO3 Alpha-code",
                 "Total Population, as of 1 July (thousands)"]
    ).rename(columns={
        "ISO3 Alpha-code": "ISO3",
        "Total Population, as of 1 July (thousands)": "Population"
    })

    df["Population"]   = pd.to_numeric(df["Population"], errors="coerce") * 1_000
    df                  = df[df["Population"] > 0]
    df["Year"]          = df["Year"].astype(int)
    df["Log_Pop"]       = np.log10(df["Population"])
    df["Log_Pop_c"]     = df["Log_Pop"] - mu_logpop
    df["Log_Pop_sq"]    = df["Log_Pop_c"] ** 2
    pop_scen[sheet]     = df[["ISO3", "Year", "Population",
                              "Log_Pop", "Log_Pop_c", "Log_Pop_sq"]]

# ------------------------------ 3. metadata & ISO-3 mapping --------------------
meta = (pd.read_csv(META_CSV)
          .rename(columns=lambda c: c.strip())
          [["Country Code", "TableName", "Region"]]
          .rename(columns={"Country Code": "ISO3",
                           "TableName": "Country"}))

# ------------------------------ 4. load posterior ------------------------------
idata  = az.from_netcdf(IDATA_NC)
post   = idata.posterior

# ------------------------------------------------------------------ 4-A  detect key type
coord_values = post.coords["Country"].values.astype(str)
looks_like_iso3 = (
    len(coord_values[0]) == 3 and          # exactly three characters
    coord_values[0].isupper() and          # all caps
    coord_values.dtype.kind == "U"         # numpy Unicode
)
# ------------------------------------------------------------------ 4-B  build lookup tables
if looks_like_iso3:
    # posterior already keyed by ISO-3
    key_type      = "iso"
    posterior_key = coord_values                         # ISO-3 strings
    meta_key      = meta[["ISO3", "Country"]]            # keep both
else:
    # posterior keyed by *country name*; map name → ISO-3 via metadata
    key_type      = "name"
    posterior_key = coord_values                         # country names
    name2iso      = dict(zip(meta["Country"], meta["ISO3"]))
    meta_key      = meta                                 # unchanged

# sets for fast membership tests
posterior_set = set(posterior_key)

print(f"[INFO] posterior keyed by {key_type}; {len(posterior_set)} countries in model")

# ------------------------------ 5. per-scenario forecasts ----------------------
rows = []

for scen, pop_df in pop_scen.items():

    # attach country names + ISO3 so we have both no matter what
    pop_df = pop_df.merge(meta_key, how="left", on="ISO3").dropna(subset=["Country"])

    # choose join key that matches the posterior
    if key_type == "iso":
        pop_df = pop_df[pop_df["ISO3"].isin(posterior_set)]
        group_key = "ISO3"
    else:                        # keyed by name
        pop_df = pop_df[pop_df["Country"].isin(posterior_set)]
        group_key = "Country"

    # ---- iterate country groups
    for key, grp in pop_df.groupby(group_key):
        # predictors
        x   = grp["Log_Pop_c"].values
        x2  = grp["Log_Pop_sq"].values
        yrs = grp["Year"].values

        # posterior draws: always index by the actual coordinate name
        a = post["alpha_country"].sel(Country=key).values.reshape(-1, 1)
        b = post["beta_country" ].sel(Country=key).values.reshape(-1, 1)
        g = post["gamma_country"].sel(Country=key).values.reshape(-1, 1)

        log_draws = a + b * x + g * x2                       # (draws, years)

        LOG_CAP = 14.3                          # ≈ log10(2×10^14 USD)
        log_draws = np.clip(log_draws, a_min=None, a_max=LOG_CAP)

        # median & 95 % HDI on log-scale
        med_log       = np.median(log_draws, axis=0)
        hdi_low, hdi_up = az.hdi(log_draws, 0.95).T

        rows.append(pd.DataFrame({
            "Scenario":     scen,
            "ISO3":         grp["ISO3"].iloc[0],
            "Country":      grp["Country"].iloc[0],
            "Year":         yrs,
            "Pred_Median":  safe_pow10(med_log),
            "Pred_Lower":   safe_pow10(hdi_low),
            "Pred_Upper":   safe_pow10(hdi_up),
        }))

# ---- concatenate
if not rows:
    raise RuntimeError("No rows were generated. Verify that posterior country "
                       "identifiers match the WPP metadata.")

df_scen = pd.concat(rows, ignore_index=True)
df_scen.to_csv(SCEN_CSV, index=False)
print(f"✓ per-scenario forecasts → {SCEN_CSV}")
