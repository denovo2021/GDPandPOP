# ---------------------- prediction_rcs.py (RCS version, 2085 peak) ----------------------
"""
Generate 2024–2100 GDP forecasts for UN WPP-2024 variants using the
hierarchical RCS model:

  log10(GDP)_it = log10(GDP_base)_i
                  + beta_country_i * (x_t - x_0i)
                  + theta_region_i^T * [RCS(x_t) - RCS(x_0i)],

where x is centered log10(population). We apply light, transparent guardrails
for far-horizon plausibility (post-2085 taper; time-varying soft floor/ceiling).
"""

# --------------------------------- imports ------------------------------------
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path

# --------------------------------- paths --------------------------------------
ROOT      = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
HIST_CSV  = ROOT / "merged.csv"
WPP_XLSX  = ROOT / "BasicData/UN/WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_FULL.xlsx"
META_CSV  = ROOT / "BasicData/API_SP.POP.TOTL_DS2_en_csv_v2_3401680/Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_3401680.csv"
IDATA_NC  = ROOT / "hierarchical_model_with_rcs.nc"        # posterior from the fit
KNOTS_NPY = ROOT / "rcs_knots_hier.npy"                    # saved during fitting
SCEN_CSV  = ROOT / "gdp_predictions_scenarios_rcs.csv"

# --------------------------------- helpers ------------------------------------
def safe_pow10(logx, clip=308.0):
    """Return 10**logx; clip exponent to avoid float overflow (~1e308)."""
    return np.power(10.0, np.minimum(logx, clip))

def rcs_design(x_in: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted cubic spline (natural cubic) with linear tails.
       Returns an (N, K-2) design matrix."""
    k = np.asarray(knots); K = k.size
    if K < 3:
        return np.zeros((x_in.size, 0))
    def d(u, j):
        return np.maximum(u - k[j], 0.0) ** 3
    cols = []
    for j in range(1, K-1):
        term = (d(x_in, j)
                - d(x_in, K-1) * (k[K-1]-k[j])/(k[K-1]-k[0])
                + d(x_in, 0)   * (k[j]   -k[0])/(k[K-1]-k[0]))
        cols.append(term)
    return np.column_stack(cols)

# --------------- 1) historical centering and mild clamping envelope ----------
df_hist = (
    pd.read_csv(HIST_CSV)
      .dropna(subset=["Population", "GDP", "Year", "Region", "Country Code"])
      .assign(Log_Population=lambda d: np.log10(d.Population))
)
mu_logpop = df_hist["Log_Population"].mean()

# envelope for centered x (keeps extrapolation linear but bounded)
x_hist_c   = df_hist["Log_Population"] - mu_logpop
x_lo, x_hi = x_hist_c.min(), x_hist_c.max()
CLAMP_BUF  = 0.25                           # ±0.25 dex buffer beyond the range
X_MIN, X_MAX = x_lo - CLAMP_BUF, x_hi + CLAMP_BUF

# --------------- base-year anchors (logGDP_base, x0 per ISO3) ----------------
BASE_YEAR = 2023
df_hist_full = df_hist.copy()
# choose base row: prefer BASE_YEAR; otherwise last available ≤ BASE_YEAR
df_base = (df_hist_full.sort_values(["Country Code","Year"])
           .groupby("Country Code", as_index=False)
           .apply(lambda g: g[g["Year"]<=BASE_YEAR].tail(1) if (g["Year"]<=BASE_YEAR).any() else g.tail(1))
           .reset_index(drop=True))

df_base = df_base.rename(columns={"Country Code":"ISO3"})
df_base["x0"]          = df_base["Log_Population"] - mu_logpop
df_base["logGDP_base"] = np.log10(df_base["GDP"])

# --------------- spline knots (reuse from fit; fallback to quantiles) --------
try:
    knots = np.load(KNOTS_NPY)
except Exception:
    knots = np.quantile(x_hist_c, [0.05, 0.35, 0.65, 0.95])

# -------------------------- 2) read WPP variants ------------------------------
xl = pd.ExcelFile(WPP_XLSX)
sheet_want = [
    s for s in xl.sheet_names
    if ("variant" in s.lower() or "fertility" in s.lower())
    and s not in {"High variant", "Low variant"}     # exclude Hi/Lo if desired
]
pop_scen = {}

for sheet in sheet_want:
    df = xl.parse(
        sheet, header=16,
        usecols=["Year", "ISO3 Alpha-code",
                 "Total Population, as of 1 July (thousands)"]
    ).rename(columns={
        "ISO3 Alpha-code": "ISO3",
        "Total Population, as of 1 July (thousands)": "Population"
    })

    df["Population"] = pd.to_numeric(df["Population"], errors="coerce") * 1_000
    df = df[df["Population"] > 0]
    df["Year"] = df["Year"].astype(int)
    df["Log_Pop"]   = np.log10(df["Population"])
    df["Log_Pop_c"] = df["Log_Pop"] - mu_logpop
    pop_scen[sheet] = df.dropna(subset=["ISO3"])

# ---------------------- 3) metadata & ISO-3 mapping ---------------------------
meta = (pd.read_csv(META_CSV)
          .rename(columns=lambda c: c.strip())
          [["Country Code", "TableName", "Region"]]
          .rename(columns={"Country Code": "ISO3", "TableName": "Country"}))

# ----------------------- 4) load posterior (RCS model) ------------------------
idata = az.from_netcdf(IDATA_NC)
post  = idata.posterior

# detect whether Country coord is ISO3 or names
coord_vals = post.coords["Country"].values.astype(str)
looks_like_iso3 = (len(coord_vals[0]) == 3 and coord_vals[0].isupper())
key_type = "iso" if looks_like_iso3 else "name"
posterior_set = set(coord_vals)
print(f"[INFO] posterior keyed by {key_type}; {len(posterior_set)} countries")

# ---------------------- 5) per-scenario forecasts (anchor+delta) --------------
rows = []

# global population peak year (World Bank)
PEAK_YEAR    = 2085
ANCHOR_START = 2035   # ramp baseline for soft negatives and pre-peak floor

for scen, df_pred in pop_scen.items():
    # attach names/regions + anchors
    merged = (df_pred
              .merge(meta, on="ISO3", how="left")
              .merge(df_base[["ISO3", "logGDP_base", "x0"]], on="ISO3", how="left")
              .dropna(subset=["Country", "Region", "logGDP_base", "x0"]))

    # keep only countries available in posterior
    if key_type == "iso":
        merged = merged[merged["ISO3"].isin(posterior_set)]
        group_key = "ISO3"
    else:
        merged = merged[merged["Country"].isin(posterior_set)]
        group_key = "Country"

    # group by country (or ISO3 key)
    for key, grp in merged.groupby(group_key):
        yrs = grp["Year"].values

        # centered x, mild clamping to training envelope
        x_raw = grp["Log_Pop_c"].values
        x     = np.clip(x_raw, X_MIN, X_MAX)
        x0    = float(grp["x0"].iloc[0])   # base-year center; should be in-range
        Z     = rcs_design(x,  knots)      # (T, m)
        Z0    = rcs_design(np.array([x0]), knots)  # (1, m)
        m     = Z.shape[1]

        # posterior draws (chain*draw flattened implicitly by reshape)
        a = post["alpha_country"].sel(Country=key).values.reshape(-1, 1)   # (D,1) unused in delta
        b = post["beta_country" ].sel(Country=key).values.reshape(-1, 1)   # (D,1)

        # region-level spline weights
        if "theta_region" in post.data_vars:
            region_name = grp["Region"].iloc[0]
            th_da = post["theta_region"].sel(Region=region_name)            # (chain,draw,Spline)
            th = (th_da
                  .stack(sample=("chain","draw"))
                  .transpose("sample","Spline")
                  .values)                                                 # (D, m) or (D,0)
        else:
            th = np.zeros((b.shape[0], 0), dtype=float)                     # no spline in posterior
            m  = 0

        # difference-anchored delta (intercept cancels): Δ = b·(x-x0) + θ·(RCS(x)-RCS(x0))
        spline_diff = th @ (Z.T - Z0.T) if m > 0 else 0.0                  # (D, T)
        delta = b * (x - x0) + spline_diff                                 # (D, T)

        # --------------------- smooth far-horizon guards ----------------------
        # (1) Smooth taper only AFTER the global peak (convex onset)
        g = np.clip((yrs - PEAK_YEAR) / (2100 - PEAK_YEAR), 0.0, 1.0)       # 0→1 from 2085→2100
        g = g**1.5
        TAPER_MIN = 0.60                                                    # keep 60% by 2100
        taper = 1.0 - (1.0 - TAPER_MIN) * g                                 # 1.0 → 0.60
        delta = delta * taper[None, :]

        # (2) Soft allowance for negative deltas BEFORE the peak
        #     Allow 20%→100% of negatives from 2035→2085 (convex ramp)
        g_neg = np.clip((yrs - ANCHOR_START) / (PEAK_YEAR - ANCHOR_START), 0.0, 1.0)**1.5
        allow_neg_frac = 0.20 + 0.80 * g_neg
        pos = np.maximum(delta, 0.0)
        neg = np.minimum(delta, 0.0)
        delta = pos + allow_neg_frac[None, :] * neg

        # (3) Time-varying soft floor/ceiling vs base (piecewise multiplicative bounds)
        #     Pre-peak: floor 0.90× → 0.80× (2035→2085)
        #     Post-peak: 0.80× → 0.50× (2085→2100)
        t_pre   = np.clip((yrs - ANCHOR_START) / (PEAK_YEAR - ANCHOR_START), 0.0, 1.0)
        min_pre = (1.0 - t_pre) * 0.90 + t_pre * 0.80
        min_post= (1.0 - g)    * 0.80 + g     * 0.50
        min_mult= np.where(yrs < PEAK_YEAR, min_pre, min_post)              # (T,)
        lower   = np.log10(min_mult)                                        # 0 → -0.301
        upper   = np.log10(4.0)                                             # +0.602 (~4× cap)

        delta = np.minimum(delta, upper)
        delta = np.maximum(delta, lower[None, :])

        # (4) De-pin tiny fraction at bounds to avoid collapsed HDIs
        EPS = 1e-3
        delta = np.where(upper - delta < EPS, upper - EPS, delta)
        delta = np.where(delta - lower[None, :] < EPS, lower[None, :] + EPS, delta)

        # final log10 GDP and summary
        log_base  = float(grp["logGDP_base"].iloc[0])
        log_draws = log_base + delta
        LOG_CAP = 15.0  # numeric guardrail (rarely binds after guards)
        log_draws = np.clip(log_draws, a_min=None, a_max=LOG_CAP)

        med_log        = np.median(log_draws, axis=0)
        hdi_lo, hdi_hi = az.hdi(log_draws, 0.95).T

        rows.append(pd.DataFrame({
            "Scenario":    scen,
            "ISO3":        grp["ISO3"].iloc[0],
            "Country":     grp["Country"].iloc[0],
            "Region":      grp["Region"].iloc[0],
            "Year":        yrs,
            "Pred_Median": safe_pow10(med_log),
            "Pred_Lower":  safe_pow10(hdi_lo),
            "Pred_Upper":  safe_pow10(hdi_hi),
        }))

# concatenate & write
if not rows:
    raise RuntimeError("No scenario rows produced. Check country identifiers.")
df_scen = pd.concat(rows, ignore_index=True)
df_scen.to_csv(SCEN_CSV, index=False)
print(f"✓ per-scenario forecasts → {SCEN_CSV}")
