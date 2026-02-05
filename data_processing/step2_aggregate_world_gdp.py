# step2_aggregate_world_gdp.py
import json
import numpy as np
import pandas as pd
import arviz as az
import sys
from pathlib import Path

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    PATH_MODEL_HIERARCHICAL_AGE, PATH_SCALE_JSON, PATH_MERGED_AGE,
    PATH_POP_PREDICTIONS, PATH_AGE_PREDICTIONS, PATH_GDP_WORLD_FAN
)

NC_PATH = PATH_MODEL_HIERARCHICAL_AGE
SCALE_JSON = PATH_SCALE_JSON
HIST_CSV = PATH_MERGED_AGE
POP_CSV = PATH_POP_PREDICTIONS
AGE_CSV = PATH_AGE_PREDICTIONS  # From Step 1
OUT_FAN_CSV = PATH_GDP_WORLD_FAN

THINNING = 10 

def rcs_design(x, knots):
    k = np.asarray(knots); K = k.size
    if K < 3: return np.zeros((x.size, 0))
    def d(u, j): return np.maximum(u - k[j], 0.0) ** 3
    cols = []
    for j in range(1, K - 1):
        cols.append(d(x, j) - d(x, K - 1) * (k[K - 1] - k[j]) / (k[K - 1] - k[0]) + d(x, 0) * (k[j] - k[0]) / (k[K - 1] - k[0]))
    return np.column_stack(cols)

def main():
    print("--- Step 2: World Aggregation ---")
    with open(SCALE_JSON, "r") as f: scales = json.load(f)
    
    MU_GLOBAL = scales["MU_GLOBAL"]; s_x = scales["s_x"]
    knots = np.array(scales["knots"]); coef_rcs = np.array(scales["coef_rcs"])
    s_dWA = scales["s_dWA"]; s_dOD = scales["s_dOD"]; s_dt = scales["s_dt"]
    coef_dt_proj = np.array(scales["coef_dt_proj"])
    dt_clip_future = 100.0 # Relaxed clipping

    print("Loading Posterior...")
    idata = az.from_netcdf(NC_PATH); post = idata.posterior
    def extract(var):
        vals = post[var].stack(sample=("chain","draw")).values
        return vals[..., ::THINNING] if vals.ndim > 1 else vals[::THINNING]

    beta0 = extract("beta0")
    theta = post["theta"].stack(sample=("chain","draw")).values
    if theta.shape[0] != len(knots)-2: theta = theta.T
    theta = theta[:, ::THINNING]
    tau_r = post["tau_region"].stack(sample=("chain","draw")).values[:, ::THINNING]
    d_wa = extract("delta_washare"); d_od = extract("delta_olddep")
    
    n_samples = beta0.shape[0]

    # Load Data
    df_hist = pd.read_csv(HIST_CSV)
    df_pop = pd.read_csv(POP_CSV)
    df_age = pd.read_csv(AGE_CSV)
    for d in [df_pop, df_age]: d["Year"] = pd.to_numeric(d["Year"], errors="coerce").astype(int)

    # Merge
    print("Merging scenarios...")
    df_scen = pd.merge(df_pop, df_age, on=["ISO3", "Year", "Scenario"], how="inner")
    df_scen = df_scen.dropna(subset=["Population", "WAshare", "OldDep"])
    
    if df_scen.empty:
        raise ValueError("Merged dataset empty! Check Step 1 output.")

    regions = list(post.coords["Region"].values)
    reg2idx = {r: i for i, r in enumerate(regions)}
    
    # Initialize World Accumulator
    scenarios = df_scen["Scenario"].unique()
    years = np.sort(df_scen["Year"].unique())
    world_agg = { (sc, yr): np.zeros(n_samples) for sc in scenarios for yr in years }

    print(f"Projecting GDP for {len(df_scen['ISO3'].unique())} countries...")
    for iso in df_scen['ISO3'].unique():
        hist_sub = df_hist[df_hist["ISO3"]==iso]
        if hist_sub.empty: continue
        anchor = hist_sub[hist_sub["Year"]==2023]
        if anchor.empty: anchor = hist_sub.sort_values("Year").tail(1)
        if anchor.empty or pd.isna(anchor["GDP"].values[0]): continue
            
        # Base Params
        y_base = float(np.log10(anchor["GDP"].values[0]))
        pop_base = float(np.log10(anchor["Population"].values[0]))
        wa_base = float(anchor["WAshare"].values[0])
        od_base = float(anchor["OldDep"].values[0])
        year_base = int(anchor["Year"].values[0])
        reg_name = anchor["Region"].values[0]
        if reg_name not in reg2idx: continue
        r_idx = reg2idx[reg_name]

        x_base_c = pop_base - MU_GLOBAL; x_base_s = x_base_c / (s_x + 1e-8)
        Z_base = rcs_design(np.array([x_base_c]), knots)
        X_orth_base = np.column_stack([np.ones(1), np.array([x_base_s])])
        Z_tilde_base = Z_base - X_orth_base @ coef_rcs
        spline_base = (Z_tilde_base @ theta).flatten()

        t_base_dec = (year_base - 2000)/10.0
        X_vec_base = np.concatenate([ [1.0], [x_base_s], Z_tilde_base[0] ])
        dt_s_orth_base = 0.0 - np.dot(X_vec_base, coef_dt_proj)
        time_term_base = tau_r[r_idx, :] * dt_s_orth_base

        scen_sub = df_scen[df_scen["ISO3"]==iso]
        for scen in scen_sub["Scenario"].unique():
            rows = scen_sub[(scen_sub["Scenario"]==scen) & (scen_sub["Year"] > year_base)].copy()
            if rows.empty: continue
            
            pop_t = np.log10(rows["Population"].values)
            wa_t = rows["WAshare"].values; od_t = rows["OldDep"].values
            yr_t = rows["Year"].values
            
            x_t_c = pop_t - MU_GLOBAL; x_t_s = x_t_c / (s_x + 1e-8)
            Z_t = rcs_design(x_t_c, knots)
            X_orth_t = np.column_stack([np.ones(len(x_t_s)), x_t_s])
            Z_tilde_t = Z_t - X_orth_t @ coef_rcs
            
            term_spline_t = Z_tilde_t @ theta
            term_lin_pop = np.outer(x_t_s - x_base_s, beta0)
            delta_spline = term_spline_t - spline_base

            t_dec = (yr_t - 2000)/10.0
            dt_raw = t_dec - (year_base - 2000)/10.0
            dt_s = dt_raw / (s_dt + 1e-8)
            X_vec_t = np.column_stack([np.ones(len(x_t_s)), x_t_s, Z_tilde_t])
            dt_proj = X_vec_t @ coef_dt_proj
            dt_s_orth = dt_s - dt_proj
            dt_s_c = np.clip(dt_s_orth, -dt_clip_future, dt_clip_future)
            
            term_time_t = np.outer(dt_s_c, tau_r[r_idx, :])
            delta_time = term_time_t - time_term_base
            delta_age = np.outer((wa_t - wa_base)/(s_dWA+1e-8), d_wa) + np.outer((od_t - od_base)/(s_dOD+1e-8), d_od)
            
            # Summation (No Tapering)
            delta_total = term_lin_pop + delta_spline + delta_time + delta_age
            pred_gdp = 10**(y_base + delta_total)

            for i, yr in enumerate(yr_t):
                if (scen, yr) in world_agg:
                    world_agg[(scen, yr)] += pred_gdp[i, :]

    print("Computing Quantiles...")
    fan_rows = []
    for yr in years:
        pooled = []
        for sc in scenarios:
            vals = world_agg[(sc, yr)]
            if np.sum(vals) > 1e-9: pooled.append(vals)
        if not pooled: continue
        grand = np.concatenate(pooled)
        fan_rows.append({
            "Year": yr, "Median": np.median(grand),
            "p95_lo": np.quantile(grand, 0.025), "p95_hi": np.quantile(grand, 0.975),
            "p80_lo": np.quantile(grand, 0.10), "p80_hi": np.quantile(grand, 0.90),
            "p50_lo": np.quantile(grand, 0.25), "p50_hi": np.quantile(grand, 0.75),
        })

    pd.DataFrame(fan_rows).to_csv(OUT_FAN_CSV, index=False)
    print(f"✓ Saved {OUT_FAN_CSV}")

if __name__ == "__main__":
    main()