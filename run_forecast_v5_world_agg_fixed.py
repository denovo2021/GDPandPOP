# run_forecast_v5_world_agg_fixed.py
import json
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path

# --- Settings ---
ROOT = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
NC_PATH     = ROOT / "hierarchical_model_with_rcs_age_v5_1_ncp_stable.nc"
SCALE_JSON  = ROOT / "scale_rcs_age.json"

HIST_CSV    = ROOT / "merged_age.csv"
POP_CSV     = ROOT / "pop_predictions_scenarios.csv"
AGE_CSV     = ROOT / "age_predictions_scenarios.csv"
OUT_FAN_CSV = ROOT / "gdp_world_fan_v5_full_uncertainty.csv"

THINNING = 10 

# --- Helper ---
def rcs_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    k = np.asarray(knots); K = k.size
    if K < 3: return np.zeros((x.size, 0))
    def d(u, j): return np.maximum(u - k[j], 0.0) ** 3
    cols = []
    for j in range(1, K - 1):
        cols.append(d(x, j)
                    - d(x, K - 1) * (k[K - 1] - k[j]) / (k[K - 1] - k[0])
                    + d(x, 0)     * (k[j]     - k[0]) / (k[K - 1] - k[0]))
    return np.column_stack(cols)

def main():
    print("--- Starting World Aggregation (Fixed) ---")
    with open(SCALE_JSON, "r") as f:
        scales = json.load(f)
    
    MU_GLOBAL = scales["MU_GLOBAL"]
    s_x       = scales["s_x"]
    knots     = np.array(scales["knots"])
    coef_rcs  = np.array(scales["coef_rcs"])
    s_dWA     = scales["s_dWA"]
    s_dOD     = scales["s_dOD"]
    s_dt         = scales["s_dt"]
    coef_dt_proj = np.array(scales["coef_dt_proj"])
    dt_clip_future = 100.0

    print(f"Loading Posterior from {NC_PATH}...")
    idata = az.from_netcdf(NC_PATH)
    post = idata.posterior
    
    def extract(var):
        vals = post[var].stack(sample=("chain","draw")).values
        if vals.ndim > 1: return vals[..., ::THINNING]
        else: return vals[::THINNING]

    beta0 = extract("beta0")
    theta = post["theta"].stack(sample=("chain","draw")).values
    if theta.shape[0] != len(knots)-2: theta = theta.T
    theta = theta[:, ::THINNING]
    tau_r = post["tau_region"].stack(sample=("chain","draw")).values[:, ::THINNING]
    d_wa  = extract("delta_washare")
    d_od  = extract("delta_olddep")
    
    print(f"Posterior samples (thinned): {beta0.shape[0]}")

    # Load & Merge
    print("Loading Data...")
    df_hist = pd.read_csv(HIST_CSV)
    df_pop = pd.read_csv(POP_CSV)
    df_age = pd.read_csv(AGE_CSV)

    for d in [df_pop, df_age]:
        d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    df_pop = df_pop.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))
    df_age = df_age.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))

    print("Merging Scenarios...")
    df_scen = pd.merge(df_pop, df_age, on=["ISO3", "Year", "Scenario"], how="inner")
    df_scen = df_scen.dropna(subset=["Population", "WAshare", "OldDep"])
    
    if df_scen.empty:
        raise ValueError("Merged dataframe is empty! Check Scenario names in POP/AGE CSVs.")
    
    print(f"Merged Data Rows: {len(df_scen)}")
    print(f"Scenarios found: {df_scen['Scenario'].unique()}")

    regions = list(post.coords["Region"].values)
    reg2idx = {r: i for i, r in enumerate(regions)}
    countries = df_scen["ISO3"].unique()
    
    # Initialize Accumulator
    scenarios_list = df_scen["Scenario"].unique()
    years_list = np.sort(df_scen["Year"].unique())
    min_year, max_year = years_list.min(), years_list.max()
    
    world_agg = {}
    for sc in scenarios_list:
        for yr in range(min_year, max_year + 1):
            world_agg[(sc, yr)] = np.zeros(beta0.shape[0])

    print(f"Aggregating GDP for {len(countries)} countries...")
    
    count_processed = 0
    for iso in countries:
        hist_sub = df_hist[df_hist["ISO3"]==iso]
        if hist_sub.empty: continue
        
        anchor = hist_sub[hist_sub["Year"]==2023]
        if anchor.empty: anchor = hist_sub.sort_values("Year").tail(1)
        if anchor.empty or pd.isna(anchor["GDP"].values[0]): continue
            
        y_base     = float(np.log10(anchor["GDP"].values[0]))
        pop_base   = float(np.log10(anchor["Population"].values[0]))
        wa_base    = float(anchor["WAshare"].values[0])
        od_base    = float(anchor["OldDep"].values[0])
        year_base  = int(anchor["Year"].values[0])
        reg_name   = anchor["Region"].values[0]
        
        if reg_name not in reg2idx: continue
        r_idx = reg2idx[reg_name]

        # Base Calc
        x_base_c = pop_base - MU_GLOBAL
        x_base_s = x_base_c / (s_x + 1e-8)
        Z_base = rcs_design(np.array([x_base_c]), knots)
        X_orth_base = np.column_stack([np.ones(1), np.array([x_base_s])])
        Z_tilde_base = Z_base - X_orth_base @ coef_rcs
        spline_base = (Z_tilde_base @ theta).flatten()

        t_base_dec = (year_base - 2000)/10.0
        X_vec_base = np.concatenate([ [1.0], [x_base_s], Z_tilde_base[0] ])
        dt_s_orth_base = 0.0 - np.dot(X_vec_base, coef_dt_proj)
        time_term_base = tau_r[r_idx, :] * dt_s_orth_base

        # Future Calc
        scen_sub = df_scen[df_scen["ISO3"]==iso]
        if scen_sub.empty: continue
        
        count_processed += 1
        
        for scen in scen_sub["Scenario"].unique():
            rows = scen_sub[(scen_sub["Scenario"]==scen) & (scen_sub["Year"] > year_base)].copy()
            if rows.empty: continue
            
            pop_t = np.log10(rows["Population"].values)
            wa_t  = rows["WAshare"].values
            od_t  = rows["OldDep"].values
            yr_t  = rows["Year"].values
            
            x_t_c = pop_t - MU_GLOBAL
            x_t_s = x_t_c / (s_x + 1e-8)
            Z_t = rcs_design(x_t_c, knots)
            X_orth_t = np.column_stack([np.ones(len(x_t_s)), x_t_s])
            Z_tilde_t = Z_t - X_orth_t @ coef_rcs
            
            term_spline_t = Z_tilde_t @ theta
            term_lin_pop = np.outer(x_t_s - x_base_s, beta0)
            delta_spline = term_spline_t - spline_base

            t_dec = (yr_t - 2000)/10.0
            dt_raw = t_dec - (year_base - 2000)/10.0
            dt_s   = dt_raw / (s_dt + 1e-8)
            
            X_vec_t = np.column_stack([np.ones(len(x_t_s)), x_t_s, Z_tilde_t])
            dt_proj = X_vec_t @ coef_dt_proj
            dt_s_orth = dt_s - dt_proj
            dt_s_c = np.clip(dt_s_orth, -dt_clip_future, dt_clip_future)
            
            term_time_t = np.outer(dt_s_c, tau_r[r_idx, :])
            delta_time  = term_time_t - time_term_base
            
            d_wa_val = (wa_t - wa_base) / (s_dWA + 1e-8)
            d_od_val = (od_t - od_base) / (s_dOD + 1e-8)
            delta_age = np.outer(d_wa_val, d_wa) + np.outer(d_od_val, d_od)
            
            delta_total = (term_lin_pop + delta_spline) + delta_time + delta_age
            pred_gdp = 10**(y_base + delta_total)
            
            for i, yr in enumerate(yr_t):
                if (scen, yr) in world_agg:
                    world_agg[(scen, yr)] += pred_gdp[i, :]

    print(f"Processed {count_processed} countries.")

    # --- Quantiles ---
    print("Computing World Quantiles...")
    fan_rows = []
    
    # フィルタリングせず、データにあるすべてのシナリオを使用
    valid_scens = scenarios_list 
    
    for yr in range(min_year, max_year + 1):
        pooled_samples = []
        for sc in valid_scens:
            if (sc, yr) in world_agg:
                vals = world_agg[(sc, yr)]
                # 合計が0の場合はスキップ（データなしの可能性）
                if np.sum(vals) > 1e-9:
                    pooled_samples.append(vals)
        
        if not pooled_samples: continue
        
        grand_arr = np.concatenate(pooled_samples)
        
        fan_rows.append({
            "Year": yr,
            "Median": np.median(grand_arr),
            "p95_lo": np.quantile(grand_arr, 0.025),
            "p95_hi": np.quantile(grand_arr, 0.975),
            "p80_lo": np.quantile(grand_arr, 0.10),
            "p80_hi": np.quantile(grand_arr, 0.90),
            "p50_lo": np.quantile(grand_arr, 0.25),
            "p50_hi": np.quantile(grand_arr, 0.75),
        })

    if not fan_rows:
        print("ERROR: No data aggregated! CSV will be empty.")
    else:
        df_fan = pd.DataFrame(fan_rows)
        df_fan.to_csv(OUT_FAN_CSV, index=False)
        print(f"✓ Saved World Fan Data ({len(df_fan)} years) to {OUT_FAN_CSV}")

if __name__ == "__main__":
    main()