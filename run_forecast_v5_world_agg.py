# run_forecast_v5_world_agg.py
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
OUT_FAN_CSV = ROOT / "gdp_world_fan_v5_full_uncertainty.csv" # 出力ファイル名変更

THINNING = 10 # サンプルを間引く（計算速度用。精度は十分確保されます）

# --- Helper: RCS Basis ---
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
    print("Loading scales and model data...")
    with open(SCALE_JSON, "r") as f:
        scales = json.load(f)
    
    # Restore transform params
    MU_GLOBAL = scales["MU_GLOBAL"]
    s_x       = scales["s_x"]
    knots     = np.array(scales["knots"])
    coef_rcs  = np.array(scales["coef_rcs"])
    s_dWA     = scales["s_dWA"]
    s_dOD     = scales["s_dOD"]
    s_dt         = scales["s_dt"]
    coef_dt_proj = np.array(scales["coef_dt_proj"])
    
    # Relaxed Clipping (No limit for future)
    dt_clip_future = 100.0

    # Load Posterior
    print(f"Loading Posterior from {NC_PATH}...")
    idata = az.from_netcdf(NC_PATH)
    post = idata.posterior
    
    # Stack chains and draw, then THIN to reduce memory/time
    # shape becomes (n_samples_thinned, )
    def extract(var):
        vals = post[var].stack(sample=("chain","draw")).values
        if vals.ndim > 1:
            return vals[..., ::THINNING] # Last dim is sample
        else:
            return vals[::THINNING]

    beta0 = extract("beta0")
    # theta: (chain, draw, Spline) -> stack -> (Spline, Samples)
    theta = post["theta"].stack(sample=("chain","draw")).values
    if theta.shape[0] != len(knots)-2: 
        theta = theta.T # Ensure (Spline, Samples)
    theta = theta[:, ::THINNING]
    
    tau_r = post["tau_region"].stack(sample=("chain","draw")).values[:, ::THINNING]
    d_wa  = extract("delta_washare")
    d_od  = extract("delta_olddep")
    
    n_samples = beta0.shape[0]
    print(f"Posterior samples (thinned): {n_samples}")

    # Load & Merge Data
    print("Loading and Merging Data...")
    df_hist = pd.read_csv(HIST_CSV)
    df_pop = pd.read_csv(POP_CSV)
    df_age = pd.read_csv(AGE_CSV)

    for d in [df_pop, df_age]:
        d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    df_pop = df_pop.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))
    df_age = df_age.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))

    df_scen = pd.merge(df_pop, df_age, on=["ISO3", "Year", "Scenario"], how="inner")
    df_scen = df_scen.dropna(subset=["Population", "WAshare", "OldDep"])
    
    regions = list(post.coords["Region"].values)
    reg2idx = {r: i for i, r in enumerate(regions)}
    
    countries = df_scen["ISO3"].unique()
    
    # --- World Accumulator ---
    # world_agg[Scenario][Year] = np.zeros(n_samples)
    # これに各国・各年の分布(1000個の値)を足し合わせていく
    scenarios_list = df_scen["Scenario"].unique()
    years_list = np.sort(df_scen["Year"].unique())
    min_year, max_year = years_list.min(), years_list.max()
    
    # 辞書キー: (Scenario, Year) -> Value: Array of sums
    world_agg = {}
    for sc in scenarios_list:
        for yr in range(min_year, max_year + 1):
            world_agg[(sc, yr)] = np.zeros(n_samples)

    print(f"Projecting and Aggregating World GDP for {len(countries)} countries...")
    
    for iso in countries:
        hist_sub = df_hist[df_hist["ISO3"]==iso]
        if hist_sub.empty: continue
        
        anchor = hist_sub[hist_sub["Year"]==2023]
        if anchor.empty:
            anchor = hist_sub.sort_values("Year").tail(1)
            
        if anchor.empty or pd.isna(anchor["GDP"].values[0]): continue
            
        # Base Values
        y_base     = float(np.log10(anchor["GDP"].values[0]))
        pop_base   = float(np.log10(anchor["Population"].values[0]))
        wa_base    = float(anchor["WAshare"].values[0])
        od_base    = float(anchor["OldDep"].values[0])
        year_base  = int(anchor["Year"].values[0])
        reg_name   = anchor["Region"].values[0]
        
        if reg_name not in reg2idx: continue
        r_idx = reg2idx[reg_name]

        # Base Calculations
        x_base_c = pop_base - MU_GLOBAL
        x_base_s = x_base_c / (s_x + 1e-8)
        
        Z_base = rcs_design(np.array([x_base_c]), knots)
        X_orth_base = np.column_stack([np.ones(1), np.array([x_base_s])])
        Z_tilde_base = Z_base - X_orth_base @ coef_rcs
        spline_base = (Z_tilde_base @ theta).flatten()

        t_base_dec = (year_base - 2000)/10.0
        X_vec_base = np.concatenate([ [1.0], [x_base_s], Z_tilde_base[0] ])
        dt_s_base_raw = 0.0
        dt_s_orth_base = dt_s_base_raw - np.dot(X_vec_base, coef_dt_proj)
        dt_s_c_base = dt_s_orth_base
        time_term_base = tau_r[r_idx, :] * dt_s_c_base

        # Loop Scenarios
        scen_sub = df_scen[df_scen["ISO3"]==iso]
        
        for scen in scen_sub["Scenario"].unique():
            rows = scen_sub[(scen_sub["Scenario"]==scen) & (scen_sub["Year"] > year_base)].copy()
            if rows.empty: continue
            
            pop_t = np.log10(rows["Population"].values)
            wa_t  = rows["WAshare"].values
            od_t  = rows["OldDep"].values
            yr_t  = rows["Year"].values
            
            # Pop Delta
            x_t_c = pop_t - MU_GLOBAL
            x_t_s = x_t_c / (s_x + 1e-8)
            Z_t = rcs_design(x_t_c, knots)
            X_orth_t = np.column_stack([np.ones(len(x_t_s)), x_t_s])
            Z_tilde_t = Z_t - X_orth_t @ coef_rcs
            
            term_spline_t = Z_tilde_t @ theta
            term_lin_pop = np.outer(x_t_s - x_base_s, beta0)
            delta_spline = term_spline_t - spline_base

            # Time Delta (Relaxed)
            t_dec = (yr_t - 2000)/10.0
            dt_raw = t_dec - t_base_dec
            dt_s   = dt_raw / (s_dt + 1e-8)
            
            X_vec_t = np.column_stack([np.ones(len(x_t_s)), x_t_s, Z_tilde_t])
            dt_proj = X_vec_t @ coef_dt_proj
            dt_s_orth = dt_s - dt_proj
            dt_s_c = np.clip(dt_s_orth, -dt_clip_future, dt_clip_future)
            
            term_time_t = np.outer(dt_s_c, tau_r[r_idx, :])
            delta_time  = term_time_t - time_term_base
            
            # Age Delta
            d_wa_val = (wa_t - wa_base) / (s_dWA + 1e-8)
            d_od_val = (od_t - od_base) / (s_dOD + 1e-8)
            delta_age = np.outer(d_wa_val, d_wa) + np.outer(d_od_val, d_od)
            
            # Total Delta (No Tapering)
            delta_total = (term_lin_pop + delta_spline) + delta_time + delta_age
            
            # Forecast (Raw Values)
            pred_log_gdp = y_base + delta_total
            pred_gdp     = 10**pred_log_gdp # (N_years, N_samples)
            
            # Add to World Accumulator
            for i, yr in enumerate(yr_t):
                if (scen, yr) in world_agg:
                    world_agg[(scen, yr)] += pred_gdp[i, :]

    # --- Compute World Quantiles (Pooling Scenarios) ---
    print("Computing World Quantiles across scenarios...")
    
    fan_rows = []
    
    # 対象とするシナリオ（主要8種など）
    target_scenarios = [
        "Medium", "Constant-fertility", "Instant-replacement zero migration",
        "Momentum", "Zero-migration", "No change",
        "No fertility below age 18", "Accelerated adolescent-birth-rate decline"
    ]
    # データにあるものだけ使う
    valid_scens = [s for s in target_scenarios if any((s, 2030) in world_agg for s in scenarios_list)]
    if not valid_scens: valid_scens = scenarios_list # Fallback
    
    for yr in range(min_year, max_year + 1):
        # Collect samples from ALL valid scenarios
        # 各シナリオの分布(1000個)を全部結合して、巨大な分布(8000個)を作る
        # これが「Scenario Pooling」の正体
        pooled_samples = []
        for sc in valid_scens:
            if (sc, yr) in world_agg:
                vals = world_agg[(sc, yr)]
                if vals.sum() > 0: # 0でなければ
                    pooled_samples.append(vals)
        
        if not pooled_samples: continue
        
        grand_arr = np.concatenate(pooled_samples)
        
        row = {
            "Year": yr,
            "Median": np.median(grand_arr),
            "p95_lo": np.quantile(grand_arr, 0.025),
            "p95_hi": np.quantile(grand_arr, 0.975),
            "p80_lo": np.quantile(grand_arr, 0.10),
            "p80_hi": np.quantile(grand_arr, 0.90),
            "p50_lo": np.quantile(grand_arr, 0.25),
            "p50_hi": np.quantile(grand_arr, 0.75),
        }
        fan_rows.append(row)

    df_fan = pd.DataFrame(fan_rows)
    df_fan.to_csv(OUT_FAN_CSV, index=False)
    print(f"✓ Saved World Fan Data (Full Uncertainty) to {OUT_FAN_CSV}")

if __name__ == "__main__":
    main()