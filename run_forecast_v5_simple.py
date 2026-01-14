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
OUT_GDP_CSV = ROOT / "gdp_predictions_scenarios_rcs_age.csv"

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
    
    # ★修正1: 将来予測ではClipを緩める（そうしないと技術進歩が数年で止まってしまう）
    # dt_clip      = scales["dt_clip"] 
    dt_clip_future = 100.0  # 実質制限なし

    # Load Posterior
    print(f"Loading Posterior from {NC_PATH}...")
    idata = az.from_netcdf(NC_PATH)
    post = idata.posterior
    
    beta0 = post["beta0"].stack(sample=("chain","draw")).values
    theta = post["theta"].stack(sample=("chain","draw")).values.T
    if theta.shape[0] != len(knots)-2: theta = theta.T
    tau_r = post["tau_region"].stack(sample=("chain","draw")).values 
    d_wa  = post["delta_washare"].stack(sample=("chain","draw")).values
    d_od  = post["delta_olddep"].stack(sample=("chain","draw")).values
    
    # Load & Merge Data
    print("Loading History and Scenarios...")
    df_hist = pd.read_csv(HIST_CSV)
    df_pop = pd.read_csv(POP_CSV)
    df_age = pd.read_csv(AGE_CSV)

    for d in [df_pop, df_age]:
        d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    df_pop = df_pop.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))
    df_age = df_age.dropna(subset=["Year"]).assign(Year=lambda d: d["Year"].astype(int))

    print("Merging Population and Age Scenarios...")
    df_scen = pd.merge(df_pop, df_age, on=["ISO3", "Year", "Scenario"], how="inner")
    df_scen = df_scen.dropna(subset=["Population", "WAshare", "OldDep"])
    
    regions = list(post.coords["Region"].values)
    reg2idx = {r: i for i, r in enumerate(regions)}
    
    out_rows = []
    countries = df_scen["ISO3"].unique()
    
    print(f"Projecting for {len(countries)} countries (No Tapering, Relaxed Clipping)...")
    
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

        # --- Base Calculations ---
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
        # BaseのClipは学習時と同じ定義でOKだが、Baseはdt=0近辺なので影響小
        dt_s_c_base = dt_s_orth_base 
        time_term_base = tau_r[r_idx, :] * dt_s_c_base

        # --- Loop Scenarios ---
        scen_sub = df_scen[df_scen["ISO3"]==iso]
        
        for scen in scen_sub["Scenario"].unique():
            rows = scen_sub[(scen_sub["Scenario"]==scen) & (scen_sub["Year"] > year_base)].copy()
            if rows.empty: continue
            
            # Future Arrays
            pop_t = np.log10(rows["Population"].values)
            wa_t  = rows["WAshare"].values
            od_t  = rows["OldDep"].values
            yr_t  = rows["Year"].values
            
            # 1. Pop Delta
            x_t_c = pop_t - MU_GLOBAL
            x_t_s = x_t_c / (s_x + 1e-8)
            Z_t = rcs_design(x_t_c, knots)
            X_orth_t = np.column_stack([np.ones(len(x_t_s)), x_t_s])
            Z_tilde_t = Z_t - X_orth_t @ coef_rcs
            
            term_spline_t = Z_tilde_t @ theta
            term_lin_pop = np.outer(x_t_s - x_base_s, beta0)
            delta_spline = term_spline_t - spline_base

            # 2. Time Delta
            t_dec = (yr_t - 2000)/10.0
            dt_raw = t_dec - t_base_dec
            dt_s   = dt_raw / (s_dt + 1e-8)
            
            X_vec_t = np.column_stack([np.ones(len(x_t_s)), x_t_s, Z_tilde_t])
            dt_proj = X_vec_t @ coef_dt_proj
            dt_s_orth = dt_s - dt_proj
            
            # ★修正2: 将来についてはClipを実質解除 (±100 SD)
            # これにより2040年以降も技術進歩係数(tau)が効き続ける
            dt_s_c = np.clip(dt_s_orth, -dt_clip_future, dt_clip_future)
            
            term_time_t = np.outer(dt_s_c, tau_r[r_idx, :])
            delta_time  = term_time_t - time_term_base
            
            # 3. Age Structure Delta
            d_wa_val = (wa_t - wa_base) / (s_dWA + 1e-8)
            d_od_val = (od_t - od_base) / (s_dOD + 1e-8)
            delta_age = np.outer(d_wa_val, d_wa) + np.outer(d_od_val, d_od)
            
            # ★修正3: Tapering (減衰) を完全に削除
            # 過去の成長を打ち消すバグを回避し、素直な予測線を出す
            delta_total = (term_lin_pop + delta_spline) + delta_time + delta_age
            
            pred_log_gdp = y_base + delta_total
            pred_gdp     = 10**pred_log_gdp
            
            med_gdp = np.median(pred_gdp, axis=1)
            
            for i, yr in enumerate(yr_t):
                out_rows.append({
                    "ISO3": iso,
                    "Scenario": scen,
                    "Year": yr,
                    "Pred_Median": med_gdp[i]
                })

    if not out_rows:
        print("Warning: No projections generated!")
    else:
        df_out = pd.DataFrame(out_rows)
        df_out.to_csv(OUT_GDP_CSV, index=False)
        print(f"Saved UN-TAPERED projections to {OUT_GDP_CSV}")

if __name__ == "__main__":
    main()