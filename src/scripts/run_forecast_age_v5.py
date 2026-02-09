# run_forecast_age_v5.py
# -----------------------------------------------------------------------------
# Re-projects GDP (2024-2100) using the Age+Time+RCS model v5.1 parameters.
# Incorporates:
#   - Anchor (2023) + Delta approach
#   - Population RCS delta (Orthogonalized)
#   - Time Drift delta (Region-specific tau * dt)
#   - Age Structure delta (Working-Age & Old-Dep)
# -----------------------------------------------------------------------------
import json
import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path

# --- Settings ---
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import (
    PATH_MODEL_HIERARCHICAL_AGE, PATH_SCALE_JSON, PATH_MERGED_AGE,
    PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE
)

NC_PATH = PATH_MODEL_HIERARCHICAL_AGE
SCALE_JSON = PATH_SCALE_JSON
WPP_CSV = PATH_MERGED_AGE  # Contains historical + future WPP scenarios
OUT_GDP_CSV = PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE

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
    coef_rcs  = np.array(scales["coef_rcs"])      # (2, m) for Z_tilde
    s_dWA     = scales["s_dWA"]
    s_dOD     = scales["s_dOD"]
    # Note: s_dt and coef_dt_proj are for training orthogonalization.
    # For prediction, we use the posterior 'tau_region' which is the net effect 
    # on the orthogonalized time axis. However, mathematically, 
    # if we simply use the estimated 'tau_region' coeff on unscaled time?
    # -> The model trained on: tau_r * dt_s_c. 
    # -> We must replicate the exact transform (dt -> dt_s -> dt_s_orth -> clip).
    s_dt         = scales["s_dt"]
    coef_dt_proj = np.array(scales["coef_dt_proj"]) # ((2+m),)
    dt_clip      = scales["dt_clip"]

    # Load Posterior
    idata = az.from_netcdf(NC_PATH)
    post = idata.posterior
    # Extract chains/draws to (Samples, ...)
    beta0 = post["beta0"].stack(sample=("chain","draw")).values # (S,)
    theta = post["theta"].stack(sample=("chain","draw")).values.T # (S, m)
    # alpha_c is NOT needed for Anchor+Delta method
    # tau_region: (Region, S)
    tau_r = post["tau_region"].stack(sample=("chain","draw")).values 
    # age effects
    d_wa  = post["delta_washare"].stack(sample=("chain","draw")).values # (S,)
    d_od  = post["delta_olddep"].stack(sample=("chain","draw")).values  # (S,)
    
    n_samples = beta0.shape[0]
    print(f"Posterior samples: {n_samples}")

    # Load Data (Historical + Projections)
    # We need to find the "Anchor" (Year 2023 or latest) for each Country-Scenario?
    # Usually Anchor is Historical (2023), same for all scenarios.
    df = pd.read_csv(WPP_CSV)
    df = df.dropna(subset=["ISO3","Year","Scenario","Population","WAshare","OldDep"])
    
    # Identify Regions
    # Map Region string to index used in model
    regions = list(post.coords["Region"].values)
    reg2idx = {r: i for i, r in enumerate(regions)}
    
    # Prepare Output
    out_rows = []

    # Process by Country (Anchor is 2023 from 'History' or 'Medium' if history missing?)
    # Assuming 'merged_age.csv' has Scenario='History' or we pick 2023 from 'Medium'
    countries = df["ISO3"].unique()
    
    print(f"Projecting for {len(countries)} countries...")
    
    for iso in countries:
        sub = df[df["ISO3"]==iso]
        # Get Anchor (2023)
        anchor = sub[sub["Year"]==2023].head(1)
        if anchor.empty:
            # Fallback to latest available if 2023 missing
            anchor = sub[sub["Year"] <= 2023].sort_values("Year").tail(1)
        
        if anchor.empty or pd.isna(anchor["GDP"].values[0]):
            continue
            
        y_base     = float(np.log10(anchor["GDP"].values[0]))
        pop_base   = float(np.log10(anchor["Population"].values[0]))
        wa_base    = float(anchor["WAshare"].values[0])
        od_base    = float(anchor["OldDep"].values[0])
        year_base  = int(anchor["Year"].values[0])
        reg_name   = anchor["Region"].values[0]
        if reg_name not in reg2idx: continue
        r_idx      = reg2idx[reg_name]

        # Base Transforms
        x_base_c = pop_base - MU_GLOBAL
        x_base_s = x_base_c / (s_x + 1e-8)
        
        # Base Spline (Orthogonalized)
        Z_base = rcs_design(np.array([x_base_c]), knots) # (1, m)
        X_orth_base = np.column_stack([np.ones(1), np.array([x_base_s])])
        Z_tilde_base = Z_base - X_orth_base @ coef_rcs # (1, m)
        spline_base = (Z_tilde_base @ theta.T).flatten() # (S,)

        # Base Time Term (Orthogonalized & Clipped)
        t_base_dec = (year_base - 2000)/10.0
        # Wait: The model defined dt = t - t_base.
        # In the training code: df["t_dec_b"] was the country's specific base year.
        # Here, the anchor IS the base. So dt_base = 0.
        # dt_s_base = 0.
        # BUT orthogonalization vector X_base includes [1, x, Z].
        # So even if dt=0, the orthogonal component might be non-zero?
        # Let's look at training: dt_s = (t - t_base)/s_dt. At base, t=t_base -> dt_s=0.
        # dt_s_orth = dt_s - X_base @ coef_dt. 
        # So yes, we must subtract the projection of 0 onto X_base.
        X_vec_base = np.concatenate([ [1.0], [x_base_s], Z_tilde_base[0] ]) # (2+m,)
        dt_s_base_raw = 0.0
        dt_s_orth_base = dt_s_base_raw - np.dot(X_vec_base, coef_dt_proj)
        dt_s_c_base = np.clip(dt_s_orth_base, -dt_clip, dt_clip)
        time_term_base = tau_r[r_idx, :] * dt_s_c_base # (S,)

        # Loop Scenarios
        scens = sub["Scenario"].unique()
        for scen in scens:
            # Future rows
            rows = sub[(sub["Scenario"]==scen) & (sub["Year"] > year_base)].copy()
            if rows.empty: continue
            
            # Arrays for vectorization
            pop_t = np.log10(rows["Population"].values)
            wa_t  = rows["WAshare"].values
            od_t  = rows["OldDep"].values
            yr_t  = rows["Year"].values
            
            # 1. Pop Delta
            x_t_c = pop_t - MU_GLOBAL
            x_t_s = x_t_c / (s_x + 1e-8)
            
            # Spline Delta
            Z_t = rcs_design(x_t_c, knots) # (N, m)
            X_orth_t = np.column_stack([np.ones(len(x_t_s)), x_t_s])
            Z_tilde_t = Z_t - X_orth_t @ coef_rcs # (N, m)
            
            # Compute spline term per sample
            # (N, m) @ (m, S) -> (N, S)
            term_spline_t = Z_tilde_t @ theta.T 
            
            # Linear Pop term
            # beta0 * (x_t - x_base) ? No, full calc:
            # y = alpha + beta*x + spline + ...
            # y - y_base = beta*(x_t - x_b) + (spline_t - spline_b) + ...
            term_lin_pop = np.outer(x_t_s - x_base_s, beta0) # (N, S)
            
            delta_spline = term_spline_t - spline_base # (N, S)

            # 2. Time Delta
            # t_dec
            t_dec = (yr_t - 2000)/10.0
            # dt relative to base
            dt_raw = t_dec - t_base_dec
            dt_s   = dt_raw / (s_dt + 1e-8)
            
            # Orthogonalize: We need X_base vector for each future point? 
            # NO. In training, X_base was defined per-observation.
            # "X_base = np.column_stack([np.ones, x_s, Z_tilde])"
            # So we must project dt_s onto the CURRENT [1, x_t, Z_t].
            X_vec_t = np.column_stack([np.ones(len(x_t_s)), x_t_s, Z_tilde_t]) # (N, 2+m)
            # dt_s_orth = dt_s - (X_vec_t @ coef)
            dt_proj = X_vec_t @ coef_dt_proj # (N,)
            dt_s_orth = dt_s - dt_proj
            dt_s_c = np.clip(dt_s_orth, -dt_clip, dt_clip)
            
            # Time Term Delta
            # term_time_t = tau * dt_s_c
            # delta_time = tau * dt_s_c - (tau * dt_s_c_base)
            term_time_t = np.outer(dt_s_c, tau_r[r_idx, :]) # (N, S)
            delta_time  = term_time_t - time_term_base      # (N, S)
            
            # 3. Age Structure Delta
            # dWA_s = (WA_t - WA_base)/s_dWA
            # Note: The model used base-anchored deltas directly. 
            # In training: df["dWA"] = df["WA"] - df["WA_base"].
            # So here:
            d_wa_val = (wa_t - wa_base) / (s_dWA + 1e-8)
            d_od_val = (od_t - od_base) / (s_dOD + 1e-8)
            
            term_age = np.outer(d_wa_val, d_wa) + np.outer(d_od_val, d_od) # (N, S)
            
            # Total Delta Log GDP
            # Delta = beta*dx + dSpline + dTime + dAge
            # Note: "beta0 * (x_t_s - x_base_s)" is already in term_lin_pop? 
            # Yes, term_lin_pop is beta0 * x_t - beta0 * x_b.
            
            delta_log_gdp = term_lin_pop + delta_spline + delta_time + term_age
            
            # Forecast
            pred_log_gdp = y_base + delta_log_gdp # (N, S)
            pred_gdp     = 10**pred_log_gdp       # (N, S) in raw units (dollars)
            
            # Summarize (Median)
            med_gdp = np.median(pred_gdp, axis=1)
            
            # Store
            for i, yr in enumerate(yr_t):
                out_rows.append({
                    "ISO3": iso,
                    "Scenario": scen,
                    "Year": yr,
                    "Pred_Median": med_gdp[i]
                })

    # Save
    df_out = pd.DataFrame(out_rows)
    df_out.to_csv(OUT_GDP_CSV, index=False)
    print(f"Saved projections to {OUT_GDP_CSV}")

if __name__ == "__main__":
    main()