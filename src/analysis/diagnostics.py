# diagnostics.py
# =============================== Key Metrics for Manuscript =============================
"""
Key summary metrics for the manuscript (RCS model version), without importing models
or writing NetCDF files. We reconstruct pointwise log-likelihood from posterior
draws + raw data, then run LOO/WAIC. This avoids Windows file locks entirely.

1. MAE of the hierarchical RCS model (log10 scale).
2. DELPD (RCS - linear) with SE; LOO preferred, WAIC fallback.
3. Median world-GDP growth 2025-2040 (95% HDI) pooled across UN scenarios.
5. Share of forecast variance attributable to between-scenario heterogeneity.
6. Marginal R^2 explained by population (linear + RCS terms).
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
import numpy as np
import pandas as pd
import arviz as az
from scipy.special import gammaln, logsumexp

from src.config import PATH_MERGED, PATH_KNOTS, PATH_MODEL_HIERARCHICAL, DIR_OUTPUT

# ------------------------------- config ----------------------------------------
THIN_RCS  = 10     # posterior thinning (RCS)
THIN_LIN  = 10     # posterior thinning (linear)
OBS_CHUNK = 1000   # observation chunk size for log-likelihood

# Optional: HDF5 lock-off (Windows/OneDrive)
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

# ------------------------------- paths ----------------------------------------
PATH_POST_RCS     = PATH_MODEL_HIERARCHICAL
PATH_POST_LINEAR  = DIR_OUTPUT / "hierarchical_model_linear.nc"
PATH_SCENARIO     = DIR_OUTPUT / "gdp_predictions_scenarios_rcs.csv"

HAS_LINEAR = PATH_POST_LINEAR.exists()
if not HAS_LINEAR:
    print(f"[warn] Linear model file not found: {PATH_POST_LINEAR} - DELPD will be skipped.")

# ---- global centering mean (use the same mean everywhere) -----------------------
df_all_for_mu = pd.read_csv(PATH_MERGED, index_col=0)
if "Log_Population" not in df_all_for_mu.columns:
    df_all_for_mu["Log_Population"] = np.log10(df_all_for_mu["Population"])
MU_GLOBAL_LOGPOP = df_all_for_mu["Log_Population"].mean()

# ------------------------------- helpers --------------------------------------
def rcs_design(x_in: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Harrell's restricted cubic spline with linear tails. Returns (N, K-2)."""
    k = np.asarray(knots); K = k.size
    if K < 3:
        return np.zeros((x_in.size, 0))
    def d(u, j):  # truncated cubic
        return np.maximum(u - k[j], 0.0) ** 3
    cols = []
    for j in range(1, K-1):
        term = (d(x_in, j)
                - d(x_in, K-1) * (k[K-1]-k[j])/(k[K-1]-k[0])
                + d(x_in, 0)   * (k[j]   -k[0])/(k[K-1]-k[0]))
        cols.append(term)
    return np.column_stack(cols)

def get_ic(idata):
    try:
        ic = az.loo(idata, var_name="Log_GDP_obs", pointwise=True)
        ic.ic_type = "loo"
        if hasattr(ic, "pareto_k"):
            pk = np.asarray(ic.pareto_k)
            frac_bad = np.mean(pk > 0.7)
            if frac_bad > 0:
                print(f"[warn] LOO: {frac_bad:.1%} of points have Pareto k > 0.7")
        return ic
    except Exception as e:
        print(f"[info] LOO failed ({e}); falling back to WAIC.")
        ic = az.waic(idata, var_name="Log_GDP_obs")
        ic.ic_type = "waic"
        return ic

def to_elpd(ic):
    """Extract (ELPD, SE) for both LOO/WAIC objects."""
    if hasattr(ic, "elpd_loo"):  return float(ic.elpd_loo),  float(ic.se)
    if hasattr(ic, "elpd_waic"): return float(ic.elpd_waic), float(ic.se)
    if hasattr(ic, "loo"):       return float(ic.loo),       float(ic.se)
    if hasattr(ic, "waic"):      return -0.5*float(ic.waic), 0.5*float(ic.se)
    raise AttributeError("No ELPD field found")

# --- add this helper near the top (below to_elpd) -----------------------------
def extract_nu(post, draw_idx):
    """
    Return (chain, draw) array of ν for Student-t from whatever the model stored.
    Prefers 'nu'; falls back to 'nu_minus1' (+1), or 'nu_raw' (+1, clipped).
    """
    if "nu" in post.data_vars:
        return post["nu"].values[:, draw_idx]                         # (C,D2)
    if "nu_minus1" in post.data_vars:
        return post["nu_minus1"].values[:, draw_idx] + 1.0            # (C,D2)
    if "nu_raw" in post.data_vars:
        nu = post["nu_raw"].values[:, draw_idx] + 1.0                 # (C,D2)
        # if your fit clipped ν in the model, mirror it here (safe & stable):
        return np.clip(nu, 2.0, 30.0)
    if "nu_minus1_log__" in post.data_vars:                           # rare
        nu_minus1 = np.exp(post["nu_minus1_log__"].values[:, draw_idx])
        return nu_minus1 + 1.0
    raise KeyError("Neither 'nu', 'nu_minus1', 'nu_raw' found in posterior.")

def student_t_logpdf(y, mu, sigma, nu):
    """
    Vectorized Student-t logpdf.
    y: (K,), mu: (C,D,K), sigma: (C,D)[,(1)], nu: (C,D)[,(1)]
    returns: (C,D,K)
    """
    if sigma.ndim == 2: sigma = sigma[:, :, None]
    if nu.ndim == 2:    nu    = nu[:, :, None]
    z = (y[None, None, :] - mu) / sigma
    return (gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi)
            - np.log(sigma)
            - ((nu + 1.0) / 2.0) * np.log1p((z**2) / nu))

def build_rcs_loglik(idata: az.InferenceData,
                     df_raw: pd.DataFrame,
                     knots: np.ndarray | None,
                     thin: int = 10,
                     obs_chunk: int = 1000) -> az.InferenceData:
    """
    Build log_likelihood InferenceData for the RCS model from posterior draws + raw data.
    No model import; fully in-memory. Ensures x/Z are computed *after* filtering.
    """
    post = idata.posterior
    C = post.sizes["chain"]; D = post.sizes["draw"]
    draw_idx = np.arange(0, D, thin, dtype=int)
    D2 = draw_idx.size

    # Posterior arrays (thinned)
    a_full = post["alpha_country"].values[:, draw_idx, :]       # (C,D2,nC)
    b_full = post["beta_country" ].values[:, draw_idx, :]       # (C,D2,nC)
    sigma  = post["sigma"].values[:, draw_idx]                  # (C,D2)
    nu = extract_nu(post, draw_idx)                             # (C,D2)   

    has_theta = "theta_region" in post.data_vars
    if has_theta:
        th_full = post["theta_region"].values[:, draw_idx, :, :]  # (C,D2,nR,m)
        m = th_full.shape[-1]
    else:
        m = 0

    # Posterior coord orders
    post_countries = post.coords["Country"].values.tolist()
    c_idx = pd.Series(range(len(post_countries)), index=post_countries)
    post_regions = post.coords["Region"].values.tolist()
    r_idx = pd.Series(range(len(post_regions)), index=post_regions)

    # Prepare raw data → filter to modeled countries FIRST
    df = (df_raw.dropna(subset=["Region","Country Name","Population","GDP"]).copy())
    if "Log_Population" not in df.columns: df["Log_Population"] = np.log10(df["Population"])
    if "Log_GDP"        not in df.columns: df["Log_GDP"]        = np.log10(df["GDP"])
    df = df[df["Country Name"].isin(c_idx.index)].copy()  # filter here
    df["ci"] = df["Country Name"].map(c_idx)
    df["ri"] = df["Region"].map(r_idx)
    df = df.reset_index(drop=True)

    # Center with the same global mean style
    x  = (df["Log_Population"] - MU_GLOBAL_LOGPOP).values

    if knots is None:
        # Fallback: compute knots from *this* centered x
        knots = np.quantile(x, [0.05, 0.35, 0.65, 0.95])
    Z  = rcs_design(x, knots)                                # (N, m), aligned

    y_obs = df["Log_GDP"].values
    ci = df["ci"].values.astype(int)
    ri = df["ri"].values.astype(int)
    N  = len(df)

    # Build log-likelihood in chunks
    ll_parts = []
    for s in range(0, N, obs_chunk):
        e = min(s + obs_chunk, N)
        k = e - s
        ci_s = ci[s:e]
        ri_s = ri[s:e]
        x_s  = x [s:e]
        y_s  = y_obs[s:e]
        Z_s  = Z [s:e, :]   # (k, m)

        a = a_full[:, :, ci_s]                  # (C,D2,k)
        b = b_full[:, :, ci_s]                  # (C,D2,k)
        mu = a + b * x_s[None, None, :]         # (C,D2,k)

        if has_theta and m > 0:
            th_obs = th_full[:, :, ri_s, :]     # (C,D2,k,m)
            mu += np.einsum("cdkm,km->cdk", th_obs, Z_s)

        ll_chunk = student_t_logpdf(y_s, mu, sigma, nu)  # (C,D2,k)
        ll_parts.append(ll_chunk)

    ll = np.concatenate(ll_parts, axis=2)       # (C,D2,N)

    da = az.numpy_to_data_array(
        ll, dims=["chain","draw","obs_id"],
        coords={"chain": np.arange(C),
                "draw":  np.arange(D2),
                "obs_id": np.arange(N)}
    )
    return az.from_dict(log_likelihood={"Log_GDP_obs": da})

def build_linear_loglik(idata: az.InferenceData,
                        df_raw: pd.DataFrame,
                        thin: int = 10,
                        obs_chunk: int = 1000) -> az.InferenceData:
    """
    Build log_likelihood InferenceData for the linear model (no spline).
    """
    post = idata.posterior
    C = post.sizes["chain"]; D = post.sizes["draw"]
    draw_idx = np.arange(0, D, thin, dtype=int)
    D2 = draw_idx.size

    a_full = post["alpha_country"].values[:, draw_idx, :]  # (C,D2,nC)
    b_full = post["beta_country" ].values[:, draw_idx, :]  # (C,D2,nC)
    sigma  = post["sigma"].values[:, draw_idx]             # (C,D2)
    nu = extract_nu(post, draw_idx)                        # (C,D2)

    post_countries = post.coords["Country"].values.tolist()
    c_idx = pd.Series(range(len(post_countries)), index=post_countries)

    # Filter FIRST, then compute x aligned to df
    df = (df_raw.dropna(subset=["Region","Country Name","Population","GDP"]).copy())
    if "Log_Population" not in df.columns: df["Log_Population"] = np.log10(df["Population"])
    if "Log_GDP"        not in df.columns: df["Log_GDP"]        = np.log10(df["GDP"])
    df = df[df["Country Name"].isin(c_idx.index)].copy()
    df["ci"] = df["Country Name"].map(c_idx)
    df = df.reset_index(drop=True)

    x  = (df["Log_Population"] - MU_GLOBAL_LOGPOP).values

    y_obs = df["Log_GDP"].values
    ci = df["ci"].values.astype(int)
    N  = len(df)

    ll_parts = []
    for s in range(0, N, obs_chunk):
        e = min(s + obs_chunk, N)
        k = e - s
        ci_s = ci[s:e]
        x_s  = x [s:e]
        y_s  = y_obs[s:e]

        a = a_full[:, :, ci_s]                 # (C,D2,k)
        b = b_full[:, :, ci_s]                 # (C,D2,k)
        mu = a + b * x_s[None, None, :]        # (C,D2,k)

        ll_chunk = student_t_logpdf(y_s, mu, sigma, nu)  # (C,D2,k)
        ll_parts.append(ll_chunk)

    ll = np.concatenate(ll_parts, axis=2)      # (C,D2,N)

    da = az.numpy_to_data_array(
        ll, dims=["chain","draw","obs_id"],
        coords={"chain": np.arange(C),
                "draw":  np.arange(D2),
                "obs_id": np.arange(ll.shape[2])}
    )
    return az.from_dict(log_likelihood={"Log_GDP_obs": da})

# ----------------------------------- 1) MAE -----------------------------------
idata_rcs = az.from_netcdf(PATH_POST_RCS)

# posterior means for α, β, θ (RCS)
a_hat = idata_rcs.posterior["alpha_country"].mean(("chain","draw")).values
b_hat = idata_rcs.posterior["beta_country" ]   .mean(("chain","draw")).values
th_da = (idata_rcs.posterior["theta_region"].mean(("chain","draw"))
         if "theta_region" in idata_rcs.posterior.data_vars else None)

# Load the training panel (same structure as used to fit the model)
df_fit = (pd.read_csv(PATH_MERGED, index_col=0)
          .dropna(subset=["Region","Country Name","Population","GDP"]))
if "Log_Population" not in df_fit.columns: df_fit["Log_Population"] = np.log10(df_fit["Population"])
if "Log_GDP"        not in df_fit.columns: df_fit["Log_GDP"]        = np.log10(df_fit["GDP"])

# Align rows to the posterior’s Country coordinate order (by country name)
post_countries = idata_rcs.posterior.coords["Country"].values.tolist()
c_idx = pd.Series(range(len(post_countries)), index=post_countries)
df_fit = df_fit[df_fit["Country Name"].isin(c_idx.index)].copy()
df_fit["ci"] = df_fit["Country Name"].map(c_idx)
df_fit = df_fit.reset_index(drop=True)

# Same centering as in the R² block: center by the data-side mean, then recompute the RCS basis
mu_fit = df_fit["Log_Population"].mean()
x  = (df_fit["Log_Population"] - mu_fit).values
# Do not reuse saved knots here (they can misalign); recompute from x (same procedure as during training)
knots0 = np.quantile(x, [0.05, 0.35, 0.65, 0.95])
Z0 = rcs_design(x, knots0)

# Tile region-level θ across rows and compute θ · s(x)
if th_da is not None:
    post_regions = idata_rcs.posterior.coords["Region"].values
    th_df = pd.DataFrame(th_da.transpose("Region","Spline").values, index=post_regions)
    th_row = th_df.loc[df_fit["Region"].values].values            # (N, m)
    spline = np.sum(th_row * Z0, axis=1)                          # (N,)
else:
    spline = 0.0

# Plug-in fitted mean: μ̂ = α̂ + β̂ x + θ̂ · s(x)  (the mean of the Student-t is μ)
mu_hat = (a_hat[df_fit["ci"].values] +
          b_hat[df_fit["ci"].values] * x +
          spline)
y_obs  = df_fit["Log_GDP"].values

mae = float(np.mean(np.abs(y_obs - mu_hat)))
print(f"[1] MAE (log10 units): {mae:.3f}")
print(f"[1] corr(y_obs, fitted) = {np.corrcoef(y_obs, mu_hat)[0,1]:.3f}")

# ------------------------------ 2) ΔELPD (RCS – linear) -----------------------
# Build log-likelihood idatas *without* importing models or writing files.
df_raw_all = pd.read_csv(PATH_MERGED, index_col=0)

# knots: reuse saved; else fallback from data (centered x after filtering inside function)
knots_for_ll = np.load(PATH_KNOTS) if PATH_KNOTS.exists() else None

idata_rcs_ll = build_rcs_loglik(az.from_netcdf(PATH_POST_RCS),
                                df_raw_all, knots=knots_for_ll,
                                thin=THIN_RCS, obs_chunk=OBS_CHUNK)

if HAS_LINEAR:
    idata_lin_ll = build_linear_loglik(az.from_netcdf(PATH_POST_LINEAR),
                                       df_raw_all, thin=THIN_LIN, obs_chunk=OBS_CHUNK)

    ic_rcs = get_ic(idata_rcs_ll)
    ic_lin = get_ic(idata_lin_ll)
    elpd_r, se_r = to_elpd(ic_rcs)
    elpd_l, se_l = to_elpd(ic_lin)
    d_elpd  = elpd_r - elpd_l
    se_diff = float(np.sqrt(se_r**2 + se_l**2))
    log_w = np.array([elpd_r, elpd_l], dtype=float)
    w_rcs = float(np.exp(log_w[0] - logsumexp(log_w)))
    w_lin = 1.0 - w_rcs

    print("\n[2] Hierarchical model comparison (manual log-likelihood)")
    print(f"   RCS      ({ic_rcs.ic_type.upper()}): ELPD = {elpd_r:8.1f}   SE = {se_r:6.1f}")
    print(f"   Linear   ({ic_lin.ic_type.upper()}): ELPD = {elpd_l:8.1f}   SE = {se_l:6.1f}")
    print(f"   ΔELPD                        : {d_elpd:8.1f}   (SE = {se_diff:6.1f})")
    print(f"   Akaike weights →  RCS: {w_rcs:.2f} | Linear: {w_lin:.2f}")
else:
    print("\n[2] Hierarchical model comparison skipped (linear model file missing).")

# --------------------- 3) World GDP growth 2025→2040 (pooled) -----------------
df_scen = pd.read_csv(PATH_SCENARIO)
yr_a, yr_b = 2025, 2040

totals = (df_scen[df_scen["Year"].isin([yr_a, yr_b])]
          .groupby(["Scenario","Year"])[["Pred_Median","Pred_Lower","Pred_Upper"]]
          .sum()
          .unstack("Year"))

growth_pct = 100 * (totals["Pred_Median", yr_b] - totals["Pred_Median", yr_a]) / totals["Pred_Median", yr_a]

se_a = (np.log10(totals["Pred_Upper", yr_a] / totals["Pred_Lower", yr_a]) / (2*1.96)).clip(lower=1e-6)
se_b = (np.log10(totals["Pred_Upper", yr_b] / totals["Pred_Lower", yr_b]) / (2*1.96)).clip(lower=1e-6)

df_ic = (pd.DataFrame({"growth_pct": growth_pct, "se_log": np.sqrt(se_a**2 + se_b**2)})
         .replace([np.inf,-np.inf], np.nan).dropna())

if df_ic.empty:
    g_vals = growth_pct.dropna().values
    if g_vals.size < 2:
        raise RuntimeError("Need ≥2 variants to estimate world-growth spread.")
    g_med = np.median(g_vals)
    hdi_lo, hdi_hi = np.quantile(g_vals, [0.025, 0.975])
    heter_pct = 100.0
else:
    log_g = np.log10(df_ic["growth_pct"].values)
    se    = df_ic["se_log"].values
    w0    = 1 / se**2
    Q     = np.sum(w0*log_g**2) - (np.sum(w0*log_g)**2)/np.sum(w0)
    tau2  = max(0, (Q - (len(w0)-1)) / (np.sum(w0) - np.sum(w0**2)/np.sum(w0)))
    w_re  = 1 / (se**2 + tau2)
    mu_re = np.sum(w_re*log_g) / np.sum(w_re)
    var_re= 1 / np.sum(w_re)
    draws = np.random.normal(mu_re, np.sqrt(var_re + tau2), 40_000)
    g_med = np.median(10**draws)
    hdi_lo, hdi_hi = np.quantile(10**draws, [0.025, 0.975])
    heter_pct = 100 * tau2 / (tau2 + var_re)

print(f"[3] World GDP growth {yr_a}-{yr_b}: {g_med:.0f}% "
      f"(95 % HDI {hdi_lo:.0f}%–{hdi_hi:.0f}%), "
      f"between-scenario heterogeneity = {heter_pct:.0f}%")

# ----------------- 4) Between-scenario heterogeneity (share) -------------------
df_scen["log_median"] = np.log10(df_scen["Pred_Median"])

def heterogeneity_share(g):
    se = (np.log10(g["Pred_Upper"]) - np.log10(g["Pred_Lower"])) / (2*1.96)
    se = pd.to_numeric(se, errors="coerce").replace([np.inf, -np.inf], np.nan)
    se = se.fillna(se.median()).clip(lower=1e-6)  # guard against zeros/NaN
    w  = 1 / se**2
    w_sum = np.sum(w)
    if w_sum <= 0 or not np.isfinite(w_sum):
        return np.nan
    Q  = np.sum(w * g["log_median"]**2) - (np.sum(w * g["log_median"])**2) / w_sum
    tau2 = max(0.0, (Q - (len(w)-1)) / (w_sum - np.sum(w**2)/w_sum))
    within = float(np.mean(se)**2)
    return tau2 / (tau2 + within) if (tau2 + within) > 0 else np.nan

# pandas の警告回避（include_groups=False は新しめなので、素直に list→mean）
shares = []
for (cty, yr), g in df_scen.groupby(["Country","Year"]):
    shares.append(heterogeneity_share(g))
share = np.nanmean(shares)
print(f"\n[4] Between-scenario heterogeneity (country-year avg): {share*100:.0f}%")

share = (df_scen.groupby(["Country","Year"]).apply(heterogeneity_share).mean())
print(f"\n[4] Between-scenario heterogeneity (country-year avg): {share*100:.0f}%")

# ------------------------- 6) Marginal R² (population) ------------------------
a_hat = idata_rcs.posterior["alpha_country"].mean(("chain","draw")).values
b_hat = idata_rcs.posterior["beta_country" ].mean(("chain","draw")).values
th_da = (idata_rcs.posterior["theta_region"].mean(("chain","draw"))
         if "theta_region" in idata_rcs.posterior.data_vars else None)

df_raw2 = (pd.read_csv(PATH_MERGED, index_col=0)
           .dropna(subset=["Region","Country Name","Population","GDP"]))

if "Log_Population" not in df_raw2.columns: df_raw2["Log_Population"] = np.log10(df_raw2["Population"])
if "Log_GDP"        not in df_raw2.columns: df_raw2["Log_GDP"]        = np.log10(df_raw2["GDP"])

# Filter FIRST to align with posterior countries
post_countries2 = idata_rcs.posterior.coords["Country"].values.tolist()
c_idx2 = pd.Series(range(len(post_countries2)), index=post_countries2)
df_raw2 = df_raw2[df_raw2["Country Name"].isin(c_idx2.index)].copy()
df_raw2["ci"] = df_raw2["Country Name"].map(c_idx2)
df_raw2 = df_raw2.reset_index(drop=True)

mu_logpop2 = df_raw2["Log_Population"].mean()
x2  = (df_raw2["Log_Population"] - mu_logpop2).values
try:
    knots2 = np.load(PATH_KNOTS)
except Exception:
    knots2 = np.quantile(x2, [0.05, 0.35, 0.65, 0.95])
Z2 = rcs_design(x2, knots2)

if th_da is not None:
    post_regions2 = idata_rcs.posterior.coords["Region"].values
    th_df2 = pd.DataFrame(th_da.transpose("Region","Spline").values, index=post_regions2)
    th_row2 = th_df2.loc[df_raw2["Region"].values].values
else:
    th_row2 = np.zeros((len(df_raw2), 0))

mu_hat2 = (a_hat[df_raw2["ci"].values] +
           b_hat[df_raw2["ci"].values] * x2 +
           np.sum(th_row2 * Z2, axis=1))

y_obs2 = df_raw2["Log_GDP"].values
mask2  = np.isfinite(mu_hat2) & np.isfinite(y_obs2)
r2_marg = 1.0 - np.var(y_obs2[mask2] - mu_hat2[mask2], ddof=1) / np.var(y_obs2[mask2], ddof=1)
print(f"[5] Population terms explain ≈ {100*r2_marg:.1f}% of cross-country log-GDP variance.")
