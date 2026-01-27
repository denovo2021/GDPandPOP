# 03_hierarchical_model_rcs.py
# Layer 3 (Final): Full Hierarchical Model with RCS + Demographic Age Effects
# -----------------------------------------------------------------------------
"""
Full hierarchical GDP model with:
  - Restricted Cubic Splines (RCS) for population-GDP relationship
  - Demographic effects: Working-Age share (WAshare) and Old-Age Dependency (OldDep)
  - Region-level time drift

3-Level Hierarchical Structure:
  Level 1 (Global):
      alpha0 ~ Normal(y_bar, 1)         <- global mean intercept
      beta0 ~ Normal(1, 0.15)           <- global elasticity

  Level 2 (Region):
      alpha_region = alpha0 + a_r       <- region intercepts (sum-to-zero)
      tau_region                        <- region-level time drift (non-centered)

  Level 3 (Country within Region):
      alpha_country ~ Normal(alpha_region, sigma_alpha_country)  <- non-centered

Key technical choices:
  - x := (log10(Pop) - mu) / SD; global RCS basis orthogonalized to [1, x]
  - Non-centered parameterization for hierarchical parameters (better sampling)
  - dt orthogonalized to [1, x, RCS] + clipping (+/-2)
  - Demographic deltas standardized for comparability
  - Student-t likelihood with nu=12 (robust to outliers)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json, warnings
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from config import (
    PATH_MERGED, PATH_MERGED_AGE, PATH_MODEL_HIERARCHICAL,
    PATH_KNOTS, PATH_SCALE_JSON
)

# -----------------------
# Helper: RCS design
# -----------------------
def rcs_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline with linear tails; returns (N, K-2) basis."""
    k = np.asarray(knots); K = k.size
    if K < 3:
        return np.zeros((x.size, 0))
    def d(u, j): return np.maximum(u - k[j], 0.0) ** 3
    cols = []
    for j in range(1, K - 1):
        cols.append(d(x, j)
                    - d(x, K - 1) * (k[K - 1] - k[j]) / (k[K - 1] - k[0])
                    + d(x, 0)     * (k[j]     - k[0]) / (k[K - 1] - k[0]))
    return np.column_stack(cols)

# ============ Data loading ============
df = pd.read_csv(PATH_MERGED_AGE)
df = df.dropna(subset=["ISO3","Country Name","Region","Year","GDP","Population","WAshare","OldDep"]).copy()
df["Year"] = df["Year"].astype(int)

df_mu = pd.read_csv(PATH_MERGED, index_col=0)
if "Log_Population" not in df_mu.columns:
    df_mu["Log_Population"] = np.log10(df_mu["Population"])
MU_GLOBAL = float(df_mu["Log_Population"].mean())

if "Log_Population" not in df.columns:
    df["Log_Population"] = np.log10(df["Population"])
if "Log_GDP" not in df.columns:
    df["Log_GDP"] = np.log10(df["GDP"])

# Base anchors per ISO3 (<=2023 else last)
base = (df.sort_values(["ISO3","Year"])
          .groupby("ISO3", as_index=False)
          .apply(lambda g: g[g["Year"]<=2023].tail(1) if (g["Year"]<=2023).any() else g.tail(1))
          .reset_index(drop=True)[["ISO3","WAshare","OldDep","Year"]]
          .rename(columns={"WAshare":"WAshare_base","OldDep":"OldDep_base","Year":"Year_base"}))
df = df.merge(base, on="ISO3", how="left")

# Deltas and time
df["dWA"] = df["WAshare"] - df["WAshare_base"]
df["dOD"] = df["OldDep"]  - df["OldDep_base"]
df["t_dec"]   = (df["Year"] - 2000) / 10.0
df["t_dec_b"] = (df["Year_base"] - 2000) / 10.0
df["dt_dec"]  = df["t_dec"] - df["t_dec_b"]

# ============ Transforms / Orthogonalization ============
eps = 1e-8

# x scaling and RCS basis (knots persisted)
x_c = df["Log_Population"].to_numpy() - MU_GLOBAL
s_x = float(np.std(x_c))
x_s = x_c / (s_x + eps)

try:
    knots = np.load(PATH_KNOTS)
except Exception:
    knots = np.quantile(x_c, [0.05, 0.35, 0.65, 0.95])
    np.save(PATH_KNOTS, knots)

Z = rcs_design(x_c, knots)           # (N, m)
m = Z.shape[1]
X_ortho = np.column_stack([np.ones_like(x_s), x_s])   # (N, 2)
coef_rcs = np.linalg.lstsq(X_ortho, Z, rcond=None)[0] # (2, m)
Z_tilde = Z - X_ortho @ coef_rcs                      # RCS perpendicular to [1, x]

# Standardize deltas (store their scales)
s_dWA = float(np.nanstd(df["dWA"].to_numpy()) or 0.1)
s_dOD = float(np.nanstd(df["dOD"].to_numpy()) or 0.1)
df["dWA_s"] = df["dWA"] / (s_dWA + eps)
df["dOD_s"] = df["dOD"] / (s_dOD + eps)

# Time transform: standardized dt for comparability
s_dt  = float(np.nanstd(df["dt_dec"].to_numpy()) or 1.0)
dt_s  = df["dt_dec"].to_numpy() / (s_dt + eps)

# Orthogonalize dt to [1, x_s, Z_tilde] and clip (+/-2)
X_base = np.column_stack([np.ones_like(x_s), x_s, Z_tilde])  # (N, 2+m)
coef_dt = np.linalg.lstsq(X_base, dt_s, rcond=None)[0]       # ((2+m),)
dt_s_orth = dt_s - X_base @ coef_dt
dt_clip = 2.0
dt_s_c = np.clip(dt_s_orth, -dt_clip, dt_clip)

# ============ Indices / Coords ============
df["reg_id"] = df["Region"].astype("category").cat.codes
df["cty_id"] = df["Country Name"].astype("category").cat.codes
regions   = df["Region"].astype("category").cat.categories
countries = df["Country Name"].astype("category").cat.categories
ri = df["reg_id"].to_numpy().astype(int)
ci = df["cty_id"].to_numpy().astype(int)
y  = df["Log_GDP"].to_numpy()
coords = {"Region": regions, "Country": countries, "Spline": np.arange(m)}
cty_reg = (
    df[["cty_id","reg_id"]].drop_duplicates().sort_values("cty_id")["reg_id"].to_numpy().astype(int)
)

# Persist transforms for forecasting
scale_dict = {
    "MU_GLOBAL": MU_GLOBAL, "s_x": s_x,
    "s_dWA": s_dWA, "s_dOD": s_dOD, "s_dt": s_dt,
    "knots": knots.tolist(),
    "coef_rcs": coef_rcs.tolist(),
    "coef_dt_proj": coef_dt.tolist(),
    "dt_clip": float(dt_clip)
}
Path(PATH_SCALE_JSON).write_text(json.dumps(scale_dict, ensure_ascii=False, indent=2), encoding="utf-8")

# Backend selection
try:
    import numpyro  # noqa: F401
    _backend = "numpyro"
except Exception:
    _backend = "nutpie"

# ============ Model (v5.1 NCP) ============
with pm.Model(coords=coords) as mdl:
    # Global mean and slope
    y_bar  = float(df["Log_GDP"].mean())
    alpha0 = pm.Normal("alpha0", y_bar, 1.0)
    beta0  = pm.Normal("beta0", 1.0, 0.15)

    # Region intercepts (sum-to-zero)
    sigma_alpha_r = pm.HalfNormal("sigma_alpha_region", 0.5)
    a_r_raw = pm.Normal("a_region_raw", 0.0, sigma_alpha_r, dims="Region")
    a_r     = a_r_raw - pm.math.mean(a_r_raw)
    alpha_r = pm.Deterministic("alpha_region", alpha0 + a_r, dims="Region")

    # Country intercepts: NON-CENTERED (alpha_c = alpha_r + sigma_alpha_by_region * z)
    sigma_alpha_c_r = pm.HalfNormal("sigma_alpha_by_region", 0.09, dims="Region")  # tightened pooling
    z_alpha_c = pm.Normal("z_alpha_country", 0.0, 1.0, dims="Country")
    alpha_c = pm.Deterministic(
        "alpha_country",
        alpha_r[cty_reg] + sigma_alpha_c_r[cty_reg] * z_alpha_c,
        dims="Country"
    )

    # Global RCS (orthogonalized)
    theta_sd = pm.HalfNormal("theta_sd", 0.08)
    theta    = pm.Normal("theta", 0.0, theta_sd, dims="Spline")
    spline   = (Z_tilde * theta).sum(axis=1)

    # Region-level time drift ONLY: NON-CENTERED (tau_r = tau0 + sigma_tau_r * z)
    tau0        = pm.Normal("tau0", 0.0, 0.12)
    sigma_tau_r = pm.HalfNormal("sigma_tau_region", 0.10, dims="Region")
    z_tau_r     = pm.Normal("z_tau_region", 0.0, 1.0, dims="Region")
    tau_r       = pm.Deterministic("tau_region", tau0 + sigma_tau_r * z_tau_r, dims="Region")

    # Demographic effects
    delta_washare = pm.Normal("delta_washare", 0.0, 0.3)
    delta_olddep  = pm.Normal("delta_olddep",  0.0, 0.3)

    # Linear predictor
    mu = (alpha_c[ci]
          + beta0 * x_s
          + spline
          + delta_washare * df["dWA_s"].to_numpy()
          + delta_olddep  * df["dOD_s"].to_numpy()
          + tau_r[ri] * dt_s_c)

    # Robust likelihood (StudentT with nu=12 for robustness)
    sigma = pm.HalfNormal("sigma", 0.10)
    nu = 12.0
    pm.StudentT("Log_GDP_obs", nu=nu, mu=mu, sigma=sigma, observed=y)

    # Bayesian R-squared
    fitted_var = pt.var(mu)
    R2 = pm.Deterministic("R2", fitted_var / (fitted_var + sigma**2))

# ------------------------------ sampling -----------------------------
# Optimized settings (based on regional_model convergence):
#   - draws: 4,000 is sufficient for stable posterior estimates
#   - tune: 2,000 is usually enough for adaptation
#   - target_accept: 0.95 balances accuracy and speed
#   - chains: 4 is standard for convergence diagnostics

if __name__ == '__main__':
    with mdl:
        idata = pm.sample(
            draws=4_000, tune=2_000, chains=4, cores=12,
            target_accept=0.95, nuts_sampler=_backend,
            random_seed=61, return_inferencedata=True,
            idata_kwargs=dict(log_likelihood=True),
        )

# ============ Save ============
    az.to_netcdf(idata, PATH_MODEL_HIERARCHICAL)
    print("Saved:", PATH_MODEL_HIERARCHICAL)

# ============ Load & diagnostics ============
idata = az.from_netcdf(PATH_MODEL_HIERARCHICAL)
print("Loaded:", PATH_MODEL_HIERARCHICAL)

# Helper: list only variables that exist
def existing_vars(idata, names):
    present = set(getattr(idata, "posterior").data_vars)
    return [n for n in names if n in present]

summary_vars = existing_vars(idata, [
    "alpha0", "beta0", "sigma", "R2",
    "sigma_alpha_region", "alpha_region",
    "sigma_alpha_by_region", "alpha_country",
    "tau0", "sigma_tau_region", "tau_region",
    "theta_sd", "theta",
    "delta_washare", "delta_olddep"
])
if summary_vars:
    summ = az.summary(idata, var_names=summary_vars, hdi_prob=0.95, extend=True)
    print(summ[["mean","sd","r_hat","ess_bulk","ess_tail"]].head(80))

# Global R-hat / ESS
rhat_max = float(az.rhat(idata).to_array().max())
ess_min  = float(az.ess(idata, method="bulk").to_array().min())
print(f"[check] max R-hat = {rhat_max:.3f} | min ESS(bulk) = {ess_min:.0f}")

# Divergences & BFMI
print("divergences =", int(np.asarray(idata.sample_stats["diverging"]).sum()))
bfmi = az.bfmi(idata)
try:
    bfmi_arr = np.asarray(getattr(bfmi, "values", bfmi))
except Exception:
    bfmi_arr = np.asarray(list(bfmi.values()))
print("BFMI per chain:", bfmi_arr)
print("BFMI (mean):", float(np.nanmean(bfmi_arr)))

# -------- Utilities for log-likelihood & memory-safe LOO --------
def ensure_loglik(idata, model):
    """Ensure 'log_likelihood' is present; attach if missing."""
    try:
        _ = idata.log_likelihood; return idata
    except AttributeError:
        pass
    try:
        return pm.compute_log_likelihood(idata=idata, model=model, extend_inferencedata=True)
    except TypeError:
        return pm.compute_log_likelihood(model, idata)

def loo_memory_safe(idata, mdl, target_max_bytes=1.0e9, pointwise=True, cast32=True):
    """
    Run PSIS-LOO on a *thinned view* (no in-place assignment; avoids xarray alignment copies).
    Thinning is chosen so obs*draws_thin*(bytes_per_float) <= target_max_bytes.
    If cast32=True, log_likelihood is downcast to float32 before LOO (halves memory).
    """
    idata = ensure_loglik(idata, mdl)
    # take a handle to the single log_likelihood variable (e.g., "Log_GDP_obs")
    ll_name, ll_da = next(iter(idata.log_likelihood.data_vars.items()))
    # dims typically ("chain","draw","obs"); get sizes
    dims = ll_da.dims
    n_obs = ll_da.sizes[dims[-1]]
    n_chain = idata.posterior.sizes["chain"]
    n_draw  = idata.posterior.sizes["draw"]
    total   = n_chain * n_draw

    bytes_per = 4 if cast32 else 8
    max_draws = int(max(1, (target_max_bytes // bytes_per) // n_obs))
    thin = max(1, int(np.ceil(total / max_draws)))

    # Build a *new* InferenceData composed of thinned views (no Dataset assignment)
    post_thin = idata.posterior.isel(draw=slice(None, None, thin))
    ll_thin   = idata.log_likelihood.isel(draw=slice(None, None, thin))
    if cast32:
        # Downcast only the log-likelihood to reduce psislw workspace
        ll_thin = ll_thin.astype("float32")

    id_th = az.InferenceData(posterior=post_thin, log_likelihood=ll_thin)

    with np.errstate(over="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = az.loo(id_th, pointwise=pointwise)

    used = id_th.posterior.sizes["chain"] * id_th.posterior.sizes["draw"]
    ll_da2 = next(iter(id_th.log_likelihood.data_vars.values()))
    print(f"[loo] thin={thin} -> draws {used}/{total}, obs={n_obs}, dtype={ll_da2.dtype}")
    return loo

# ---- Run memory-safe LOO (set pointwise=False if you only need aggregate fit) ----
loo = loo_memory_safe(idata, mdl, target_max_bytes=1.0e9, pointwise=True, cast32=True)
print(loo)
try:
    pk = np.asarray(getattr(loo, "pareto_k", np.array([])))
    if pk.size:
        print("Pareto-k > 0.7:", int((pk > 0.7).sum()))
        print("Pareto-k in (0.5, 0.7]:", int(((pk > 0.5) & (pk <= 0.7)).sum()))
except Exception:
    pass

# Orthogonality sanity checks (should be ~0)
X1 = np.column_stack([np.ones_like(x_s), x_s])
ortho_RCS = (X1.T @ Z_tilde) / len(x_s)
print("max |<[1,x], Z_tilde>| / N =", float(np.max(np.abs(ortho_RCS))))
proj_dt = (X_base.T @ dt_s_c) / len(x_s)
print("max |<[1,x,Z], dt_s_c>| / N =", float(np.max(np.abs(proj_dt))))

# Optional PPC
DO_PPC = False
if DO_PPC:
    ppc = pm.sample_posterior_predictive(idata, var_names=["Log_GDP_obs"], return_inferencedata=True)
    yhat = ppc.posterior_predictive["Log_GDP_obs"].mean(dim=("chain","draw")).values
    resid = y - yhat
    print("RMSE (log10):", float(np.sqrt(np.mean(resid**2))))
