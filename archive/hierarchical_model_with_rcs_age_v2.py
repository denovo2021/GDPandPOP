# hierarchical_model_with_rcs_age_v2.py
"""
Hierarchical RCS GDP model with demographic composition & time drift (stabilized, improved)
------------------------------------------------------------------------------------------
Key geometry choices to improve convergence and precision:
  • Center & scale: x = (log10(Pop) − μ) / SD; orthogonalize RCS basis vs [1, x]
  • Region-level partial pooling for intercepts (α_r) and slopes (β_r), sum-to-zero deviations
  • Country-level joint deviations (α, β) per region via 2×2 Cholesky (SDs + rho), non-centered
  • Hierarchical shrinkage for spline weights: global → region scales
  • Time drift τ: region mean + country deviation; priors matched to posterior scale
  • Student-t likelihood with fixed ν=4 for robust yet geometrically simpler tails
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

# ── paths ────────────────────────────────────────────────────────────────────
PROJ = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
PATH_MERGED      = PROJ / "merged_age.csv"      # historical panel + WAshare / OldDep
PATH_MERGED_RAW  = PROJ / "merged.csv"
PATH_KNOTS       = PROJ / "rcs_knots_hier.npy"  # RCS knots (saved/loaded here)
PATH_OUT         = PROJ / "hierarchical_model_with_rcs_age_v2.nc"
PATH_SCALE_JSON  = PROJ / "scale_rcs_age.json"  # saved μ, std(ΔWA,ΔOD,Δt) for forecast

# ── helpers ──────────────────────────────────────────────────────────────────
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

# ═════════════════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ═════════════════════════════════════════════════════════════════════════════
df = pd.read_csv(PATH_MERGED)
df = df.dropna(subset=["ISO3","Country Name","Region","Year","GDP","Population","WAshare","OldDep"]).copy()
df["Year"] = df["Year"].astype(int)

# global μ for centering x
df_mu = pd.read_csv(PATH_MERGED_RAW, index_col=0)
if "Log_Population" not in df_mu.columns:
    df_mu["Log_Population"] = np.log10(df_mu["Population"])
MU_GLOBAL = float(df_mu["Log_Population"].mean())

# logs
if "Log_Population" not in df.columns:
    df["Log_Population"] = np.log10(df["Population"])
if "Log_GDP" not in df.columns:
    df["Log_GDP"] = np.log10(df["GDP"])

# base anchors per ISO3 (≤2023 else last)
base = (df.sort_values(["ISO3","Year"])
          .groupby("ISO3", as_index=False)
          .apply(lambda g: g[g["Year"]<=2023].tail(1) if (g["Year"]<=2023).any() else g.tail(1))
          .reset_index(drop=True)[["ISO3","WAshare","OldDep","Year"]]
          .rename(columns={"WAshare":"WAshare_base","OldDep":"OldDep_base","Year":"Year_base"}))
df = df.merge(base, on="ISO3", how="left")

# deltas and time (anchor-and-delta)
df["dWA"]  = df["WAshare"] - df["WAshare_base"]
df["dOD"]  = df["OldDep"]  - df["OldDep_base"]
df["t_dec"]   = (df["Year"] - 2000) / 10.0
df["t_dec_b"] = (df["Year_base"] - 2000) / 10.0
df["dt_dec"]  = df["t_dec"] - df["t_dec_b"]

# RCS design with linear tails; knots saved/loaded
try:
    knots = np.load(PATH_KNOTS)
except Exception:
    x_tmp = (np.log10(df["Population"].to_numpy()) - MU_GLOBAL)
    knots = np.quantile(x_tmp, [0.05, 0.35, 0.65, 0.95])
    np.save(PATH_KNOTS, knots)

x_c = (df["Log_Population"].to_numpy() - MU_GLOBAL)
s_x = float(np.std(x_c))
x_s = x_c / (s_x + 1e-8)                 # scaled predictor
Z   = rcs_design(x_c, knots)
m   = Z.shape[1]

# Orthogonalize Z against [1, x_s] to reduce α–β–spline confounding
X_ortho = np.column_stack([np.ones_like(x_s), x_s])
# least-squares projection
coef = np.linalg.lstsq(X_ortho, Z, rcond=None)[0]  # shape (2, m)
Z_tilde = Z - X_ortho @ coef                        # orthogonalized basis

# ── scaling for geometric stability ──────────────────────────────────────────
eps  = 1e-8
s_dWA = float(np.nanstd(df["dWA"].to_numpy()) or 0.1)
s_dOD = float(np.nanstd(df["dOD"].to_numpy()) or 0.1)
s_dt  = float(np.nanstd(df["dt_dec"].to_numpy()) or 1.0)
df["dWA_s"] = df["dWA"]   / (s_dWA + eps)
df["dOD_s"] = df["dOD"]   / (s_dOD + eps)
df["dt_s"]  = df["dt_dec"]/ (s_dt  + eps)

scale_dict = {"MU_GLOBAL": MU_GLOBAL, "s_dWA": s_dWA, "s_dOD": s_dOD, "s_dt": s_dt, "s_x": s_x, "knots": knots.tolist()}
with open(PATH_SCALE_JSON, "w", encoding="utf-8") as f:
    json.dump(scale_dict, f, ensure_ascii=False, indent=2)

# indices and coords
df["reg_id"] = df["Region"].astype("category").cat.codes
df["cty_id"] = df["Country Name"].astype("category").cat.codes
regions   = df["Region"].astype("category").cat.categories
countries = df["Country Name"].astype("category").cat.categories
ri = df["reg_id"].to_numpy().astype(int)
ci = df["cty_id"].to_numpy().astype(int)
y  = df["Log_GDP"].to_numpy()

coords = {
    "Region": regions,
    "Country": countries,
    "Spline": np.arange(m),
    "AB": ["alpha","beta"],
}

# country -> region mapping (constant per Country)
cty_reg = (df[["cty_id","reg_id"]]
           .drop_duplicates()
           .sort_values("cty_id")["reg_id"]
           .to_numpy()
           .astype(int))

# ═════════════════════════════════════════════════════════════════════════════
#  MODEL
# ═════════════════════════════════════════════════════════════════════════════
with pm.Model(coords=coords) as mdl:
    # Global centers for α, β
    y_bar  = float(df["Log_GDP"].mean())
    alpha0 = pm.Normal("alpha0", y_bar, 1.5)
    beta0  = pm.Normal("beta0",  0.0,  0.5)

    # Region-level deviations (partial pooling) with sum-to-zero constraint
    sigma_alpha_r = pm.HalfNormal("sigma_alpha_region", 0.6)
    sigma_beta_r  = pm.HalfNormal("sigma_beta_region",  0.3)

    a_r = pm.Normal("a_region_raw", 0.0, sigma_alpha_r, dims="Region")
    b_r = pm.Normal("b_region_raw", 0.0, sigma_beta_r,  dims="Region")

    a_r_c = a_r - pm.math.mean(a_r)  # sum-to-zero across regions
    b_r_c = b_r - pm.math.mean(b_r)

    alpha_r = pm.Deterministic("alpha_region", alpha0 + a_r_c, dims="Region")
    beta_r  = pm.Deterministic("beta_region",  beta0  + b_r_c, dims="Region")

    # Hierarchical shrinkage for spline weights (global → region)
    theta_sd_global = pm.HalfNormal("theta_sd_global", 0.12)
    theta_sd_region = pm.HalfNormal("theta_sd_region", theta_sd_global, dims="Region")
    theta_r = pm.Normal("theta_region", 0.0, theta_sd_region[..., None], dims=("Region","Spline"))

    # Country-level joint (α, β) deviations per region via 2×2 Cholesky (manual)
    sd_a_r = pm.HalfNormal("sd_alpha_region", 0.5, dims="Region")
    sd_b_r = pm.HalfNormal("sd_beta_region",  0.5, dims="Region")
    rho_un = pm.Normal("rho_region_unconstrained", 0.0, 1.0, dims="Region")
    rho_r  = pm.Deterministic("rho_region", pm.math.tanh(rho_un), dims="Region")  # (-1,1)

    # Cholesky components of L = D * L_R (n=2)
    L11 = sd_a_r
    L21 = rho_r * sd_a_r
    L22 = sd_b_r * pm.math.sqrt(1 - rho_r**2)

    # Standard normals for countries in R^2
    z_ab = pm.Normal("z_ab_country", 0.0, 1.0, dims=("Country","AB"))

    # Map each country to its region-specific L and transform
    L11_c = L11[cty_reg]   # (C,)
    L21_c = L21[cty_reg]   # (C,)
    L22_c = L22[cty_reg]   # (C,)

    dev_alpha = L11_c * z_ab[..., 0]
    dev_beta  = L21_c * z_ab[..., 0] + L22_c * z_ab[..., 1]

    alpha_c = pm.Deterministic("alpha_country", alpha_r[cty_reg] + dev_alpha, dims="Country")
    beta_c  = pm.Deterministic("beta_country",  beta_r[cty_reg]  + dev_beta,  dims="Country")

    # Demography deltas (standardized) and time drift (hierarchical)
    delta_washare = pm.Normal("delta_washare", 0.0, 0.5)
    delta_olddep  = pm.Normal("delta_olddep",  0.0, 0.5)

    tau0   = pm.Normal("tau0", 0.0, 0.2)
    sigma_tau_r = pm.HalfNormal("sigma_tau_region", 0.25, dims="Region")
    tau_r  = pm.Normal("tau_region", tau0, sigma_tau_r, dims="Region")

    tau_sd = pm.HalfNormal("tau_sd", 0.3)
    z_tau  = pm.Normal("z_tau_country", 0.0, 1.0, dims="Country")
    tau_c  = pm.Deterministic("tau_country", tau_r[cty_reg] + tau_sd * z_tau, dims="Country")

    # Residual scale & fixed degrees of freedom for Student-t
    sigma = pm.HalfStudentT("sigma", 4, 0.1)
    nu = 4.0

    # RCS contribution using orthogonalized basis
    theta_rows = theta_r[ri, :]                  # (N, m)
    spline     = (theta_rows * Z_tilde).sum(axis=1)

    # Linear predictor (all standardized inputs)
    mu = (alpha_c[ci]
          + beta_c[ci] * x_s
          + spline
          + delta_washare * df["dWA_s"].to_numpy()
          + delta_olddep  * df["dOD_s"].to_numpy()
          + tau_c[ci]     * df["dt_s"].to_numpy())

    pm.StudentT("Log_GDP_obs", nu=nu, mu=mu, sigma=sigma, observed=y)

    # ── sampling (geometry-aware) ────────────────────────────────────────────
    idata = pm.sample(
        draws=8000, tune=9000, chains=4,
        target_accept=0.998, nuts_sampler="nutpie", random_seed=42,
        return_inferencedata=True, idata_kwargs=dict(log_likelihood=True)
    )

# save & diagnostics
az.to_netcdf(idata, PATH_OUT)
print("✓ saved →", PATH_OUT)

summ = az.summary(
    idata,
    var_names=[
        "sigma","tau_sd","delta_washare","delta_olddep",
        "alpha_region","beta_region",
        "sd_alpha_region","sd_beta_region","rho_region",
        "alpha_country","beta_country"
    ],
    hdi_prob=0.95,
    extend=True
)
print(summ[["mean","sd","r_hat","ess_bulk","ess_tail"]].head(18))
rhat_max = float(az.rhat(idata).to_array().max().values)
ess_min  = float(az.ess(idata, method="bulk").to_array().min().values)
print(f"[check] max r_hat = {rhat_max:.3f} | min ESS(bulk) = {ess_min:.0f}")
