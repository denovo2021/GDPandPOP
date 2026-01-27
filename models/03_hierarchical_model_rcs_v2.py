# 03_hierarchical_model_rcs_v2.py
# Layer 3 (Final): Improved Hierarchical Model with RCS + Demographics
# -----------------------------------------------------------------------------
"""
IMPROVEMENTS OVER v1:
  1. Consistent scaling: RCS computed on standardized x, knots on standardized scale
  2. Basis normalization: RCS columns scaled to unit variance for numerical stability
  3. Learnable nu: Degrees of freedom estimated from data (not fixed at 12)
  4. Optional country-level slope variation (controlled by flag)
  5. Improved prior calibration based on data characteristics
  6. Better documentation of mathematical structure

Mathematical Model:
  log₁₀(GDP)_{crt} = α_c + β₀·x_s + Σⱼ θⱼ·Z̃ⱼ(x_s) + δ_WA·ΔWA + δ_OD·ΔOD + τ_r·Δt + ε

  where:
    c = country, r = region, t = time
    x_s = (log₁₀(Pop) - μ) / σ        [standardized log-population]
    Z̃(x_s) = orthonormalized RCS basis
    ΔWA, ΔOD = standardized demographic changes from base year
    Δt = orthogonalized, clipped time deviation

Hierarchical Structure:
  Level 1 (Global):    α₀, β₀, θ, δ_WA, δ_OD
  Level 2 (Region):    α_r = α₀ + a_r,  τ_r
  Level 3 (Country):   α_c = α_r + σ_c · z_c   [non-centered]

Key Technical Choices:
  - RCS knots placed on standardized scale for consistency
  - RCS basis orthonormalized: unit variance, zero correlation with [1, x_s]
  - All predictors standardized for comparable coefficient interpretation
  - Non-centered parameterization for hierarchical terms (better HMC geometry)
  - Student-t likelihood with learnable nu (robust yet adaptive)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import warnings
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az

from config import (
    PATH_MERGED, PATH_MERGED_AGE, DIR_OUTPUT, DIR_CACHE
)

# Output paths for v2
PATH_MODEL_V2 = DIR_OUTPUT / "hierarchical_model_rcs_v2.nc"
PATH_KNOTS_V2 = DIR_CACHE / "rcs_knots_v2.npy"
PATH_SCALE_V2 = DIR_CACHE / "scale_rcs_v2.json"

# =============================================================================
# Configuration Flags
# =============================================================================
ESTIMATE_NU = True           # Learn degrees of freedom (vs fixed nu=12)
COUNTRY_SLOPES = False       # Add country-level slope variation (increases complexity)
N_KNOTS = 5                  # Number of RCS knots (5 gives 3 basis functions)
KNOT_QUANTILES = [0.05, 0.275, 0.50, 0.725, 0.95]  # Symmetric placement

# =============================================================================
# Improved RCS Implementation
# =============================================================================
def rcs_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Harrell's Restricted Cubic Spline basis with linear tails.

    For K knots, produces K-2 basis functions that are:
    - Cubic between adjacent knots
    - Linear beyond the boundary knots (crucial for extrapolation)

    Parameters
    ----------
    x : array (N,)
        Predictor values (should be on same scale as knots)
    knots : array (K,)
        Knot positions, must be sorted ascending

    Returns
    -------
    Z : array (N, K-2)
        RCS basis matrix
    """
    k = np.asarray(knots)
    K = k.size

    if K < 3:
        return np.zeros((x.size, 0))

    # Truncated power basis
    def d(u, j):
        return np.maximum(u - k[j], 0.0) ** 3

    # Compute restricted basis functions
    # Each rcs_j enforces second derivative = 0 at boundary knots
    cols = []
    for j in range(1, K - 1):
        # Harrell's formula ensures linear tails
        basis_j = (
            d(x, j)
            - d(x, K - 1) * (k[K - 1] - k[j]) / (k[K - 1] - k[0])
            + d(x, 0)     * (k[j]     - k[0]) / (k[K - 1] - k[0])
        )
        cols.append(basis_j)

    return np.column_stack(cols) if cols else np.zeros((x.size, 0))


def orthonormalize_rcs(Z: np.ndarray, X_linear: np.ndarray) -> tuple:
    """
    Orthonormalize RCS basis with respect to linear terms.

    This ensures:
    1. RCS coefficients capture pure nonlinearity (orthogonal to linear trend)
    2. Basis columns have unit variance (comparable coefficient scales)
    3. Basis columns are mutually orthogonal (cleaner interpretation)

    Parameters
    ----------
    Z : array (N, m)
        Raw RCS basis
    X_linear : array (N, 2)
        [1, x] matrix for linear terms

    Returns
    -------
    Z_ortho : array (N, m)
        Orthonormalized RCS basis
    transform_params : dict
        Parameters needed to apply same transform to new data
    """
    N, m = Z.shape

    if m == 0:
        return Z, {"proj_coef": np.zeros((2, 0)), "scales": np.array([])}

    # Step 1: Remove linear component (project out [1, x])
    proj_coef = np.linalg.lstsq(X_linear, Z, rcond=None)[0]  # (2, m)
    Z_resid = Z - X_linear @ proj_coef

    # Step 2: Standardize each column to unit variance
    scales = np.std(Z_resid, axis=0, ddof=1)
    scales = np.where(scales < 1e-10, 1.0, scales)  # Prevent division by zero
    Z_scaled = Z_resid / scales

    # Step 3: Optional QR orthogonalization for mutual orthogonality
    # (Skip if only 1-2 columns, as computational overhead not worth it)
    if m > 2:
        Q, R = np.linalg.qr(Z_scaled)
        Z_ortho = Q * np.sqrt(N)  # Rescale to maintain unit variance
        # Store R for inverse transform if needed
        transform_params = {
            "proj_coef": proj_coef,
            "scales": scales,
            "R_qr": R / np.sqrt(N)
        }
    else:
        Z_ortho = Z_scaled
        transform_params = {
            "proj_coef": proj_coef,
            "scales": scales,
        }

    return Z_ortho, transform_params


def apply_rcs_transform(x_new: np.ndarray, knots: np.ndarray,
                        transform_params: dict, x_mean: float, x_std: float) -> np.ndarray:
    """
    Apply RCS transformation to new data using saved parameters.
    Essential for consistent forecasting.
    """
    # Standardize new x
    x_s = (x_new - x_mean) / x_std

    # Compute raw RCS on standardized scale
    Z_raw = rcs_design(x_s, knots)

    if Z_raw.shape[1] == 0:
        return Z_raw

    # Apply saved orthonormalization
    X_linear = np.column_stack([np.ones_like(x_s), x_s])
    Z_resid = Z_raw - X_linear @ transform_params["proj_coef"]
    Z_scaled = Z_resid / transform_params["scales"]

    if "R_qr" in transform_params:
        # Apply QR transform
        Z_ortho = Z_scaled @ np.linalg.inv(transform_params["R_qr"])
    else:
        Z_ortho = Z_scaled

    return Z_ortho


# =============================================================================
# Data Loading and Preparation
# =============================================================================
print("Loading data...")
df = pd.read_csv(PATH_MERGED_AGE)
df = df.dropna(subset=["ISO3", "Country Name", "Region", "Year",
                       "GDP", "Population", "WAshare", "OldDep"]).copy()
df["Year"] = df["Year"].astype(int)

# Load reference data for global mean (ensures consistency with other models)
df_ref = pd.read_csv(PATH_MERGED, index_col=0)
if "Log_Population" not in df_ref.columns:
    df_ref["Log_Population"] = np.log10(df_ref["Population"])

# Compute log transforms
df["Log_Population"] = np.log10(df["Population"])
df["Log_GDP"] = np.log10(df["GDP"])

# Global standardization parameters
MU_GLOBAL = float(df_ref["Log_Population"].mean())
SD_GLOBAL = float(df_ref["Log_Population"].std())

# Standardized predictor (this is the scale for RCS)
x_raw = df["Log_Population"].to_numpy()
x_s = (x_raw - MU_GLOBAL) / SD_GLOBAL

# =============================================================================
# RCS Basis Construction (on standardized scale)
# =============================================================================
print(f"Constructing RCS with {N_KNOTS} knots...")

# Place knots on standardized scale
knots = np.quantile(x_s, KNOT_QUANTILES[:N_KNOTS])
print(f"  Knots (standardized): {knots}")
print(f"  Knots (original log scale): {knots * SD_GLOBAL + MU_GLOBAL}")

# Compute and orthonormalize RCS basis
Z_raw = rcs_design(x_s, knots)
X_linear = np.column_stack([np.ones_like(x_s), x_s])
Z_tilde, rcs_transform = orthonormalize_rcs(Z_raw, X_linear)
m = Z_tilde.shape[1]
print(f"  RCS basis dimensions: ({Z_tilde.shape[0]}, {m})")

# Verify orthogonality
ortho_check = np.abs(X_linear.T @ Z_tilde / len(x_s))
print(f"  Orthogonality check (should be ~0): max={ortho_check.max():.2e}")

# =============================================================================
# Demographic Variables (anchored to base year)
# =============================================================================
print("Preparing demographic variables...")

# Base year anchors per country (2023 or latest available)
base = (df.sort_values(["ISO3", "Year"])
        .groupby("ISO3", as_index=False)
        .apply(lambda g: g[g["Year"] <= 2023].tail(1) if (g["Year"] <= 2023).any() else g.tail(1))
        .reset_index(drop=True)[["ISO3", "WAshare", "OldDep", "Year"]]
        .rename(columns={"WAshare": "WAshare_base", "OldDep": "OldDep_base", "Year": "Year_base"}))
df = df.merge(base, on="ISO3", how="left")

# Compute deltas from base year
df["dWA"] = df["WAshare"] - df["WAshare_base"]
df["dOD"] = df["OldDep"] - df["OldDep_base"]

# Time variables
df["t_dec"] = (df["Year"] - 2000) / 10.0
df["t_dec_b"] = (df["Year_base"] - 2000) / 10.0
df["dt_dec"] = df["t_dec"] - df["t_dec_b"]

# Standardize demographic deltas
eps = 1e-8
s_dWA = float(np.nanstd(df["dWA"].to_numpy()) or 0.1)
s_dOD = float(np.nanstd(df["dOD"].to_numpy()) or 0.1)
df["dWA_s"] = df["dWA"] / (s_dWA + eps)
df["dOD_s"] = df["dOD"] / (s_dOD + eps)

# Time: standardize and orthogonalize to [1, x_s, Z_tilde]
s_dt = float(np.nanstd(df["dt_dec"].to_numpy()) or 1.0)
dt_s = df["dt_dec"].to_numpy() / (s_dt + eps)

X_for_dt = np.column_stack([np.ones_like(x_s), x_s, Z_tilde])
coef_dt = np.linalg.lstsq(X_for_dt, dt_s, rcond=None)[0]
dt_s_orth = dt_s - X_for_dt @ coef_dt
dt_clip = 2.0
dt_s_c = np.clip(dt_s_orth, -dt_clip, dt_clip)

# =============================================================================
# Indices and Coordinates
# =============================================================================
df["reg_id"] = df["Region"].astype("category").cat.codes
df["cty_id"] = df["Country Name"].astype("category").cat.codes
regions = df["Region"].astype("category").cat.categories
countries = df["Country Name"].astype("category").cat.categories

ri = df["reg_id"].to_numpy().astype(int)
ci = df["cty_id"].to_numpy().astype(int)
y = df["Log_GDP"].to_numpy()

# Country-to-region mapping
cty_reg = (df[["cty_id", "reg_id"]]
           .drop_duplicates()
           .sort_values("cty_id")["reg_id"]
           .to_numpy().astype(int))

n_regions = len(regions)
n_countries = len(countries)
n_obs = len(df)

print(f"Data: {n_obs} observations, {n_countries} countries, {n_regions} regions")

coords = {
    "Region": regions,
    "Country": countries,
    "Spline": np.arange(m)
}

# =============================================================================
# Save Transform Parameters for Forecasting
# =============================================================================
scale_dict = {
    "MU_GLOBAL": MU_GLOBAL,
    "SD_GLOBAL": SD_GLOBAL,
    "s_dWA": s_dWA,
    "s_dOD": s_dOD,
    "s_dt": s_dt,
    "knots": knots.tolist(),
    "rcs_proj_coef": rcs_transform["proj_coef"].tolist(),
    "rcs_scales": rcs_transform["scales"].tolist(),
    "coef_dt_proj": coef_dt.tolist(),
    "dt_clip": float(dt_clip),
    "N_KNOTS": N_KNOTS,
    "KNOT_QUANTILES": KNOT_QUANTILES[:N_KNOTS]
}
if "R_qr" in rcs_transform:
    scale_dict["rcs_R_qr"] = rcs_transform["R_qr"].tolist()

np.save(PATH_KNOTS_V2, knots)
Path(PATH_SCALE_V2).write_text(
    json.dumps(scale_dict, ensure_ascii=False, indent=2),
    encoding="utf-8"
)
print(f"Saved transform parameters to {PATH_SCALE_V2}")

# =============================================================================
# Backend Selection
# =============================================================================
try:
    import numpyro
    _backend = "numpyro"
except ImportError:
    _backend = "nutpie"
print(f"Using sampling backend: {_backend}")

# =============================================================================
# Model Specification
# =============================================================================
print("Building model...")

with pm.Model(coords=coords) as model_v2:

    # =========================================================================
    # Level 1: Global Parameters
    # =========================================================================

    # Global intercept (at mean log-population)
    y_bar = float(df["Log_GDP"].mean())
    y_sd = float(df["Log_GDP"].std())
    alpha0 = pm.Normal("alpha0", mu=y_bar, sigma=1.0)

    # Global linear elasticity
    # Prior centered on 1 (GDP proportional to population)
    # SD of 0.15 allows for reasonable deviation
    beta0 = pm.Normal("beta0", mu=1.0, sigma=0.15)

    # RCS coefficients (pure nonlinearity)
    # Hierarchical shrinkage: theta_j ~ Normal(0, theta_sd)
    theta_sd = pm.HalfNormal("theta_sd", sigma=0.15)
    theta = pm.Normal("theta", mu=0.0, sigma=theta_sd, dims="Spline")

    # Demographic effects (standardized coefficients)
    # Prior SD of 0.3 corresponds to ~0.3 log-units per SD change
    delta_washare = pm.Normal("delta_washare", mu=0.0, sigma=0.3)
    delta_olddep = pm.Normal("delta_olddep", mu=0.0, sigma=0.3)

    # =========================================================================
    # Level 2: Region Parameters
    # =========================================================================

    # Region intercept deviations (sum-to-zero constraint)
    sigma_alpha_r = pm.HalfNormal("sigma_alpha_region", sigma=0.5)
    a_r_raw = pm.Normal("a_region_raw", mu=0.0, sigma=sigma_alpha_r, dims="Region")
    a_r = a_r_raw - pm.math.mean(a_r_raw)  # Sum-to-zero
    alpha_r = pm.Deterministic("alpha_region", alpha0 + a_r, dims="Region")

    # Region-level time drift (non-centered parameterization)
    tau0 = pm.Normal("tau0", mu=0.0, sigma=0.12)
    sigma_tau_r = pm.HalfNormal("sigma_tau_region", sigma=0.10)
    z_tau_r = pm.Normal("z_tau_region", mu=0.0, sigma=1.0, dims="Region")
    tau_r = pm.Deterministic("tau_region", tau0 + sigma_tau_r * z_tau_r, dims="Region")

    # =========================================================================
    # Level 3: Country Parameters
    # =========================================================================

    # Country intercepts (non-centered, nested within regions)
    sigma_alpha_c = pm.HalfNormal("sigma_alpha_country", sigma=0.15)
    z_alpha_c = pm.Normal("z_alpha_country", mu=0.0, sigma=1.0, dims="Country")
    alpha_c = pm.Deterministic(
        "alpha_country",
        alpha_r[cty_reg] + sigma_alpha_c * z_alpha_c,
        dims="Country"
    )

    # Optional: Country-level slope variation
    if COUNTRY_SLOPES:
        sigma_beta_c = pm.HalfNormal("sigma_beta_country", sigma=0.05)
        z_beta_c = pm.Normal("z_beta_country", mu=0.0, sigma=1.0, dims="Country")
        beta_c = pm.Deterministic(
            "beta_country",
            beta0 + sigma_beta_c * z_beta_c,
            dims="Country"
        )
        linear_term = beta_c[ci] * x_s
    else:
        linear_term = beta0 * x_s

    # =========================================================================
    # Linear Predictor
    # =========================================================================

    # Spline contribution
    spline_term = pm.math.dot(Z_tilde, theta)

    # Full linear predictor
    mu = (
        alpha_c[ci]                              # Country intercept
        + linear_term                            # Linear elasticity
        + spline_term                            # RCS nonlinearity
        + delta_washare * df["dWA_s"].to_numpy() # Working-age effect
        + delta_olddep * df["dOD_s"].to_numpy()  # Old-age dependency effect
        + tau_r[ri] * dt_s_c                     # Time drift
    )

    # =========================================================================
    # Likelihood
    # =========================================================================

    # Residual scale
    sigma = pm.HalfNormal("sigma", sigma=0.15)

    # Degrees of freedom (learnable or fixed)
    if ESTIMATE_NU:
        # Gamma prior for nu gives mass on reasonable values (3-30)
        nu_minus2 = pm.Gamma("nu_minus2", alpha=2.0, beta=0.1)
        nu = pm.Deterministic("nu", nu_minus2 + 2.0)  # Ensure nu > 2
    else:
        nu = 12.0

    # Student-t likelihood (robust to outliers)
    obs = pm.StudentT("Log_GDP_obs", nu=nu, mu=mu, sigma=sigma, observed=y)

    # =========================================================================
    # Derived Quantities
    # =========================================================================

    # Bayesian R-squared
    fitted_var = pt.var(mu)
    R2 = pm.Deterministic("R2", fitted_var / (fitted_var + sigma**2))

    # Effective elasticity at different population levels
    # (Useful for understanding nonlinear relationship)
    x_eval = np.array([-1.5, 0.0, 1.5])  # Low, mean, high population
    Z_eval = apply_rcs_transform(
        x_eval * SD_GLOBAL + MU_GLOBAL, knots, rcs_transform, MU_GLOBAL, SD_GLOBAL
    )

    # Marginal elasticity = d(log GDP)/d(log Pop) = beta0 + sum(theta * dZ/dx)
    # For RCS, this varies with x

# =============================================================================
# Sampling
# =============================================================================
if __name__ == '__main__':
    print("\nStarting MCMC sampling...")

    with model_v2:
        idata = pm.sample(
            draws=4_000,
            tune=2_000,
            chains=4,
            cores=4,  # Adjust based on your CPU
            target_accept=0.95,
            nuts_sampler=_backend,
            random_seed=42,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True}
        )

    # Save results
    az.to_netcdf(idata, PATH_MODEL_V2)
    print(f"\nSaved model to {PATH_MODEL_V2}")

    # =========================================================================
    # Diagnostics
    # =========================================================================
    print("\n" + "=" * 60)
    print("MCMC Diagnostics")
    print("=" * 60)

    # Key parameters to summarize
    summary_vars = [
        "alpha0", "beta0", "sigma", "R2",
        "sigma_alpha_region", "sigma_alpha_country",
        "tau0", "sigma_tau_region",
        "theta_sd", "theta",
        "delta_washare", "delta_olddep"
    ]
    if ESTIMATE_NU:
        summary_vars.append("nu")
    if COUNTRY_SLOPES:
        summary_vars.append("sigma_beta_country")

    # Filter to existing variables
    existing = [v for v in summary_vars if v in idata.posterior.data_vars]

    summ = az.summary(idata, var_names=existing, hdi_prob=0.95)
    print(summ[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "r_hat", "ess_bulk"]].head(30))

    # Global diagnostics
    rhat_max = float(az.rhat(idata).to_array().max())
    ess_min = float(az.ess(idata, method="bulk").to_array().min())
    n_divergent = int(np.asarray(idata.sample_stats["diverging"]).sum())

    print(f"\nmax R-hat: {rhat_max:.4f} (should be < 1.01)")
    print(f"min ESS: {ess_min:.0f} (should be > 400)")
    print(f"Divergences: {n_divergent} (should be 0)")

    # BFMI
    bfmi = az.bfmi(idata)
    bfmi_arr = np.array(list(bfmi.values()) if isinstance(bfmi, dict) else bfmi)
    print(f"BFMI: {bfmi_arr} (should be > 0.3)")

    # =========================================================================
    # Model Fit
    # =========================================================================
    print("\n" + "=" * 60)
    print("Model Fit")
    print("=" * 60)

    r2_samples = idata.posterior["R2"].values.flatten()
    print(f"R²: {np.mean(r2_samples):.3f} ({np.percentile(r2_samples, 2.5):.3f}, {np.percentile(r2_samples, 97.5):.3f})")

    if ESTIMATE_NU:
        nu_samples = idata.posterior["nu"].values.flatten()
        print(f"ν (df): {np.mean(nu_samples):.1f} ({np.percentile(nu_samples, 2.5):.1f}, {np.percentile(nu_samples, 97.5):.1f})")

    # Elasticity interpretation
    beta_samples = idata.posterior["beta0"].values.flatten()
    print(f"\nGlobal elasticity (β₀): {np.mean(beta_samples):.3f} ± {np.std(beta_samples):.3f}")
    print("  Interpretation: 1% population increase → {:.2f}% GDP increase (linear component)".format(
        np.mean(beta_samples)))

    # Demographic effects interpretation
    dwa_samples = idata.posterior["delta_washare"].values.flatten()
    dod_samples = idata.posterior["delta_olddep"].values.flatten()
    print(f"\nWorking-age share effect: {np.mean(dwa_samples):.3f} ± {np.std(dwa_samples):.3f}")
    print(f"  1 SD increase in working-age share → {10**np.mean(dwa_samples):.2f}x GDP multiplier")
    print(f"\nOld-age dependency effect: {np.mean(dod_samples):.3f} ± {np.std(dod_samples):.3f}")
    print(f"  1 SD increase in old-age dependency → {10**np.mean(dod_samples):.2f}x GDP multiplier")
