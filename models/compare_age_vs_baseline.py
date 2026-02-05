# compare_age_vs_baseline.py
# ---------------------------------------------------------------------------
# Compare the age-augmented hierarchical RCS GDP model vs the baseline RCS model
# using PSIS-LOO on the SAME historical dataset (1960–2023).
#
# - Builds pointwise Student-t log-likelihood for each model from posterior samples
#   (thinned for memory), WITHOUT requiring a posterior group in the InferenceData.
# - Computes LOO with explicit reff=1.0 (since we provide only log_likelihood).
# - Reports ELPD, SE, ΔELPD, Pareto-k diagnostics, and pseudo-BMA weights.
#
# Requirements (project root):
#   hierarchical_model_with_rcs_age.nc     (age + time slope model)
#   hierarchical_model_with_rcs.nc         (baseline RCS-only model)
#   merged_age.csv                         (historical panel with WAshare/OldDep)
#   merged.csv                             (for MU_GLOBAL centering if needed)
#   rcs_knots_hier.npy                     (RCS knots used in both models)
#
# Output: console report
# ---------------------------------------------------------------------------

import numpy as np
import pandas as pd
import arviz as az
from pathlib import Path
from scipy.special import gammaln

# ------------------------------- config ---------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DIR_OUTPUT, PATH_MERGED_AGE, PATH_MERGED, PATH_KNOTS

ID_AGE = DIR_OUTPUT / "hierarchical_model_with_rcs_age.nc"
ID_RCS = DIR_OUTPUT / "hierarchical_model_with_rcs.nc"
MERGEDA = PATH_MERGED_AGE  # includes WAshare / OldDep
MERGED = PATH_MERGED  # for MU_GLOBAL
KNOTS = PATH_KNOTS

# Thinning to reduce memory (every THIN-th draw)
THIN = 10

# --------------------------- helpers ------------------------------------------
def rcs_design(x, knots):
    """Restricted cubic spline design with linear tails; result (N, K-2)."""
    k = np.asarray(knots); K = k.size
    if K < 3:
        return np.zeros((x.size, 0))
    def d(u, j): return np.maximum(u - k[j], 0.0) ** 3
    cols = []
    for j in range(1, K - 1):
        term = (d(x, j)
                - d(x, K - 1) * (k[K - 1] - k[j]) / (k[K - 1] - k[0])
                + d(x, 0)     * (k[j]     - k[0]) / (k[K - 1] - k[0]))
        cols.append(term)
    return np.column_stack(cols)

def student_t_logpdf(y, mu, sigma, nu):
    """
    Vectorized Student-t logpdf.
    y : (N,)
    mu: (S, N)
    sigma: (S, 1)
    nu: (S, 1)
    returns: (S, N)
    """
    z = (y[None, :] - mu) / sigma
    return (gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0)
            - 0.5 * np.log(nu * np.pi) - np.log(sigma)
            - ((nu + 1.0) / 2.0) * np.log1p((z * z) / nu))

def flatten_thin(arr, chain_dim="chain", draw_dim="draw", thin=10):
    """
    Flatten posterior (chain, draw, ...) -> (S, ...), taking every 'thin' draw.
    Accepts xarray.DataArray .values directly; caller should slice indices.
    """
    return arr  # we'll handle thinning by indexing arrays slice-wise

def build_loglik_age(idata_age, df, MU, knots):
    """
    Build (chain, draw, obs) log-likelihood for the age-augmented model.

    idata_age: InferenceData with posterior (alpha_country, beta_country, theta_region,
                delta_washare, delta_olddep, tau_country, sigma, nu or nu_raw)
    df: pandas DataFrame of training rows aligned to model coords
        required columns: Region, Country Name, Log_Population, Log_GDP,
                          dWA, dOD, t_dec
    MU, knots: centering mean and RCS knots
    """
    post = idata_age.posterior
    C = post.sizes["chain"]
    D = post.sizes["draw"]
    draw_idx = np.arange(0, D, THIN, dtype=int)  # thinning
    D2 = draw_idx.size
    S = C * D2

    # maps
    reg_list = post.coords["Region"].values.tolist()
    cty_list = post.coords["Country"].values.tolist()
    reg_map = {r: i for i, r in enumerate(reg_list)}
    cty_map = {c: i for i, c in enumerate(cty_list)}

    # restrict df to countries/regions present
    dfm = df[df["Country Name"].isin(cty_list)].copy()
    dfm = dfm[dfm["Region"].isin(reg_list)].copy()
    dfm = dfm.reset_index(drop=True)

    # design
    x  = (dfm["Log_Population"].to_numpy() - MU)
    Z  = rcs_design(x, knots)                          # (N, m)
    y  = dfm["Log_GDP"].to_numpy()
    t  = dfm["t_dec"].to_numpy()
    dWA= dfm["dWA"].to_numpy()
    dOD= dfm["dOD"].to_numpy()
    ri = dfm["Region"].map(reg_map).to_numpy().astype(int)
    ci = dfm["Country Name"].map(cty_map).to_numpy().astype(int)
    N  = y.shape[0]
    m  = Z.shape[1]

    # extract and thin
    a  = post["alpha_country"].values[:, draw_idx, :].reshape(S, -1)      # (S, nC)
    b  = post["beta_country" ].values[:, draw_idx, :].reshape(S, -1)      # (S, nC)
    th = post["theta_region" ].values[:, draw_idx, :, :].reshape(S, len(reg_list), m)  # (S, nR, m)
    tau= post["tau_country"  ].values[:, draw_idx, :].reshape(S, -1)      # (S, nC)
    sig= post["sigma"].values[:, draw_idx].reshape(S, 1)                   # (S, 1)
    if "nu" in post.data_vars:
        nu = post["nu"].values[:, draw_idx].reshape(S, 1)
    else:
        nu = np.clip(post["nu_raw"].values[:, draw_idx].reshape(S, 1) + 1.0, 2.0, 30.0)

    dW = post["delta_washare"].values[:, draw_idx].reshape(S, 1)
    dO = post["delta_olddep" ].values[:, draw_idx].reshape(S, 1)

    # assemble μ for every sample S and observation N
    theta_rows = th[:, ri, :]                                # (S, N, m)
    spline     = (theta_rows * Z[None, :, :]).sum(axis=2)    # (S, N)
    mu_age = (a[:, ci] + b[:, ci] * x[None, :] + spline
              + dW @ dWA[None, :]
              + dO @ dOD[None, :]
              + tau[:, ci] * t[None, :])                     # (S, N)

    ll = student_t_logpdf(y, mu_age, sig, nu)                # (S, N)
    # reshape back to (chain, draw_thinned, obs)
    ll_cdo = ll.reshape(C, D2, N)
    return dfm, ll_cdo

def build_loglik_rcs(idata_rcs, df, MU, knots):
    """
    Build (chain, draw, obs) log-likelihood for the baseline RCS model (no age or time slope).
    required df columns: Region, Country Name, Log_Population, Log_GDP
    """
    post = idata_rcs.posterior
    C = post.sizes["chain"]
    D = post.sizes["draw"]
    draw_idx = np.arange(0, D, THIN, dtype=int)
    D2 = draw_idx.size
    S = C * D2

    reg_list = post.coords["Region"].values.tolist()
    cty_list = post.coords["Country"].values.tolist()
    reg_map = {r: i for i, r in enumerate(reg_list)}
    cty_map = {c: i for i, c in enumerate(cty_list)}

    dfm = df[df["Country Name"].isin(cty_list)].copy()
    dfm = dfm[dfm["Region"].isin(reg_list)].copy()
    dfm = dfm.reset_index(drop=True)

    x  = (dfm["Log_Population"].to_numpy() - MU)
    Z  = rcs_design(x, knots)                          # (N, m)
    y  = dfm["Log_GDP"].to_numpy()
    ri = dfm["Region"].map(reg_map).to_numpy().astype(int)
    ci = dfm["Country Name"].map(cty_map).to_numpy().astype(int)
    N  = y.shape[0]
    m  = Z.shape[1]

    a  = post["alpha_country"].values[:, draw_idx, :].reshape(S, -1)      # (S, nC)
    b  = post["beta_country" ].values[:, draw_idx, :].reshape(S, -1)
    th = post["theta_region" ].values[:, draw_idx, :, :].reshape(S, len(reg_list), m)
    sig= post["sigma"].values[:, draw_idx].reshape(S, 1)
    if "nu" in post.data_vars:
        nu = post["nu"].values[:, draw_idx].reshape(S, 1)
    else:
        nu = np.clip(post["nu_raw"].values[:, draw_idx].reshape(S, 1) + 1.0, 2.0, 30.0)

    theta_rows = th[:, ri, :]                                # (S, N, m)
    spline     = (theta_rows * Z[None, :, :]).sum(axis=2)    # (S, N)
    mu_rcs     = a[:, ci] + b[:, ci] * x[None, :] + spline   # (S, N)

    ll = student_t_logpdf(y, mu_rcs, sig, nu)                # (S, N)
    ll_cdo = ll.reshape(C, D2, N)
    return dfm, ll_cdo

# ------------------------------- main ------------------------------------------
# 0) global centering and knots
df_mu = pd.read_csv(MERGED, index_col=0)
if "Log_Population" not in df_mu.columns:
    df_mu["Log_Population"] = np.log10(df_mu["Population"])
MU = df_mu["Log_Population"].mean()
knots = np.load(KNOTS)

# 1) training panel (with age) for both models
df = pd.read_csv(MERGEDA).dropna(subset=["Region","Country Name","Population","GDP"])
if "Log_Population" not in df.columns: df["Log_Population"] = np.log10(df["Population"])
if "Log_GDP"        not in df.columns: df["Log_GDP"]        = np.log10(df["GDP"])
df["t_dec"] = (df["Year"] - 2000) / 10.0
# If age columns missing for some rows, fill anchors with 0 deltas (those rows will be omitted in age model)
if "WAshare" in df.columns and "OldDep" in df.columns:
    # If you saved anchors, merge and compute deltas
    try:
        anch = pd.read_csv(PROJ/"age_base_anchors.csv")
        df = df.merge(anch[["ISO3","WAshare_base","OldDep_base"]], on="ISO3", how="left")
        df["dWA"] = df["WAshare"] - df["WAshare_base"]
        df["dOD"] = df["OldDep"]  - df["OldDep_base"]
    except Exception:
        # fallback: zero deltas (rows with NA will be dropped in age build if needed)
        df["dWA"] = np.nan
        df["dOD"] = np.nan
else:
    df["dWA"] = np.nan
    df["dOD"] = np.nan

# 2) load models
idata_age = az.from_netcdf(ID_AGE)
idata_rcs = az.from_netcdf(ID_RCS)

# 3) build LOO datasets
df_age, ll_age_cdo = build_loglik_age(idata_age, df, MU, knots)
df_rcs, ll_rcs_cdo = build_loglik_rcs(idata_rcs, df, MU, knots)

# Ensure SAME observations for fair ΔELPD: intersect on a key (country, year)
key_age = pd.Series(range(len(df_age)), index=pd.MultiIndex.from_arrays([df_age["Country Name"], df_age["Year"]]))
key_rcs = pd.Series(range(len(df_rcs)), index=pd.MultiIndex.from_arrays([df_rcs["Country Name"], df_rcs["Year"]]))
common_idx = key_age.index.intersection(key_rcs.index)

idx_age = key_age.loc[common_idx].to_numpy()
idx_rcs = key_rcs.loc[common_idx].to_numpy()

ll_age_cdo = ll_age_cdo[:, :, idx_age]   # (C, D2, N_common)
ll_rcs_cdo = ll_rcs_cdo[:, :, idx_rcs]   # (C, D2, N_common)

idata_ll_age = az.from_dict(log_likelihood={"Log_GDP_obs": ll_age_cdo})
idata_ll_rcs = az.from_dict(log_likelihood={"Log_GDP_obs": ll_rcs_cdo})

# 4) LOO (explicit reff since we have only log_likelihood)
ic_age = az.loo(idata_ll_age, var_name="Log_GDP_obs", pointwise=True, reff=1.0)
ic_rcs = az.loo(idata_ll_rcs, var_name="Log_GDP_obs", pointwise=True, reff=1.0)

print("\n[LOO]")
print(f"  age-augmented: ELPD = {float(ic_age.elpd_loo):.1f}   SE = {float(ic_age.se):.1f}")
print(f"  baseline RCS : ELPD = {float(ic_rcs.elpd_loo):.1f}   SE = {float(ic_rcs.se):.1f}")

d_elpd  = float(ic_age.elpd_loo - ic_rcs.elpd_loo)
se_diff = float(np.sqrt(ic_age.se**2 + ic_rcs.se**2))
print(f"  ΔELPD (age − rcs) = {d_elpd:.1f}   (SE = {se_diff:.1f})")

# pseudo-BMA weights (scale-free)
w_age = float(np.exp(ic_age.elpd_loo) / (np.exp(ic_age.elpd_loo) + np.exp(ic_rcs.elpd_loo)))
w_rcs = 1.0 - w_age
print(f"  pseudo-BMA weights → age: {w_age:.2f} | rcs: {w_rcs:.2f}")

# Pareto-k diagnostics
def pk_report(ic, label):
    if hasattr(ic, "pareto_k"):
        pk = np.asarray(ic.pareto_k)
        frac_bad = float((pk > 0.7).mean())
        print(f"  {label}: Pareto k > 0.7 = {frac_bad*100:.1f}%")
pk_report(ic_age, "age-augmented")
pk_report(ic_rcs, "baseline")

# ---- stable pseudo-BMA weights (log-sum-exp normalization) ----
log_w = np.array([float(ic_age.elpd_loo), float(ic_rcs.elpd_loo)])
log_w = log_w - log_w.max()              # subtract max to avoid overflow
w     = np.exp(log_w)
w    /= w.sum()
w_age, w_rcs = float(w[0]), float(w[1])
print(f"  pseudo-BMA weights (stable) → age: {w_age:.3f} | rcs: {w_rcs:.3f}")

# ---- per-observation ELPD (nice for manuscripts) ----
N_common = ll_age_cdo.shape[-1]          # # of aligned observations
print(f"  ΔELPD per obs = {d_elpd / N_common:.4f}   (age: {float(ic_age.elpd_loo)/N_common:.4f},"
      f" rcs: {float(ic_rcs.elpd_loo)/N_common:.4f})")
