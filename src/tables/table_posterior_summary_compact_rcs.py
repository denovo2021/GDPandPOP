# ===================== table_posterior_summary_rcs_by_region.py =====================
# Region-level posterior summary with Bayesian diagnostics (r_hat, ESS).
# Outputs one row per Region with:
#   alpha_median / hdi / rhat / ess_bulk / ess_tail
#   beta_median  / hdi / rhat / ess_bulk / ess_tail
#   theta_L2_*   (optional: L2-norm of spline weights per region)

import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PATH_MODEL_HIERARCHICAL, DIR_TABLES

IDATA = PATH_MODEL_HIERARCHICAL
OUT_CSV = DIR_TABLES / "table_posterior_summary_rcs.csv"

idata = az.from_netcdf(IDATA)
post  = idata.posterior
regions = post.coords["Region"].values

def summarize_da(da: xr.DataArray, prefix: str) -> pd.DataFrame:
    """Summarize a (chain, draw, Region) DataArray into a tidy DF."""
    # median
    med = (da.stack(sample=("chain","draw"))
             .quantile(0.5, dim="sample")
             .to_series())
    # 95% HDI
    hdi_ds = az.hdi(da, hdi_prob=0.95)          # Dataset
    hdi_da = hdi_ds.to_array().squeeze()        # DataArray dims: ('Region','hdi')
    hdi_lo = hdi_da.sel(hdi="lower").to_series()
    hdi_hi = hdi_da.sel(hdi="higher").to_series()
    # diagnostics
    rhat_s = az.rhat(da).to_array().squeeze().to_series()
    essb_s = az.ess(da, method="bulk").to_array().squeeze().to_series()
    esst_s = az.ess(da, method="tail").to_array().squeeze().to_series()
    # align to Region ordering
    out = pd.DataFrame({
        "Region": regions,
        f"{prefix}_median":   med.reindex(regions).values,
        f"{prefix}_hdi_lo":   hdi_lo.reindex(regions).values,
        f"{prefix}_hdi_hi":   hdi_hi.reindex(regions).values,
        f"{prefix}_rhat":     rhat_s.reindex(regions).values,
        f"{prefix}_ess_bulk": essb_s.reindex(regions).values,
        f"{prefix}_ess_tail": esst_s.reindex(regions).values,
    })
    return out

# alpha_region & beta_region (required)
if "alpha_region" not in post.data_vars or "beta_region" not in post.data_vars:
    raise RuntimeError("alpha_region and/or beta_region missing in posterior.")

alpha_df = summarize_da(post["alpha_region"], "alpha")
beta_df  = summarize_da(post["beta_region"],  "beta")

# theta_region magnitude (optional): L2 = sqrt(sum_j theta^2) over Spline
theta_df = None
if "theta_region" in post.data_vars:
    l2_da = np.sqrt((post["theta_region"] ** 2).sum(dim="Spline"))  # (chain, draw, Region)
    theta_df = summarize_da(l2_da, "theta_L2")

# merge & save
out = alpha_df.merge(beta_df, on="Region", how="inner")
if theta_df is not None:
    out = out.merge(theta_df, on="Region", how="left")

# nice column order
cols = [
    "Region",
    "alpha_median","alpha_hdi_lo","alpha_hdi_hi","alpha_rhat","alpha_ess_bulk","alpha_ess_tail",
    "beta_median","beta_hdi_lo","beta_hdi_hi","beta_rhat","beta_ess_bulk","beta_ess_tail",
]
if theta_df is not None:
    cols += ["theta_L2_median","theta_L2_hdi_lo","theta_L2_hdi_hi",
             "theta_L2_rhat","theta_L2_ess_bulk","theta_L2_ess_tail"]

out = out[cols].sort_values("Region").reset_index(drop=True)
out.to_csv(OUT_CSV, index=False)
print(f"✓ region-level posterior summary with diagnostics → {OUT_CSV}")
print(out)
