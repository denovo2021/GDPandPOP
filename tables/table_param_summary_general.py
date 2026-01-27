# table_posterior_summary_compact.py  – robust for any naming style
import arviz as az
import pandas as pd
import numpy as np
import re
from pathlib import Path

ROOT  = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
IDATA = ROOT / "hierarchical_model_with_quadratic.nc"
OUT   = ROOT / "table_posterior_summary_compact.csv"

idata  = az.from_netcdf(IDATA)
poster = idata.posterior
all_vars = list(poster.data_vars)

# ---------------- 1. global & region parameters ----------------
pat_global = re.compile(r"^(alpha|beta|gamma|sigma)$")
pat_region = re.compile(r"_(region)$")       # e.g. alpha_region

vars_global = [v for v in all_vars if pat_global.match(v)]
vars_region = [v for v in all_vars if pat_region.search(v)]

tbl_main = az.summary(
    idata,
    var_names=vars_global + vars_region,
    hdi_prob=0.95,
    round_to=3
)

# harmonise column names so concat works cleanly
tbl_main = (
    tbl_main.rename(columns={"hdi_2.5%": "p2.5", "hdi_97.5%": "p97.5"})
            # ensure quartile columns exist (filled with NaN)
            .assign(p25=np.nan, p75=np.nan)
            # keep original diagnostics (r_hat, ess_bulk, ess_tail, etc.)
            # reorder so p-columns sit next to each other
            [["mean", "sd", "p2.5", "p25", "p75", "p97.5",
              "r_hat", "ess_bulk", "ess_tail"]]
)

# ---------------- 2. country-level distribution summaries -------
rows = []
for v in all_vars:
    if "Country" in poster[v].dims:
        core = re.match(r"^(alpha|beta|gamma)", v).group(1)
        flat = poster[v].values.reshape(-1)
        rows.append({
            "parameter": core,
            "mean":  round(float(np.mean(flat)), 3),
            "sd":    round(float(np.std(flat, ddof=1)), 3),
            "p2.5":  round(float(np.percentile(flat, 2.5)), 3),
            "p25":   round(float(np.percentile(flat, 25)), 3),
            "p75":   round(float(np.percentile(flat, 75)), 3),
            "p97.5": round(float(np.percentile(flat, 97.5)), 3),
            # diagnostics not applicable to pooled rows → NaN placeholders
            "r_hat":     np.nan,
            "ess_bulk":  np.nan,
            "ess_tail":  np.nan,
        })

tbl_dist = (
    pd.DataFrame(rows)
      .groupby("parameter")
      .first()
      .set_index(pd.Index([f"{p}_country (all)" for p in ["alpha","beta","gamma"]]))
)

# ---------- 3. combine & save -------------------------------------------------
table1 = (
    pd.concat([tbl_main, tbl_dist])
      .rename_axis("parameter")
      .reset_index()
)

table1.to_csv(OUT, index=False)
print(f"✓ compact Table 1 written → {OUT}")