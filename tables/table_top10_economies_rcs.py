# ============================ table_top10_economies_rcs.py ============================
"""
Top-10 economies table (RCS forecast version).

Reads scenario-level forecasts produced by prediction_rcs.py
(gdp_predictions_scenarios.csv) and aggregates across variants with a
robust (trimmed) median before selecting the top-N countries for each
target year.

Why trimmed median?
- Even after excluding High/Low variants, a few scenario paths can be
  outlying for specific country–year cells. Trimming (e.g., 10%) keeps
  the ranking stable without discarding data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ------------------------------- paths -----------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PATH_GDP_PREDICTIONS_RCS, DIR_TABLES

SCEN_CSV = PATH_GDP_PREDICTIONS_RCS  # from prediction_rcs.py
OUT_CSV = DIR_TABLES / "table_top10_economies_rcs.csv"

# ------------------------------ config -----------------------------------------
YEARS   = [2035, 2040, 2050, 2060, 2070, 2080, 2090, 2100]
TOP_N   = 10
TRIM    = 0.10   # symmetric trimming proportion (e.g., 0.10 = drop top/bottom 10%)

# --------------------------- helper: trimmed stats ------------------------------
def trimmed_stat(series: pd.Series, q: float = 0.5, trim: float = 0.10) -> float:
    """Symmetrically trim the series and return the quantile (default: median)."""
    vals = np.sort(series.to_numpy(dtype=float))
    k = int(np.floor(trim * len(vals)))
    if len(vals) - 2*k <= 0:
        return np.nan
    core = vals[k: len(vals)-k]
    return float(np.quantile(core, q))

# ------------------------------ load -------------------------------------------
# Expect columns: Scenario, Year, Country, Pred_Median, Pred_Lower, Pred_Upper
df = pd.read_csv(
    SCEN_CSV,
    usecols=["Scenario", "Year", "Country", "Pred_Median", "Pred_Lower", "Pred_Upper"]
)

if df.empty:
    raise RuntimeError("Scenario CSV is empty. Run prediction_rcs.py first.")

# ---------------- aggregate across variants (robust) ----------------------------
# For each Country–Year, compute trimmed medians of the three columns.
agg = (
    df.groupby(["Country", "Year"])
      .agg(
          Pred_Median=("Pred_Median", lambda s: trimmed_stat(s, 0.5, TRIM)),
          Pred_Lower =("Pred_Lower",  lambda s: trimmed_stat(s, 0.5, TRIM)),
          Pred_Upper =("Pred_Upper",  lambda s: trimmed_stat(s, 0.5, TRIM)),
          n_variants =("Scenario",    "nunique"),
      )
      .reset_index()
)

# Optional sanity check: drop rows that could not be aggregated
agg = agg.dropna(subset=["Pred_Median"])

# ------------------------- build top-N per target year --------------------------
def top_n(year: int, n: int = TOP_N) -> pd.DataFrame:
    sub = agg[agg["Year"] == year].nlargest(n, "Pred_Median").copy()
    sub.insert(0, "Rank", range(1, len(sub)+1))
    return sub[["Rank", "Country", "Pred_Median", "Pred_Lower", "Pred_Upper", "Year"]]

tbl = pd.concat([top_n(y) for y in YEARS], axis=0, ignore_index=True)

# ---------------------------- format (trillion USD) -----------------------------
for col in ["Pred_Median", "Pred_Lower", "Pred_Upper"]:
    tbl[col] = (tbl[col] / 1e12).round(1)

# ------------------------------- save ------------------------------------------
tbl.to_csv(OUT_CSV, index=False)
print(f"✓ top-10 table → {OUT_CSV}")
