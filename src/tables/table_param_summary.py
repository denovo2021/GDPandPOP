import arviz as az
import re
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.config import PATH_MODEL_HIERARCHICAL, DIR_TABLES

IDATA = PATH_MODEL_HIERARCHICAL
OUT_CSV = DIR_TABLES / "table_posterior_summary_rcs.csv"

idata  = az.from_netcdf(IDATA)
poster = idata.posterior

# ----------------------- collect variable names that match patterns -----------
want_patterns = [r"^alpha($|_)", r"^beta($|_)", r"^gamma($|_)",
                 r"^sigma($|_)"]
vars_present  = [v for v in poster.data_vars
                 if any(re.match(p, v) for p in want_patterns)]

if not vars_present:
    raise RuntimeError("No alpha/beta/gamma/sigma variables found!")

summary = az.summary(
    idata,
    var_names=vars_present,
    hdi_prob=0.95,
    round_to=3
)
summary.to_csv(OUT_CSV)
print(f"✓ posterior summary → {OUT_CSV}")
