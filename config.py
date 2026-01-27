# config.py
# Centralized path configuration for GDPandPOP project
# -------------------------------------------------------------------
# All paths are defined relative to the project root.
# Users should place CSV data files directly in the project root.
# -------------------------------------------------------------------

from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).parent.resolve()

# =====================================================================
# Input Data Paths
# =====================================================================
# Place these CSV files in the project root for reproducibility:
#   - merged.csv: Main GDP/Population panel data
#   - merged_age.csv: Extended data with age structure variables
#   - pop_predictions_scenarios.csv: UN WPP population projections
#   - age_predictions_scenarios.csv: UN WPP age structure projections

PATH_MERGED = PROJECT_ROOT / "merged.csv"
PATH_MERGED_AGE = PROJECT_ROOT / "merged_age.csv"
PATH_POP_PREDICTIONS = PROJECT_ROOT / "pop_predictions_scenarios.csv"
PATH_AGE_PREDICTIONS = PROJECT_ROOT / "age_predictions_scenarios.csv"

# =====================================================================
# Output Directories
# =====================================================================
DIR_OUTPUT = PROJECT_ROOT / "outputs"
DIR_FIGURES = DIR_OUTPUT / "figures"
DIR_TABLES = DIR_OUTPUT / "tables"
DIR_CACHE = DIR_OUTPUT / "cache"

# =====================================================================
# Model Output Files
# =====================================================================
# NetCDF files containing posterior samples from each model layer
PATH_MODEL_SIMPLE = DIR_OUTPUT / "simple_model_rcs.nc"
PATH_MODEL_REGIONAL = DIR_OUTPUT / "regional_model_rcs.nc"
PATH_MODEL_HIERARCHICAL = DIR_OUTPUT / "hierarchical_model_rcs.nc"

# Supporting files for the hierarchical model
PATH_KNOTS = DIR_CACHE / "rcs_knots_hier.npy"
PATH_SCALE_JSON = DIR_CACHE / "scale_rcs_age_v3.json"

# =====================================================================
# Figure Output Files
# =====================================================================
PATH_FIG_SIMPLE_MODEL = DIR_FIGURES / "simple_model_rcs.png"
PATH_FIG_REGIONAL_MODEL = DIR_FIGURES / "regional_model_rcs.png"
PATH_FIG_HIERARCHICAL_MODEL = DIR_FIGURES / "hierarchical_model_rcs.png"
PATH_FIG_GDP_FAN = DIR_FIGURES / "fig_gdp_fan_chart.png"
PATH_FIG_U5MR = DIR_FIGURES / "fig_u5mr_projection.png"

# =====================================================================
# Legacy Paths (for backward compatibility during transition)
# =====================================================================
# These map old OneDrive paths to new project-relative paths
_LEGACY_PROJ = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")

def migrate_path(old_path):
    """Convert legacy OneDrive path to project-relative path."""
    old_path = Path(old_path)
    try:
        rel = old_path.relative_to(_LEGACY_PROJ)
        return PROJECT_ROOT / rel
    except ValueError:
        return old_path  # Not a legacy path, return as-is


# =====================================================================
# Utility Functions
# =====================================================================
def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [DIR_OUTPUT, DIR_FIGURES, DIR_TABLES, DIR_CACHE]:
        d.mkdir(parents=True, exist_ok=True)


# Auto-create output directories on import
ensure_dirs()
