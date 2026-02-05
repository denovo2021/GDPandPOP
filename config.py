# config.py
# Centralized path configuration for GDPandPOP project
# -------------------------------------------------------------------
# All paths are defined relative to the project root.
# Data files are in MacroMetrics/, outputs go to outputs/.
# -------------------------------------------------------------------

from pathlib import Path

# Project root (directory containing this file)
PROJECT_ROOT = Path(__file__).parent.resolve()

# =====================================================================
# Input Data Directories
# =====================================================================
DIR_DATA = PROJECT_ROOT / "MacroMetrics"
DIR_UN = DIR_DATA / "UN"
DIR_BASIC_DATA = DIR_DATA / "BasicData"
DIR_WB_POP_META = DIR_DATA / "API_SP.POP.TOTL_DS2_en_csv_v2_3401680"

# =====================================================================
# Input Data Files - Core Panel Data
# =====================================================================
PATH_MERGED = DIR_DATA / "merged.csv"
PATH_MERGED_AGE = DIR_DATA / "merged_age.csv"
PATH_MERGED_HEALTH = DIR_DATA / "merged_health.csv"

# =====================================================================
# Input Data Files - Population/Age Projections
# =====================================================================
PATH_POP_PREDICTIONS = DIR_DATA / "pop_predictions_scenarios.csv"
PATH_AGE_PREDICTIONS = DIR_DATA / "age_predictions_scenarios.csv"
PATH_WPP_XLSX = DIR_UN / "WPP2024_GEN_F01_DEMOGRAPHIC_INDICATORS_FULL.xlsx"
PATH_WB_METADATA = DIR_WB_POP_META / "Metadata_Country_API_SP.POP.TOTL_DS2_en_csv_v2_3401680.csv"

# =====================================================================
# Output Directories
# =====================================================================
DIR_OUTPUT = PROJECT_ROOT / "outputs"
DIR_FIGURES = DIR_OUTPUT / "figures"
DIR_TABLES = DIR_OUTPUT / "tables"
DIR_CACHE = DIR_OUTPUT / "cache"

# =====================================================================
# Model Output Files - NetCDF posteriors
# =====================================================================
PATH_MODEL_SIMPLE = DIR_OUTPUT / "simple_model_rcs.nc"
PATH_MODEL_REGIONAL = DIR_OUTPUT / "regional_model_rcs.nc"
PATH_MODEL_HIERARCHICAL = DIR_OUTPUT / "hierarchical_model_rcs.nc"
PATH_MODEL_HIERARCHICAL_AGE = DIR_OUTPUT / "hierarchical_model_with_rcs_age_v5_1_ncp_stable.nc"
PATH_MODEL_HIERARCHICAL_QUAD = DIR_OUTPUT / "hierarchical_model_with_quadratic.nc"
PATH_MODEL_REGIONAL_QUAD = DIR_OUTPUT / "regional_model_with_quadratic.nc"
PATH_MODEL_SIMPLE_QUAD = DIR_OUTPUT / "simple_model_with_quadratic.nc"

# Health outcome models
PATH_MODEL_U5MR_ELASTICITY = DIR_OUTPUT / "u5mr_elasticity.nc"
PATH_MODEL_GHED_ELASTICITY = DIR_OUTPUT / "ghed_elasticity.nc"

# =====================================================================
# Cache / Scaling Files
# =====================================================================
PATH_KNOTS = DIR_CACHE / "rcs_knots_hier.npy"
PATH_SCALE_JSON = DIR_CACHE / "scale_rcs_age.json"
PATH_SCALE_JSON_V3 = DIR_CACHE / "scale_rcs_age_v3.json"

# =====================================================================
# Prediction Output Files
# =====================================================================
PATH_GDP_PREDICTIONS = DIR_OUTPUT / "gdp_predictions_2024.csv"
PATH_GDP_PREDICTIONS_META = DIR_OUTPUT / "gdp_predictions_meta.csv"
PATH_GDP_PREDICTIONS_SCENARIOS = DIR_OUTPUT / "gdp_predictions_scenarios.csv"
PATH_GDP_PREDICTIONS_SCENARIOS_RCS_AGE = DIR_OUTPUT / "gdp_predictions_scenarios_rcs_age.csv"

# World aggregation fan charts
PATH_GDP_WORLD_FAN = DIR_OUTPUT / "gdp_world_fan_v5_full_uncertainty.csv"
PATH_GDP_WORLD_FAN_CONSTANT = DIR_OUTPUT / "gdp_world_fan_rcs_age_constant.csv"

# RCS predictions
PATH_GDP_PREDICTIONS_RCS = DIR_OUTPUT / "gdp_predictions_scenarios_rcs.csv"

# U5MR predictions
PATH_U5MR_PREDICTIONS = DIR_OUTPUT / "u5mr_predictions_scenarios_rcs.csv"
PATH_U5MR_WORLD_FAN = DIR_OUTPUT / "u5mr_world_fan.csv"

# =====================================================================
# Figure Output Files
# =====================================================================
PATH_FIG_SIMPLE_MODEL = DIR_FIGURES / "simple_model_rcs.png"
PATH_FIG_REGIONAL_MODEL = DIR_FIGURES / "regional_model_rcs.png"
PATH_FIG_HIERARCHICAL_MODEL = DIR_FIGURES / "hierarchical_model_rcs.png"
PATH_FIG_GDP_FAN = DIR_FIGURES / "fig_gdp_fan_chart.png"
PATH_FIG_U5MR = DIR_FIGURES / "fig_u5mr_projection.png"
PATH_FIG_GLOBAL_GDP_FAN = DIR_FIGURES / "Figure2_Global_GDP_Fan_v5_Final.png"

# =====================================================================
# Utility Functions
# =====================================================================
def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [DIR_OUTPUT, DIR_FIGURES, DIR_TABLES, DIR_CACHE]:
        d.mkdir(parents=True, exist_ok=True)


# Auto-create output directories on import
ensure_dirs()
