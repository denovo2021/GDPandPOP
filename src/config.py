# config.py
# Centralized path configuration for GDPandPOP project
# -------------------------------------------------------------------
# All paths are defined relative to the project root.
# Data files are in data/, outputs go to results/.
# This file should be located at src/config.py after reorganization.
# -------------------------------------------------------------------

from pathlib import Path

# Project root (parent of src/ directory)
# This works whether config.py is at root or in src/
_THIS_FILE = Path(__file__).resolve()
if _THIS_FILE.parent.name == "src":
    PROJECT_ROOT = _THIS_FILE.parent.parent
else:
    PROJECT_ROOT = _THIS_FILE.parent

# =====================================================================
# Input Data Directories
# =====================================================================
# Support both data/ (new) and MacroMetrics/ (legacy) locations
_DATA_NEW = PROJECT_ROOT / "data"
_DATA_LEGACY = PROJECT_ROOT / "MacroMetrics"

# Use data/ if it has the main files, otherwise fall back to MacroMetrics/
if (_DATA_NEW / "merged.csv").exists():
    DIR_DATA = _DATA_NEW
elif (_DATA_LEGACY / "merged.csv").exists():
    DIR_DATA = _DATA_LEGACY
else:
    DIR_DATA = _DATA_NEW  # Default to new location

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
DIR_OUTPUT = PROJECT_ROOT / "results"
DIR_FIGURES = PROJECT_ROOT / "figures"
DIR_FIGURES_MAIN = DIR_FIGURES / "main"
DIR_FIGURES_SUPP = DIR_FIGURES / "supplementary"
DIR_FIGURES_DIAG = DIR_FIGURES / "diagnostics"
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
PATH_FIG_SIMPLE_MODEL = DIR_FIGURES_DIAG / "simple_model_rcs.png"
PATH_FIG_REGIONAL_MODEL = DIR_FIGURES_DIAG / "regional_model_rcs.png"
PATH_FIG_HIERARCHICAL_MODEL = DIR_FIGURES_DIAG / "hierarchical_model_rcs.png"
PATH_FIG_GDP_FAN = DIR_FIGURES_MAIN / "fig_gdp_fan_chart.png"
PATH_FIG_U5MR = DIR_FIGURES_MAIN / "fig_u5mr_projection.png"
PATH_FIG_GLOBAL_GDP_FAN = DIR_FIGURES_MAIN / "Figure2_Global_GDP_Fan_v5_Final.png"

# =====================================================================
# Publication Figure Settings (for Tier 1 journals)
# =====================================================================
# Resolution settings
FIG_DPI_SCREEN = 150      # For quick preview
FIG_DPI_PRINT = 300       # Minimum for publication
FIG_DPI_HIGHRES = 600     # High-resolution for journals

# Figure dimensions (in inches) - based on journal column widths
# Single column: ~85mm = 3.35 inches
# Double column: ~170mm = 6.7 inches
# Full page width: ~180mm = 7.1 inches
FIG_WIDTH_SINGLE = 3.35
FIG_WIDTH_DOUBLE = 6.7
FIG_WIDTH_FULL = 7.1

# Font settings
FIG_FONT_FAMILY = "Arial"  # Or "Helvetica" - standard for scientific journals
FIG_FONT_SIZE_SMALL = 8
FIG_FONT_SIZE_NORMAL = 10
FIG_FONT_SIZE_LARGE = 12
FIG_FONT_SIZE_TITLE = 14

# Default matplotlib rcParams for publication
PUBLICATION_RC_PARAMS = {
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': FIG_FONT_SIZE_NORMAL,
    'axes.titlesize': FIG_FONT_SIZE_LARGE,
    'axes.labelsize': FIG_FONT_SIZE_NORMAL,
    'xtick.labelsize': FIG_FONT_SIZE_SMALL,
    'ytick.labelsize': FIG_FONT_SIZE_SMALL,
    'legend.fontsize': FIG_FONT_SIZE_SMALL,
    'figure.titlesize': FIG_FONT_SIZE_TITLE,
    'figure.dpi': FIG_DPI_SCREEN,
    'savefig.dpi': FIG_DPI_PRINT,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'pdf.fonttype': 42,  # TrueType fonts in PDF (required by some journals)
    'ps.fonttype': 42,
}

# =====================================================================
# Utility Functions
# =====================================================================
def ensure_dirs():
    """Create output directories if they don't exist."""
    dirs = [
        DIR_OUTPUT,
        DIR_FIGURES,
        DIR_FIGURES_MAIN,
        DIR_FIGURES_SUPP,
        DIR_FIGURES_DIAG,
        DIR_TABLES,
        DIR_CACHE,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def setup_publication_style():
    """Apply publication-quality matplotlib settings."""
    import matplotlib.pyplot as plt
    plt.rcParams.update(PUBLICATION_RC_PARAMS)


def save_figure(fig, name: str, formats=('png', 'pdf'), dpi=None, directory=None):
    """
    Save figure in multiple formats for publication.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    name : str
        Base filename (without extension).
    formats : tuple
        File formats to save (default: png and pdf).
    dpi : int, optional
        Resolution. If None, uses FIG_DPI_PRINT for raster, vector for pdf.
    directory : Path, optional
        Output directory. If None, uses DIR_FIGURES_MAIN.
    """
    if directory is None:
        directory = DIR_FIGURES_MAIN

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = directory / f"{name}.{fmt}"
        if fmt in ('pdf', 'svg', 'eps'):
            # Vector formats don't need high DPI
            fig.savefig(filepath, format=fmt, bbox_inches='tight')
        else:
            # Raster formats (png, tiff) need high DPI
            fig.savefig(filepath, format=fmt, dpi=dpi or FIG_DPI_PRINT, bbox_inches='tight')
        print(f"[OK] Saved {filepath}")


# Auto-create output directories on import
ensure_dirs()
