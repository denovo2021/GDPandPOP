# GDPandPOP: Hierarchical Bayesian Modeling for Global GDP Projections

## Project Overview

This project estimates the association between population, age structure, and technological drift on GDP using a hierarchical Bayesian model with panel data from 180 countries (1960-2023), and forecasts GDP and Under-5 Mortality Rate (U5MR) up to 2100.

## Methodology

- **Framework**: Hierarchical Bayesian modeling using PyMC (v5)
- **Features**: Restricted Cubic Splines (RCS), Non-Centered Parameterization (NCP), Orthogonalization
- **Data Sources**: World Bank, UN World Population Prospects 2024

## Project Structure

```
GDPandPOP/
|-- data/                    # Raw data files
|   |-- merged.csv           # Historical GDP/Population panel (1960-2023)
|   |-- merged_age.csv       # Extended with age structure variables
|   |-- pop_predictions_scenarios.csv  # UN WPP 2024 population projections
|   |-- age_predictions_scenarios.csv  # Age structure projections
|   +-- UN/                  # UN WPP source files
|
|-- src/                     # Source code
|   |-- config.py            # Central path configuration
|   |-- models/              # Bayesian model definitions
|   |   |-- 01_simple_model_rcs.py      # Layer 1: Global pooling
|   |   |-- 02_regional_model_rcs.py    # Layer 2: Regional partial pooling
|   |   +-- 03_hierarchical_model_rcs.py # Layer 3: Full hierarchical (main)
|   |-- analysis/            # Projection and aggregation
|   |-- visualization/       # Publication-quality figures
|   |-- data_processing/     # ETL pipeline scripts
|   |-- utils/               # Utility functions & RCS basis
|   |-- tables/              # Table generation scripts
|   +-- scripts/             # Execution scripts
|
|-- figures/                 # Generated figures
|   |-- main/                # Main manuscript figures (Fig 1-5)
|   |-- supplementary/       # Supplementary figures
|   +-- diagnostics/         # Model diagnostic plots
|
|-- results/                 # Model outputs
|   |-- *.nc                 # NetCDF posterior samples
|   |-- *.csv                # Forecast outputs
|   +-- cache/               # Cached computations
|
|-- docs/                    # Documentation
|   |-- main_manuscript_GDPAndPOP.docx
|   +-- README.md
|
+-- archive/                 # Deprecated model versions
```

## Quick Start

```bash
# Install dependencies
uv sync

# Run the main hierarchical model
uv run python -m src.models.03_hierarchical_model_rcs

# Generate forecasts
uv run python -m src.analysis.forecast_gdp

# Create publication figures
uv run python -m src.visualization.publication_figures
```

## Data Pipeline

1. `src/utils/build_age_covariates_from_owid.py` - Build age covariates from OWID data
2. `src/data_processing/add_age_to_merged.py` - Merge age data with panel
3. `src/models/03_hierarchical_model_rcs.py` - Fit Bayesian model
4. `src/analysis/forecast_gdp.py` - Generate projections

## Three-Layer Model Hierarchy

1. **Simple Model** (`01_simple_model_rcs.py`): Global pooling with RCS
2. **Regional Model** (`02_regional_model_rcs.py`): Regional partial pooling
3. **Hierarchical Model** (`03_hierarchical_model_rcs.py`): Full country-within-region hierarchy with demographics

## Visualization Standards

Figures are generated to meet Tier 1 journal standards (The Lancet, Nature Medicine):
- Resolution: 300-600 DPI
- Formats: PNG, PDF, TIFF
- Fonts: Arial/Helvetica, 8-14pt
- Dimensions: Single column (85mm), Double column (170mm)

See `src/visualization/visualization_utils.py` for helper functions.

## Requirements

- Python 3.11+
- PyMC 5.x
- ArviZ
- Pandas, NumPy, Matplotlib, Seaborn

## License

[Add license information]
