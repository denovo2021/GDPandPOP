# Global GDP and Population Projection (2025-2100)

This repository contains the source code for the paper:
**"Projecting the Demographic Bounds of Global Health Financing to 2100: A Hierarchical Bayesian Analysis"**

## Repository Structure

```
GDPandPOP/
├── config.py                     # Centralized path configuration
├── requirements.txt              # Python dependencies
│
├── models/                       # Core statistical models (Layers 1-3)
│   ├── 01_simple_model_rcs.py   # Layer 1: Global pooling with RCS
│   ├── 02_regional_model_rcs.py # Layer 2: Regional partial pooling
│   └── 03_hierarchical_model_rcs.py  # Layer 3: Full hierarchical (MAIN)
│
├── data_processing/              # ETL pipeline
│   ├── step1_prepare_age_data.py
│   ├── step2_aggregate_world_gdp.py
│   └── step3_plot_projections.py
│
├── analysis/                     # Forecasting & diagnostics
│   ├── forecast_gdp.py          # GDP projection engine
│   ├── forecast_u5mr.py         # U5MR projection
│   └── diagnostics.py           # Model diagnostics (LOO, MAE, R2)
│
├── visualization/                # Figure generation
│   ├── fig_gdp_fan_chart.py     # Figure 2: Global GDP fan chart
│   ├── fig_u5mr_projection.py   # Figure 3: U5MR projection
│   └── publication_figures.py   # All publication figures
│
├── tables/                       # Table generation
│   ├── table_posterior_summary_compact_rcs.py
│   └── table_top10_economies_rcs.py
│
├── utils/                        # Shared utilities
│   └── calculation.py
│
├── outputs/                      # Generated outputs (gitignored)
│   ├── figures/
│   ├── tables/
│   └── cache/
│
├── data/                         # Data documentation
│   └── README.md
│
└── archive/                      # Deprecated versions
    └── README.md
```

## Model Layers

The project implements a three-layer hierarchical Bayesian framework:

| Layer | Model | Description |
|-------|-------|-------------|
| 1 | Simple (Global Pooling) | Single global elasticity, RCS for nonlinearity |
| 2 | Regional (Partial Pooling) | Region-specific intercepts/slopes with shrinkage |
| 3 | **Hierarchical (Full)** | Country-within-region hierarchy + demographics |

The **Layer 3 model** (`03_hierarchical_model_rcs.py`) is the primary analysis model featuring:
- Restricted Cubic Splines (RCS) for flexible population-GDP relationship
- Demographic effects: Working-age share and old-age dependency ratio
- Region-level time drift with non-centered parameterization
- Student-t likelihood for robustness to outliers

## Data Requirements

Place the following CSV files in the project root:
- `merged.csv` - Historical GDP/Population panel data
- `merged_age.csv` - Extended data with age structure variables
- `pop_predictions_scenarios.csv` - UN WPP 2024 population projections
- `age_predictions_scenarios.csv` - UN WPP 2024 age structure projections

See `data/README.md` for detailed column specifications and data sources.

## Installation

```bash
# Clone repository
git clone https://github.com/[username]/GDPandPOP.git
cd GDPandPOP

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Fit the Hierarchical Model
```bash
python models/03_hierarchical_model_rcs.py
```
This produces `outputs/hierarchical_model_rcs.nc` containing posterior samples.

### 2. Generate GDP Projections
```bash
python analysis/forecast_gdp.py
```
This produces `outputs/gdp_predictions_scenarios_rcs_age.csv`.

### 3. Create Figures
```bash
python visualization/publication_figures.py
```
Figures are saved to `outputs/figures/`.

## Key Outputs

| Output | Description |
|--------|-------------|
| Figure 1 | Global association curve (population-GDP elasticity) |
| Figure 2 | Global GDP fan chart (2025-2100 projections) |
| Figure 3 | Global U5MR projection |
| Table 1 | Posterior summary of model parameters |
| Table 2 | Model comparison across layers |

## Requirements

- Python 3.9+
- PyMC >= 5.10
- ArviZ >= 0.17
- See `requirements.txt` for full list

## License

MIT License

## Citation

If you use this code, please cite:
```
[Citation will be added upon publication]
```
