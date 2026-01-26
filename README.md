# Global GDP and Population Projection (2025-2100)

This repository contains the source code and data processing scripts for the paper:
**"Projecting the Demographic Bounds of Global Health Financing to 2100: A Hierarchical Bayesian Analysis"**

## Repository Structure

The project relies on a hierarchical Bayesian framework implemented in PyMC (v5). The analysis proceeds in three stages:

### 1. Statistical Modeling
* `simple_model_with_rcs.py`: Baseline Global Pooling Model (Figure S1).
* `regional_model_with_rcs.py`: Regional Stratified Model (Figure 1, Table S2).
* `hierarchical_model_with_rcs_age_v3.py`: **Final Model** (Hierarchical Bayesian Model with Age Structure & Time Drift).

### 2. Projection & Analysis
* `step1_fix_age_data.py`: Prepares demographic covariates (WPP 2024).
* `step2_aggregate_world_gdp.py`: Generates probabilistic global GDP projections (Figure 2).
* `run_fig3_u5mr_v5_smart.py`: Generates U5MR projections (Figure 3).
* `calculation_final_metrics.py`: Calculates model diagnostics (MAE, LOO, R2).

### 3. Visualization
* `fig2_gdp_fan_v5_aggregated.py`: Plotting script for Global GDP Fan Chart.
* `fig_ranking_chart.py`: **(New)** Generates regional/country GDP ranking charts for 2030, 2050, 2075.

## Requirements
* Python 3.9+
* PyMC >= 5.0
* ArviZ
* Pandas, NumPy, Matplotlib, Seaborn

## Usage
To reproduce the main results:
1. Run `hierarchical_model_with_rcs_age_v3.py` to sample the posterior.
2. Run `step2_aggregate_world_gdp.py` to generate forecast CSVs.
3. Run `fig2_gdp_fan_v5_aggregated.py` to visualize.

## License
MIT License
