#!/usr/bin/env python
"""
organize_project.py - Safely reorganize GDPandPOP project structure

This script reorganizes the project into a clean structure suitable for
academic publication. It verifies file existence before moving and creates
backup references.

Target Structure:
    data/               - Raw data files (renamed from MacroMetrics)
    src/
        models/         - Model definition scripts
        analysis/       - Projection and aggregation scripts
        visualization/  - Plotting scripts
        data_processing/- ETL pipeline scripts
        utils/          - Utility functions
        config.py       - Central configuration
    figures/
        main/           - Main manuscript figures
        supplementary/  - Supplementary figures
        diagnostics/    - Model diagnostic figures
    docs/               - Manuscripts and documentation
    results/            - Model outputs (.nc, .csv files)
    archive/            - Deprecated code (unchanged)

Usage:
    python organize_project.py --dry-run    # Preview changes
    python organize_project.py              # Execute reorganization
"""

import os
import shutil
import argparse
from pathlib import Path
from datetime import datetime

# Project root
PROJECT_ROOT = Path(__file__).parent.resolve()

# Define the reorganization mapping
# Format: (source_path, destination_path, description)
MOVES = [
    # =========================================================================
    # DATA: Rename MacroMetrics -> data
    # =========================================================================
    ("MacroMetrics", "data", "Raw data directory"),

    # =========================================================================
    # SRC: Consolidate source code
    # =========================================================================
    # Models - already well organized, just move to src/
    ("models/01_simple_model_rcs.py", "src/models/01_simple_model_rcs.py", "Simple model"),
    ("models/02_regional_model_rcs.py", "src/models/02_regional_model_rcs.py", "Regional model"),
    ("models/03_hierarchical_model_rcs.py", "src/models/03_hierarchical_model_rcs.py", "Hierarchical model (main)"),
    ("models/03_hierarchical_model_rcs_v2.py", "src/models/03_hierarchical_model_rcs_v2.py", "Hierarchical model v2"),
    ("models/compare_age_vs_baseline.py", "src/models/compare_age_vs_baseline.py", "Age vs baseline comparison"),
    ("models/__init__.py", "src/models/__init__.py", "Models package init"),

    # Analysis scripts
    ("analysis/diagnostics.py", "src/analysis/diagnostics.py", "Model diagnostics"),
    ("analysis/forecast_gdp.py", "src/analysis/forecast_gdp.py", "GDP forecasting"),
    ("analysis/forecast_u5mr.py", "src/analysis/forecast_u5mr.py", "U5MR forecasting"),
    ("analysis/__init__.py", "src/analysis/__init__.py", "Analysis package init"),

    # Data processing
    ("data_processing/add_age_to_merged.py", "src/data_processing/add_age_to_merged.py", "Add age to merged"),
    ("data_processing/ghed_elasticity_fit.py", "src/data_processing/ghed_elasticity_fit.py", "GHED elasticity"),
    ("data_processing/rebuild_world_fan_constant_countries_logpool.py", "src/data_processing/rebuild_world_fan_constant_countries_logpool.py", "World fan logpool"),
    ("data_processing/rebuild_world_fan_constant_set.py", "src/data_processing/rebuild_world_fan_constant_set.py", "World fan constant set"),
    ("data_processing/rebuild_world_fan_logpool.py", "src/data_processing/rebuild_world_fan_logpool.py", "World fan logpool v2"),
    ("data_processing/step1_prepare_age_data.py", "src/data_processing/step1_prepare_age_data.py", "Step 1: Age data"),
    ("data_processing/step2_aggregate_world_gdp.py", "src/data_processing/step2_aggregate_world_gdp.py", "Step 2: World GDP"),
    ("data_processing/step3_plot_projections.py", "src/data_processing/step3_plot_projections.py", "Step 3: Plot projections"),
    ("data_processing/table_u5mr_regions.py", "src/data_processing/table_u5mr_regions.py", "U5MR regions table"),
    ("data_processing/u5mr_elasticity_fit.py", "src/data_processing/u5mr_elasticity_fit.py", "U5MR elasticity fit"),
    ("data_processing/u5mr_forecast_from_gdp.py", "src/data_processing/u5mr_forecast_from_gdp.py", "U5MR forecast"),
    ("data_processing/u5mr_hindcast.py", "src/data_processing/u5mr_hindcast.py", "U5MR hindcast"),
    ("data_processing/__init__.py", "src/data_processing/__init__.py", "Data processing package init"),

    # Visualization - consolidate all plotting scripts
    ("visualization/fig_gdp_fan_chart.py", "src/visualization/fig_gdp_fan_chart.py", "GDP fan chart"),
    ("visualization/fig_u5mr_projection.py", "src/visualization/fig_u5mr_projection.py", "U5MR projection"),
    ("visualization/publication_figures.py", "src/visualization/publication_figures.py", "Publication figures"),
    ("visualization/__init__.py", "src/visualization/__init__.py", "Visualization package init"),
    ("figures/fig2_gdp_fan_v5.py", "src/visualization/fig2_gdp_fan_v5.py", "Figure 2 GDP fan"),
    ("figures/fig_delta_gdp_fan.py", "src/visualization/fig_delta_gdp_fan.py", "Delta GDP fan"),
    ("figures/fig_global_gdp_fan.py", "src/visualization/fig_global_gdp_fan.py", "Global GDP fan"),
    ("figures/fig_global_gdp_fan_age.py", "src/visualization/fig_global_gdp_fan_age.py", "Global GDP fan age"),

    # Utils
    ("utils/build_age_covariates_from_owid.py", "src/utils/build_age_covariates_from_owid.py", "Build age covariates"),
    ("utils/build_pop_predictions.py", "src/utils/build_pop_predictions.py", "Build pop predictions"),
    ("utils/calculation.py", "src/utils/calculation.py", "Calculations"),
    ("utils/diagnose_world_fan_coverage.py", "src/utils/diagnose_world_fan_coverage.py", "World fan diagnostics"),
    ("utils/ingest_health_data.py", "src/utils/ingest_health_data.py", "Ingest health data"),
    ("utils/late_year_diagnostics.py", "src/utils/late_year_diagnostics.py", "Late year diagnostics"),
    ("utils/rcs_basis.py", "src/utils/rcs_basis.py", "RCS basis functions"),
    ("utils/__init__.py", "src/utils/__init__.py", "Utils package init"),

    # Tables
    ("tables/table_param_summary.py", "src/tables/table_param_summary.py", "Parameter summary"),
    ("tables/table_param_summary_general.py", "src/tables/table_param_summary_general.py", "Parameter summary general"),
    ("tables/table_posterior_summary_compact_rcs.py", "src/tables/table_posterior_summary_compact_rcs.py", "Posterior summary"),
    ("tables/table_top10_economies.py", "src/tables/table_top10_economies.py", "Top 10 economies"),
    ("tables/table_top10_economies_rcs.py", "src/tables/table_top10_economies_rcs.py", "Top 10 economies RCS"),
    ("tables/__init__.py", "src/tables/__init__.py", "Tables package init"),

    # Scripts (execution scripts)
    ("scripts/hierarchical_model_with_rcs.py", "src/scripts/hierarchical_model_with_rcs.py", "Hierarchical RCS script"),
    ("scripts/hierarchical_model_with_rcs_age.py", "src/scripts/hierarchical_model_with_rcs_age.py", "Hierarchical RCS age"),
    ("scripts/make_table1_v5.py", "src/scripts/make_table1_v5.py", "Make Table 1"),
    ("scripts/plot_fanchart.py", "src/scripts/plot_fanchart.py", "Plot fan chart"),
    ("scripts/prediction.py", "src/scripts/prediction.py", "Prediction"),
    ("scripts/prediction_rcs.py", "src/scripts/prediction_rcs.py", "Prediction RCS"),
    ("scripts/prediction_rcs_age.py", "src/scripts/prediction_rcs_age.py", "Prediction RCS age"),
    ("scripts/run_fig3_u5mr_v5.py", "src/scripts/run_fig3_u5mr_v5.py", "Run Figure 3 U5MR"),
    ("scripts/run_forecast_age_v5.py", "src/scripts/run_forecast_age_v5.py", "Run forecast age"),
    ("scripts/run_forecast_v5_simple.py", "src/scripts/run_forecast_v5_simple.py", "Run forecast simple"),
    ("scripts/run_forecast_v5_world_agg.py", "src/scripts/run_forecast_v5_world_agg.py", "Run forecast world agg"),
    ("scripts/run_forecast_v5_world_agg_fixed.py", "src/scripts/run_forecast_v5_world_agg_fixed.py", "Run forecast world agg fixed"),

    # Config
    ("config.py", "src/config.py", "Central configuration"),

    # =========================================================================
    # DOCUMENTS: Consolidate documentation
    # =========================================================================
    ("main_manuscript.docx", "docs/main_manuscript.docx", "Main manuscript (old)"),
    ("main_manuscript_GDPAndPOP.docx", "docs/main_manuscript_GDPAndPOP.docx", "Main manuscript"),
    ("README.md", "docs/README.md", "Project README"),
    ("data/README.md", "docs/data_README.md", "Data README"),

    # =========================================================================
    # RESULTS: Model outputs
    # =========================================================================
    ("outputs/simple_model_rcs.nc", "results/simple_model_rcs.nc", "Simple model posterior"),
    ("outputs/regional_model_rcs.nc", "results/regional_model_rcs.nc", "Regional model posterior"),
    ("outputs/cache", "results/cache", "Cache files"),
    ("outputs/tables", "results/tables", "Generated tables"),

    # Cache files from utils
    ("utils/rcs_knots_hier.npy", "results/cache/rcs_knots_hier.npy", "RCS knots hierarchical"),
    ("utils/rcs_knots_region.npy", "results/cache/rcs_knots_region.npy", "RCS knots regional"),
    ("utils/scale_rcs_age.json", "results/cache/scale_rcs_age.json", "Scale RCS age"),
    ("utils/scale_rcs_age_v3.json", "results/cache/scale_rcs_age_v3.json", "Scale RCS age v3"),
]

# Figure moves - separate for clarity
FIGURE_MOVES = [
    # Main figures (publication quality)
    ("figures/fig1_global_association.pdf", "figures/main/fig1_global_association.pdf", "Figure 1"),
    ("figures/fig1_global_association.png", "figures/main/fig1_global_association.png", "Figure 1 PNG"),
    ("figures/fig2_regional_effects.pdf", "figures/main/fig2_regional_effects.pdf", "Figure 2"),
    ("figures/fig2_regional_effects.png", "figures/main/fig2_regional_effects.png", "Figure 2 PNG"),
    ("figures/fig3_demographic_effects.pdf", "figures/main/fig3_demographic_effects.pdf", "Figure 3"),
    ("figures/fig3_demographic_effects.png", "figures/main/fig3_demographic_effects.png", "Figure 3 PNG"),
    ("figures/fig4_time_trends.pdf", "figures/main/fig4_time_trends.pdf", "Figure 4"),
    ("figures/fig4_time_trends.png", "figures/main/fig4_time_trends.png", "Figure 4 PNG"),
    ("figures/fig5_hierarchical_overview.pdf", "figures/main/fig5_hierarchical_overview.pdf", "Figure 5"),
    ("figures/fig5_hierarchical_overview.png", "figures/main/fig5_hierarchical_overview.png", "Figure 5 PNG"),

    # Global fan charts
    ("figures/fig_global_fanchart.png", "figures/main/fig_global_fanchart.png", "Global fan chart"),
    ("figures/fig_global_fanchart_rcs.png", "figures/main/fig_global_fanchart_rcs.png", "Global fan chart RCS"),
    ("figures/fig_global_gdp_fan.png", "figures/main/fig_global_gdp_fan.png", "Global GDP fan"),
    ("figures/fig_global_gdp_fan_age.png", "figures/main/fig_global_gdp_fan_age.png", "Global GDP fan age"),
    ("figures/fig_global_u5mr_fan.png", "figures/main/fig_global_u5mr_fan.png", "Global U5MR fan"),
    ("figures/Figure2_Global_GDP_Fan_v5.png", "figures/main/Figure2_Global_GDP_Fan_v5.png", "Figure 2 v5"),
    ("figures/Figure2_Global_GDP_Fan_v5_Final.png", "figures/main/Figure2_Global_GDP_Fan_v5_Final.png", "Figure 2 v5 final"),

    # Diagnostic figures
    ("figures/simple_model.png", "figures/diagnostics/simple_model.png", "Simple model diagnostics"),
    ("figures/simple_model_with_rcs.png", "figures/diagnostics/simple_model_with_rcs.png", "Simple RCS diagnostics"),
    ("figures/simple_model_with_quadratic.png", "figures/diagnostics/simple_model_with_quadratic.png", "Simple quadratic diagnostics"),
    ("figures/regional_model.png", "figures/diagnostics/regional_model.png", "Regional model diagnostics"),
    ("figures/regional_model_with_rcs.png", "figures/diagnostics/regional_model_with_rcs.png", "Regional RCS diagnostics"),
    ("figures/regional_model_with_rcs_posterior.png", "figures/diagnostics/regional_model_with_rcs_posterior.png", "Regional posterior"),
    ("figures/regional_model_with_quadratic.png", "figures/diagnostics/regional_model_with_quadratic.png", "Regional quadratic"),
    ("figures/hierarchical_model.png", "figures/diagnostics/hierarchical_model.png", "Hierarchical diagnostics"),
    ("figures/hierarchical_model_with_quadratic.png", "figures/diagnostics/hierarchical_model_with_quadratic.png", "Hierarchical quadratic"),

    # Supplementary
    ("figures/fig_global_association.png", "figures/supplementary/fig_global_association.png", "Global association"),
    ("figures/fig_global_association_rcs.png", "figures/supplementary/fig_global_association_rcs.png", "Global association RCS"),

    # Outputs figures
    ("outputs/figures/simple_model_rcs.png", "figures/diagnostics/outputs_simple_model_rcs.png", "Output simple model"),
    ("outputs/figures/regional_model_rcs.png", "figures/diagnostics/outputs_regional_model_rcs.png", "Output regional model"),
    ("outputs/figures/regional_model_rcs_posterior.png", "figures/diagnostics/outputs_regional_model_rcs_posterior.png", "Output regional posterior"),
]


def create_directory_structure(dry_run=False):
    """Create the target directory structure."""
    directories = [
        "src/models",
        "src/analysis",
        "src/visualization",
        "src/data_processing",
        "src/utils",
        "src/tables",
        "src/scripts",
        "figures/main",
        "figures/supplementary",
        "figures/diagnostics",
        "docs",
        "results/cache",
        "results/tables",
    ]

    for d in directories:
        path = PROJECT_ROOT / d
        if dry_run:
            print(f"  [CREATE DIR] {d}/")
        else:
            path.mkdir(parents=True, exist_ok=True)
            print(f"  [OK] Created {d}/")


def safe_move(src, dst, description, dry_run=False):
    """Safely move a file or directory with verification."""
    src_path = PROJECT_ROOT / src
    dst_path = PROJECT_ROOT / dst

    if not src_path.exists():
        print(f"  [SKIP] {src} (not found)")
        return False

    if dst_path.exists():
        print(f"  [SKIP] {dst} (already exists)")
        return False

    if dry_run:
        print(f"  [MOVE] {src} -> {dst}")
        return True

    # Ensure parent directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    # Move file or directory
    shutil.move(str(src_path), str(dst_path))
    print(f"  [OK] {src} -> {dst}")
    return True


def create_src_init_files(dry_run=False):
    """Create __init__.py files for new src package structure."""
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/analysis/__init__.py",
        "src/visualization/__init__.py",
        "src/data_processing/__init__.py",
        "src/utils/__init__.py",
        "src/tables/__init__.py",
        "src/scripts/__init__.py",
    ]

    for f in init_files:
        path = PROJECT_ROOT / f
        if dry_run:
            if not path.exists():
                print(f"  [CREATE] {f}")
        else:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('"""Package initialization."""\n')
                print(f"  [OK] Created {f}")


def create_root_readme(dry_run=False):
    """Create a new root README.md with project structure documentation."""
    readme_content = '''# GDPandPOP: Hierarchical Bayesian Modeling for Global GDP Projections

## Project Overview

This project estimates the association between population, age structure, and technological drift on GDP using a hierarchical Bayesian model with panel data from 180 countries (1960-2023), and forecasts GDP and Under-5 Mortality Rate (U5MR) up to 2100.

## Methodology

- **Framework**: Hierarchical Bayesian modeling using PyMC (v5)
- **Features**: Restricted Cubic Splines (RCS), Non-Centered Parameterization (NCP), Orthogonalization
- **Data**: World Bank, UN World Population Prospects 2024

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

1. `src/data_processing/step1_prepare_age_data.py` - Prepare age structure data
2. `src/data_processing/add_age_to_merged.py` - Merge age data with panel
3. `src/models/03_hierarchical_model_rcs.py` - Fit Bayesian model
4. `src/analysis/forecast_gdp.py` - Generate projections

## Requirements

- Python 3.11+
- PyMC 5.x
- ArviZ
- Pandas, NumPy, Matplotlib, Seaborn

## License

[Add license information]
'''

    path = PROJECT_ROOT / "README.md"
    if dry_run:
        print("  [CREATE] README.md (new)")
    else:
        # Backup old README if it exists
        if path.exists():
            backup = PROJECT_ROOT / "docs" / "README_old.md"
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(path), str(backup))
        path.write_text(readme_content, encoding='utf-8')
        print("  [OK] Created new README.md")


def cleanup_empty_dirs(dry_run=False):
    """Remove empty directories after reorganization."""
    dirs_to_check = [
        "models",
        "analysis",
        "visualization",
        "data_processing",
        "utils",
        "tables",
        "scripts",
        "outputs/figures",
        "outputs",
    ]

    for d in dirs_to_check:
        path = PROJECT_ROOT / d
        if path.exists() and path.is_dir():
            try:
                # Check if directory is empty (no files, only maybe __pycache__)
                contents = list(path.iterdir())
                contents = [c for c in contents if c.name != "__pycache__"]
                if not contents:
                    if dry_run:
                        print(f"  [REMOVE] {d}/ (empty)")
                    else:
                        shutil.rmtree(str(path))
                        print(f"  [OK] Removed empty {d}/")
            except Exception as e:
                print(f"  [WARN] Could not check/remove {d}/: {e}")


def main():
    parser = argparse.ArgumentParser(description="Reorganize GDPandPOP project structure")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without executing")
    args = parser.parse_args()

    print("=" * 60)
    print("GDPandPOP Project Reorganization")
    print("=" * 60)

    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")

    # Create backup log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = PROJECT_ROOT / f"reorganization_log_{timestamp}.txt"

    # Step 1: Create directory structure
    print("\n[1/6] Creating directory structure...")
    create_directory_structure(args.dry_run)

    # Step 2: Create __init__.py files
    print("\n[2/6] Creating package init files...")
    create_src_init_files(args.dry_run)

    # Step 3: Move source files
    print("\n[3/6] Moving source files...")
    for src, dst, desc in MOVES:
        safe_move(src, dst, desc, args.dry_run)

    # Step 4: Move figure files
    print("\n[4/6] Moving figure files...")
    for src, dst, desc in FIGURE_MOVES:
        safe_move(src, dst, desc, args.dry_run)

    # Step 5: Create new README
    print("\n[5/6] Creating new README...")
    create_root_readme(args.dry_run)

    # Step 6: Cleanup empty directories
    print("\n[6/6] Cleaning up empty directories...")
    cleanup_empty_dirs(args.dry_run)

    print("\n" + "=" * 60)
    if args.dry_run:
        print("DRY RUN COMPLETE - Run without --dry-run to execute")
    else:
        print("REORGANIZATION COMPLETE")
        print(f"\nNext steps:")
        print("  1. Update imports in Python files (see update_imports.py)")
        print("  2. Review and test the new structure")
        print("  3. Commit changes to git")
    print("=" * 60)


if __name__ == "__main__":
    main()
