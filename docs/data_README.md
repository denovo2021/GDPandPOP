# Data Documentation

This directory documents the data sources and preparation steps for the GDPandPOP project.

## Required Input Files

Place the following CSV files in the **project root** directory for the analysis scripts to run:

### 1. `merged.csv`
Main GDP-Population panel data.

| Column | Description |
|--------|-------------|
| ISO3 | ISO 3166-1 alpha-3 country code |
| Country Name | Full country name |
| Region | World Bank region classification |
| Year | Calendar year |
| Population | Total population |
| GDP | Gross Domestic Product (current USD) |

### 2. `merged_age.csv`
Extended data with demographic age structure variables.

| Column | Description |
|--------|-------------|
| (All columns from merged.csv) | ... |
| WAshare | Working-age population share (15-64 years) |
| OldDep | Old-age dependency ratio (65+ / 15-64) |

### 3. `pop_predictions_scenarios.csv`
UN World Population Prospects 2024 population projections.

| Column | Description |
|--------|-------------|
| ISO3 | Country code |
| Year | Projection year (2024-2100) |
| Scenario | UN variant (Low, Medium, High, etc.) |
| Population | Projected total population |

### 4. `age_predictions_scenarios.csv`
UN WPP 2024 age structure projections.

| Column | Description |
|--------|-------------|
| ISO3 | Country code |
| Year | Projection year |
| Scenario | UN variant |
| WAshare | Projected working-age share |
| OldDep | Projected old-age dependency ratio |

## Data Sources

### Historical Data
- **GDP**: World Bank World Development Indicators (WDI)
  - Indicator: NY.GDP.MKTP.CD (GDP, current USD)
  - Source: https://data.worldbank.org/

- **Population**: UN World Population Prospects 2024
  - Source: https://population.un.org/wpp/

### Projection Data
- **UN WPP 2024**: World Population Prospects 2024 Revision
  - Probabilistic projections for 9 scenarios
  - Source: https://population.un.org/wpp/Download/

## Data Preparation

Historical data was merged and cleaned using the scripts in `data_processing/`:

1. `step1_prepare_age_data.py` - Merge age structure variables
2. `step2_aggregate_world_gdp.py` - Compute world aggregates
3. `step3_plot_projections.py` - Validate and visualize projections

## Column Specifications

### Region Classification
Uses World Bank regional classification:
- East Asia & Pacific
- Europe & Central Asia
- Latin America & Caribbean
- Middle East & North Africa
- North America
- South Asia
- Sub-Saharan Africa

### Year Coverage
- Historical: 1960-2023 (varies by country)
- Projections: 2024-2100
