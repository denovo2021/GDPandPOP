# Archive Directory

This directory contains deprecated model versions preserved for reference.

## Archived Files

### Quadratic Models (Superseded by RCS)

These models used quadratic polynomial terms for the population-GDP relationship.
They were replaced by Restricted Cubic Splines (RCS) models which provide:
- More flexible nonlinear relationships
- Linear extrapolation behavior outside knot range
- Better interpretability

| File | Description |
|------|-------------|
| `simple_model_quadratic.py` | Layer 1: Global pooling with quadratic term |
| `regional_model_quadratic.py` | Layer 2: Regional partial pooling with quadratic |
| `hierarchical_model_quadratic.py` | Layer 3: Full hierarchy with quadratic |
| `simple_model_with_quadratic.py` | Alternative quadratic implementation |
| `regional_model_with_quadratic.py` | Alternative quadratic implementation |
| `hierarchical_model_with_quadratic.py` | Alternative quadratic implementation |

### Development Versions

| File | Description |
|------|-------------|
| `hierarchical_model_with_rcs_age_v2.py` | Earlier RCS version (documents evolution) |

## Current Models

The current production models are in `/models/`:

- `01_simple_model_rcs.py` - Layer 1: Global pooling with RCS
- `02_regional_model_rcs.py` - Layer 2: Regional partial pooling with RCS
- `03_hierarchical_model_rcs.py` - Layer 3: Full hierarchical with RCS + demographics

## Why Keep These?

These files are preserved for:
1. **Reproducibility**: Reviewers may request comparison with simpler models
2. **Documentation**: Shows methodological evolution
3. **Debugging**: Reference for understanding model development decisions
