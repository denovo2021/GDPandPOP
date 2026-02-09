# rcs_basis.py
# Restricted Cubic Spline Basis Functions for GDP-Population Modeling
# =============================================================================
"""
MATHEMATICAL FOUNDATIONS OF RESTRICTED CUBIC SPLINES (RCS)
=============================================================================

1. WHY RCS FOR GDP-POPULATION MODELING?
---------------------------------------
The relationship between log(Population) and log(GDP) is approximately linear
(constant elasticity), but with important nonlinearities:
  - Diminishing returns at very large populations (e.g., India, China)
  - Higher variance for small populations (e.g., microstates)
  - Potential non-monotonicity in extreme ranges

RCS offers key advantages for economic projections:
  a) LINEAR EXTRAPOLATION beyond observed data range
     - Critical for forecasting (2025-2100)
     - Prevents explosive/imploding predictions
  b) SMOOTH, differentiable function (unlike piecewise linear)
  c) INTERPRETABLE as deviation from linear trend
  d) PARSIMONIOUS: K knots → K-2 parameters

2. MATHEMATICAL SPECIFICATION
-----------------------------
For K knots k₁ < k₂ < ... < k_K, the j-th basis function (j = 1, ..., K-2) is:

    rcs_j(x) = d_j(x) - d_{K-1}(x) · λ_j + d_0(x) · (1 - λ_j)

where:
    d_j(x) = max(x - k_j, 0)³     [truncated cubic]
    λ_j = (k_K - k_j) / (k_K - k₁)

This construction ensures:
  - Second derivative = 0 at boundary knots → LINEAR TAILS
  - Continuous first and second derivatives throughout
  - Cubic polynomial between adjacent interior knots

3. FULL MODEL EQUATION
----------------------
The complete RCS regression for log-GDP is:

    log₁₀(GDP) = α + β·x + Σⱼ θⱼ·rcs_j(x) + ε

where x = (log₁₀(Pop) - μ) / σ  [standardized]

The MARGINAL ELASTICITY (how GDP responds to population) is:

    ∂log(GDP)/∂log(Pop) = β + Σⱼ θⱼ · ∂rcs_j/∂x

This elasticity VARIES with x, capturing nonlinear population effects.

4. KNOT PLACEMENT
-----------------
Recommended: Place knots at data quantiles for even spacing in data density.
For K=4 (our default): [5th, 35th, 65th, 95th] percentiles
For K=5 (more flexible): [5th, 27.5th, 50th, 72.5th, 95th] percentiles

Boundary knots (5th, 95th) determine where linear tails begin—crucial for
extrapolation behavior in forecasting.

5. ORTHOGONALIZATION
--------------------
For stable estimation, we orthogonalize RCS to the linear term [1, x]:

    Z̃ = Z - [1, x] · ([1, x]ᵀ[1, x])⁻¹ · [1, x]ᵀZ

This ensures:
  - θ coefficients capture PURE NONLINEARITY (not confounded with linear trend)
  - β represents the average linear elasticity
  - Better numerical conditioning for MCMC sampling

6. SCALING FOR NUMERICAL STABILITY
----------------------------------
RCS basis columns can have vastly different scales (cubic terms grow fast).
We standardize each column to unit variance:

    Z̃_scaled = Z̃ / diag(std(Z̃))

This ensures θ coefficients are on comparable scales with informative priors.

References
----------
- Harrell, F.E. (2015). Regression Modeling Strategies. Springer.
- Durrleman, S. & Simon, R. (1989). Flexible regression models with cubic splines.
  Statistics in Medicine, 8(5), 551-561.
"""

import numpy as np
from typing import Tuple, Dict, Optional


def rcs_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Compute Harrell's Restricted Cubic Spline basis matrix.

    Parameters
    ----------
    x : array-like, shape (N,)
        Predictor values. Should be on the same scale as knots.

    knots : array-like, shape (K,)
        Knot positions, sorted in ascending order.
        K must be >= 3 for meaningful splines.

    Returns
    -------
    Z : ndarray, shape (N, K-2)
        RCS basis matrix. Each column is a basis function.
        Returns empty array (N, 0) if K < 3.

    Notes
    -----
    The j-th basis function (j = 1, ..., K-2) is:

        rcs_j(x) = d_j(x) - d_{K-1}(x) · (k_K - k_j)/(k_K - k₁)
                         + d_0(x) · (k_j - k₁)/(k_K - k₁)

    where d_j(x) = max(x - k_j, 0)³

    This ensures LINEAR EXTRAPOLATION beyond boundary knots—critical for
    out-of-sample forecasting.

    Examples
    --------
    >>> x = np.linspace(-2, 2, 100)
    >>> knots = np.array([-1.5, -0.5, 0.5, 1.5])  # 4 knots
    >>> Z = rcs_design(x, knots)
    >>> Z.shape
    (100, 2)  # K-2 = 2 basis functions
    """
    x = np.asarray(x).ravel()
    k = np.asarray(knots).ravel()
    K = k.size
    N = x.size

    if K < 3:
        return np.zeros((N, 0))

    # Truncated cubic power function
    def d(u: np.ndarray, j: int) -> np.ndarray:
        return np.maximum(u - k[j], 0.0) ** 3

    # Build basis functions
    cols = []
    for j in range(1, K - 1):
        # Harrell's formula for restricted cubic spline
        lambda_j = (k[K - 1] - k[j]) / (k[K - 1] - k[0])
        basis_j = d(x, j) - d(x, K - 1) * lambda_j + d(x, 0) * (1 - lambda_j)
        cols.append(basis_j)

    return np.column_stack(cols) if cols else np.zeros((N, 0))


def rcs_derivative(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """
    Compute the derivative of RCS basis functions.

    Useful for computing marginal effects (how the slope varies with x).

    Parameters
    ----------
    x : array-like, shape (N,)
        Predictor values.
    knots : array-like, shape (K,)
        Knot positions.

    Returns
    -------
    dZ : ndarray, shape (N, K-2)
        Derivative of each basis function at each x.

    Notes
    -----
    The derivative of the j-th basis function is:

        d/dx rcs_j(x) = 3·[max(x-k_j,0)² - λ_j·max(x-k_{K-1},0)²
                                         + (1-λ_j)·max(x-k_0,0)²]

    For x beyond the boundary knots, this becomes constant (linear tail).
    """
    x = np.asarray(x).ravel()
    k = np.asarray(knots).ravel()
    K = k.size
    N = x.size

    if K < 3:
        return np.zeros((N, 0))

    # Derivative of truncated cubic: 3 * max(u - k_j, 0)²
    def d_deriv(u: np.ndarray, j: int) -> np.ndarray:
        return 3.0 * np.maximum(u - k[j], 0.0) ** 2

    cols = []
    for j in range(1, K - 1):
        lambda_j = (k[K - 1] - k[j]) / (k[K - 1] - k[0])
        deriv_j = (d_deriv(x, j)
                   - d_deriv(x, K - 1) * lambda_j
                   + d_deriv(x, 0) * (1 - lambda_j))
        cols.append(deriv_j)

    return np.column_stack(cols) if cols else np.zeros((N, 0))


def orthonormalize_basis(Z: np.ndarray,
                         X_linear: np.ndarray,
                         standardize: bool = True) -> Tuple[np.ndarray, Dict]:
    """
    Orthogonalize RCS basis with respect to linear terms and optionally standardize.

    Parameters
    ----------
    Z : ndarray, shape (N, m)
        Raw RCS basis matrix.
    X_linear : ndarray, shape (N, 2)
        Matrix [1, x] for intercept and linear term.
    standardize : bool, default True
        Whether to scale columns to unit variance.

    Returns
    -------
    Z_ortho : ndarray, shape (N, m)
        Orthogonalized (and optionally standardized) basis.
    transform_info : dict
        Contains parameters needed to apply the same transformation to new data:
        - 'proj_coef': Coefficients for projecting out linear terms
        - 'col_means': Column means (for centering)
        - 'col_stds': Column standard deviations (for scaling)

    Notes
    -----
    Orthogonalization ensures that:
    1. RCS captures PURE nonlinearity (orthogonal to linear trend)
    2. The linear coefficient β has clean interpretation as average elasticity
    3. θ coefficients represent deviation from linearity

    Standardization ensures:
    1. All basis columns have comparable scale
    2. Priors on θ are meaningful (same prior SD has same effect for all j)
    3. Better numerical conditioning for MCMC
    """
    N, m = Z.shape

    if m == 0:
        return Z, {"proj_coef": np.zeros((X_linear.shape[1], 0)),
                   "col_means": np.array([]),
                   "col_stds": np.array([])}

    # Step 1: Remove linear component via projection
    # Z_ortho = Z - X @ (X'X)^{-1} X'Z
    proj_coef = np.linalg.lstsq(X_linear, Z, rcond=None)[0]
    Z_resid = Z - X_linear @ proj_coef

    # Step 2: Center (should already be ~0 mean after orthogonalization)
    col_means = Z_resid.mean(axis=0)
    Z_centered = Z_resid - col_means

    # Step 3: Standardize to unit variance (if requested)
    if standardize:
        col_stds = Z_centered.std(axis=0, ddof=1)
        col_stds = np.where(col_stds < 1e-10, 1.0, col_stds)  # Prevent div by zero
        Z_ortho = Z_centered / col_stds
    else:
        col_stds = np.ones(m)
        Z_ortho = Z_centered

    transform_info = {
        "proj_coef": proj_coef,
        "col_means": col_means,
        "col_stds": col_stds
    }

    return Z_ortho, transform_info


def apply_basis_transform(x_new: np.ndarray,
                          knots: np.ndarray,
                          transform_info: Dict,
                          x_center: float,
                          x_scale: float) -> np.ndarray:
    """
    Apply saved RCS transformation to new data.

    Essential for making predictions on new observations (e.g., future years).

    Parameters
    ----------
    x_new : array-like
        New predictor values (on original scale, e.g., log10(Population)).
    knots : array-like
        Knot positions (on standardized scale).
    transform_info : dict
        Output from orthonormalize_basis().
    x_center : float
        Mean used for centering (e.g., MU_GLOBAL).
    x_scale : float
        Standard deviation used for scaling (e.g., SD_GLOBAL).

    Returns
    -------
    Z_new : ndarray
        Transformed RCS basis for new data.

    Notes
    -----
    The transformation sequence must match exactly what was done to training data:
    1. Standardize x_new using saved center and scale
    2. Compute raw RCS on standardized scale
    3. Apply saved projection coefficients to orthogonalize
    4. Apply saved centering and scaling
    """
    x_new = np.asarray(x_new).ravel()

    # Standardize to same scale as training
    x_s = (x_new - x_center) / x_scale

    # Compute raw RCS basis
    Z_raw = rcs_design(x_s, knots)

    if Z_raw.shape[1] == 0:
        return Z_raw

    # Apply orthogonalization: remove linear component
    X_linear = np.column_stack([np.ones_like(x_s), x_s])
    Z_resid = Z_raw - X_linear @ transform_info["proj_coef"]

    # Apply centering and scaling
    Z_centered = Z_resid - transform_info["col_means"]
    Z_scaled = Z_centered / transform_info["col_stds"]

    return Z_scaled


def compute_marginal_elasticity(x: np.ndarray,
                                knots: np.ndarray,
                                beta: float,
                                theta: np.ndarray,
                                transform_info: Dict,
                                x_center: float,
                                x_scale: float) -> np.ndarray:
    """
    Compute marginal elasticity at given population levels.

    The marginal elasticity is:
        ∂log(GDP)/∂log(Pop) = β + Σⱼ θⱼ · (∂Z̃ⱼ/∂x) · (1/x_scale)

    Parameters
    ----------
    x : array-like
        Population values (on log10 scale, not standardized).
    knots : array-like
        Knot positions (on standardized scale).
    beta : float
        Linear elasticity coefficient.
    theta : array-like
        RCS coefficients.
    transform_info : dict
        From orthonormalize_basis().
    x_center, x_scale : float
        Standardization parameters.

    Returns
    -------
    elasticity : ndarray
        Marginal elasticity at each x value.

    Notes
    -----
    Due to the chain rule, derivatives of the standardized/orthogonalized
    basis require accounting for all transformations.
    """
    x = np.asarray(x).ravel()
    theta = np.asarray(theta).ravel()

    # Standardize x
    x_s = (x - x_center) / x_scale

    # Derivative of raw RCS
    dZ_raw = rcs_derivative(x_s, knots)

    if dZ_raw.shape[1] == 0:
        return np.full(x.shape, beta)

    # The derivative of orthogonalized basis:
    # d(Z_ortho)/dx = d(Z_raw - X @ proj_coef)/dx / col_stds
    # = (dZ_raw/dx - [0, 1] @ proj_coef) / col_stds
    # Note: d(X_linear)/dx = [0, 1] (intercept derivative is 0)
    dZ_ortho_dx = (dZ_raw - np.outer(np.ones_like(x_s), transform_info["proj_coef"][1, :]))
    dZ_ortho_dx = dZ_ortho_dx / transform_info["col_stds"]

    # Marginal elasticity (chain rule: need to account for x_scale)
    # ∂y/∂(log Pop) = β + θᵀ · ∂Z̃/∂x_s · ∂x_s/∂(log Pop)
    #               = β + θᵀ · ∂Z̃/∂x_s · (1/x_scale)
    elasticity = beta + (dZ_ortho_dx @ theta) / x_scale

    return elasticity


def suggest_knots(x: np.ndarray,
                  n_knots: int = 4,
                  boundary_quantiles: Tuple[float, float] = (0.05, 0.95)) -> np.ndarray:
    """
    Suggest knot positions based on data distribution.

    Parameters
    ----------
    x : array-like
        Predictor values.
    n_knots : int, default 4
        Number of knots. Recommendations:
        - K=3: Very limited nonlinearity, use for near-linear relationships
        - K=4: Moderate flexibility (default), good for most applications
        - K=5: More flexible, use if strong nonlinearity expected
        - K>5: Rarely needed, risk of overfitting
    boundary_quantiles : tuple, default (0.05, 0.95)
        Quantiles for boundary knots. These determine where linear
        extrapolation begins—crucial for forecasting.

    Returns
    -------
    knots : ndarray, shape (n_knots,)
        Suggested knot positions at evenly-spaced quantiles.

    Notes
    -----
    Harrell recommends placing knots at fixed quantiles rather than
    optimizing their positions, as this provides more stable inference.
    """
    x = np.asarray(x).ravel()

    if n_knots < 3:
        raise ValueError("Need at least 3 knots for RCS")

    # Compute quantiles with boundary constraints
    q_lo, q_hi = boundary_quantiles
    interior_quantiles = np.linspace(q_lo, q_hi, n_knots)

    knots = np.quantile(x, interior_quantiles)

    return knots


# =============================================================================
# Diagnostic Functions
# =============================================================================

def check_knot_coverage(x: np.ndarray, knots: np.ndarray) -> Dict:
    """
    Check how well knots cover the data distribution.

    Returns statistics about data coverage and potential extrapolation issues.
    """
    x = np.asarray(x).ravel()
    knots = np.asarray(knots).ravel()

    n_below = np.sum(x < knots[0])
    n_above = np.sum(x > knots[-1])
    n_total = len(x)

    coverage = {
        "n_observations": n_total,
        "n_below_boundary": n_below,
        "n_above_boundary": n_above,
        "pct_extrapolation": 100 * (n_below + n_above) / n_total,
        "x_range": (x.min(), x.max()),
        "knot_range": (knots[0], knots[-1]),
        "warning": ""
    }

    if coverage["pct_extrapolation"] > 15:
        coverage["warning"] = ("High extrapolation fraction ({:.1f}%). "
                               "Consider adjusting boundary quantiles.".format(
                               coverage["pct_extrapolation"]))

    return coverage


def verify_orthogonality(Z_ortho: np.ndarray, X_linear: np.ndarray,
                         tol: float = 1e-6) -> Dict:
    """
    Verify that orthogonalized basis is truly orthogonal to linear terms.

    Parameters
    ----------
    Z_ortho : ndarray
        Orthogonalized RCS basis.
    X_linear : ndarray
        Linear terms [1, x].
    tol : float
        Tolerance for checking orthogonality.

    Returns
    -------
    result : dict
        Orthogonality diagnostics.
    """
    N = Z_ortho.shape[0]
    inner_products = X_linear.T @ Z_ortho / N

    max_inner = np.abs(inner_products).max()
    is_orthogonal = max_inner < tol

    return {
        "max_inner_product": max_inner,
        "is_orthogonal": is_orthogonal,
        "tolerance": tol,
        "inner_product_matrix": inner_products
    }
