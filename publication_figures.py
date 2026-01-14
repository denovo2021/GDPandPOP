# publication_figures.py
# Generate publication-quality figures and tables for hierarchical GDP-Population analysis
# Target: Academic journal (LaTeX/PDF compatible)
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path
import json

# =============================================================================
# Publication-quality matplotlib settings
# =============================================================================
def set_publication_style():
    """Configure matplotlib for academic journal publication."""
    rcParams.update({
        # Font settings (compatible with LaTeX)
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,

        # Figure settings
        'figure.figsize': (6.5, 4.5),  # Single column width for journals
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Line and marker settings
        'lines.linewidth': 1.5,
        'lines.markersize': 4,

        # Axes settings
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,

        # Legend
        'legend.frameon': False,
        'legend.loc': 'best',

        # LaTeX compatibility
        'text.usetex': False,  # Set True if LaTeX is installed
        'mathtext.fontset': 'stix',
    })

set_publication_style()

# =============================================================================
# Paths
# =============================================================================
PROJ = Path(r"C:/Users/aaagc/OneDrive/ドキュメント/GDPandPOP")
PATH_DATA = PROJ / "merged_age.csv"
PATH_DATA_RAW = PROJ / "merged.csv"
PATH_KNOTS = PROJ / "rcs_knots_hier.npy"
PATH_SCALE = PROJ / "scale_rcs_age_v3.json"

# Model files
PATH_SIMPLE = PROJ / "simple_model_with_rcs.nc"
PATH_REGIONAL = PROJ / "regional_model_with_rcs.nc"
PATH_HIERARCHICAL = PROJ / "hierarchical_model_with_rcs_age_v3.nc"

# Output paths
OUT_FIG = PROJ / "figures"
OUT_TABLE = PROJ / "tables"
OUT_FIG.mkdir(exist_ok=True)
OUT_TABLE.mkdir(exist_ok=True)

# =============================================================================
# Helper: RCS design matrix
# =============================================================================
def rcs_design(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Restricted cubic spline with linear tails."""
    k = np.asarray(knots)
    K = k.size
    if K < 3:
        return np.zeros((x.size, 0))
    def d(u, j):
        return np.maximum(u - k[j], 0.0) ** 3
    cols = []
    for j in range(1, K - 1):
        cols.append(d(x, j)
                    - d(x, K - 1) * (k[K - 1] - k[j]) / (k[K - 1] - k[0])
                    + d(x, 0)     * (k[j]     - k[0]) / (k[K - 1] - k[0]))
    return np.column_stack(cols) if cols else np.zeros((x.size, 0))

# =============================================================================
# Load data and model
# =============================================================================
print("Loading data and model results...")

# Load data
df = pd.read_csv(PATH_DATA_RAW, index_col=0)
df = df.dropna(subset=["Region", "Population", "GDP"])
df["Log_Population"] = np.log10(df["Population"])
df["Log_GDP"] = np.log10(df["GDP"])
log_pop_mean = df["Log_Population"].mean()
df["Log_Population_c"] = df["Log_Population"] - log_pop_mean

# Load knots
try:
    knots = np.load(PATH_KNOTS)
except:
    knots = np.quantile(df["Log_Population_c"].values, [0.05, 0.35, 0.65, 0.95])

# Load model results
idata_simple = az.from_netcdf(PATH_SIMPLE)
print(f"Loaded: {PATH_SIMPLE.name}")

try:
    idata_regional = az.from_netcdf(PATH_REGIONAL)
    print(f"Loaded: {PATH_REGIONAL.name}")
except:
    idata_regional = None
    print(f"Not found: {PATH_REGIONAL.name}")

try:
    idata_hier = az.from_netcdf(PATH_HIERARCHICAL)
    print(f"Loaded: {PATH_HIERARCHICAL.name}")
except:
    idata_hier = None
    print(f"Not found: {PATH_HIERARCHICAL.name}")

# =============================================================================
# Figure 1: Global Association Curve (Primary Figure)
# =============================================================================
def plot_global_association(idata, df, knots, filename="fig1_global_association.pdf"):
    """
    Create publication-quality global association curve.
    Shows the population-GDP elasticity with 95% credible interval.
    """
    print(f"Creating: {filename}")

    post = idata.posterior

    # Extract posterior samples
    alpha_s = post["alpha"].stack(sample=("chain", "draw")).values
    beta_s = post["beta"].stack(sample=("chain", "draw")).values

    # Handle theta (RCS coefficients)
    theta_da = post["theta"]
    theta_dim = [d for d in theta_da.dims if d not in ("chain", "draw")][0]
    theta_s = (theta_da
               .stack(sample=("chain", "draw"))
               .transpose("sample", theta_dim)
               .values)

    # Prediction grid
    x_range = df["Log_Population_c"].values
    x_grid = np.linspace(x_range.min() - 0.2, x_range.max() + 0.2, 300)
    Z_grid = rcs_design(x_grid, knots)

    # Compute predicted log-GDP for each posterior sample
    y_pred = (alpha_s[:, None]
              + beta_s[:, None] * x_grid[None, :]
              + theta_s @ Z_grid.T)

    # Summary statistics
    y_median = np.median(y_pred, axis=0)
    y_lo = np.percentile(y_pred, 2.5, axis=0)
    y_hi = np.percentile(y_pred, 97.5, axis=0)
    y_lo_50 = np.percentile(y_pred, 25, axis=0)
    y_hi_50 = np.percentile(y_pred, 75, axis=0)

    # Convert back to original scale for axis labels
    x_grid_orig = x_grid + log_pop_mean
    x_obs_orig = df["Log_Population"].values

    # Create figure
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Color scheme (colorblind-friendly)
    color_data = '#666666'
    color_line = '#2166AC'  # Blue
    color_ci95 = '#92C5DE'  # Light blue
    color_ci50 = '#4393C3'  # Medium blue

    # Plot observed data
    ax.scatter(x_obs_orig, df["Log_GDP"].values,
               alpha=0.15, s=8, c=color_data,
               edgecolors='none', rasterized=True,
               label='Observed data')

    # Plot credible intervals (95% then 50% for layering)
    ax.fill_between(x_grid_orig, y_lo, y_hi,
                    alpha=0.3, color=color_ci95,
                    edgecolor='none',
                    label='95% CI')
    ax.fill_between(x_grid_orig, y_lo_50, y_hi_50,
                    alpha=0.4, color=color_ci50,
                    edgecolor='none',
                    label='50% CI')

    # Plot median line
    ax.plot(x_grid_orig, y_median,
            color=color_line, linewidth=2,
            label='Posterior median')

    # Axis labels with units
    ax.set_xlabel(r'Population (log$_{10}$ scale)')
    ax.set_ylabel(r'GDP (log$_{10}$ scale, current USD)')

    # Custom tick labels showing actual population values
    pop_ticks = [5, 6, 7, 8, 9, 10]  # log10(population)
    pop_labels = ['100K', '1M', '10M', '100M', '1B', '10B']
    ax.set_xticks(pop_ticks)
    ax.set_xticklabels(pop_labels)

    # GDP tick labels
    gdp_ticks = [7, 8, 9, 10, 11, 12, 13]
    gdp_labels = ['$10M', '$100M', '$1B', '$10B', '$100B', '$1T', '$10T']
    ax.set_yticks(gdp_ticks)
    ax.set_yticklabels(gdp_labels)

    # Legend
    ax.legend(loc='upper left', frameon=False)

    # Add elasticity annotation
    beta_mean = float(post["beta"].mean())
    beta_sd = float(post["beta"].std())
    ax.text(0.97, 0.05,
            f'Elasticity: {beta_mean:.2f} ({beta_mean-1.96*beta_sd:.2f}, {beta_mean+1.96*beta_sd:.2f})',
            transform=ax.transAxes,
            ha='right', va='bottom',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # Tight layout
    fig.tight_layout()

    # Save in multiple formats
    fig.savefig(OUT_FIG / filename, dpi=600)
    fig.savefig(OUT_FIG / filename.replace('.pdf', '.png'), dpi=600)
    print(f"  Saved: {OUT_FIG / filename}")

    plt.close(fig)
    return fig

# =============================================================================
# Figure 2: Regional Comparison (Supplementary)
# =============================================================================
def plot_regional_comparison(idata, df, filename="fig2_regional_effects.pdf"):
    """
    Forest plot showing regional intercept and slope effects.
    """
    if idata is None:
        print("Regional model not loaded, skipping regional comparison plot.")
        return None

    print(f"Creating: {filename}")

    post = idata.posterior

    # Check available variables
    available_vars = list(post.data_vars)

    # Extract regional parameters
    if "alpha_region" not in available_vars:
        print("  alpha_region not found, skipping.")
        return None

    alpha_r = post["alpha_region"]
    beta_r = post["beta_region"] if "beta_region" in available_vars else None

    regions = alpha_r.coords["Region"].values
    n_regions = len(regions)

    # Compute summaries
    alpha_mean = alpha_r.mean(dim=("chain", "draw")).values
    alpha_hdi = az.hdi(alpha_r, hdi_prob=0.95)["alpha_region"].values

    if beta_r is not None:
        beta_mean = beta_r.mean(dim=("chain", "draw")).values
        beta_hdi = az.hdi(beta_r, hdi_prob=0.95)["beta_region"].values
        n_cols = 2
    else:
        n_cols = 1

    # Create figure
    fig, axes = plt.subplots(1, n_cols, figsize=(3.5 * n_cols, 0.4 * n_regions + 1.5))
    if n_cols == 1:
        axes = [axes]

    # Sort by alpha mean
    order = np.argsort(alpha_mean)[::-1]

    y_pos = np.arange(n_regions)

    # Panel A: Intercepts
    ax = axes[0]
    ax.errorbar(alpha_mean[order], y_pos,
                xerr=[alpha_mean[order] - alpha_hdi[order, 0],
                      alpha_hdi[order, 1] - alpha_mean[order]],
                fmt='o', color='#2166AC', capsize=3, capthick=1, markersize=5)
    ax.axvline(x=alpha_mean.mean(), color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([regions[i] for i in order])
    ax.set_xlabel('Intercept (log$_{10}$ GDP)')
    ax.set_title('(A) Regional Intercepts', fontsize=11, fontweight='bold')
    ax.invert_yaxis()

    # Panel B: Slopes (if available)
    if beta_r is not None:
        ax = axes[1]
        ax.errorbar(beta_mean[order], y_pos,
                    xerr=[beta_mean[order] - beta_hdi[order, 0],
                          beta_hdi[order, 1] - beta_mean[order]],
                    fmt='o', color='#D6604D', capsize=3, capthick=1, markersize=5)
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([])
        ax.set_xlabel('Elasticity (slope)')
        ax.set_title('(B) Regional Elasticities', fontsize=11, fontweight='bold')
        ax.invert_yaxis()

    fig.tight_layout()

    # Save
    fig.savefig(OUT_FIG / filename, dpi=600)
    fig.savefig(OUT_FIG / filename.replace('.pdf', '.png'), dpi=600)
    print(f"  Saved: {OUT_FIG / filename}")

    plt.close(fig)
    return fig

# =============================================================================
# Figure 3: Demographic Effects (Working-Age Share & Old-Age Dependency)
# =============================================================================
def plot_demographic_effects(idata, filename="fig3_demographic_effects.pdf"):
    """
    Create coefficient plot for demographic effects.
    Shows the effect of working-age share and old-age dependency on GDP.
    """
    if idata is None:
        print("Hierarchical model not loaded, skipping demographic effects plot.")
        return None

    print(f"Creating: {filename}")

    post = idata.posterior
    available_vars = list(post.data_vars)

    # Check for demographic variables
    demo_vars = []
    demo_labels = []
    demo_descriptions = []

    if "delta_washare" in available_vars:
        demo_vars.append("delta_washare")
        demo_labels.append("Working-Age Share")
        demo_descriptions.append("Effect of 1 SD increase in\nworking-age population share")

    if "delta_olddep" in available_vars:
        demo_vars.append("delta_olddep")
        demo_labels.append("Old-Age Dependency")
        demo_descriptions.append("Effect of 1 SD increase in\nold-age dependency ratio")

    if not demo_vars:
        print("  No demographic variables found.")
        return None

    # Extract posterior samples
    n_vars = len(demo_vars)
    means = []
    hdis = []

    for var in demo_vars:
        samples = post[var].values.flatten()
        means.append(np.mean(samples))
        hdis.append([np.percentile(samples, 2.5), np.percentile(samples, 97.5)])

    means = np.array(means)
    hdis = np.array(hdis)

    # Create figure
    fig, ax = plt.subplots(figsize=(5, 2.5 + 0.5 * n_vars))

    y_pos = np.arange(n_vars)
    colors = ['#2166AC', '#B2182B']  # Blue for positive expected, Red for negative expected

    for i, (mean, hdi, label) in enumerate(zip(means, hdis, demo_labels)):
        # Color based on whether CI includes zero
        if hdi[0] > 0:
            color = '#2166AC'  # Significantly positive
            marker = 's'
        elif hdi[1] < 0:
            color = '#B2182B'  # Significantly negative
            marker = 's'
        else:
            color = '#666666'  # Not significant
            marker = 'o'

        ax.errorbar(mean, i, xerr=[[mean - hdi[0]], [hdi[1] - mean]],
                    fmt=marker, color=color, capsize=4, capthick=1.5,
                    markersize=8, markeredgecolor='white', markeredgewidth=0.5)

    # Reference line at zero
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(demo_labels)
    ax.set_xlabel('Effect on log$_{10}$(GDP)')
    ax.set_title('Demographic Effects on GDP', fontsize=11, fontweight='bold')

    # Add annotation box
    textstr = '\n'.join([
        'Standardized coefficients:',
        '  Square = significant (95% CI)',
        '  Circle = not significant'
    ])
    props = dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', alpha=0.8, edgecolor='none')
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    # Invert y-axis for top-to-bottom reading
    ax.invert_yaxis()

    fig.tight_layout()

    # Save
    fig.savefig(OUT_FIG / filename, dpi=600)
    fig.savefig(OUT_FIG / filename.replace('.pdf', '.png'), dpi=600)
    print(f"  Saved: {OUT_FIG / filename}")

    plt.close(fig)
    return fig

# =============================================================================
# Figure 4: Regional Time Trends
# =============================================================================
def plot_time_trends(idata, filename="fig4_time_trends.pdf"):
    """
    Create plot showing regional time drift effects.
    Shows how GDP growth rates vary by region over time.
    """
    if idata is None:
        print("Hierarchical model not loaded, skipping time trends plot.")
        return None

    print(f"Creating: {filename}")

    post = idata.posterior
    available_vars = list(post.data_vars)

    # Check for time trend variables
    if "tau_region" not in available_vars:
        print("  tau_region not found, skipping.")
        return None

    tau_r = post["tau_region"]
    regions = tau_r.coords["Region"].values
    n_regions = len(regions)

    # Compute summaries
    tau_mean = tau_r.mean(dim=("chain", "draw")).values
    tau_hdi = az.hdi(tau_r, hdi_prob=0.95)["tau_region"].values

    # Also get global tau if available
    if "tau0" in available_vars:
        tau0_mean = float(post["tau0"].mean())
        tau0_samples = post["tau0"].values.flatten()
        tau0_hdi = [np.percentile(tau0_samples, 2.5), np.percentile(tau0_samples, 97.5)]
    else:
        tau0_mean = tau_mean.mean()
        tau0_hdi = None

    # Sort by tau mean
    order = np.argsort(tau_mean)[::-1]

    # Create figure
    fig, ax = plt.subplots(figsize=(5.5, 0.4 * n_regions + 1.5))

    y_pos = np.arange(n_regions)

    # Color scheme: diverging from global mean
    for i, idx in enumerate(order):
        if tau_hdi[idx, 0] > 0:
            color = '#2166AC'  # Significantly positive (accelerating)
        elif tau_hdi[idx, 1] < 0:
            color = '#B2182B'  # Significantly negative (decelerating)
        else:
            color = '#666666'  # Not significant

        ax.errorbar(tau_mean[idx], i,
                    xerr=[[tau_mean[idx] - tau_hdi[idx, 0]],
                          [tau_hdi[idx, 1] - tau_mean[idx]]],
                    fmt='o', color=color, capsize=3, capthick=1, markersize=6)

    # Reference lines
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.axvline(x=tau0_mean, color='#4DAF4A', linestyle='--', linewidth=1.2,
               alpha=0.8, label=f'Global mean: {tau0_mean:.3f}')

    # Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([regions[i] for i in order])
    ax.set_xlabel('Time trend coefficient (per decade)')
    ax.set_title('Regional GDP Growth Trends', fontsize=11, fontweight='bold')

    # Legend
    ax.legend(loc='lower right', fontsize=8)

    # Add interpretation note
    ax.text(0.02, 0.02,
            'Positive = faster GDP growth\nNegative = slower GDP growth',
            transform=ax.transAxes, fontsize=8, va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f0f0f0', alpha=0.8, edgecolor='none'))

    ax.invert_yaxis()

    fig.tight_layout()

    # Save
    fig.savefig(OUT_FIG / filename, dpi=600)
    fig.savefig(OUT_FIG / filename.replace('.pdf', '.png'), dpi=600)
    print(f"  Saved: {OUT_FIG / filename}")

    plt.close(fig)
    return fig

# =============================================================================
# Figure 5: Combined Hierarchical Structure (Overview)
# =============================================================================
def plot_hierarchical_overview(idata, filename="fig5_hierarchical_overview.pdf"):
    """
    Create a multi-panel figure showing the hierarchical model structure.
    Panel A: Global elasticity
    Panel B: Regional variation
    Panel C: Country-level shrinkage example
    """
    if idata is None:
        print("Hierarchical model not loaded, skipping overview plot.")
        return None

    print(f"Creating: {filename}")

    post = idata.posterior
    available_vars = list(post.data_vars)

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))

    # Panel A: Global elasticity (beta0) posterior
    ax = axes[0, 0]
    if "beta0" in available_vars:
        beta0 = post["beta0"].values.flatten()
        ax.hist(beta0, bins=50, density=True, alpha=0.7, color='#2166AC', edgecolor='white')
        ax.axvline(x=np.mean(beta0), color='#B2182B', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(beta0):.3f}')
        ax.axvline(x=1.0, color='gray', linestyle='--', linewidth=1.5,
                   label='Reference: 1.0')
        ax.set_xlabel('Elasticity')
        ax.set_ylabel('Density')
        ax.set_title('(A) Global Elasticity', fontweight='bold')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'beta0 not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(A) Global Elasticity', fontweight='bold')

    # Panel B: Regional intercepts
    ax = axes[0, 1]
    if "alpha_region" in available_vars:
        alpha_r = post["alpha_region"]
        regions = alpha_r.coords["Region"].values
        alpha_mean = alpha_r.mean(dim=("chain", "draw")).values
        alpha_std = alpha_r.std(dim=("chain", "draw")).values

        order = np.argsort(alpha_mean)
        y_pos = np.arange(len(regions))

        ax.barh(y_pos, alpha_mean[order], xerr=alpha_std[order],
                color='#4393C3', alpha=0.8, capsize=2)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([regions[i][:15] for i in order], fontsize=7)
        ax.set_xlabel('Intercept')
        ax.set_title('(B) Regional Intercepts', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'alpha_region not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(B) Regional Intercepts', fontweight='bold')

    # Panel C: Variance components
    ax = axes[1, 0]
    variance_vars = ["sigma_alpha_region", "sigma_alpha_by_region", "sigma"]
    variance_labels = ["Between-Region", "Within-Region\n(Country)", "Residual"]
    var_means = []
    var_hdis = []

    for v in variance_vars:
        if v in available_vars:
            samples = post[v].values.flatten()
            var_means.append(np.mean(samples))
            var_hdis.append([np.percentile(samples, 2.5), np.percentile(samples, 97.5)])
        else:
            var_means.append(np.nan)
            var_hdis.append([np.nan, np.nan])

    var_means = np.array(var_means)
    var_hdis = np.array(var_hdis)

    valid_idx = ~np.isnan(var_means)
    if valid_idx.any():
        colors = ['#66C2A5', '#FC8D62', '#8DA0CB']
        y_pos = np.arange(sum(valid_idx))
        valid_means = var_means[valid_idx]
        valid_hdis = var_hdis[valid_idx]
        valid_labels = [l for l, v in zip(variance_labels, valid_idx) if v]
        valid_colors = [c for c, v in zip(colors, valid_idx) if v]

        for i, (m, hdi, label, color) in enumerate(zip(valid_means, valid_hdis, valid_labels, valid_colors)):
            ax.barh(i, m, color=color, alpha=0.8)
            ax.errorbar(m, i, xerr=[[m - hdi[0]], [hdi[1] - m]],
                       fmt='none', color='black', capsize=3)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(valid_labels)
        ax.set_xlabel('Standard Deviation')
        ax.set_title('(C) Variance Components', fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'Variance components\nnot available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(C) Variance Components', fontweight='bold')

    # Panel D: R-squared
    ax = axes[1, 1]
    if "R2" in available_vars:
        r2 = post["R2"].values.flatten()
        ax.hist(r2, bins=50, density=True, alpha=0.7, color='#7570B3', edgecolor='white')
        ax.axvline(x=np.mean(r2), color='#E7298A', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(r2):.3f}')
        ax.axvline(x=np.percentile(r2, 2.5), color='gray', linestyle=':', linewidth=1)
        ax.axvline(x=np.percentile(r2, 97.5), color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('R²')
        ax.set_ylabel('Density')
        ax.set_title('(D) Model Fit (R²)', fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
    else:
        ax.text(0.5, 0.5, 'R² not available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('(D) Model Fit (R²)', fontweight='bold')

    fig.tight_layout()

    # Save
    fig.savefig(OUT_FIG / filename, dpi=600)
    fig.savefig(OUT_FIG / filename.replace('.pdf', '.png'), dpi=600)
    print(f"  Saved: {OUT_FIG / filename}")

    plt.close(fig)
    return fig

# =============================================================================
# Table 1: Posterior Summary (Main Parameters)
# =============================================================================
def create_posterior_table(idata, model_name, filename="table1_posterior_summary.csv"):
    """
    Create publication-ready posterior summary table.
    """
    print(f"Creating: {filename}")

    # Key parameters to include
    key_vars = ["alpha", "beta", "sigma", "R2", "theta"]

    # Filter to existing variables
    available = list(idata.posterior.data_vars)
    vars_to_summarize = [v for v in key_vars if v in available]

    if not vars_to_summarize:
        print("  No key variables found.")
        return None

    # Create summary
    summary = az.summary(idata, var_names=vars_to_summarize, hdi_prob=0.95)

    # Format for publication
    summary_pub = summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "r_hat"]].copy()
    summary_pub.columns = ["Mean", "SD", "2.5%", "97.5%", "ESS", "R-hat"]

    # Round appropriately
    for col in ["Mean", "SD", "2.5%", "97.5%"]:
        summary_pub[col] = summary_pub[col].round(3)
    summary_pub["ESS"] = summary_pub["ESS"].round(0).astype(int)
    summary_pub["R-hat"] = summary_pub["R-hat"].round(3)

    # Save CSV
    summary_pub.to_csv(OUT_TABLE / filename)
    print(f"  Saved: {OUT_TABLE / filename}")

    # Also create LaTeX version
    latex_file = filename.replace('.csv', '.tex')
    latex_str = summary_pub.to_latex(
        caption=f"Posterior summary for {model_name}",
        label=f"tab:{model_name.lower().replace(' ', '_')}",
        column_format="l" + "r" * len(summary_pub.columns),
        escape=False
    )
    (OUT_TABLE / latex_file).write_text(latex_str)
    print(f"  Saved: {OUT_TABLE / latex_file}")

    return summary_pub

# =============================================================================
# Table 2: Model Comparison
# =============================================================================
def create_model_comparison_table(models_dict, filename="table2_model_comparison.csv"):
    """
    Create model comparison table with R2 and fit statistics.
    """
    print(f"Creating: {filename}")

    rows = []
    for name, idata in models_dict.items():
        if idata is None:
            continue

        row = {"Model": name}

        # R2
        if "R2" in idata.posterior.data_vars:
            r2 = idata.posterior["R2"]
            row["R2_mean"] = float(r2.mean())
            row["R2_sd"] = float(r2.std())

        # Number of parameters (approximate)
        n_params = sum(np.prod(v.shape[2:]) for v in idata.posterior.data_vars.values())
        row["n_params"] = n_params

        rows.append(row)

    if not rows:
        print("  No models to compare.")
        return None

    comparison_df = pd.DataFrame(rows)

    # Format
    if "R2_mean" in comparison_df.columns:
        comparison_df["R2"] = comparison_df.apply(
            lambda r: f"{r['R2_mean']:.3f} ({r['R2_sd']:.3f})" if pd.notna(r.get('R2_mean')) else "-",
            axis=1
        )
        comparison_df = comparison_df.drop(columns=["R2_mean", "R2_sd"])

    # Save
    comparison_df.to_csv(OUT_TABLE / filename, index=False)
    print(f"  Saved: {OUT_TABLE / filename}")

    # LaTeX version
    latex_file = filename.replace('.csv', '.tex')
    latex_str = comparison_df.to_latex(
        index=False,
        caption="Model comparison across hierarchical layers",
        label="tab:model_comparison",
        column_format="l" + "c" * (len(comparison_df.columns) - 1)
    )
    (OUT_TABLE / latex_file).write_text(latex_str)
    print(f"  Saved: {OUT_TABLE / latex_file}")

    return comparison_df

# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Generating Publication Figures and Tables")
    print("="*60 + "\n")

    # Figure 1: Global Association Curve (Primary)
    plot_global_association(idata_simple, df, knots,
                           filename="fig1_global_association.pdf")

    # Figure 2: Regional Comparison (if available)
    if idata_regional is not None:
        plot_regional_comparison(idata_regional, df,
                                filename="fig2_regional_effects.pdf")

    # Figure 3: Demographic Effects (from hierarchical model)
    if idata_hier is not None:
        plot_demographic_effects(idata_hier,
                                filename="fig3_demographic_effects.pdf")

    # Figure 4: Time Trends (from hierarchical model)
    if idata_hier is not None:
        plot_time_trends(idata_hier,
                        filename="fig4_time_trends.pdf")

    # Figure 5: Hierarchical Overview (from hierarchical model)
    if idata_hier is not None:
        plot_hierarchical_overview(idata_hier,
                                  filename="fig5_hierarchical_overview.pdf")

    # Table 1: Posterior Summary
    create_posterior_table(idata_simple, "Simple RCS Model",
                          filename="table1_posterior_simple.csv")

    if idata_regional is not None:
        create_posterior_table(idata_regional, "Regional RCS Model",
                              filename="table1_posterior_regional.csv")

    if idata_hier is not None:
        create_posterior_table(idata_hier, "Hierarchical RCS Model",
                              filename="table1_posterior_hierarchical.csv")

    # Table 2: Model Comparison
    models = {
        "Layer 1: Simple (Pooled)": idata_simple,
        "Layer 2: Regional (Partial Pooling)": idata_regional,
        "Layer 3: Hierarchical (Full)": idata_hier,
    }
    create_model_comparison_table(models, filename="table2_model_comparison.csv")

    print("\n" + "="*60)
    print("Done! Output files:")
    print(f"  Figures: {OUT_FIG}")
    print(f"  Tables:  {OUT_TABLE}")
    print("="*60)
