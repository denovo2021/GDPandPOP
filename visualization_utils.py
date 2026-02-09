"""
visualization_utils.py - Publication-quality figure utilities

This module provides helper functions and settings for creating figures
suitable for submission to Tier 1 journals (e.g., The Lancet, Nature Medicine).

Standards enforced:
- High resolution (300-600 DPI)
- Professional fonts (Arial/Helvetica)
- Appropriate dimensions for journal columns
- Multiple output formats (PNG, PDF, TIFF)

Usage:
    from visualization_utils import setup_style, save_publication_figure

    # At the start of your script
    setup_style()

    # Create your figure
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ...

    # Save in publication formats
    save_publication_figure(fig, "figure1", formats=['png', 'pdf', 'tiff'])
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from typing import Tuple, List, Optional, Union

# =============================================================================
# Publication Standards Configuration
# =============================================================================

# Resolution settings
DPI_SCREEN = 150      # For interactive viewing
DPI_PRINT = 300       # Minimum for publication (most journals)
DPI_HIGHRES = 600     # High-resolution (Nature, Science)

# Figure dimensions (in inches)
# Based on standard journal column widths:
#   - Single column: ~85mm (3.35 in)
#   - 1.5 column: ~114mm (4.5 in)
#   - Double column: ~170mm (6.7 in)
#   - Full page: ~180mm (7.1 in)

FIGSIZE_SINGLE = (3.35, 2.5)      # Single column, square-ish
FIGSIZE_SINGLE_TALL = (3.35, 4.0) # Single column, tall
FIGSIZE_DOUBLE = (6.7, 4.0)       # Double column
FIGSIZE_DOUBLE_WIDE = (6.7, 3.0)  # Double column, wide
FIGSIZE_FULL = (7.1, 5.0)         # Full page width

# Font settings
FONT_FAMILY = 'sans-serif'
FONT_SANS_SERIF = ['Arial', 'Helvetica', 'DejaVu Sans', 'Liberation Sans']
FONT_SIZE_TINY = 6
FONT_SIZE_SMALL = 8
FONT_SIZE_NORMAL = 10
FONT_SIZE_LARGE = 12
FONT_SIZE_TITLE = 14

# Color palettes suitable for colorblind readers
# Based on Wong (2011) Nature Methods colorblind-safe palette
COLORS_WONG = {
    'blue': '#0072B2',
    'orange': '#E69F00',
    'green': '#009E73',
    'yellow': '#F0E442',
    'sky_blue': '#56B4E9',
    'vermillion': '#D55E00',
    'purple': '#CC79A7',
    'black': '#000000',
}

# Sequential palette for fan charts
COLORS_FAN = {
    'median': '#2166AC',      # Dark blue
    'ci_50': '#4393C3',       # Medium blue
    'ci_80': '#92C5DE',       # Light blue
    'ci_95': '#D1E5F0',       # Very light blue
    'historical': '#333333',   # Dark gray
}

# =============================================================================
# Style Setup Functions
# =============================================================================

def setup_style(context: str = 'paper'):
    """
    Configure matplotlib for publication-quality figures.

    Parameters
    ----------
    context : str
        'paper' for print/PDF, 'talk' for presentations, 'poster' for posters
    """
    # Base font sizes by context
    font_scales = {
        'paper': 1.0,
        'talk': 1.3,
        'poster': 1.5,
    }
    scale = font_scales.get(context, 1.0)

    rc_params = {
        # Font settings
        'font.family': FONT_FAMILY,
        'font.sans-serif': FONT_SANS_SERIF,
        'font.size': FONT_SIZE_NORMAL * scale,

        # Axes
        'axes.titlesize': FONT_SIZE_LARGE * scale,
        'axes.labelsize': FONT_SIZE_NORMAL * scale,
        'axes.linewidth': 0.8,
        'axes.spines.top': False,
        'axes.spines.right': False,

        # Ticks
        'xtick.labelsize': FONT_SIZE_SMALL * scale,
        'ytick.labelsize': FONT_SIZE_SMALL * scale,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.major.size': 4,
        'ytick.major.size': 4,

        # Legend
        'legend.fontsize': FONT_SIZE_SMALL * scale,
        'legend.frameon': False,

        # Figure
        'figure.titlesize': FONT_SIZE_TITLE * scale,
        'figure.dpi': DPI_SCREEN,
        'figure.facecolor': 'white',
        'figure.edgecolor': 'white',

        # Saving
        'savefig.dpi': DPI_PRINT,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'savefig.facecolor': 'white',
        'savefig.edgecolor': 'white',

        # PDF/PS settings (for vector output)
        'pdf.fonttype': 42,  # TrueType fonts
        'ps.fonttype': 42,

        # Lines
        'lines.linewidth': 1.5,
        'lines.markersize': 5,

        # Grid (off by default for clean look)
        'axes.grid': False,
    }

    plt.rcParams.update(rc_params)

    # Use seaborn-inspired color cycle
    plt.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
        COLORS_WONG['blue'],
        COLORS_WONG['orange'],
        COLORS_WONG['green'],
        COLORS_WONG['vermillion'],
        COLORS_WONG['purple'],
        COLORS_WONG['sky_blue'],
    ])


def reset_style():
    """Reset matplotlib to default settings."""
    mpl.rcdefaults()


# =============================================================================
# Figure Creation Helpers
# =============================================================================

def create_figure(
    figsize: Tuple[float, float] = FIGSIZE_DOUBLE,
    nrows: int = 1,
    ncols: int = 1,
    **kwargs
) -> Tuple[plt.Figure, Union[plt.Axes, List[plt.Axes]]]:
    """
    Create a publication-ready figure with subplots.

    Parameters
    ----------
    figsize : tuple
        Figure size in inches (width, height)
    nrows, ncols : int
        Number of subplot rows and columns
    **kwargs : dict
        Additional arguments passed to plt.subplots()

    Returns
    -------
    fig, ax : Figure and Axes objects
    """
    fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, ax


def add_panel_label(
    ax: plt.Axes,
    label: str,
    loc: str = 'upper left',
    fontsize: int = FONT_SIZE_LARGE,
    fontweight: str = 'bold'
):
    """
    Add a panel label (A, B, C, etc.) to a subplot.

    Parameters
    ----------
    ax : plt.Axes
        The axes to label
    label : str
        The label text (e.g., 'A', 'B', 'a)', '(i)')
    loc : str
        Location: 'upper left', 'upper right', etc.
    fontsize : int
        Font size for the label
    fontweight : str
        Font weight ('normal', 'bold')
    """
    loc_coords = {
        'upper left': (-0.15, 1.05),
        'upper right': (1.05, 1.05),
        'lower left': (-0.15, -0.15),
        'lower right': (1.05, -0.15),
    }

    x, y = loc_coords.get(loc, (-0.15, 1.05))

    ax.text(
        x, y, label,
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight=fontweight,
        va='top',
        ha='left'
    )


# =============================================================================
# Saving Functions
# =============================================================================

def save_publication_figure(
    fig: plt.Figure,
    name: str,
    output_dir: Optional[Path] = None,
    formats: List[str] = ['png', 'pdf'],
    dpi: Optional[int] = None,
    close: bool = True
):
    """
    Save figure in publication-quality formats.

    Parameters
    ----------
    fig : plt.Figure
        The figure to save
    name : str
        Base filename (without extension)
    output_dir : Path, optional
        Output directory. If None, uses current directory.
    formats : list
        List of formats to save: 'png', 'pdf', 'tiff', 'svg', 'eps'
    dpi : int, optional
        Resolution for raster formats. If None:
        - 300 for png
        - 600 for tiff (required by some journals)
    close : bool
        Whether to close the figure after saving

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3])
    >>> save_publication_figure(fig, "figure1", formats=['png', 'pdf', 'tiff'])
    """
    if output_dir is None:
        output_dir = Path.cwd()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default DPI settings by format
    default_dpi = {
        'png': 300,
        'tiff': 600,
        'jpg': 300,
        'jpeg': 300,
    }

    for fmt in formats:
        filepath = output_dir / f"{name}.{fmt}"

        # Determine appropriate DPI
        if fmt in ('pdf', 'svg', 'eps'):
            # Vector formats - DPI doesn't matter
            fig.savefig(filepath, format=fmt, bbox_inches='tight')
        else:
            # Raster formats - use appropriate DPI
            save_dpi = dpi or default_dpi.get(fmt, DPI_PRINT)
            fig.savefig(filepath, format=fmt, dpi=save_dpi, bbox_inches='tight')

        print(f"[OK] Saved: {filepath}")

    if close:
        plt.close(fig)


# =============================================================================
# Fan Chart Helpers
# =============================================================================

def plot_fan_chart(
    ax: plt.Axes,
    x: list,
    median: list,
    ci_50_lower: list,
    ci_50_upper: list,
    ci_80_lower: list,
    ci_80_upper: list,
    ci_95_lower: list,
    ci_95_upper: list,
    historical_x: Optional[list] = None,
    historical_y: Optional[list] = None,
    colors: Optional[dict] = None,
    label_prefix: str = ''
):
    """
    Plot a fan chart with confidence intervals.

    Parameters
    ----------
    ax : plt.Axes
        The axes to plot on
    x : array-like
        X values (e.g., years)
    median : array-like
        Median projection
    ci_*_lower/upper : array-like
        Confidence interval bounds (50%, 80%, 95%)
    historical_x, historical_y : array-like, optional
        Historical data points
    colors : dict, optional
        Color dictionary (uses COLORS_FAN if None)
    label_prefix : str
        Prefix for legend labels
    """
    if colors is None:
        colors = COLORS_FAN

    # Plot confidence bands (widest first)
    ax.fill_between(x, ci_95_lower, ci_95_upper,
                    alpha=0.3, color=colors['ci_95'], label=f'{label_prefix}95% CI')
    ax.fill_between(x, ci_80_lower, ci_80_upper,
                    alpha=0.4, color=colors['ci_80'], label=f'{label_prefix}80% CI')
    ax.fill_between(x, ci_50_lower, ci_50_upper,
                    alpha=0.5, color=colors['ci_50'], label=f'{label_prefix}50% CI')

    # Plot median
    ax.plot(x, median, color=colors['median'], linewidth=2, label=f'{label_prefix}Median')

    # Plot historical data if provided
    if historical_x is not None and historical_y is not None:
        ax.plot(historical_x, historical_y, color=colors['historical'],
                linewidth=1.5, linestyle='-', label='Historical')


# =============================================================================
# Convenience exports
# =============================================================================

__all__ = [
    # Constants
    'DPI_SCREEN', 'DPI_PRINT', 'DPI_HIGHRES',
    'FIGSIZE_SINGLE', 'FIGSIZE_SINGLE_TALL', 'FIGSIZE_DOUBLE',
    'FIGSIZE_DOUBLE_WIDE', 'FIGSIZE_FULL',
    'FONT_SIZE_TINY', 'FONT_SIZE_SMALL', 'FONT_SIZE_NORMAL',
    'FONT_SIZE_LARGE', 'FONT_SIZE_TITLE',
    'COLORS_WONG', 'COLORS_FAN',

    # Functions
    'setup_style', 'reset_style',
    'create_figure', 'add_panel_label',
    'save_publication_figure',
    'plot_fan_chart',
]
