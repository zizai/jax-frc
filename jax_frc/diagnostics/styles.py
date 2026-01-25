"""Consistent styling for JAX-FRC plots.

This module provides seaborn-based styling and field-specific
colormaps and labels for consistent, publication-quality plots.
"""

import seaborn as sns
import matplotlib.pyplot as plt


# Field-specific colormaps
FIELD_COLORS = {
    'psi': 'RdBu',      # Diverging for flux (positive/negative)
    'B': 'viridis',     # Sequential for magnitude
    'n': 'plasma',      # Sequential for density
    'p': 'inferno',     # Sequential for pressure
}

# Consistent labels with units
LABELS = {
    'psi': r'$\psi$ [Wb]',
    'B': r'$|B|$ [T]',
    'n': r'$n$ [m$^{-3}$]',
    'p': r'$p$ [Pa]',
    't': r'$t$ [s]',
    'r': r'$r$ [m]',
    'z': r'$z$ [m]',
    'energy': r'Energy [J]',
    'v_r': r'$v_r$ [m/s]',
    'v_z': r'$v_z$ [m/s]',
}


def apply_style():
    """Apply consistent seaborn style to all plots.

    Call this once at module import or before creating figures.
    Uses whitegrid style for clean scientific plots.
    """
    sns.set_theme(
        style="whitegrid",
        palette="deep",
        font_scale=1.1,
        rc={"figure.figsize": (10, 6)}
    )


def get_cmap(field_name: str) -> str:
    """Get colormap for a field, with fallback to viridis."""
    return FIELD_COLORS.get(field_name, 'viridis')


def get_label(field_name: str) -> str:
    """Get label for a field, with fallback to field name."""
    return LABELS.get(field_name, field_name)
