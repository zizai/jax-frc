"""Plot generation functions for validation reports."""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def create_scalar_comparison_plot(metrics: dict, resolution: int) -> Figure:
    """Create grouped bar chart comparing JAX vs AGATE scalar values.

    Args:
        metrics: Dict of metric name -> {jax_value, agate_value, ...}
        resolution: Grid resolution for title

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    names = list(metrics.keys())
    jax_vals = [m['jax_value'] for m in metrics.values()]
    agate_vals = [m['agate_value'] for m in metrics.values()]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, jax_vals, width, label='JAX', color='steelblue')
    ax.bar(x + width/2, agate_vals, width, label='AGATE', color='coral')

    ax.set_xlabel('Metric')
    ax.set_ylabel('Value')
    ax.set_title(f'Scalar Metrics Comparison (Resolution {resolution})')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    fig.tight_layout()
    return fig
