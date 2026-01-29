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


def create_error_threshold_plot(
    field_errors: dict,
    scalar_metrics: dict,
    l2_tol: float,
    rel_tol: float
) -> Figure:
    """Create horizontal bar chart showing errors relative to thresholds.

    Args:
        field_errors: Dict of field name -> L2 error value
        scalar_metrics: Dict of metric name -> {relative_error, threshold, passed}
        l2_tol: L2 error threshold for field comparisons
        rel_tol: Relative error threshold (unused, thresholds come from scalar_metrics)

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Combine all errors with their thresholds
    items = []
    for name, error in field_errors.items():
        items.append((name, error, l2_tol, error <= l2_tol))
    for name, data in scalar_metrics.items():
        items.append((name, data['relative_error'], data['threshold'], data['passed']))

    names = [i[0] for i in items]
    errors = [i[1] for i in items]
    thresholds = [i[2] for i in items]
    passed = [i[3] for i in items]

    # Normalize errors to percentage of threshold
    normalized = [e/t * 100 if t > 0 else 0 for e, t in zip(errors, thresholds)]
    colors = ['green' if p else 'red' for p in passed]

    y_pos = np.arange(len(names))
    ax.barh(y_pos, normalized, color=colors, alpha=0.7)
    ax.axvline(x=100, color='black', linestyle='--', linewidth=2, label='Threshold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel('Error (% of threshold)')
    ax.set_title('Error vs Threshold Summary')
    ax.legend()
    fig.tight_layout()
    return fig
