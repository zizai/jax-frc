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


def create_field_comparison_plot(
    jax_field: np.ndarray,
    agate_field: np.ndarray,
    field_name: str,
    resolution: int,
    l2_error_value: float
) -> Figure:
    """Create side-by-side contour plots: JAX, AGATE, and Difference.

    Args:
        jax_field: 2D array of JAX field values
        agate_field: 2D array of AGATE field values
        field_name: Name of the field for title
        resolution: Grid resolution for title
        l2_error_value: Pre-computed L2 error for annotation

    Returns:
        matplotlib Figure object
    """
    # Ensure numpy arrays
    jax_field = np.asarray(jax_field)
    agate_field = np.asarray(agate_field)

    # Compute difference
    diff_field = jax_field - agate_field

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Common colorbar range for JAX and AGATE
    vmin = min(float(jax_field.min()), float(agate_field.min()))
    vmax = max(float(jax_field.max()), float(agate_field.max()))

    # JAX field
    im1 = axes[0].imshow(jax_field.T, origin='lower', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    axes[0].set_title(f'JAX {field_name}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('z')
    plt.colorbar(im1, ax=axes[0])

    # AGATE field
    im2 = axes[1].imshow(agate_field.T, origin='lower', cmap='viridis',
                          vmin=vmin, vmax=vmax)
    axes[1].set_title(f'AGATE {field_name}')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('z')
    plt.colorbar(im2, ax=axes[1])

    # Difference (symmetric colormap)
    diff_abs_max = max(abs(float(diff_field.min())), abs(float(diff_field.max())))
    if diff_abs_max == 0:
        diff_abs_max = 1e-10  # Avoid zero range
    im3 = axes[2].imshow(diff_field.T, origin='lower', cmap='RdBu_r',
                          vmin=-diff_abs_max, vmax=diff_abs_max)
    axes[2].set_title('Difference (JAX - AGATE)')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('z')
    plt.colorbar(im3, ax=axes[2])

    # Add L2 error annotation
    fig.suptitle(f'{field_name.replace("_", " ").title()} Comparison '
                 f'(Resolution {resolution}) - L2 Error: {l2_error_value:.4g}')

    fig.tight_layout()
    return fig


def create_timeseries_comparison_plot(
    jax_times: np.ndarray,
    jax_values: np.ndarray,
    agate_times: np.ndarray,
    agate_values: np.ndarray,
    metric_name: str,
    resolution: list[int]
) -> Figure:
    """Create time-series comparison plot for an aggregate metric.

    Args:
        jax_times: JAX simulation times
        jax_values: JAX metric values
        agate_times: AGATE reference times
        agate_values: AGATE metric values
        metric_name: Name of the metric
        resolution: Grid resolution as [nx, ny, nz]

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: JAX vs AGATE
    ax1.plot(jax_times, jax_values, 'b-', label='JAX-FRC', linewidth=2)
    ax1.plot(agate_times, agate_values, 'r--', label='AGATE', linewidth=2)
    ax1.set_ylabel(metric_name)
    ax1.legend()
    ax1.set_title(f"{metric_name} Evolution (Resolution {resolution[0]})")
    ax1.grid(True, alpha=0.3)

    # Bottom: Residual
    jax_interp = np.interp(agate_times, jax_times, jax_values)
    residuals = jax_interp - agate_values
    ax2.plot(agate_times, residuals, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Residual (JAX - AGATE)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def create_field_error_evolution_plot(
    snapshot_errors: list[dict],
    threshold: float,
    resolution: list[int]
) -> Figure:
    """Create plot of per-field L2 error vs snapshot time.

    Args:
        snapshot_errors: List of {time, errors} dicts from validate_all_snapshots
        threshold: Error threshold
        resolution: Grid resolution as [nx, ny, nz]

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times = [s["time"] for s in snapshot_errors]
    fields = list(snapshot_errors[0]["errors"].keys())

    colors = ['b', 'g', 'r', 'purple']
    for field, color in zip(fields, colors):
        errors = [s["errors"][field]["l2_error"] for s in snapshot_errors]
        ax.plot(times, errors, '-o', label=field, markersize=3, color=color)

    ax.axhline(y=threshold, color='k', linestyle='--', linewidth=2,
               label=f'Threshold ({threshold})')
    ax.set_xlabel('Time')
    ax.set_ylabel('L2 Error')
    ax.set_title(f'Per-Field L2 Error Evolution (Resolution {resolution[0]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
