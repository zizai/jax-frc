"""Plotting utilities for validation comparisons."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence


def plot_comparison(
    x: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
    labels: Sequence[str] = ('Simulation', 'Expected'),
    title: str = 'Comparison',
    xlabel: str = 'x',
    ylabel: str = 'y',
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create 1D comparison plot of actual vs expected values.

    Generates an overlay plot comparing simulation results against expected
    (e.g., analytic) values. The actual values are shown as a solid blue line
    and expected values as a dashed red line.

    Args:
        x: Independent variable array (e.g., spatial coordinates or time).
        actual: Simulation or computed values to compare.
        expected: Reference or analytic values for comparison.
        labels: Two-element sequence of legend labels for actual and expected
            curves. Defaults to ('Simulation', 'Expected').
        title: Plot title. Defaults to 'Comparison'.
        xlabel: Label for x-axis. Defaults to 'x'.
        ylabel: Label for y-axis. Defaults to 'y'.
        figsize: Figure dimensions as (width, height) in inches.
            Defaults to (10, 6).

    Returns:
        matplotlib Figure object containing the comparison plot with legend
        and grid.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, actual, 'b-', linewidth=2, label=labels[0])
    ax.plot(x, expected, 'r--', linewidth=2, label=labels[1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_error(
    x: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
    title: str = 'Error',
    xlabel: str = 'x',
    ylabel: str = 'Error',
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Plot the difference between actual and expected values.

    Generates an error plot showing the pointwise difference (actual - expected)
    along with a zero reference line for visual comparison. Useful for
    identifying systematic errors or spatial patterns in simulation accuracy.

    Args:
        x: Independent variable array (e.g., spatial coordinates or time).
        actual: Simulation or computed values.
        expected: Reference or analytic values.
        title: Plot title. Defaults to 'Error'.
        xlabel: Label for x-axis. Defaults to 'x'.
        ylabel: Label for y-axis. Defaults to 'Error'.
        figsize: Figure dimensions as (width, height) in inches.
            Defaults to (10, 4).

    Returns:
        matplotlib Figure object containing the error plot with a dashed
        zero reference line and grid.
    """
    fig, ax = plt.subplots(figsize=figsize)

    error = actual - expected
    ax.plot(x, error, 'k-', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
