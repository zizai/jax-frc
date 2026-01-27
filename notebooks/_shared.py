"""Shared utilities for validation notebooks.

Provides reusable plotting, animation, and interactive widget helpers.
"""

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize
import numpy as np
from IPython.display import HTML
import ipywidgets as widgets


# Consistent style for all plots
NUMERICAL_STYLE = {'color': '#1f77b4', 'linewidth': 2, 'label': 'Numerical'}
ANALYTIC_STYLE = {'color': '#ff7f0e', 'linewidth': 2, 'linestyle': '--', 'label': 'Analytic'}
INITIAL_STYLE = {'color': '#7f7f7f', 'linewidth': 1, 'linestyle': ':', 'label': 'Initial'}


def plot_comparison(x, numerical, analytic, xlabel='x', ylabel='y', title=None,
                    initial=None, figsize=(10, 6)):
    """Create overlay plot comparing numerical and analytic solutions.

    Parameters
    ----------
    x : array-like
        Coordinate values for x-axis
    numerical : array-like
        Numerical solution values
    analytic : array-like
        Analytic solution values
    xlabel, ylabel : str
        Axis labels
    title : str, optional
        Plot title
    initial : array-like, optional
        Initial condition to show as reference
    figsize : tuple
        Figure size in inches

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=figsize)

    if initial is not None:
        ax.plot(x, initial, **INITIAL_STYLE)

    ax.plot(x, numerical, **NUMERICAL_STYLE)
    ax.plot(x, analytic, **ANALYTIC_STYLE)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_error(x, numerical, analytic, xlabel='x', title='Error Distribution',
               figsize=(10, 4)):
    """Plot absolute and relative error between numerical and analytic solutions.

    Parameters
    ----------
    x : array-like
        Coordinate values
    numerical, analytic : array-like
        Solutions to compare

    Returns
    -------
    fig, axes : matplotlib figure and axes (2 subplots)
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    abs_error = np.abs(numerical - analytic)
    rel_error = abs_error / (np.abs(analytic) + 1e-10)

    axes[0].plot(x, abs_error, color='#d62728', linewidth=2)
    axes[0].set_xlabel(xlabel, fontsize=12)
    axes[0].set_ylabel('Absolute Error', fontsize=12)
    axes[0].set_title('Absolute Error', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x, rel_error * 100, color='#9467bd', linewidth=2)
    axes[1].set_xlabel(xlabel, fontsize=12)
    axes[1].set_ylabel('Relative Error (%)', fontsize=12)
    axes[1].set_title('Relative Error', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig, axes


def animate_evolution(x, history_numerical, analytic_fn, times,
                      xlabel='x', ylabel='y', title='Time Evolution',
                      figsize=(12, 5), interval=100):
    """Create side-by-side animation of numerical vs analytic evolution.

    Parameters
    ----------
    x : array-like
        Spatial coordinates
    history_numerical : list of arrays
        Numerical solution at each saved time
    analytic_fn : callable
        Function(x, t) returning analytic solution
    times : array-like
        Time values corresponding to history snapshots
    interval : int
        Milliseconds between frames

    Returns
    -------
    anim : matplotlib FuncAnimation
        Call HTML(anim.to_jshtml()) to display in notebook
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Find global y-limits
    y_min = min(np.min(h) for h in history_numerical)
    y_max = max(np.max(h) for h in history_numerical)
    margin = 0.1 * (y_max - y_min)

    # Initial frame
    line_num, = axes[0].plot(x, history_numerical[0], **NUMERICAL_STYLE)
    line_ana, = axes[1].plot(x, analytic_fn(x, times[0]), **ANALYTIC_STYLE)

    for ax in axes:
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3)

    axes[0].set_title('Numerical', fontsize=12)
    axes[1].set_title('Analytic', fontsize=12)

    time_text = fig.suptitle(f'{title} | t = {times[0]:.2e}', fontsize=14)

    def update(frame):
        line_num.set_ydata(history_numerical[frame])
        line_ana.set_ydata(analytic_fn(x, times[frame]))
        time_text.set_text(f'{title} | t = {times[frame]:.2e}')
        return line_num, line_ana, time_text

    anim = FuncAnimation(fig, update, frames=len(times),
                         interval=interval, blit=False)
    plt.close(fig)  # Prevent static display
    return anim


def animate_overlay(x, history_numerical, analytic_fn, times,
                    xlabel='x', ylabel='y', title='Comparison',
                    initial=None, figsize=(10, 6), interval=100):
    """Create overlaid animation showing numerical and analytic together.

    Parameters
    ----------
    x : array-like
        Spatial coordinates
    history_numerical : list of arrays
        Numerical solution at each saved time
    analytic_fn : callable
        Function(x, t) returning analytic solution
    times : array-like
        Time values corresponding to history snapshots
    initial : array-like, optional
        Initial condition to show as static reference

    Returns
    -------
    anim : matplotlib FuncAnimation
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Find global y-limits
    y_min = min(np.min(h) for h in history_numerical)
    y_max = max(np.max(h) for h in history_numerical)
    margin = 0.1 * (y_max - y_min)

    if initial is not None:
        ax.plot(x, initial, **INITIAL_STYLE)

    line_num, = ax.plot(x, history_numerical[0], **NUMERICAL_STYLE)
    line_ana, = ax.plot(x, analytic_fn(x, times[0]), **ANALYTIC_STYLE)

    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    time_text = ax.set_title(f'{title} | t = {times[0]:.2e}', fontsize=14)

    def update(frame):
        line_num.set_ydata(history_numerical[frame])
        line_ana.set_ydata(analytic_fn(x, times[frame]))
        ax.set_title(f'{title} | t = {times[frame]:.2e}', fontsize=14)
        return line_num, line_ana

    anim = FuncAnimation(fig, update, frames=len(times),
                         interval=interval, blit=False)
    plt.close(fig)
    return anim


def metrics_summary(metrics, thresholds=None):
    """Display metrics as a formatted summary table.

    Parameters
    ----------
    metrics : dict
        Metric name -> value
    thresholds : dict, optional
        Metric name -> threshold for pass/fail

    Returns
    -------
    widget : ipywidgets HTML or VBox
    """
    rows = []
    for name, value in metrics.items():
        if thresholds and name in thresholds:
            threshold = thresholds[name]
            passed = value <= threshold
            status = '✓ PASS' if passed else '✗ FAIL'
            color = 'green' if passed else 'red'
            row = f'<tr><td>{name}</td><td>{value:.4e}</td><td>{threshold:.4e}</td><td style="color:{color}">{status}</td></tr>'
        else:
            row = f'<tr><td>{name}</td><td>{value:.4e}</td><td>-</td><td>-</td></tr>'
        rows.append(row)

    html = f'''
    <table style="border-collapse: collapse; font-family: monospace;">
        <tr style="background: #f0f0f0;">
            <th style="padding: 8px; border: 1px solid #ddd;">Metric</th>
            <th style="padding: 8px; border: 1px solid #ddd;">Value</th>
            <th style="padding: 8px; border: 1px solid #ddd;">Threshold</th>
            <th style="padding: 8px; border: 1px solid #ddd;">Status</th>
        </tr>
        {''.join(rows)}
    </table>
    '''
    return widgets.HTML(html)


def compute_metrics(numerical, analytic):
    """Compute standard validation metrics.

    Parameters
    ----------
    numerical, analytic : array-like
        Solutions to compare

    Returns
    -------
    dict with keys: l2_error, linf_error, mean_error, max_rel_error
    """
    diff = numerical - analytic
    analytic_norm = np.linalg.norm(analytic)

    return {
        'l2_error': np.linalg.norm(diff) / (analytic_norm + 1e-10),
        'linf_error': np.max(np.abs(diff)),
        'mean_error': np.mean(np.abs(diff)),
        'max_rel_error': np.max(np.abs(diff) / (np.abs(analytic) + 1e-10)),
    }
