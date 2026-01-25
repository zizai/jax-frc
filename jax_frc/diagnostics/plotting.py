"""Main plotting functions for JAX-FRC diagnostics.

High-level functions that accept SimulationResult objects and produce
standard visualizations with consistent styling.
"""

from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from jax_frc.results import SimulationResult
from jax_frc.diagnostics.styles import apply_style, get_label
from jax_frc.diagnostics.file_output import (
    is_notebook,
    create_output_dir,
    save_figure,
)


# Apply style on module import
apply_style()


def plot_time_traces(
    result: SimulationResult,
    save_dir: Optional[str] = "auto",
    show: Optional[bool] = None,
    format: str = 'png',
) -> plt.Figure:
    """Plot time evolution of key quantities.

    Creates a multi-panel figure showing magnetic energy, kinetic energy,
    and peak field strength over time.

    Args:
        result: SimulationResult object with history data
        save_dir: Directory to save plot. "auto" creates timestamped dir,
                  None skips saving.
        show: Whether to display interactively. Default: True in scripts,
              False in notebooks.
        format: Output format ('png', 'pdf', or 'both')

    Returns:
        Matplotlib Figure object
    """
    if show is None:
        show = not is_notebook()

    history = result.history
    if history is None:
        raise ValueError("Result has no history data for time traces")

    # Handle both dict and array history formats
    if isinstance(history, dict):
        time = np.asarray(history.get('time', np.arange(result.n_steps)))
        traces = {
            k: np.asarray(v) for k, v in history.items()
            if k != 'time' and hasattr(v, '__len__')
        }
    else:
        # Assume it's an array - use step indices
        time = np.arange(len(history) if hasattr(history, '__len__') else result.n_steps)
        traces = {'values': np.asarray(history)}

    n_traces = len(traces)
    if n_traces == 0:
        n_traces = 1
        traces = {'(no data)': np.zeros_like(time)}

    fig, axes = plt.subplots(n_traces, 1, figsize=(10, 3 * n_traces), sharex=True)
    if n_traces == 1:
        axes = [axes]

    for ax, (name, values) in zip(axes, traces.items()):
        ax.plot(time, values, linewidth=2)
        ax.set_ylabel(get_label(name) if name in ['t', 'energy'] else name)
        ax.set_title(name.replace('_', ' ').title())

    axes[-1].set_xlabel(get_label('t'))
    fig.suptitle(f"{result.model_name} - Time Evolution", fontsize=14)
    fig.tight_layout()

    # Save if requested
    if save_dir is not None:
        if save_dir == "auto":
            save_dir = create_output_dir(result.model_name)
        save_figure(fig, "time_traces", save_dir, format=format)

    if show:
        plt.show()

    return fig
