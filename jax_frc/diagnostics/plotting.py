"""Main plotting functions for JAX-FRC diagnostics.

High-level functions that accept SimulationResult objects and produce
standard visualizations with consistent styling.
"""

from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt

from jax_frc.results import SimulationResult
from jax_frc.diagnostics.styles import apply_style, get_label, get_cmap
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


def plot_fields(
    result: SimulationResult,
    t: Optional[Union[int, float]] = None,
    save_dir: Optional[str] = "auto",
    show: Optional[bool] = None,
    format: str = 'png',
) -> plt.Figure:
    """Plot 2D field contours at a specific time.

    Creates contour plots of available fields (psi, B, density, pressure).
    Uses appropriate colormaps for each field type.

    Args:
        result: SimulationResult object with field data
        t: Time to plot. If float, finds nearest time. If int, uses as index.
           If None, plots final state.
        save_dir: Directory to save plots. "auto" creates timestamped dir.
        show: Whether to display interactively.
        format: Output format ('png', 'pdf', or 'both')

    Returns:
        Matplotlib Figure object (last one created if multiple fields)
    """
    if show is None:
        show = not is_notebook()

    # Collect available fields
    fields = {}
    if result.psi is not None:
        fields['psi'] = np.asarray(result.psi)
    if result.density is not None:
        fields['n'] = np.asarray(result.density)
    if result.pressure is not None:
        fields['p'] = np.asarray(result.pressure)
    if result.B is not None:
        B_r, B_theta, B_z = result.B
        B_mag = np.sqrt(
            np.asarray(B_r)**2 +
            np.asarray(B_theta)**2 +
            np.asarray(B_z)**2
        )
        fields['B'] = B_mag

    if not fields:
        raise ValueError("Result has no field data to plot")

    # Create coordinate arrays
    nr, nz = result.grid_shape[:2]
    dr, dz = result.grid_spacing[:2]
    r = np.arange(nr) * dr
    z = np.arange(nz) * dz

    # Handle save directory
    actual_save_dir = None
    if save_dir is not None:
        if save_dir == "auto":
            actual_save_dir = create_output_dir(result.model_name)
        else:
            actual_save_dir = save_dir

    # Time label for title
    time_label = f"t = {result.final_time:.2e} s"
    if t is not None:
        if isinstance(t, float):
            time_label = f"t = {t:.2e} s"
        else:
            time_label = f"step {t}"

    # Plot each field
    last_fig = None
    for name, data in fields.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        # Transpose if needed to get (r, z) orientation
        if data.shape[0] == nr and data.shape[1] == nz:
            plot_data = data.T  # Transpose for imshow/contour
        else:
            plot_data = data

        im = ax.contourf(r, z, plot_data, levels=50, cmap=get_cmap(name))
        ax.contour(r, z, plot_data, levels=10, colors='k', linewidths=0.5, alpha=0.3)

        ax.set_xlabel(get_label('r'))
        ax.set_ylabel(get_label('z'))
        ax.set_title(f"{get_label(name)} - {time_label}")
        ax.set_aspect('equal')

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(get_label(name))

        fig.tight_layout()

        if actual_save_dir:
            save_figure(fig, f"{name}_field", actual_save_dir, format=format)

        last_fig = fig

    if show:
        plt.show()

    return last_fig
