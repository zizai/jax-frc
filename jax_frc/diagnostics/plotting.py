"""Main plotting functions for JAX-FRC diagnostics.

High-level functions that accept SimulationResult objects and produce
standard visualizations with consistent styling.
"""

from typing import Optional, Union
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
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


def plot_profiles(
    result: SimulationResult,
    t: Optional[Union[int, float]] = None,
    axis: str = 'r',
    save_dir: Optional[str] = "auto",
    show: Optional[bool] = None,
    format: str = 'png',
) -> plt.Figure:
    """Plot 1D profiles through the midplane.

    Creates line plots showing field values along a radial or axial slice.

    Args:
        result: SimulationResult object with field data
        t: Time to plot (see plot_fields for format)
        axis: Axis for profile ('r' for radial, 'z' for axial)
        save_dir: Directory to save plots
        show: Whether to display interactively
        format: Output format

    Returns:
        Matplotlib Figure object
    """
    if show is None:
        show = not is_notebook()

    # Collect available fields
    fields = {}
    if result.density is not None:
        fields['Density'] = np.asarray(result.density)
    if result.pressure is not None:
        fields['Pressure'] = np.asarray(result.pressure)
    if result.psi is not None:
        fields['Psi'] = np.asarray(result.psi)

    if not fields:
        raise ValueError("Result has no field data for profiles")

    # Create coordinate arrays
    nr, nz = result.grid_shape[:2]
    dr, dz = result.grid_spacing[:2]
    r = np.arange(nr) * dr
    z = np.arange(nz) * dz

    # Get midplane index
    mid_r = nr // 2
    mid_z = nz // 2

    n_fields = len(fields)
    fig, axes = plt.subplots(n_fields, 1, figsize=(10, 3 * n_fields), sharex=True)
    if n_fields == 1:
        axes = [axes]

    for ax, (name, data) in zip(axes, fields.items()):
        if axis == 'r':
            # Radial profile at z midplane
            profile = data[:, mid_z]
            coord = r
            xlabel = get_label('r')
        else:
            # Axial profile at r midplane
            profile = data[mid_r, :]
            coord = z
            xlabel = get_label('z')

        ax.plot(coord, profile, linewidth=2)
        ax.set_ylabel(name)
        ax.set_title(f"{name} profile along {axis}")

    axes[-1].set_xlabel(xlabel)

    time_label = f"t = {result.final_time:.2e} s"
    fig.suptitle(f"{result.model_name} - Profiles ({time_label})", fontsize=14)
    fig.tight_layout()

    # Save if requested
    if save_dir is not None:
        if save_dir == "auto":
            save_dir = create_output_dir(result.model_name)
        save_figure(fig, f"profiles_{axis}", save_dir, format=format)

    if show:
        plt.show()

    return fig


def plot_particles(
    result: SimulationResult,
    t: Optional[Union[int, float]] = None,
    save_dir: Optional[str] = "auto",
    show: Optional[bool] = None,
    format: str = 'png',
) -> Optional[plt.Figure]:
    """Plot particle phase space and velocity distributions.

    Only applicable to hybrid kinetic simulations. Returns None for other models.

    Args:
        result: SimulationResult object with particle data
        t: Time to plot (see plot_fields for format)
        save_dir: Directory to save plots
        show: Whether to display interactively
        format: Output format

    Returns:
        Matplotlib Figure object, or None if no particle data
    """
    import seaborn as sns

    if result.particles is None:
        return None

    if show is None:
        show = not is_notebook()

    particles = result.particles
    x = np.asarray(particles.get('x', particles.get('r', [])))
    v = np.asarray(particles.get('v', particles.get('vr', [])))
    w = np.asarray(particles.get('w', []))

    if len(x) == 0:
        return None

    # Handle different particle data formats
    if x.ndim == 2:
        r = x[:, 0] if x.shape[1] > 0 else x
        z = x[:, 2] if x.shape[1] > 2 else np.zeros_like(r)
    else:
        r = x
        z = np.zeros_like(r)

    if v.ndim == 2:
        vr = v[:, 0] if v.shape[1] > 0 else v
        vz = v[:, 2] if v.shape[1] > 2 else np.zeros_like(vr)
    else:
        vr = v
        vz = np.zeros_like(vr)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Phase space: r vs vr
    ax = axes[0, 0]
    ax.scatter(r, vr, alpha=0.5, s=5)
    ax.set_xlabel(get_label('r'))
    ax.set_ylabel(get_label('v_r'))
    ax.set_title('Phase Space (r, vr)')

    # Phase space: z vs vz
    ax = axes[0, 1]
    ax.scatter(z, vz, alpha=0.5, s=5)
    ax.set_xlabel(get_label('z'))
    ax.set_ylabel(get_label('v_z'))
    ax.set_title('Phase Space (z, vz)')

    # Velocity distribution
    ax = axes[1, 0]
    sns.histplot(vr, ax=ax, kde=True, bins=30)
    ax.set_xlabel(get_label('v_r'))
    ax.set_title('Radial Velocity Distribution')

    # Weight distribution
    ax = axes[1, 1]
    if len(w) > 0:
        sns.histplot(w, ax=ax, kde=True, bins=30)
        ax.set_xlabel('Weight')
        ax.set_title('Delta-f Weight Distribution')
    else:
        ax.text(0.5, 0.5, 'No weight data', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('Weight Distribution')

    time_label = f"t = {result.final_time:.2e} s"
    fig.suptitle(f"{result.model_name} - Particles ({time_label})", fontsize=14)
    fig.tight_layout()

    # Save if requested
    if save_dir is not None:
        if save_dir == "auto":
            save_dir = create_output_dir(result.model_name)
        save_figure(fig, "particles", save_dir, format=format)

    if show:
        plt.show()

    return fig
