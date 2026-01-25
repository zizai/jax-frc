# Visualization Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add plotting functions to reduce boilerplate when visualizing simulation results.

**Architecture:** A diagnostics plotting module with high-level functions that accept `SimulationResult` objects, auto-detect the physics model, and produce consistent visualizations. Uses seaborn for styling, matplotlib for 2D contours.

**Tech Stack:** Python, matplotlib, seaborn, numpy/jax.numpy

---

## Task 1: Create styles.py with seaborn theme

**Files:**
- Create: `jax_frc/diagnostics/styles.py`
- Test: `tests/test_plotting.py`

**Step 1: Write the failing test**

Create `tests/test_plotting.py`:

```python
"""Tests for diagnostics plotting module."""

import pytest


class TestStyles:
    """Tests for styles module."""

    def test_apply_style_sets_seaborn_theme(self):
        """Verify apply_style configures seaborn."""
        from jax_frc.diagnostics.styles import apply_style
        import matplotlib.pyplot as plt

        apply_style()

        # Check that figure size was set
        assert plt.rcParams['figure.figsize'] == [10.0, 6.0]

    def test_field_colors_has_required_keys(self):
        """Verify all expected field colormaps are defined."""
        from jax_frc.diagnostics.styles import FIELD_COLORS

        required = ['psi', 'B', 'n', 'p']
        for key in required:
            assert key in FIELD_COLORS

    def test_labels_has_required_keys(self):
        """Verify all expected labels are defined."""
        from jax_frc.diagnostics.styles import LABELS

        required = ['psi', 'B', 'n', 'p', 't']
        for key in required:
            assert key in LABELS
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_plotting.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'jax_frc.diagnostics.styles'"

**Step 3: Write minimal implementation**

Create `jax_frc/diagnostics/styles.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_plotting.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/diagnostics/styles.py tests/test_plotting.py
git commit -m "feat(diagnostics): add styles module with seaborn theme"
```

---

## Task 2: Create output.py for file management

**Files:**
- Create: `jax_frc/diagnostics/file_output.py`
- Modify: `tests/test_plotting.py`

**Step 1: Write the failing test**

Add to `tests/test_plotting.py`:

```python
import tempfile
import os
from pathlib import Path


class TestFileOutput:
    """Tests for file output management."""

    def test_create_output_dir_creates_timestamped_directory(self):
        """Verify output directory is created with model name and timestamp."""
        from jax_frc.diagnostics.file_output import create_output_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            result_dir = create_output_dir(
                model_name="resistive_mhd",
                base_dir=tmpdir
            )

            assert os.path.isdir(result_dir)
            assert "resistive_mhd" in result_dir
            # Should contain date pattern YYYY-MM-DD
            assert "2026" in result_dir or "202" in result_dir

    def test_is_notebook_returns_bool(self):
        """Verify notebook detection returns boolean."""
        from jax_frc.diagnostics.file_output import is_notebook

        result = is_notebook()
        assert isinstance(result, bool)
        # In pytest, we're not in a notebook
        assert result is False

    def test_save_figure_creates_file(self):
        """Verify save_figure writes PNG file."""
        from jax_frc.diagnostics.file_output import save_figure
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_figure(fig, "test_plot", tmpdir, format='png')

            assert os.path.isfile(path)
            assert path.endswith('.png')

        plt.close(fig)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_plotting.py::TestFileOutput -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'jax_frc.diagnostics.file_output'"

**Step 3: Write minimal implementation**

Create `jax_frc/diagnostics/file_output.py`:

```python
"""File output management for diagnostic plots.

Handles directory creation, figure saving, and environment detection.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt


def is_notebook() -> bool:
    """Detect if running in a Jupyter notebook.

    Returns:
        True if in notebook environment, False otherwise.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if 'ZMQInteractiveShell' in shell_name:
            return True
        return False
    except (ImportError, NameError):
        return False


def create_output_dir(
    model_name: str,
    base_dir: str = "outputs"
) -> str:
    """Create timestamped output directory for a simulation run.

    Args:
        model_name: Name of the physics model (e.g., "resistive_mhd")
        base_dir: Base directory for outputs (default: "outputs")

    Returns:
        Full path to created directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_name = f"{model_name}_{timestamp}"
    full_path = os.path.join(base_dir, dir_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


def save_figure(
    fig: plt.Figure,
    name: str,
    directory: str,
    format: str = 'png',
    dpi: int = 150
) -> str:
    """Save a matplotlib figure with consistent settings.

    Args:
        fig: Matplotlib figure to save
        name: Base name for the file (without extension)
        directory: Directory to save in
        format: Output format ('png', 'pdf', or 'both')
        dpi: Resolution for raster formats

    Returns:
        Path to saved file (or first file if 'both').
    """
    os.makedirs(directory, exist_ok=True)
    paths = []

    if format in ('png', 'both'):
        path = os.path.join(directory, f"{name}.png")
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        paths.append(path)

    if format in ('pdf', 'both'):
        path = os.path.join(directory, f"{name}.pdf")
        fig.savefig(path, bbox_inches='tight')
        paths.append(path)

    return paths[0] if paths else ""


def generate_index_html(directory: str) -> str:
    """Generate an index.html file listing all plots in directory.

    Args:
        directory: Directory containing plot files

    Returns:
        Path to generated index.html
    """
    images = sorted([
        f for f in os.listdir(directory)
        if f.endswith(('.png', '.pdf'))
    ])

    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Simulation Plots</title>
    <style>
        body { font-family: sans-serif; margin: 20px; }
        h1 { color: #333; }
        .gallery { display: flex; flex-wrap: wrap; gap: 20px; }
        .plot { border: 1px solid #ddd; padding: 10px; }
        .plot img { max-width: 400px; height: auto; }
        .plot p { margin: 5px 0; font-size: 14px; }
    </style>
</head>
<body>
    <h1>Simulation Results</h1>
    <div class="gallery">
"""

    for img in images:
        if img.endswith('.png'):
            html_content += f"""        <div class="plot">
            <img src="{img}" alt="{img}">
            <p>{img}</p>
        </div>
"""

    html_content += """    </div>
</body>
</html>
"""

    index_path = os.path.join(directory, "index.html")
    with open(index_path, 'w') as f:
        f.write(html_content)

    return index_path
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_plotting.py::TestFileOutput -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/diagnostics/file_output.py tests/test_plotting.py
git commit -m "feat(diagnostics): add file output management"
```

---

## Task 3: Create plotting.py with plot_time_traces

**Files:**
- Create: `jax_frc/diagnostics/plotting.py`
- Modify: `tests/test_plotting.py`

**Step 1: Write the failing test**

Add to `tests/test_plotting.py`:

```python
import jax.numpy as jnp
from jax_frc.results import SimulationResult


def make_mock_result(model_name: str = "resistive_mhd") -> SimulationResult:
    """Create a mock SimulationResult for testing."""
    nr, nz = 32, 64
    n_steps = 10

    # Create mock history as dict with time series
    history = {
        'time': jnp.linspace(0, 1e-3, n_steps),
        'magnetic_energy': jnp.linspace(1.0, 0.8, n_steps),
        'kinetic_energy': jnp.linspace(0.0, 0.2, n_steps),
        'max_B': jnp.linspace(0.5, 0.4, n_steps),
    }

    return SimulationResult(
        model_name=model_name,
        final_time=1e-3,
        n_steps=n_steps,
        grid_shape=(nr, nz),
        grid_spacing=(0.01, 0.01),
        psi=jnp.zeros((nr, nz)),
        density=jnp.ones((nr, nz)) * 1e20,
        pressure=jnp.ones((nr, nz)) * 1000,
        history=history,
    )


class TestPlotTimeTraces:
    """Tests for plot_time_traces function."""

    def test_plot_time_traces_returns_figure(self):
        """Verify plot_time_traces returns matplotlib figure."""
        from jax_frc.diagnostics.plotting import plot_time_traces
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_time_traces(result, show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) >= 1  # At least one subplot
        plt.close(fig)

    def test_plot_time_traces_saves_file(self):
        """Verify plot_time_traces saves to directory when specified."""
        from jax_frc.diagnostics.plotting import plot_time_traces
        import matplotlib.pyplot as plt

        result = make_mock_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig = plot_time_traces(result, show=False, save_dir=tmpdir)

            files = os.listdir(tmpdir)
            assert any('time_traces' in f for f in files)

        plt.close(fig)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_plotting.py::TestPlotTimeTraces -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'jax_frc.diagnostics.plotting'"

**Step 3: Write minimal implementation**

Create `jax_frc/diagnostics/plotting.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_plotting.py::TestPlotTimeTraces -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add jax_frc/diagnostics/plotting.py tests/test_plotting.py
git commit -m "feat(diagnostics): add plot_time_traces function"
```

---

## Task 4: Add plot_fields for 2D contour plots

**Files:**
- Modify: `jax_frc/diagnostics/plotting.py`
- Modify: `tests/test_plotting.py`

**Step 1: Write the failing test**

Add to `tests/test_plotting.py`:

```python
class TestPlotFields:
    """Tests for plot_fields function."""

    def test_plot_fields_returns_figure(self):
        """Verify plot_fields returns matplotlib figure."""
        from jax_frc.diagnostics.plotting import plot_fields
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_fields(result, show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fields_with_time_index(self):
        """Verify plot_fields accepts time index."""
        from jax_frc.diagnostics.plotting import plot_fields
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_fields(result, t=0, show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_fields_saves_files(self):
        """Verify plot_fields saves PNG files."""
        from jax_frc.diagnostics.plotting import plot_fields
        import matplotlib.pyplot as plt

        result = make_mock_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            fig = plot_fields(result, show=False, save_dir=tmpdir)
            files = os.listdir(tmpdir)
            # Should have at least one field plot
            assert len(files) >= 1

        plt.close(fig)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_plotting.py::TestPlotFields -v`
Expected: FAIL with "cannot import name 'plot_fields'"

**Step 3: Write implementation**

Add to `jax_frc/diagnostics/plotting.py`:

```python
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
    from jax_frc.diagnostics.styles import get_cmap

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
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_plotting.py::TestPlotFields -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/diagnostics/plotting.py tests/test_plotting.py
git commit -m "feat(diagnostics): add plot_fields for 2D contours"
```

---

## Task 5: Add plot_profiles for 1D slices

**Files:**
- Modify: `jax_frc/diagnostics/plotting.py`
- Modify: `tests/test_plotting.py`

**Step 1: Write the failing test**

Add to `tests/test_plotting.py`:

```python
class TestPlotProfiles:
    """Tests for plot_profiles function."""

    def test_plot_profiles_returns_figure(self):
        """Verify plot_profiles returns matplotlib figure."""
        from jax_frc.diagnostics.plotting import plot_profiles
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_profiles(result, show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_profiles_radial_axis(self):
        """Verify plot_profiles works with axis='r'."""
        from jax_frc.diagnostics.plotting import plot_profiles
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_profiles(result, axis='r', show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_profiles_axial_axis(self):
        """Verify plot_profiles works with axis='z'."""
        from jax_frc.diagnostics.plotting import plot_profiles
        import matplotlib.pyplot as plt

        result = make_mock_result()
        fig = plot_profiles(result, axis='z', show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_plotting.py::TestPlotProfiles -v`
Expected: FAIL with "cannot import name 'plot_profiles'"

**Step 3: Write implementation**

Add to `jax_frc/diagnostics/plotting.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_plotting.py::TestPlotProfiles -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/diagnostics/plotting.py tests/test_plotting.py
git commit -m "feat(diagnostics): add plot_profiles for 1D slices"
```

---

## Task 6: Add plot_particles for hybrid kinetic

**Files:**
- Modify: `jax_frc/diagnostics/plotting.py`
- Modify: `tests/test_plotting.py`

**Step 1: Write the failing test**

Add to `tests/test_plotting.py`:

```python
def make_hybrid_result() -> SimulationResult:
    """Create a mock SimulationResult for hybrid kinetic model."""
    nr, nz = 32, 64
    n_particles = 100

    particles = {
        'x': jnp.column_stack([
            jnp.linspace(0.1, 0.3, n_particles),  # r
            jnp.zeros(n_particles),                # theta
            jnp.linspace(-0.5, 0.5, n_particles), # z
        ]),
        'v': jnp.column_stack([
            jnp.linspace(-1e5, 1e5, n_particles),  # vr
            jnp.zeros(n_particles),                 # vtheta
            jnp.linspace(-1e5, 1e5, n_particles),  # vz
        ]),
        'w': jnp.ones(n_particles) * 0.1,
    }

    return SimulationResult(
        model_name="hybrid_kinetic",
        final_time=1e-6,
        n_steps=10,
        grid_shape=(nr, nz),
        grid_spacing=(0.01, 0.01),
        particles=particles,
        history={'time': jnp.linspace(0, 1e-6, 10)},
    )


class TestPlotParticles:
    """Tests for plot_particles function."""

    def test_plot_particles_returns_figure(self):
        """Verify plot_particles returns matplotlib figure."""
        from jax_frc.diagnostics.plotting import plot_particles
        import matplotlib.pyplot as plt

        result = make_hybrid_result()
        fig = plot_particles(result, show=False, save_dir=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_particles_skips_non_hybrid(self):
        """Verify plot_particles returns None for non-hybrid models."""
        from jax_frc.diagnostics.plotting import plot_particles
        import matplotlib.pyplot as plt

        result = make_mock_result()  # resistive_mhd, no particles
        fig = plot_particles(result, show=False, save_dir=None)

        assert fig is None
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_plotting.py::TestPlotParticles -v`
Expected: FAIL with "cannot import name 'plot_particles'"

**Step 3: Write implementation**

Add to `jax_frc/diagnostics/plotting.py`:

```python
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
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_plotting.py::TestPlotParticles -v`
Expected: PASS (2 tests)

**Step 5: Commit**

```bash
git add jax_frc/diagnostics/plotting.py tests/test_plotting.py
git commit -m "feat(diagnostics): add plot_particles for hybrid kinetic"
```

---

## Task 7: Add plot_overview and update exports

**Files:**
- Modify: `jax_frc/diagnostics/plotting.py`
- Modify: `jax_frc/diagnostics/__init__.py`
- Modify: `tests/test_plotting.py`

**Step 1: Write the failing test**

Add to `tests/test_plotting.py`:

```python
class TestPlotOverview:
    """Tests for plot_overview function."""

    def test_plot_overview_creates_all_plots(self):
        """Verify plot_overview generates multiple files."""
        from jax_frc.diagnostics.plotting import plot_overview
        import matplotlib.pyplot as plt

        result = make_mock_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_overview(result, save_dir=tmpdir, show=False)

            files = os.listdir(tmpdir)
            # Should have time traces, fields, profiles, and index.html
            assert len(files) >= 3
            assert any('time_traces' in f for f in files)

        plt.close('all')

    def test_plot_overview_creates_index_html(self):
        """Verify plot_overview generates index.html."""
        from jax_frc.diagnostics.plotting import plot_overview

        result = make_mock_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            plot_overview(result, save_dir=tmpdir, show=False)

            assert os.path.isfile(os.path.join(tmpdir, 'index.html'))

        plt.close('all')


class TestExports:
    """Test that all functions are properly exported."""

    def test_imports_from_diagnostics(self):
        """Verify all plotting functions can be imported from diagnostics."""
        from jax_frc.diagnostics import (
            plot_overview,
            plot_time_traces,
            plot_fields,
            plot_profiles,
            plot_particles,
        )

        assert callable(plot_overview)
        assert callable(plot_time_traces)
        assert callable(plot_fields)
        assert callable(plot_profiles)
        assert callable(plot_particles)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_plotting.py::TestPlotOverview -v`
Expected: FAIL with "cannot import name 'plot_overview'"

**Step 3: Write implementation**

Add to `jax_frc/diagnostics/plotting.py`:

```python
def plot_overview(
    result: SimulationResult,
    save_dir: Optional[str] = "auto",
    show: bool = False,
    format: str = 'png',
) -> str:
    """Generate all standard plots for a simulation result.

    Calls all applicable plotting functions and saves to a single directory.
    Also generates an index.html for easy browsing.

    Args:
        result: SimulationResult object
        save_dir: Directory to save all plots. "auto" creates timestamped dir.
        show: Whether to display plots interactively (default False for overview)
        format: Output format

    Returns:
        Path to output directory
    """
    from jax_frc.diagnostics.file_output import generate_index_html

    # Determine output directory
    if save_dir == "auto":
        actual_dir = create_output_dir(result.model_name)
    else:
        actual_dir = save_dir

    # Generate all applicable plots
    if result.history is not None:
        try:
            plot_time_traces(result, save_dir=actual_dir, show=False, format=format)
        except Exception as e:
            print(f"Warning: Could not generate time traces: {e}")

    # Field plots
    try:
        plot_fields(result, save_dir=actual_dir, show=False, format=format)
    except Exception as e:
        print(f"Warning: Could not generate field plots: {e}")

    # Profile plots
    try:
        plot_profiles(result, axis='r', save_dir=actual_dir, show=False, format=format)
        plot_profiles(result, axis='z', save_dir=actual_dir, show=False, format=format)
    except Exception as e:
        print(f"Warning: Could not generate profile plots: {e}")

    # Particle plots (hybrid only)
    if result.particles is not None:
        try:
            plot_particles(result, save_dir=actual_dir, show=False, format=format)
        except Exception as e:
            print(f"Warning: Could not generate particle plots: {e}")

    # Generate index.html
    generate_index_html(actual_dir)

    if show:
        plt.show()

    return actual_dir
```

Update `jax_frc/diagnostics/__init__.py`:

```python
"""Diagnostics and output for JAX-FRC simulation."""

from jax_frc.diagnostics.probes import (
    Probe,
    FluxProbe,
    EnergyProbe,
    BetaProbe,
    CurrentProbe,
    SeparatrixProbe,
    DiagnosticSet,
)
from jax_frc.diagnostics.output import (
    save_checkpoint,
    load_checkpoint,
    save_time_history,
)
from jax_frc.diagnostics.merging import MergingDiagnostics
from jax_frc.diagnostics.plotting import (
    plot_overview,
    plot_time_traces,
    plot_fields,
    plot_profiles,
    plot_particles,
)

__all__ = [
    "Probe",
    "FluxProbe",
    "EnergyProbe",
    "BetaProbe",
    "CurrentProbe",
    "SeparatrixProbe",
    "DiagnosticSet",
    "MergingDiagnostics",
    "save_checkpoint",
    "load_checkpoint",
    "save_time_history",
    "plot_overview",
    "plot_time_traces",
    "plot_fields",
    "plot_profiles",
    "plot_particles",
]
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_plotting.py::TestPlotOverview tests/test_plotting.py::TestExports -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add jax_frc/diagnostics/plotting.py jax_frc/diagnostics/__init__.py tests/test_plotting.py
git commit -m "feat(diagnostics): add plot_overview and export all plotting functions"
```

---

## Task 8: Add seaborn dependency

**Files:**
- Modify: `pyproject.toml` (if exists) or `requirements.txt`

**Step 1: Check which dependency file exists**

Run: `ls pyproject.toml requirements.txt 2>/dev/null`

**Step 2: Add seaborn dependency**

If `pyproject.toml` exists, add to dependencies section:
```toml
seaborn = ">=0.12.0"
```

If `requirements.txt` exists, add line:
```
seaborn>=0.12.0
```

**Step 3: Install and verify**

Run: `pip install seaborn`
Run: `py -c "import seaborn; print(seaborn.__version__)"`

**Step 4: Commit**

```bash
git add pyproject.toml  # or requirements.txt
git commit -m "build: add seaborn dependency for plotting"
```

---

## Task 9: Run full test suite and final commit

**Step 1: Run all plotting tests**

Run: `py -m pytest tests/test_plotting.py -v`
Expected: All tests pass

**Step 2: Run full test suite to check for regressions**

Run: `py -m pytest tests/test_boundaries.py tests/test_plotting.py -v`
Expected: All tests pass

**Step 3: Final integration commit**

```bash
git add -A
git status
# If any uncommitted changes:
git commit -m "feat(diagnostics): complete visualization module

Adds plotting functions to reduce boilerplate:
- plot_overview: generate all standard plots
- plot_time_traces: energy and field evolution
- plot_fields: 2D contour plots
- plot_profiles: 1D radial/axial slices
- plot_particles: phase space for hybrid kinetic

Uses seaborn for consistent styling, auto-generates
index.html for browsing results."
```
