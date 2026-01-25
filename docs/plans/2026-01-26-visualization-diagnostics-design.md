# Visualization and Diagnostics Design

**Date:** 2026-01-26
**Status:** Approved
**Author:** Brainstorming session

## Overview

Add a plotting module to `jax_frc/diagnostics/` that eliminates repetitive boilerplate when visualizing simulation results. Pass a `SimulationResult` object, get standard plots with consistent styling.

## Problem Statement

Current workflow involves:
- Writing ad-hoc matplotlib scripts for each simulation run
- Repetitive boilerplate for common plot types
- Inconsistent styling across plots
- Manual file organization

## Solution

A diagnostics plotting module with high-level functions that auto-detect the physics model and produce appropriate visualizations.

## Module Structure

```
jax_frc/diagnostics/
├── __init__.py          # Add new exports
├── plotting.py          # NEW - main plotting functions
├── styles.py            # NEW - seaborn theme, colors, labels
└── output.py            # NEW - file saving and directory management
```

## Main Functions

### `plotting.py`

| Function | Purpose |
|----------|---------|
| `plot_overview(result)` | Generate all standard plots for a run |
| `plot_time_traces(result)` | Energy, flux, and other quantities vs time |
| `plot_fields(result, t=None)` | 2D contour plots of ψ, B, n, p at time t |
| `plot_profiles(result, t=None, axis='r')` | 1D slices along r or z axis |
| `plot_particles(result, t=None)` | Phase space and velocity distributions (hybrid only) |

### Common Parameters

All functions accept:

- `result` - SimulationResult object (required)
- `save_dir` - Where to save (default: auto-generated timestamped directory)
- `show` - Display interactive figure (default: True in scripts, False in Jupyter)
- `format` - Output format: 'png', 'pdf', or 'both' (default: 'png')

## Plot Details

### `plot_time_traces(result)`

Single figure with subplots:
- Total magnetic energy vs time
- Total kinetic energy vs time
- Peak magnetic field strength vs time
- Total particle count (hybrid only) or mass conservation check

### `plot_fields(result, t=None)`

One figure per field variable:
- ψ(r,z) - poloidal flux with contour lines showing field lines
- |B|(r,z) - magnetic field magnitude
- n(r,z) - density (with separatrix overlay if detectable)
- p(r,z) - pressure

If `t=None`, plots the final timestep. Accepts `t` as float (finds nearest time) or integer (direct index).

### `plot_profiles(result, t=None, axis='r')`

Line plots through midplane:
- Density profile along chosen axis
- Pressure profile
- Magnetic field components
- Temperature (if available)

### `plot_particles(result, t=None)`

Hybrid kinetic model only:
- (r, vr) phase space scatter
- (z, vz) phase space scatter
- Velocity distribution histogram (vr, vθ, vz)
- Weight distribution (delta-f diagnostic)

### `plot_overview(result)`

Calls all applicable functions and saves everything to one directory.

## Styling

### `styles.py`

Uses seaborn for consistent, publication-quality appearance:

```python
import seaborn as sns

def apply_style():
    """Call once at module import to set consistent style."""
    sns.set_theme(
        style="whitegrid",
        palette="deep",
        font_scale=1.1,
        rc={"figure.figsize": (10, 6)}
    )
```

Field-specific colormaps:

```python
FIELD_COLORS = {
    'psi': 'RdBu',      # Diverging for flux (positive/negative)
    'B': 'viridis',     # Sequential for magnitude
    'n': 'plasma',      # Sequential for density
    'p': 'inferno',     # Sequential for pressure
}
```

Consistent labels with units:

```python
LABELS = {
    'psi': r'$\psi$ [Wb]',
    'B': r'$|B|$ [T]',
    'n': r'$n$ [m$^{-3}$]',
    'p': r'$p$ [Pa]',
    't': r'$t$ [s]',
}
```

## Output Management

### `output.py`

- `create_output_dir(result)` → Returns path like `outputs/resistive_mhd_2026-01-26_14-30-00/`
- `save_figure(fig, name, directory, format)` → Saves with consistent naming
- Auto-detects environment (Jupyter vs script) to set `show` default
- Generates `index.html` in each output directory with thumbnails of all plots

### Directory Structure

```
outputs/
└── resistive_mhd_2026-01-26_14-30-00/
    ├── index.html
    ├── time_traces.png
    ├── psi_t0.001.png
    ├── B_t0.001.png
    ├── density_t0.001.png
    ├── pressure_t0.001.png
    ├── profiles_r_t0.001.png
    └── ...
```

## Model Detection and Adaptability

### Detection Logic

1. Check `result.model_type` if available
2. Fall back to inspecting state variables:
   - `psi` present → Resistive MHD
   - `B_r`, `B_theta`, `B_z` present → Extended MHD
   - `particles` present → Hybrid Kinetic

### Adaptive Behavior

| Model | Time Traces | Fields | Profiles | Particles |
|-------|-------------|--------|----------|-----------|
| Resistive MHD | ✓ | ψ, B, n, p | ✓ | skipped |
| Extended MHD | ✓ | B components, n, p_e | ✓ | skipped |
| Hybrid Kinetic | ✓ | B, n (grid) | ✓ | ✓ |

### Graceful Degradation

- Missing field → skip that plot with warning (not error)
- Missing time array → use step indices
- Missing geometry → fall back to pixel coordinates with note

### Environment Detection

- Uses `get_ipython()` to detect Jupyter
- Notebooks: returns figures inline, `show=False` default
- Scripts: `show=True` triggers `plt.show()`

## Usage Examples

### Basic workflow

```python
from jax_frc.diagnostics import plot_overview

result = simulation.run()
plot_overview(result)  # Saves all plots to outputs/resistive_mhd_2026-01-26.../
```

### Selective plotting

```python
from jax_frc.diagnostics import plot_fields, plot_time_traces

# Just time traces, displayed interactively
plot_time_traces(result, show=True, save_dir=None)

# Fields at specific time, saved as PDF
plot_fields(result, t=0.0005, format='pdf')

# Fields at multiple times
for t in [0.0, 0.0005, 0.001]:
    plot_fields(result, t=t)
```

### In Jupyter

```python
from jax_frc.diagnostics import plot_fields

fig = plot_fields(result, t=0.001)  # Displays inline automatically
fig.axes[0].set_title("Custom title")
fig
```

### Custom save location

```python
plot_overview(result, save_dir="results/experiment_42/")
```

## Dependencies

- `matplotlib` (existing)
- `seaborn` (new)

## Scope

### Included

- 5 plotting functions
- 3 new files in `jax_frc/diagnostics/`
- Seaborn styling with whitegrid theme
- Auto-organized output directories
- Auto-generated index.html for browsing
- Jupyter and script support
- All three physics models

### Not Included (Future Work)

- Comparing multiple runs side-by-side
- Auto-detecting interesting events (reconnection, instabilities)
- Presentation-ready reports or export
- Animations
- 3D reconstruction of axisymmetric fields
