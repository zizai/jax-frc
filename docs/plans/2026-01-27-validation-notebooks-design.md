# Validation Notebooks Design

**Date:** 2026-01-27
**Status:** Approved
**Goal:** Create Jupyter notebooks for validation cases that serve as both interactive exploration tools and educational documentation

## Overview

Jupyter notebooks for validation cases that:
1. Show explicit logic of configuration setup, simulation progress, and result comparison
2. Provide full physics derivations and educational context
3. Visualize simulations with static plots, animations, and interactive widgets
4. Exist in parallel with the YAML/script automation pipeline (not replacing it)

## Directory Structure

```
notebooks/
├── diffusion_slab.ipynb
├── cylindrical_shock.ipynb
├── belova_case1.ipynb              # Future
├── belova_case2.ipynb              # Future
├── cylindrical_vortex.ipynb        # Future
├── cylindrical_gem.ipynb           # Future
└── _shared.py                      # Reusable plotting/widget helpers
```

Flat structure. Notebook names match validation case names exactly.

## Notebook Content Structure

Each notebook follows this consistent structure:

### 1. Title and Overview
- Case name, one-line description
- Learning objectives: what the reader will understand after completing

### 2. Physics Background (2-4 markdown cells)
- Governing equations with LaTeX
- Derivation of the analytic solution (step-by-step)
- Physical intuition: what should happen and why
- Key parameters and their physical meaning

### 3. Configuration Setup
- Import statements
- Parameter definitions with explanatory comments
- Grid setup and initial conditions
- Visualization of initial state

### 4. Run Simulation
- Build the simulation model
- Time-stepping loop with progress indication
- Optionally save intermediate snapshots for animation

### 5. Analytic Solution
- Implement the exact solution as a Python function
- Evaluate on the same grid at the same final time
- Side-by-side plot: initial vs final (both numerical and analytic)

### 6. Comparison and Metrics
- Overlay plots: numerical vs analytic
- Error metrics: L2 error, max error, conservation checks
- Interpretation: does the simulation pass validation?

### 7. Time Evolution Animation
- Animated plot showing simulation progress over time
- Analytic solution evolving alongside for comparison

### 8. Interactive Exploration
- Sliders for key parameters (e.g., diffusivity, grid resolution)
- Re-run and see how results change

## Visualization Approach

### Static plots (matplotlib)
Used for:
- Initial condition visualization
- Final state comparison (numerical vs analytic overlaid)
- Error distribution plots
- Metric summary bar charts

Style: Clean, publication-quality. Consistent color scheme (blue for numerical, orange dashed for analytic). Labeled axes with units.

### Animated plots (matplotlib.animation)
Used for:
- Time evolution of the solution
- Side-by-side: simulation on left, analytic on right, evolving together
- Displayed inline using `HTML(anim.to_jshtml())`

### Interactive exploration (ipywidgets)
Used for:
- Parameter sensitivity: sliders for key physics parameters and grid resolution
- "What-if" exploration: change a parameter, click "Run", see updated comparison
- Lightweight — runs short simulations for responsiveness

Example widget layout:
```
┌─────────────────────────────────────┐
│  κ (diffusivity): [====o====] 1e-3  │
│  σ (initial width): [==o======] 0.3 │
│  Grid points: [64] [128] [256]      │
│  [Run Simulation]                   │
├─────────────────────────────────────┤
│  [Plot: numerical vs analytic]      │
│  L2 Error: 0.032  ✓ PASS            │
└─────────────────────────────────────┘
```

## Analytic Case Details

### `diffusion_slab.ipynb` — 1D Heat Diffusion

**Physics**: Heat equation ∂T/∂t = κ ∇²T with initial Gaussian profile.

**Derivation to include**:
- Start from Fourier's law and energy conservation
- Show the Gaussian ansatz and verify it satisfies the PDE
- Derive the time-dependent width: σ(t)² = σ₀² + 2κt
- Explain: peak decreases, profile spreads, total heat conserved

**Key teaching points**:
- Why explicit timestepping has a stability limit (CFL condition)
- How L2 error scales with grid resolution (convergence)
- Mass/energy conservation as a sanity check

### `cylindrical_shock.ipynb` — Cylindrical Shock Tube

**Physics**: Riemann problem in cylindrical geometry. Discontinuous initial conditions evolve into shock, contact, and rarefaction waves.

**Derivation to include**:
- Euler equations in cylindrical coordinates
- Rankine-Hugoniot jump conditions
- Self-similar solution structure
- Geometric source terms from cylindrical divergence

**Key teaching points**:
- Shock capturing vs shock tracking
- Numerical diffusion at discontinuities
- Why cylindrical geometry changes wave speeds

## Implementation Details

### Dependencies

Added to project's dev dependencies:
- `ipywidgets` — interactive sliders and buttons
- `matplotlib` — static plots and animations (likely already present)
- `IPython` — display helpers for animations (comes with Jupyter)

### Running notebooks

```bash
# From project root
jupyter lab notebooks/
```

Or with VS Code's Jupyter extension.

### `_shared.py` contents

Lightweight reusable helpers:

```python
# notebooks/_shared.py
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def plot_comparison(x, numerical, analytic, xlabel='x', ylabel='y', title=None):
    """Overlay plot of numerical vs analytic solution."""
    ...

def animate_evolution(x, history, analytic_fn, times, ...):
    """Create side-by-side animation of simulation vs analytic."""
    ...

def parameter_explorer(run_fn, params_spec):
    """Build ipywidgets UI for parameter exploration."""
    ...
```

Notebooks can import these or define visualization code inline — both work.

### Notebook metadata

Each notebook starts with a raw cell containing metadata:
```yaml
---
case: diffusion_slab
category: analytic
dependencies: [ipywidgets, matplotlib]
---
```

### Output handling

Notebooks committed with outputs cleared (via `.gitattributes` or pre-commit hook). Users regenerate outputs by running.

## Scope

**Initial implementation**: Two analytic cases
1. `diffusion_slab.ipynb`
2. `cylindrical_shock.ipynb`

**Future expansion**: FRC cases (belova_case1, etc.), hall reconnection, MHD regression cases.

## Relationship to Existing Validation

These notebooks exist **in parallel** with the existing validation infrastructure:
- YAML configs and automation scripts handle CI/batch validation runs
- Notebooks are for exploration, teaching, and interactive documentation
- Notebooks import from `jax_frc` like any user code — no coupling to validation internals
