# Validation Module Refactor Design

**Date:** 2026-01-27
**Status:** Proposed
**Goal:** Replace YAML-based validation with explicit Python scripts that produce HTML reports with comparison plots

## Overview

Refactor the validation module to:
1. Make validation logic explicit per case (Python functions instead of YAML string formulas)
2. Output simulation results with comparison plots (numerical vs analytic/expected)
3. Support interactive exploration and documentation/reproducibility

## Directory Structure

```
validation/
├── cases/
│   ├── analytic/
│   │   ├── diffusion_slab.py      # Runnable validation script
│   │   └── cylindrical_shock.py
│   ├── frc/
│   │   ├── belova_case1.py
│   │   ├── belova_case2.py
│   │   └── belova_case4.py
│   ├── hall_reconnection/
│   │   └── cylindrical_gem.py
│   └── mhd_regression/
│       └── cylindrical_vortex.py
├── utils/
│   ├── reporting.py               # HTML report generation
│   ├── plotting.py                # Comparison plot utilities
│   └── metrics.py                 # L2 error, conservation checks, etc.
└── reports/                       # Output directory (timestamped)
```

Each script is self-contained and runnable:
```bash
python validation/cases/analytic/diffusion_slab.py
```

This produces `validation/reports/2026-01-27T10-15-00_diffusion_slab/report.html`

## Validation Script Anatomy

Each validation script follows this structure:

```python
"""
Diffusion Slab Validation
=========================
1D heat diffusion test with analytic Gaussian solution.

Physics: Heat equation ∂T/∂t = κ ∇²T with initial Gaussian profile.
The analytic solution is a spreading Gaussian that conserves total heat.
"""
import jax.numpy as jnp
from jax_frc.validation.utils import ValidationReport, plot_comparison

# === CASE METADATA ===
NAME = "diffusion_slab"
DESCRIPTION = "1D heat diffusion test with analytic solution"

# === CONFIGURATION ===
def setup_configuration():
    """Define simulation parameters."""
    return {
        'T_peak': 200.0,
        'sigma': 0.3,
        'kappa': 1.0e-3,
        'T_base': 10.0,
        't_end': 1.0e-4,
        'dt': 1.0e-6,
        'nz': 128,
    }

# === ANALYTIC SOLUTION ===
def analytic_solution(z, t, cfg):
    """Exact Gaussian diffusion solution.

    T(z,t) = T_peak * sqrt(σ² / (σ² + 2κt)) * exp(-z² / (2(σ² + 2κt))) + T_base
    """
    sigma_t_sq = cfg['sigma']**2 + 2 * cfg['kappa'] * t
    amplitude = cfg['T_peak'] * jnp.sqrt(cfg['sigma']**2 / sigma_t_sq)
    return amplitude * jnp.exp(-z**2 / (2 * sigma_t_sq)) + cfg['T_base']

# === SIMULATION ===
def run_simulation(cfg):
    """Run the diffusion simulation, return final state and geometry."""
    # Build configuration, geometry, initial state, model
    # Run time stepping loop
    return state, geometry, history

# === VALIDATION ===
def compute_metrics(state, analytic, cfg):
    """Compare simulation to analytic solution."""
    return {'l2_error': ..., 'peak_error': ..., 'mass_conservation': ...}

# === ACCEPTANCE CRITERIA ===
ACCEPTANCE = {
    'l2_error': {'threshold': 0.1, 'description': 'Relative L2 error vs analytic'},
    'mass_conservation': {'threshold': 0.01, 'description': 'Total heat conserved within 1%'},
}

def main():
    """Run validation and generate report."""
    cfg = setup_configuration()

    # Run simulation
    print(f"Running {NAME}...")
    state, geometry, history = run_simulation(cfg)

    # Compute analytic solution on same grid
    z = geometry.z_centers
    t_final = cfg['t_end']
    expected = analytic_solution(z, t_final, cfg)

    # Compare
    metrics = compute_metrics(state, expected, cfg)

    # Check acceptance
    results = {}
    for name, spec in ACCEPTANCE.items():
        value = metrics[name]
        passed = value <= spec['threshold']
        results[name] = {
            'value': value,
            'threshold': spec['threshold'],
            'passed': passed,
            'description': spec['description'],
        }

    overall_pass = all(r['passed'] for r in results.values())

    # Generate report
    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration=cfg,
        metrics=results,
        overall_pass=overall_pass,
    )

    # Add comparison plots
    report.add_plot(
        plot_comparison(z, state.T, expected,
                       labels=['Simulation', 'Analytic'],
                       title='Temperature Profile at t_final',
                       xlabel='z', ylabel='T')
    )

    # Save
    output_dir = report.save()
    print(f"Report: {output_dir}/report.html")
    print(f"Result: {'PASS' if overall_pass else 'FAIL'}")

if __name__ == "__main__":
    main()
```

## HTML Report Structure

The `ValidationReport` class generates a self-contained HTML file containing:

- **Header**: Case name, description, overall PASS/FAIL badge, timestamp
- **Physics Background**: Rendered from the module docstring (supports markdown)
- **Configuration Table**: All parameters used
- **Results Table**: Each metric with value, threshold, pass/fail status
- **Plots Section**: Embedded PNG images (numerical vs expected overlaid)
- **Timing**: Setup time, simulation time, total runtime
- **Warnings**: Any convergence issues, NaN detections, etc.

```python
@dataclass
class ValidationReport:
    name: str
    description: str
    docstring: str           # Physics background from module docstring
    configuration: dict      # All parameters used
    metrics: dict            # Results with pass/fail
    overall_pass: bool
    plots: list = field(default_factory=list)
    timing: dict = None      # Runtime stats
    warnings: list = None    # Any issues encountered

    def add_plot(self, fig, name: str = None):
        """Add matplotlib figure, converts to base64 PNG for embedding."""
        ...

    def save(self, base_dir: Path = None) -> Path:
        """Write report.html to timestamped directory."""
        ...
```

## Plotting Utilities

`validation/utils/plotting.py` provides:

- `plot_comparison(x, actual, expected, ...)` — 1D line comparison
- `plot_contour_comparison(x, y, actual, expected, ...)` — 2D side-by-side
- `plot_time_trace(t, values, ...)` — Evolution over time

## Handling Different Case Types

### Analytic Cases (diffusion_slab, cylindrical_shock)
Full numeric comparison with `analytic_solution()` function as shown above.

### Qualitative Cases (belova_case1, etc.)
No analytic solution. Check stability and conservation instead:

```python
"""
Belova Case 1: Large FRC Merging
================================
Two FRCs merge without axial compression. Expected outcome is partial merge
forming a doublet configuration (Belova et al., Phys. Plasmas, Figs 1-2).

This is a qualitative validation - we check stability and conservation,
not numeric accuracy against a known solution.
"""

ACCEPTANCE = {
    'no_numerical_instability': {'description': 'No NaN/Inf in solution'},
    'flux_conservation': {'threshold': 0.1, 'description': 'Total flux within 10%'},
    'energy_bounded': {'description': 'Energy remains bounded'},
}

def check_qualitative_features(history):
    """Visual inspection helpers - not pass/fail."""
    return {
        'forms_doublet': "See psi contour at t=30",
        'x_point_count': count_x_points(history[-1].psi),
    }
```

### Regression Cases (cylindrical_vortex)
Compare against saved baseline from previous validated run:

```python
def load_baseline():
    """Load previous validated run."""
    return jnp.load('validation/baselines/cylindrical_vortex.npz')
```

## Migration Plan

### Phase 1: Build utilities
- Create `validation/utils/reporting.py` with `ValidationReport` class
- Create `validation/utils/plotting.py` with comparison plot functions
- Keep existing `metrics.py` (already has L2 error, etc.)

### Phase 2: Convert one analytic case as template
- Implement `diffusion_slab.py` fully
- Verify it produces correct HTML report
- Use this as the reference for other cases

### Phase 3: Convert remaining cases
- `cylindrical_shock.py` (analytic)
- `belova_case1.py`, `belova_case2.py`, `belova_case4.py` (qualitative)
- `cylindrical_gem.py` (hall reconnection)
- `cylindrical_vortex.py` (regression)

### Phase 4: Cleanup
- Delete old YAML files from `validation/cases/`
- Remove or deprecate `ValidationRunner` class
- Update documentation

### What stays
- `validation/utils/metrics.py` — metric functions still useful
- `validation/reports/` — output location unchanged

### What goes
- `validation/cases/**/*.yaml` — replaced by `.py` files
- `jax_frc/validation/runner.py` — logic moves into individual scripts
- `jax_frc/validation/references.py` — analytic solutions now in scripts

## Benefits

1. **Explicit validation logic**: Physics is visible in Python functions, not hidden in YAML strings
2. **Interactive exploration**: Scientists can tweak parameters and re-run easily
3. **Self-documenting**: Each case includes physics background in docstring
4. **Flexible structure**: Accommodates analytic, qualitative, and regression cases
5. **Rich output**: HTML reports with embedded comparison plots for review and archiving
