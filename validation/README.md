# Validation Cases

This directory contains validation scripts that test JAX-FRC against known solutions and expected behaviors.

## Running a Validation

Each validation case is a standalone Python script:

```bash
# Run a specific validation
python validation/cases/analytic/magnetic_diffusion.py

# Run with quick test mode (reduced resolution/time for testing)
python validation/cases/frc/belova_case1.py --quick
```

Each script produces an HTML report in `validation/reports/` with:
- Pass/fail status for each metric
- Configuration parameters used
- Comparison plots (simulation vs expected)
- Timing information

## Case Types

### Analytic (`cases/analytic/`)
Cases with exact analytic solutions for quantitative comparison:
- **magnetic_diffusion.py** - 3D magnetic field diffusion with Gaussian initial condition

### FRC Merging (`cases/frc/`)
FRC merging cases validated qualitatively (stability, conservation):
- **belova_case1.py** - Large FRC merging without compression

### Hall Reconnection (`cases/hall_reconnection/`)
Hall MHD reconnection tests:
- **reconnection_gem.py** - GEM reconnection regression vs AGATE reference data

### MHD Regression (`cases/mhd_regression/`)
Regression tests against reference runs:
- **orszag_tang.py** - Orszag–Tang regression vs AGATE reference data

## AGATE Reference Data

Some cases auto-download reference data from Zenodo (record 15084058) and cache
it under `validation/references/agate/` the first time they run.

## Adding a New Case

1. Create `validation/cases/<category>/<name>.py`
2. Follow the template structure:

```python
"""
Case Name
=========
Physics description...
"""
import time
import sys
from pathlib import Path
import jax.numpy as jnp

# Add project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from jax_frc.configurations import MagneticDiffusionConfiguration
from jax_frc.solvers.explicit import EulerSolver
from validation.utils.reporting import ValidationReport
from validation.utils.plotting import plot_comparison, plot_error

NAME = "case_name"
DESCRIPTION = "Brief description"

def setup_configuration():
    """Define simulation parameters."""
    return MagneticDiffusionConfiguration(
        nx=64, ny=4, nz=64,
        extent=1.0,
        sigma=0.1,
        B_peak=1.0,
    )

def run_simulation(config):
    """Run simulation, return state and geometry."""
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = config.build_model()
    solver = EulerSolver()

    # Time stepping
    dt = 0.3
    for _ in range(100):
        state = solver.step(state, dt, model, geometry)

    return state, geometry

ACCEPTANCE = {
    'l2_error': 0.05,
    'max_error': 0.10,
}

def main():
    """Run validation and generate report."""
    config = setup_configuration()
    state, geometry = run_simulation(config)

    # Compute metrics...
    # Check acceptance...

    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration=vars(config),
        metrics=results,
        overall_pass=overall_pass,
    )

    # Add plots...
    report.save()

    return overall_pass

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

3. For **analytic cases**: Define `analytic_solution()` and use `plot_comparison()`
4. For **qualitative cases**: Define stability/conservation checks and use time trace plots
5. Run and verify report generation

## Coordinate System

All validation cases use 3D Cartesian coordinates:
- Scalars: shape `(nx, ny, nz)`
- Vectors: shape `(nx, ny, nz, 3)`

For 2D-like tests, use a thin y dimension (`ny=4`) with periodic boundaries.

## Utilities

The `validation/utils/` module provides:

- **ValidationReport** - Generates HTML reports with embedded plots
- **plot_comparison()** - Overlay plot of simulation vs expected
- **plot_error()** - Error distribution plot
