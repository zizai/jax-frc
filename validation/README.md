# Validation Cases

This directory contains validation scripts that test JAX-FRC against known solutions and expected behaviors.

## Running a Validation

Each validation case is a standalone Python script:

```bash
# Run a specific validation
python validation/cases/analytic/diffusion_slab.py

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
- **diffusion_slab.py** - 1D heat diffusion with Gaussian initial condition

### FRC Merging (`cases/frc/`)
FRC merging cases validated qualitatively (stability, conservation):
- **belova_case1.py** - Large FRC merging without compression

### Hall Reconnection (`cases/hall_reconnection/`)
Hall MHD reconnection tests (to be converted)

### MHD Regression (`cases/mhd_regression/`)
Regression tests against baseline runs (to be converted)

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

from jax_frc.configurations.<module> import <Configuration>
from jax_frc.solvers import Solver
from validation.utils.reporting import ValidationReport
from validation.utils.plotting import plot_comparison, plot_error

NAME = "case_name"
DESCRIPTION = "Brief description"

def setup_configuration():
    """Define simulation parameters."""
    return {...}

def analytic_solution(x, t, cfg):  # For analytic cases
    """Exact solution."""
    return ...

def run_simulation(cfg):
    """Run simulation, return state and geometry."""
    ...
    return state, geometry

ACCEPTANCE = {
    'metric_name': {'threshold': 0.1, 'description': '...'},
}

def main():
    """Run validation and generate report."""
    cfg = setup_configuration()
    state, geometry = run_simulation(cfg)

    # Compute metrics...
    # Check acceptance...

    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration=cfg,
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

## Utilities

The `validation/utils/` module provides:

- **ValidationReport** - Generates HTML reports with embedded plots
- **plot_comparison()** - Overlay plot of simulation vs expected
- **plot_error()** - Error distribution plot

## Migration from YAML

The previous YAML-based validation system (`ValidationRunner`) is deprecated.
Each YAML case should be converted to a standalone Python script following the templates above.
