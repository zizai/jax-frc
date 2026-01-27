# Validation API

The validation module provides infrastructure for validating simulations against analytic solutions and reference data.

## Overview

Validation cases are standalone scripts under `validation/cases/`.
Run them directly to generate HTML reports in `validation/reports/`.

```bash
python validation/cases/analytic/magnetic_diffusion.py
python validation/cases/analytic/frozen_flux.py
```

Each script prints metrics to stdout and saves a report with plots.

## Metrics

Available error metrics for comparing simulation results:

```python
from jax_frc.validation import l2_error, linf_error, rmse_curve, check_tolerance

# L2 (RMS) error
error = l2_error(computed, reference)

# L-infinity (max) error
error = linf_error(computed, reference)

# RMSE over time series
errors = rmse_curve(computed_history, reference_history)

# Check if value is within tolerance
passed = check_tolerance(value, expected, tolerance)
```

| Metric | Description |
|--------|-------------|
| `l2_error(a, b)` | Root-mean-square error: √(mean((a-b)²)) |
| `linf_error(a, b)` | Maximum absolute error: max(\|a-b\|) |
| `rmse_curve(a, b)` | RMSE at each time point |
| `check_tolerance(v, exp, tol)` | Check if \|v - exp\| ≤ tol |

## ValidationResult

Container for validation outcomes:

```python
from jax_frc.validation import ValidationResult, MetricResult

result = ValidationResult(
    case_name="magnetic_diffusion",
    passed=True,
    runtime_seconds=12.5,
    metrics={
        "l2_error": MetricResult(name="l2_error", value=0.02, passed=True),
        "linf_error": MetricResult(name="linf_error", value=0.08, passed=True),
    }
)
```

### MetricResult

```python
@dataclass
class MetricResult:
    name: str       # Metric name
    value: float    # Computed value
    passed: bool    # Whether within tolerance
    expected: float = 0.0
    tolerance: float = 0.0
    message: str = ""
```

## ReferenceManager

Manages reference data for validation:

```python
from jax_frc.validation import ReferenceManager, ReferenceData

# Load reference data
manager = ReferenceManager("references/")
ref = manager.load("magnetic_diffusion_analytic")

# Access reference fields
T_ref = ref.fields["T"]
time_ref = ref.time
```

## Integration with Configurations

Configurations can provide analytic solutions for validation:

```python
from jax_frc.configurations import MagneticDiffusionConfiguration

config = MagneticDiffusionConfiguration(B_peak=1.0, eta=1e-6)
geometry = config.build_geometry()

# Get analytic solution at time t
B_analytic = config.analytic_solution(geometry, t=1e-4)
```
