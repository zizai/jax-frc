# Validation API

The validation module provides infrastructure for validating simulations against analytic solutions and reference data.

## Overview

```python
from jax_frc.validation import ValidationRunner, ValidationResult

# Run validation case from YAML
runner = ValidationRunner("cases/slab_diffusion.yaml", output_dir="results/")
result = runner.run()

# Check results
print(f"Passed: {result.passed}")
print(f"Metrics: {result.metrics}")
```

## ValidationRunner

Orchestrates validation case execution:

```python
from jax_frc.validation import ValidationRunner

runner = ValidationRunner(
    case_path="cases/my_case.yaml",  # YAML case definition
    output_dir="results/"             # Output directory
)

# Dry run (no simulation, just setup verification)
result = runner.run(dry_run=True)

# Full run
result = runner.run()
```

### YAML Case Format

```yaml
name: slab_diffusion_test
description: Validate thermal diffusion against analytic solution

configuration:
  class: SlabDiffusionConfiguration
  overrides:
    nr: 64
    nz: 256
    T_peak: 100.0

runtime:
  t_end: 1e-4
  dt: 1e-7

acceptance:
  quantitative:
    - metric: l2_error
      field: T
      expected: 0.0
      tolerance: 0.05
    - metric: linf_error
      field: T
      expected: 0.0
      tolerance: 0.1
```

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
    case_name="slab_diffusion",
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
ref = manager.load("slab_diffusion_analytic")

# Access reference fields
T_ref = ref.fields["T"]
time_ref = ref.time
```

## Integration with Configurations

Configurations can provide analytic solutions for validation:

```python
from jax_frc.configurations import SlabDiffusionConfiguration

config = SlabDiffusionConfiguration(T_peak=100.0, kappa=1e10)
geometry = config.build_geometry()

# Get analytic solution at time t
T_analytic = config.analytic_solution(geometry, t=1e-4)
```
