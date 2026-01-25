# Physics Invariants

Property-based testing for validating physics conservation laws and bounds.

## Overview

The invariant system provides automated checks that physics is preserved during simulation:

```python
from tests.invariants.conservation import EnergyConservation, FluxConservation
from tests.invariants.boundedness import PositiveDensity, FiniteValues

# Check that physics is preserved
energy_check = EnergyConservation(rtol=0.01)
result = energy_check.check(state_before, state_after)
assert result.passed, result.message

# Verify bounded quantities
density_check = PositiveDensity()
result = density_check.check(state.n)
assert result.passed
```

## Available Invariants

### Conservation Laws

| Invariant | Description | Tolerance |
|-----------|-------------|-----------|
| `EnergyConservation` | Total energy within tolerance | rtol=0.01 |
| `FluxConservation` | Magnetic flux preserved | rtol=0.01 |
| `MomentumConservation` | Total momentum preserved | rtol=0.01 |
| `ParticleCountConservation` | Particle number unchanged | exact |

### Boundedness

| Invariant | Description |
|-----------|-------------|
| `PositiveDensity` | Density > 0 everywhere |
| `FiniteValues` | No NaN or Inf values |

### Consistency

| Invariant | Description |
|-----------|-------------|
| `StateConsistency` | Array shapes match geometry |

## Usage in Tests

```python
import pytest
from tests.invariants.conservation import EnergyConservation

class TestResistiveMHD:
    def test_energy_conservation(self, simulation):
        initial_state = simulation.state
        simulation.run_steps(100)
        final_state = simulation.state

        check = EnergyConservation(rtol=0.05)
        result = check.check(initial_state, final_state)
        assert result.passed, result.message
```

## Creating Custom Invariants

Extend the base `Invariant` class:

```python
from tests.invariants.base import Invariant, InvariantResult

class CustomInvariant(Invariant):
    def check(self, state_before, state_after) -> InvariantResult:
        # Your validation logic here
        passed = True
        message = "Custom check passed"
        return InvariantResult(passed=passed, message=message)
```
