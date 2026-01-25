# Property-Based Regression Testing Design

## Overview

Property-based regression testing for the JAX plasma physics simulation. Instead of comparing outputs to saved baselines, we test physical invariants that must always hold regardless of code changes.

## Goals

- Catch regressions when code changes break physics behavior
- Handle stochastic particle methods with statistical bounds
- No baseline files to manage or regenerate
- Tests remain valid when algorithms are intentionally improved

## Invariant Categories

### 1. Conservation Laws
Physical quantities that should be preserved across simulation steps.

### 2. Boundedness
Values that must stay within physical or numerical limits.

### 3. Consistency
Mathematical relationships that must hold (e.g., divergence-free magnetic field).

## File Structure

```
jax-fusion/
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # pytest fixtures, shared setup
│   ├── invariants/
│   │   ├── __init__.py
│   │   ├── conservation.py      # Energy, momentum, flux conservation
│   │   ├── boundedness.py       # Positivity, finiteness, range checks
│   │   └── consistency.py       # div(B)=0, J=curl(B), etc.
│   ├── test_resistive_mhd.py    # Invariant tests for resistive MHD
│   ├── test_extended_mhd.py     # Invariant tests for extended MHD
│   └── test_hybrid_kinetic.py   # Invariant tests for hybrid kinetic
└── test_simulations.py          # (existing file, keep as smoke tests)
```

## Invariants by Model

### Resistive MHD

| Invariant | What it checks | Tolerance |
|-----------|----------------|-----------|
| Flux boundedness | `ψ` stays finite, no NaN | exact |
| Current positivity | `J_φ` physically reasonable | warn only |
| Circuit energy | Coil energy `½LI²` bounded | 10% drift/step |
| Resistivity bounds | `0 < η < η_max` (Chodura model) | exact |

### Extended MHD

| Invariant | What it checks | Tolerance |
|-----------|----------------|-----------|
| Divergence-free B | `∇·B ≈ 0` | 1e-6 |
| Density positivity | `n > n_floor` everywhere | exact |
| Hall term stability | No exponential growth in `B` | 2x per 100 steps |
| Halo density applied | Vacuum regions have `n_halo` | exact |

### Hybrid Kinetic

| Invariant | What it checks | Tolerance |
|-----------|----------------|-----------|
| Particle count | No particles lost/created | exact |
| Weight bounds | `w ∈ [-1, 1]` for all particles | exact |
| Distribution positivity | `f = f₀(1 + w) > 0` | 99% of particles |
| Energy conservation | Total kinetic + field energy | 1% over simulation |
| Momentum conservation | Total momentum drift | 1% over simulation |

## Implementation

### Invariant Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class InvariantResult:
    passed: bool
    name: str
    value: float
    tolerance: float
    message: str

class Invariant(ABC):
    @abstractmethod
    def check(self, state_before, state_after) -> InvariantResult:
        pass
```

### Example Invariant

```python
class EnergyConservation(Invariant):
    def __init__(self, rtol=1e-3):
        self.rtol = rtol

    def check(self, state_before, state_after) -> InvariantResult:
        E_before = compute_total_energy(state_before)
        E_after = compute_total_energy(state_after)
        delta = abs(E_after - E_before) / E_before
        passed = delta < self.rtol
        return InvariantResult(
            passed=passed,
            name="EnergyConservation",
            value=delta,
            tolerance=self.rtol,
            message=f"ΔE/E = {delta:.2e} (limit: {self.rtol:.2e})"
        )
```

### Pytest Fixtures

```python
# conftest.py
import pytest
import jax

@pytest.fixture
def resistive_mhd_runner():
    """Yields (state, step_fn) for running controlled steps."""
    from resistive_mhd import initialize, step
    state = initialize(nr=32, nz=64)
    step_fn = jax.jit(step)
    return state, step_fn

@pytest.fixture
def invariant_checker():
    """Runs all invariants and collects failures."""
    def check_all(invariants, state_before, state_after):
        results = [inv.check(state_before, state_after) for inv in invariants]
        failures = [r for r in results if not r.passed]
        return results, failures
    return check_all
```

### Test Example

```python
def test_resistive_mhd_conservation(resistive_mhd_runner, invariant_checker):
    state, step_fn = resistive_mhd_runner
    invariants = [FluxBoundedness(), CircuitEnergy(rtol=0.1)]

    for _ in range(100):
        state_new = step_fn(state)
        results, failures = invariant_checker(invariants, state, state_new)
        assert not failures, format_failures(failures)
        state = state_new
```

### Failure Output

```
FAILED test_resistive_mhd.py::test_conservation
  Invariant 'CircuitEnergy' violated at step 47:
    Expected: ΔE/E < 0.1
    Got: ΔE/E = 0.23
    Values: E_before=1.23e-4, E_after=1.51e-4
```

## Usage

```bash
# Run all invariant tests
pytest tests/

# Run only resistive MHD tests
pytest tests/ -k resistive

# Verbose output with invariant values
pytest tests/ -v

# Generate report file
pytest tests/ --invariant-report=report.json
```

## Report Format

```json
{
  "model": "resistive_mhd",
  "steps": 100,
  "invariants": {
    "FluxBoundedness": {"passed": true, "min": -0.02, "max": 0.15},
    "CircuitEnergy": {"passed": true, "max_drift": 0.03}
  },
  "duration_seconds": 4.2
}
```

## CI Integration

```yaml
- name: Run invariant tests
  run: pytest tests/ -v --tb=short

- name: Upload invariant report
  uses: actions/upload-artifact@v4
  with:
    name: invariant-report
    path: report.json
```

## Rollout Plan

1. **Phase 1:** Boundedness checks (easy, catches crashes)
2. **Phase 2:** Conservation laws (catches physics bugs)
3. **Phase 3:** Consistency checks (catches subtle numerical issues)

Each category has separate pytest markers for selective runs:
- `@pytest.mark.boundedness`
- `@pytest.mark.conservation`
- `@pytest.mark.consistency`
