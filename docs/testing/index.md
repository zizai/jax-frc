# Testing

The project includes a comprehensive test suite with physics invariant validation.

## Running Tests

```bash
# Run all tests
py -m pytest tests/ -v

# Skip slow physics tests
py -m pytest tests/ -k "not slow"

# Run specific model tests
py -m pytest tests/test_resistive_mhd.py -v
py -m pytest tests/test_extended_mhd.py -v
py -m pytest tests/test_hybrid_kinetic.py -v

# Run with coverage
py -m pytest tests/ --cov=jax_frc --cov-report=term-missing
```

## Test Structure

```
tests/
├── conftest.py                    # Pytest fixtures
├── invariants/                    # Property-based testing
│   ├── boundedness.py             # Value bounds checking
│   ├── conservation.py            # Energy, momentum, flux conservation
│   └── consistency.py             # State consistency checks
├── test_resistive_mhd.py          # Resistive MHD model
├── test_extended_mhd.py           # Extended MHD model
├── test_hybrid_kinetic.py         # Hybrid kinetic model
├── test_neutral_fluid.py          # Neutral fluid model
├── test_burning_plasma.py         # Burning plasma model
├── test_burn_physics.py           # Fusion reaction rates
├── test_transport.py              # Anomalous transport
├── test_boundaries.py             # Boundary conditions
├── test_solvers.py                # Time integration
├── test_scenarios.py              # Multi-phase scenarios
├── test_merging_phase.py          # FRC merging phase
├── test_merging_diagnostics.py    # Merging-specific diagnostics
├── test_merging_integration.py    # End-to-end merging tests
├── test_validation_*.py           # Validation infrastructure
└── test_belova_comparison.py      # Literature comparison
```

## Test Categories

### Unit Tests

Test individual functions and classes in isolation:
- Operators (laplacian, gradient, divergence)
- Resistivity models
- Boundary condition application

### Model Tests

Test physics models compute correct RHS:
- Shape preservation
- Known limits (zero velocity, uniform fields)
- Stability criteria

### Integration Tests

Test components working together:
- Simulation runs without errors
- Merging scenario completes phases
- Output files written correctly

### Property-Based Tests (Invariants)

Test physics conservation laws:
- Energy conservation (within solver tolerance)
- Flux conservation at boundaries
- Positive density/pressure
- Bounded temperatures

See [Invariants](invariants.md) for detailed documentation.

### Validation Tests

Test against analytical solutions:
- Diffusion equation exact solution
- Alfvén wave propagation
- Equilibrium Grad-Shafranov

### Comparison Tests

Test against published results:
- Belova et al. merging dynamics
- Quantitative metrics within tolerance

## Writing Tests

### For Bug Fixes

```python
def test_bug_123_boundary_at_r_zero():
    """Regression test for issue #123."""
    # Setup that triggered the bug
    geometry = Geometry(r_min=0.0, ...)  # r=0 was problematic

    # This should not raise
    result = model.compute_rhs(state, geometry)

    # Verify correct behavior
    assert jnp.isfinite(result.psi).all()
```

### For New Features

```python
class TestNewFeature:
    def test_basic_functionality(self):
        """Feature works in simple case."""
        ...

    def test_edge_case(self):
        """Feature handles edge case."""
        ...

    def test_integration(self):
        """Feature works with rest of system."""
        ...
```

### For Physics Models

```python
def test_energy_conservation():
    """Total energy should be conserved (within tolerance)."""
    initial_energy = compute_energy(initial_state)
    final_energy = compute_energy(final_state)

    # Allow small numerical drift
    assert abs(final_energy - initial_energy) / initial_energy < 1e-6
```

## Slow Tests

Tests marked `@pytest.mark.slow` run full simulations. Skip with:

```bash
py -m pytest tests/ -k "not slow"
```

These are run in CI but can be skipped locally for faster iteration.
