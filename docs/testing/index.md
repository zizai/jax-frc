# Testing

The project includes a comprehensive test suite with physics invariant validation.

## Running Tests

```bash
# Run all tests
py -m pytest tests/ -v

# Run specific model tests
py -m pytest tests/test_resistive_mhd.py -v
py -m pytest tests/test_extended_mhd.py -v
py -m pytest tests/test_hybrid_kinetic.py -v

# Run with coverage
py -m pytest tests/ --cov=. --cov-report=term-missing
```

## Test Structure

```
tests/
├── conftest.py              # Pytest fixtures
├── invariants/              # Property-based testing
│   ├── boundedness.py       # Value bounds checking
│   ├── conservation.py      # Energy, momentum, flux conservation
│   └── consistency.py       # State consistency checks
├── test_resistive_mhd.py
├── test_extended_mhd.py
├── test_hybrid_kinetic.py
├── test_boundaries.py
├── test_scenarios.py
├── test_merging_phase.py
├── test_merging_diagnostics.py
└── test_merging_integration.py
```

## Property-Based Testing

See [Invariants](invariants.md) for detailed documentation on physics invariant checks.

## Legacy Test Runner

For backward compatibility:

```bash
python test_simulations.py
```
