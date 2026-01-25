# JAX Plasma Physics Simulation

[![Tests](https://github.com/your-username/jax-frc/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/jax-frc/actions)

A JAX-based GPU-accelerated implementation of plasma physics models for Field-Reversed Configuration (FRC) research.

## Key Features

- **Three Physics Models**: Resistive MHD, Extended MHD, and Hybrid Kinetic
- **Multi-Phase Scenarios**: Automatic phase transitions for complex simulations
- **FRC Merging**: Two-FRC collision with Belova et al. validation
- **Property-Based Testing**: Physics invariant validation

## Quick Start

```bash
pip install jax jaxlib numpy matplotlib
```

```python
from jax_frc import Simulation, Geometry, ResistiveMHD, RK4Solver, TimeController
import jax.numpy as jnp

geometry = Geometry(coord_system='cylindrical', nr=32, nz=64,
                    r_min=0.01, r_max=1.0, z_min=-1.0, z_max=1.0)
model = ResistiveMHD.from_config({'resistivity': {'type': 'chodura', 'eta_0': 1e-6}})
solver = RK4Solver()
time_controller = TimeController(cfl_safety=0.25, dt_max=1e-4)

sim = Simulation(geometry=geometry, model=model, solver=solver, time_controller=time_controller)
sim.initialize(psi_init=lambda r, z: (1 - r**2) * jnp.exp(-z**2))
final_state = sim.run_steps(100)
```

## Documentation

Full documentation is available in the [docs/](docs/index.md) directory:

- [Getting Started](docs/getting-started.md) - Installation and usage
- [Physics Models](docs/models/index.md) - Resistive MHD, Extended MHD, Hybrid Kinetic
- [API Reference](docs/api/index.md) - Core, Solvers, Boundaries, Diagnostics
- [Scenarios](docs/scenarios/index.md) - Multi-phase simulations and FRC merging
- [Testing](docs/testing/index.md) - Test suite and physics invariants
- [Reference](docs/reference/index.md) - Physics concepts and model comparison

## Commands

```bash
# Run examples
python examples.py

# Run tests
py -m pytest tests/ -v
```

## Project Structure

```
jax-frc/
├── jax_frc/           # Main package
├── tests/             # Test suite with invariants
├── examples/          # Example scenarios
├── docs/              # Documentation
├── resistive_mhd.py   # Standalone models
├── extended_mhd.py
└── hybrid_kinetic.py
```

## License

This project is provided for educational and research purposes.
