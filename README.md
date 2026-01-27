# JAX Plasma Physics Simulation

[![Tests](https://github.com/your-username/jax-frc/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/jax-frc/actions)

GPU-accelerated JAX-based multi-physics simulation. Design, build, control production-level FRC fusion reactor.

## Key Features

- **Five Physics Models**: Resistive MHD, Extended MHD, Hybrid Kinetic, Neutral Fluid, Burning Plasma
- **Fusion Burn Physics**: D-T/D-D/D-He3 reactions with Bosch-Hale rates
- **Anomalous Transport**: Configurable particle and energy diffusion
- **Multi-Phase Scenarios**: Automatic phase transitions for complex simulations
- **FRC Merging**: Two-FRC collision with Belova et al. validation
- **Direct Energy Conversion**: Induction-based power recovery modeling
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
- [Physics Models](docs/models/index.md) - Resistive MHD, Extended MHD, Hybrid Kinetic, Neutral Fluid, Burning Plasma
- [Supporting Modules](docs/modules/burn.md) - Burn physics, transport, comparisons
- [Developer Guide](docs/developer/architecture.md) - Architecture, JAX patterns, extending the code
- [API Reference](docs/api/index.md) - Core, Solvers, Boundaries, Diagnostics
- [Testing](docs/testing/index.md) - Test suite and physics invariants

## Commands

```bash
# Run examples
python run_example.py examples/merging.yaml

# Run tests
py -m pytest tests/ -v

# Run tests (skip slow physics tests)
py -m pytest tests/ -k "not slow"
```

## Project Structure

```
jax-frc/
├── jax_frc/           # Main package
│   ├── models/        # Physics models
│   ├── solvers/       # Time integration
│   ├── burn/          # Fusion burn physics
│   ├── transport/     # Anomalous transport
│   ├── comparisons/   # Literature validation
│   └── ...
├── tests/             # Test suite with invariants
├── examples/          # Example configurations
├── docs/              # Documentation
└── CONTRIBUTING.md    # Contribution guidelines
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR workflow.

## License

This project is provided for educational and research purposes.
