# JAX Plasma Physics Simulation

[![Tests](https://github.com/your-username/jax-frc/actions/workflows/tests.yml/badge.svg)](https://github.com/your-username/jax-frc/actions)

GPU-accelerated JAX-based multi-physics simulation. Design, build, control production-level FRC fusion reactor.

Reference implementations:
- [AGATE](https://git.smce.nasa.gov/marble/agate-open-source) 
- [Athena](https://github.com/PrincetonUniversity/athena)
- [gPLUTO](https://gitlab.com/PLUTO-code/gPLUTO)

## Key Features

- **Plasma Physics**: Resistive MHD, Extended MHD, Hybrid Kinetic, Coupled Fluid, Burning Plasma
- **Fusion Burn Physics**: D-T/D-D/D-He3 reactions with Bosch-Hale rates
- **Anomalous Transport**: Configurable particle and energy diffusion
- **FRC Fusion Cycle**: Two-FRC collision Induction-based power recovery modeling

## Installation

```bash
# (Optional) create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\activate

# Install the package and dependencies
pip install -e .

# Optional: install AGATE for regression/validation data generation
pip install -e ../agate-open-source
```

## Quick Start

```bash
python scripts/run.py examples/merging.yaml
```

```python
from jax_frc.simulation import Simulation, Geometry, State
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.solvers.explicit import RK4Solver

# Build simulation with fluent API
geometry = Geometry(nx=64, ny=64, nz=1,
                    x_min=-1.0, x_max=1.0,
                    y_min=-1.0, y_max=1.0)
model = ExtendedMHD(eta=1e-4)
solver = RK4Solver(cfl_safety=0.25, dt_max=1e-4)
state = State.zeros(64, 64, 1)

sim = Simulation.builder() \
    .geometry(geometry) \
    .model(model) \
    .solver(solver) \
    .initial_state(state) \
    .build()

# Run simulation
final_state = sim.run(t_end=1.0)
```

Or use a preset configuration:

```python
from jax_frc.simulation.presets import create_magnetic_diffusion

sim = create_magnetic_diffusion(nx=64, ny=64)
sim.run(t_end=0.1)
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
python scripts/run.py examples/merging.yaml

# Run tests
py -m pytest tests/ -v

# Run tests (skip slow physics tests)
py -m pytest tests/ -k "not slow"
```

## Project Structure

```
jax-frc/
├── jax_frc/           # Main package
│   ├── simulation/    # Orchestration layer (Simulation, State, Geometry)
│   │   └── presets/   # Pre-configured simulations
│   ├── models/        # Physics models (compute_rhs, compute_stable_dt)
│   ├── solvers/       # Time integration (step, _compute_dt, _apply_constraints)
│   ├── burn/          # Fusion burn physics
│   ├── transport/     # Anomalous transport
│   ├── comparisons/   # Literature validation
│   └── ...
├── tests/             # Test suite with invariants
├── examples/          # Example configurations
├── docs/              # Documentation
└── CONTRIBUTING.md    # Contribution guidelines
```

## Architecture

The codebase follows a **three-pillar architecture**:

- **Model** (pure physics): `compute_rhs()`, `compute_stable_dt()` - physics equations only
- **Solver** (pure numerics): timestep control, divergence cleaning, stability checks
- **Simulation** (orchestration): builder pattern, phases, callbacks

```python
# Solver owns all numerical concerns
solver = RK4Solver(
    cfl_safety=0.5,      # Timestep control
    dt_min=1e-12,
    dt_max=1e-3,
    divergence_cleaning="projection",  # Constraint enforcement
    use_checked_step=True              # NaN/Inf checking
)

# Simulation orchestrates everything
sim = Simulation.builder() \
    .geometry(geometry) \
    .model(model) \
    .solver(solver) \
    .initial_state(state) \
    .build()
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and PR workflow.

## License

This project is provided for educational and research purposes.
