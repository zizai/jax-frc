# Architecture Overview

This document explains the jax-frc system architecture for developers who want to understand or extend the codebase.

## High-Level Design

```
┌─────────────────────────────────────────────────────────────┐
│                      Simulation                              │
│  Orchestrates: Model + Solver + Geometry + TimeController   │
└─────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌────────────┐
   │ Model   │   │ Solver  │   │ Geometry │   │ Diagnostics│
   │         │   │         │   │          │   │            │
   │compute_ │   │ step()  │   │ grids    │   │ probes     │
   │  rhs()  │   │         │   │ coords   │   │ output     │
   └─────────┘   └─────────┘   └──────────┘   └────────────┘
```

## Data Flow

1. **Initialization**: `Simulation` creates `State` from initial conditions
2. **Time Loop**: For each step:
   - `Model.compute_rhs(state, geometry)` → derivatives
   - `Solver.step(state, rhs, dt)` → new state
   - `TimeController` adjusts dt based on CFL/stability
   - `Diagnostics` record measurements
3. **Output**: Final state + history returned

## Module Structure

```
jax_frc/
├── core/               # Geometry, State, Simulation orchestrator
│   ├── geometry.py     # Grid definitions, coordinate systems
│   ├── state.py        # Simulation state container
│   └── simulation.py   # Main orchestrator
├── models/             # Physics models
│   ├── base.py         # PhysicsModel protocol
│   ├── resistive_mhd.py
│   ├── extended_mhd.py
│   ├── hybrid_kinetic.py
│   ├── neutral_fluid.py
│   └── burning_plasma.py
├── solvers/            # Time integration
│   ├── explicit.py     # RK4, Euler
│   ├── semi_implicit.py
│   └── imex.py         # Implicit-explicit
├── burn/               # Fusion burn physics
│   ├── physics.py      # Reaction rates (Bosch-Hale)
│   ├── species.py      # Fuel tracking
│   └── conversion.py   # Direct energy conversion
├── transport/          # Transport models
│   └── anomalous.py    # Anomalous diffusion
├── comparisons/        # Literature validation
│   └── belova_merging.py
├── validation/         # Validation infrastructure
├── diagnostics/        # Output and probes
├── boundaries/         # Boundary conditions
└── configurations/     # Pre-built setups
```

## Key Abstractions

### PhysicsModel Protocol

All physics models implement this interface:

```python
class PhysicsModel(Protocol):
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for all state variables."""
        ...

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Return maximum stable timestep."""
        ...
```

### State Container

Immutable dataclass holding all simulation variables:

```python
@dataclass(frozen=True)
class State:
    psi: Array      # Flux function
    v: Array        # Velocity field
    p: Array        # Pressure
    rho: Array      # Density
    t: float        # Current time
```

### Solver Interface

```python
class Solver(Protocol):
    def step(self, state: State, rhs_fn: Callable, dt: float) -> State:
        """Advance state by dt."""
        ...
```

## Extension Points

### Adding a New Model

1. Create `jax_frc/models/your_model.py`
2. Implement `PhysicsModel` protocol
3. Add to `jax_frc/models/__init__.py`
4. Create tests in `tests/test_your_model.py`

See [Adding Models Tutorial](adding-models.md) for walkthrough.

### Adding Diagnostics

1. Create diagnostic class in `jax_frc/diagnostics/`
2. Implement `__call__(state, geometry) -> measurement`
3. Register in `diagnostics/__init__.py`

### Adding Boundary Conditions

1. Add to `jax_frc/boundaries/`
2. Implement `apply(state, geometry) -> state`
3. Make configurable via `from_config()` classmethod
