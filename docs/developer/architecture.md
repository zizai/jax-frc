# Architecture Overview

This document explains the jax-frc system architecture for developers who want to understand or extend the codebase.

## High-Level Design

The codebase follows a **three-pillar architecture**:

```
┌─────────────────────────────────────────────────────────────┐
│                      Simulation                              │
│     Orchestrates: Model + Solver + Geometry (Builder API)   │
└─────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌─────────┐   ┌─────────┐   ┌──────────┐   ┌────────────┐
   │ Model   │   │ Solver  │   │ Geometry │   │ Diagnostics│
   │ (pure   │   │ (pure   │   │          │   │            │
   │ physics)│   │numerics)│   │ grids    │   │ probes     │
   │compute_ │   │ step()  │   │ coords   │   │ output     │
   │  rhs()  │   │_compute │   │          │   │            │
   │         │   │  _dt()  │   │          │   │            │
   └─────────┘   └─────────┘   └──────────┘   └────────────┘
```

### Three Pillars

1. **Model** (pure physics): Only physics equations
   - `compute_rhs(state, geometry)` - time derivatives
   - `compute_stable_dt(state, geometry)` - CFL condition

2. **Solver** (pure numerics): All numerical concerns
   - Timestep control: `cfl_safety`, `dt_min`, `dt_max`
   - Constraint enforcement: `divergence_cleaning`
   - Stability checks: `use_checked_step`
   - `step(state, model, geometry)` - complete timestep

3. **Simulation** (orchestration): Builder pattern API
   - Fluent builder: `.geometry().model().solver().initial_state().build()`
   - Phases and callbacks
   - `run(t_end)` - main loop

## Data Flow

1. **Initialization**: `Simulation.builder()` creates configured simulation
2. **Time Loop**: For each step:
   - `Solver._compute_dt(state, model, geometry)` → timestep
   - `Solver.advance(state, dt, model, geometry)` → new state
   - `Solver._apply_constraints(state, geometry)` → cleaned state
   - `Diagnostics` record measurements
3. **Output**: Final state + history returned

## Module Structure

```
jax_frc/
├── simulation/         # Orchestration layer (NEW)
│   ├── simulation.py   # Simulation, SimulationBuilder
│   ├── state.py        # State container
│   ├── geometry.py     # Grid definitions
│   └── presets/        # Factory functions (create_magnetic_diffusion, etc.)
├── models/             # Physics models (pure physics)
│   ├── base.py         # PhysicsModel protocol
│   ├── resistive_mhd.py
│   ├── extended_mhd.py
│   ├── hybrid_kinetic.py
│   ├── neutral_fluid.py
│   └── burning_plasma.py
├── solvers/            # Time integration (pure numerics)
│   ├── base.py         # Solver base with timestep control
│   ├── explicit.py     # RK4, Euler
│   ├── semi_implicit.py
│   ├── imex.py         # Implicit-explicit
│   └── divergence_cleaning.py
├── core/               # Legacy (use simulation/ instead)
├── burn/               # Fusion burn physics
├── transport/          # Transport models
├── comparisons/        # Literature validation
├── validation/         # Validation infrastructure
├── diagnostics/        # Output and probes
├── boundaries/         # Boundary conditions
└── configurations/     # Pre-built setups (legacy)
```

## Key Abstractions

### PhysicsModel Protocol

All physics models implement this interface (pure physics only):

```python
class PhysicsModel(ABC):
    @abstractmethod
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for all state variables."""
        ...

    @abstractmethod
    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Return maximum stable timestep (CFL condition)."""
        ...
```

### Solver Base Class

Solvers own all numerical concerns:

```python
class Solver(ABC):
    # Timestep control
    cfl_safety: float = 0.5
    dt_min: float = 1e-12
    dt_max: float = 1e-3
    
    # Numerical options
    use_checked_step: bool = True
    divergence_cleaning: str = "projection"
    
    def step(self, state: State, model: PhysicsModel, geometry) -> State:
        """Complete step: compute dt, advance, apply constraints."""
        dt = self._compute_dt(state, model, geometry)
        new_state = self.advance(state, dt, model, geometry)
        new_state = self._apply_constraints(new_state, geometry)
        return new_state
    
    @abstractmethod
    def advance(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance state by dt (implemented by subclasses)."""
        ...
```

### SimulationBuilder

Fluent builder pattern for creating simulations:

```python
sim = Simulation.builder() \
    .geometry(Geometry(nx=64, ny=64, nz=1)) \
    .model(ExtendedMHD(eta=1e-4)) \
    .solver(RK4Solver()) \
    .initial_state(State.zeros(64, 64, 1)) \
    .build()

sim.run(t_end=1.0)
```

### State Container

Immutable dataclass holding all simulation variables:

```python
@dataclass(frozen=True)
class State:
    B: Array        # Magnetic field
    E: Array        # Electric field
    n: Array        # Density
    p: Array        # Pressure
    v: Array        # Velocity
    time: float     # Current time
    step: int       # Step counter
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
