# Architecture Overhaul Design

**Date:** 2026-01-31
**Goal:** More consistency and clarity through consolidation of APIs

## Summary

1. **Consolidate NumericalRecipe and TimeController into Solver** - Solver owns all numerics
2. **Consolidate Configuration with Simulation** - Simulation orchestrates everything
3. **Model only controls physics** - isolate numerics to Solver

## The Three Pillars

### 1. Model - Pure Physics

```python
# jax_frc/models/base.py (simplified)

class Model(ABC):
    """Base class for physics models - pure physics only."""

    @abstractmethod
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives from physics equations."""
        ...

    @abstractmethod
    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Compute CFL-stable timestep from physics wave speeds."""
        ...

    # REMOVED: apply_constraints() - now in Solver
```

**Key change:** `apply_constraints()` removed from Model, moved to Solver.

### 2. Solver - Pure Numerics

**Consolidates:** `NumericalRecipe` + `TimeController` + existing `Solver`

```python
# jax_frc/solvers/base.py (extended)

class Solver(ABC):
    """Base class for time integration - owns all numerics."""

    # Timestep control (absorbed from TimeController)
    cfl_safety: float = 0.5
    dt_min: float = 1e-12
    dt_max: float = 1e-3

    # Numerical options (absorbed from NumericalRecipe)
    use_checked_step: bool = True
    divergence_cleaning: str = "projection"

    @abstractmethod
    def advance(self, state: State, dt: float, model: Model, geometry: Geometry) -> State:
        """Advance state by dt using this integration scheme."""
        ...

    def step(self, state: State, model: Model, geometry: Geometry) -> State:
        """Complete step: compute dt, advance, apply constraints."""
        dt = self._compute_dt(state, model, geometry)
        new_state = self.advance(state, dt, model, geometry)
        new_state = self._apply_constraints(new_state, geometry)
        if self.use_checked_step:
            self._check_stability(new_state)
        return new_state

    def _compute_dt(self, state, model, geometry) -> float:
        """Compute timestep from model CFL and bounds."""
        dt_cfl = model.compute_stable_dt(state, geometry) * self.cfl_safety
        return jnp.clip(dt_cfl, self.dt_min, self.dt_max)

    def _apply_constraints(self, state, geometry) -> State:
        """Apply div(B)=0 cleaning and boundary conditions."""
        ...
```

**Concrete implementations:**
- `RK4Solver(Solver)`
- `IMEXSolver(Solver)`
- `SemiImplicitSolver(Solver)`

### 3. Simulation - Orchestration

**Absorbs Configuration** with builder pattern.

```python
# jax_frc/simulation/simulation.py

@dataclass
class Phase:
    """A simulation phase with transition condition."""
    name: str
    transition: Transition
    config: dict = field(default_factory=dict)

class SimulationBuilder:
    """Fluent builder for Simulation."""

    def geometry(self, geometry: Geometry) -> "SimulationBuilder": ...
    def model(self, model: Model) -> "SimulationBuilder": ...
    def solver(self, solver: Solver) -> "SimulationBuilder": ...
    def initial_state(self, state: State) -> "SimulationBuilder": ...
    def phases(self, phases: List[Phase]) -> "SimulationBuilder": ...
    def callbacks(self, callbacks: List[Callable]) -> "SimulationBuilder": ...
    def build(self) -> "Simulation": ...

@dataclass
class Simulation:
    """Main simulation orchestrator."""

    geometry: Geometry
    model: Model
    solver: Solver
    state: State
    phases: List[Phase] = field(default_factory=list)
    callbacks: List[Callable] = field(default_factory=list)

    @classmethod
    def builder(cls) -> SimulationBuilder:
        """Start building a Simulation."""
        return SimulationBuilder()

    def step(self) -> State:
        """Advance by one timestep."""
        self.state = self.solver.step(self.state, self.model, self.geometry)
        return self.state

    def run(self, t_end: float) -> State:
        """Run until t_end (simple mode, no phases)."""
        while self.state.time < t_end:
            self.step()
            for cb in self.callbacks:
                cb(self.state)
        return self.state

    def run_phases(self) -> List[PhaseResult]:
        """Run all phases in sequence."""
        ...
```

**Usage:**
```python
sim = Simulation.builder() \
    .geometry(Geometry(nx=64, ny=64, nz=1, ...)) \
    .model(ResistiveMHD(eta=1e-4)) \
    .solver(RK4Solver(cfl_safety=0.5)) \
    .initial_state(State.zeros(64, 64, 1)) \
    .build()

sim.run(t_end=1.0)
```

## Factory Functions

Configuration subclasses become factory functions:

```python
# jax_frc/simulation/presets/magnetic_diffusion.py

def create_magnetic_diffusion(
    nx: int = 64,
    ny: int = 64,
    nz: int = 1,
    extent: float = 1.0,
    eta: float = 1.26e-10,
    model_type: str = "extended_mhd",
) -> Simulation:
    """Create a magnetic diffusion test simulation."""
    geometry = Geometry(...)
    model = _build_model(model_type, eta)
    state = _build_initial_state(geometry, ...)
    solver = RK4Solver()

    return Simulation.builder() \
        .geometry(geometry) \
        .model(model) \
        .solver(solver) \
        .initial_state(state) \
        .build()
```

## Module Structure

```
jax_frc/
├── simulation/
│   ├── __init__.py        # Exports Simulation, SimulationBuilder, Phase
│   ├── simulation.py      # Simulation, SimulationBuilder
│   ├── phase.py           # Phase, Transition, PhaseResult
│   ├── state.py           # State dataclass
│   ├── geometry.py        # Geometry dataclass
│   └── presets/
│       ├── __init__.py
│       ├── magnetic_diffusion.py
│       ├── frozen_flux.py
│       └── frc_merging.py
├── models/
│   ├── __init__.py
│   ├── base.py            # Model ABC
│   ├── resistive_mhd.py
│   ├── extended_mhd.py
│   └── ...
├── solvers/
│   ├── __init__.py
│   ├── base.py            # Solver ABC
│   ├── rk4.py
│   ├── imex.py
│   └── ...
└── ...
```

**Imports:**
```python
from jax_frc.simulation import Simulation, State, Geometry
from jax_frc.simulation.presets import create_magnetic_diffusion
from jax_frc.models import ResistiveMHD
from jax_frc.solvers import RK4Solver
```

## Migration Path

### What Gets Removed
- `NumericalRecipe` - absorbed into `Solver`
- `TimeController` - absorbed into `Solver`
- `AbstractConfiguration` - replaced by factory functions
- `LinearConfiguration` - replaced by `Simulation` with phases
- `jax_frc/core/` directory - contents move to `jax_frc/simulation/`
- `jax_frc/configurations/` directory - becomes `jax_frc/simulation/presets/`

### What Gets Modified
- `Solver` - extended with timestep control, `_apply_constraints()`
- `Model` - `apply_constraints()` removed

### Migration Order
1. Create new `jax_frc/simulation/` structure
2. Extend `Solver` with timestep control and constraint enforcement
3. Move `apply_constraints()` from Model to Solver
4. Implement `SimulationBuilder` and new `Simulation`
5. Convert Configuration subclasses to factory functions
6. Update tests
7. Remove deprecated code

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Solver owns all numerics | Yes | Clean separation: Model = physics, Solver = numerics |
| CFL stays in Model | Yes | Wave speeds are physics knowledge |
| Constraints move to Solver | Yes | Div cleaning algorithms are numerical techniques |
| Simulation absorbs Configuration | Yes | Single entry point, eliminates dual-run confusion |
| Builder pattern for Simulation | Yes | Fluent API, explicit construction |
| Phases are Simulation feature | Yes | Keeps all execution logic in one place |
| Configurations become factory functions | Yes | Simpler, more testable, no inheritance |
| No separate SolverConfig | Yes | Keep it simple, attributes on Solver class |
