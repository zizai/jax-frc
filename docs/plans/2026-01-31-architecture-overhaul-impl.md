# Architecture Overhaul Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate NumericalRecipe + TimeController into Solver, Configuration into Simulation, and isolate physics from numerics.

**Architecture:** Three pillars - Model (pure physics), Solver (pure numerics), Simulation (orchestration with builder pattern).

**Tech Stack:** Python 3.9+, JAX, dataclasses

---

## Task 1: Extend Solver Base Class

**Files:**
- Modify: `jax_frc/solvers/base.py`
- Test: `tests/test_solver_base.py` (new)

**Step 1: Write failing test for new Solver attributes**

```python
# tests/test_solver_base.py
import pytest
from jax_frc.solvers.base import Solver

def test_solver_has_timestep_attributes():
    """Solver should have cfl_safety, dt_min, dt_max attributes."""
    # Can't instantiate ABC directly, but we can check class attributes
    assert hasattr(Solver, 'cfl_safety')
    assert hasattr(Solver, 'dt_min')
    assert hasattr(Solver, 'dt_max')
    assert hasattr(Solver, 'use_checked_step')
    assert hasattr(Solver, 'divergence_cleaning')
```

**Step 2: Run test to verify it fails**

Run: `/home/zsong/jax-frc/.venv/bin/python -m pytest tests/test_solver_base.py -v`
Expected: FAIL with AttributeError

**Step 3: Add attributes to Solver base class**

Edit `jax_frc/solvers/base.py` to add class attributes:

```python
class Solver(ABC):
    """Base class for time integration solvers - owns all numerics."""
    
    # Timestep control (absorbed from TimeController)
    cfl_safety: float = 0.5
    dt_min: float = 1e-12
    dt_max: float = 1e-3
    
    # Numerical options (absorbed from NumericalRecipe)
    use_checked_step: bool = True
    divergence_cleaning: str = "projection"
```

**Step 4: Run test to verify it passes**

Run: `/home/zsong/jax-frc/.venv/bin/python -m pytest tests/test_solver_base.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/solvers/base.py tests/test_solver_base.py
git commit -m "feat(solver): add timestep control attributes to Solver base"
```

---

## Task 2: Add _compute_dt Method to Solver

**Files:**
- Modify: `jax_frc/solvers/base.py`
- Test: `tests/test_solver_base.py`

**Step 1: Write failing test**

```python
def test_solver_compute_dt():
    """Solver._compute_dt should compute timestep from model CFL."""
    from jax_frc.solvers.explicit import RK4Solver
    from jax_frc.models.resistive_mhd import ResistiveMHD
    from jax_frc.simulation.state import State
    from jax_frc.simulation.geometry import Geometry
    
    solver = RK4Solver()
    model = ResistiveMHD(eta=1e-4)
    geometry = Geometry(nx=8, ny=8, nz=1)
    state = State.zeros(8, 8, 1)
    
    dt = solver._compute_dt(state, model, geometry)
    assert dt > 0
    assert dt <= solver.dt_max
    assert dt >= solver.dt_min
```

**Step 2: Run test to verify it fails**

Run: `/home/zsong/jax-frc/.venv/bin/python -m pytest tests/test_solver_base.py::test_solver_compute_dt -v`
Expected: FAIL with AttributeError '_compute_dt'

**Step 3: Implement _compute_dt**

```python
def _compute_dt(self, state: State, model: "PhysicsModel", geometry) -> float:
    """Compute timestep from model CFL and config bounds."""
    import jax.numpy as jnp
    dt_cfl = model.compute_stable_dt(state, geometry) * self.cfl_safety
    return float(jnp.clip(dt_cfl, self.dt_min, self.dt_max))
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add jax_frc/solvers/base.py tests/test_solver_base.py
git commit -m "feat(solver): add _compute_dt method"
```

---

## Task 3: Move apply_constraints from Model to Solver

**Files:**
- Modify: `jax_frc/solvers/base.py`
- Modify: `jax_frc/solvers/divergence_cleaning.py`
- Test: `tests/test_solver_base.py`

**Step 1: Write failing test**

```python
def test_solver_apply_constraints():
    """Solver._apply_constraints should enforce div(B)=0."""
    from jax_frc.solvers.explicit import RK4Solver
    from jax_frc.simulation.state import State
    from jax_frc.simulation.geometry import Geometry
    
    solver = RK4Solver()
    geometry = Geometry(nx=8, ny=8, nz=1)
    state = State.zeros(8, 8, 1)
    
    new_state = solver._apply_constraints(state, geometry)
    assert new_state is not None
    assert new_state.B.shape == state.B.shape
```

**Step 2: Run test to verify it fails**

**Step 3: Implement _apply_constraints in Solver**

```python
def _apply_constraints(self, state: State, geometry) -> State:
    """Apply div(B)=0 cleaning based on divergence_cleaning setting."""
    if self.divergence_cleaning == "none":
        return state
    elif self.divergence_cleaning == "projection":
        from jax_frc.solvers.divergence_cleaning import project_divergence_free
        return project_divergence_free(state, geometry)
    elif self.divergence_cleaning == "hyperbolic":
        from jax_frc.solvers.divergence_cleaning import hyperbolic_cleaning
        return hyperbolic_cleaning(state, geometry)
    return state
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add jax_frc/solvers/base.py tests/test_solver_base.py
git commit -m "feat(solver): add _apply_constraints method"
```

---

## Task 4: Update Solver.step() to Use New Methods

**Files:**
- Modify: `jax_frc/solvers/base.py`
- Test: `tests/test_solver_base.py`

**Step 1: Write failing test**

```python
def test_solver_step_computes_dt_internally():
    """Solver.step() should compute dt internally, not require it as param."""
    from jax_frc.solvers.explicit import RK4Solver
    from jax_frc.models.resistive_mhd import ResistiveMHD
    from jax_frc.simulation.state import State
    from jax_frc.simulation.geometry import Geometry
    
    solver = RK4Solver()
    model = ResistiveMHD(eta=1e-4)
    geometry = Geometry(nx=8, ny=8, nz=1)
    state = State.zeros(8, 8, 1)
    
    # New signature: step(state, model, geometry) - no dt parameter
    new_state = solver.step(state, model, geometry)
    assert new_state.time > state.time
```

**Step 2: Run test to verify it fails**

**Step 3: Update step() method**

```python
def step(self, state: State, model: "PhysicsModel", geometry) -> State:
    """Complete step: compute dt, advance, apply constraints."""
    dt = self._compute_dt(state, model, geometry)
    new_state = self.advance(state, dt, model, geometry)
    new_state = self._apply_constraints(new_state, geometry)
    if self.use_checked_step:
        self._check_state(new_state)
    return new_state
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add jax_frc/solvers/base.py tests/test_solver_base.py
git commit -m "feat(solver): update step() to compute dt internally"
```

---

## Task 5: Create Simulation Module Structure

**Files:**
- Create: `jax_frc/simulation/__init__.py`
- Create: `jax_frc/simulation/simulation.py`
- Move: `jax_frc/core/state.py` -> `jax_frc/simulation/state.py`
- Move: `jax_frc/core/geometry.py` -> `jax_frc/simulation/geometry.py`

**Step 1: Create directory and __init__.py**

```bash
mkdir -p jax_frc/simulation
```

**Step 2: Create simulation/__init__.py**

```python
"""Simulation module - orchestration layer."""
from jax_frc.simulation.state import State
from jax_frc.simulation.geometry import Geometry
from jax_frc.simulation.simulation import Simulation, SimulationBuilder

__all__ = ["State", "Geometry", "Simulation", "SimulationBuilder"]
```

**Step 3: Move state.py and geometry.py**

```bash
cp jax_frc/core/state.py jax_frc/simulation/state.py
cp jax_frc/core/geometry.py jax_frc/simulation/geometry.py
```

**Step 4: Update imports in moved files**

**Step 5: Commit**

```bash
git add jax_frc/simulation/
git commit -m "feat(simulation): create simulation module structure"
```

---

## Task 6: Implement SimulationBuilder

**Files:**
- Create: `jax_frc/simulation/simulation.py`
- Test: `tests/test_simulation_builder.py` (new)

**Step 1: Write failing test**

```python
# tests/test_simulation_builder.py
import pytest
from jax_frc.simulation import Simulation, State, Geometry
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.solvers.explicit import RK4Solver

def test_simulation_builder_basic():
    """SimulationBuilder should create Simulation with fluent API."""
    geometry = Geometry(nx=8, ny=8, nz=1)
    model = ResistiveMHD(eta=1e-4)
    solver = RK4Solver()
    state = State.zeros(8, 8, 1)
    
    sim = Simulation.builder() \
        .geometry(geometry) \
        .model(model) \
        .solver(solver) \
        .initial_state(state) \
        .build()
    
    assert sim.geometry == geometry
    assert sim.model == model
    assert sim.solver == solver
    assert sim.state == state
```

**Step 2: Run test to verify it fails**

**Step 3: Implement SimulationBuilder**

```python
# jax_frc/simulation/simulation.py
from dataclasses import dataclass, field
from typing import Optional, List, Callable

@dataclass
class Simulation:
    """Main simulation orchestrator."""
    geometry: "Geometry"
    model: "Model"
    solver: "Solver"
    state: "State"
    phases: List = field(default_factory=list)
    callbacks: List[Callable] = field(default_factory=list)
    
    @classmethod
    def builder(cls) -> "SimulationBuilder":
        return SimulationBuilder()
    
    def step(self) -> "State":
        self.state = self.solver.step(self.state, self.model, self.geometry)
        return self.state
    
    def run(self, t_end: float) -> "State":
        while self.state.time < t_end:
            self.step()
            for cb in self.callbacks:
                cb(self.state)
        return self.state


class SimulationBuilder:
    """Fluent builder for Simulation."""
    
    def __init__(self):
        self._geometry = None
        self._model = None
        self._solver = None
        self._state = None
        self._phases = []
        self._callbacks = []
    
    def geometry(self, g) -> "SimulationBuilder":
        self._geometry = g
        return self
    
    def model(self, m) -> "SimulationBuilder":
        self._model = m
        return self
    
    def solver(self, s) -> "SimulationBuilder":
        self._solver = s
        return self
    
    def initial_state(self, s) -> "SimulationBuilder":
        self._state = s
        return self
    
    def phases(self, p) -> "SimulationBuilder":
        self._phases = p
        return self
    
    def callbacks(self, c) -> "SimulationBuilder":
        self._callbacks = c
        return self
    
    def build(self) -> Simulation:
        if self._geometry is None:
            raise ValueError("geometry is required")
        if self._model is None:
            raise ValueError("model is required")
        if self._solver is None:
            raise ValueError("solver is required")
        if self._state is None:
            raise ValueError("initial_state is required")
        
        return Simulation(
            geometry=self._geometry,
            model=self._model,
            solver=self._solver,
            state=self._state,
            phases=self._phases,
            callbacks=self._callbacks,
        )
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add jax_frc/simulation/simulation.py tests/test_simulation_builder.py
git commit -m "feat(simulation): implement SimulationBuilder"
```

---

## Task 7: Create Presets Module

**Files:**
- Create: `jax_frc/simulation/presets/__init__.py`
- Create: `jax_frc/simulation/presets/magnetic_diffusion.py`
- Test: `tests/test_presets.py` (new)

**Step 1: Write failing test**

```python
# tests/test_presets.py
def test_create_magnetic_diffusion():
    """create_magnetic_diffusion should return configured Simulation."""
    from jax_frc.simulation.presets import create_magnetic_diffusion
    
    sim = create_magnetic_diffusion(nx=16, ny=16)
    
    assert sim.geometry.nx == 16
    assert sim.geometry.ny == 16
    assert sim.state is not None
```

**Step 2: Run test to verify it fails**

**Step 3: Implement preset factory function**

```python
# jax_frc/simulation/presets/magnetic_diffusion.py
from jax_frc.simulation import Simulation, State, Geometry
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.solvers.explicit import RK4Solver
import jax.numpy as jnp

def create_magnetic_diffusion(
    nx: int = 64,
    ny: int = 64,
    nz: int = 1,
    extent: float = 1.0,
    eta: float = 1.26e-10,
) -> Simulation:
    """Create magnetic diffusion test simulation."""
    geometry = Geometry(
        nx=nx, ny=ny, nz=nz,
        x_min=-extent, x_max=extent,
        y_min=-extent, y_max=extent,
        z_min=-extent, z_max=extent,
        bc_x="neumann", bc_y="neumann", bc_z="neumann",
    )
    
    model = ExtendedMHD(
        eta=eta,
        include_hall=False,
        include_electron_pressure=False,
        evolve_density=False,
        evolve_velocity=False,
        evolve_pressure=False,
    )
    
    solver = RK4Solver()
    
    # Initial Gaussian B_z profile
    x, y = geometry.x_grid, geometry.y_grid
    sigma = 0.1
    B_z = jnp.exp(-(x**2 + y**2) / (2 * sigma**2))
    B = jnp.zeros((nx, ny, nz, 3))
    B = B.at[:, :, :, 2].set(B_z)
    
    state = State.zeros(nx, ny, nz).replace(B=B)
    
    return Simulation.builder() \
        .geometry(geometry) \
        .model(model) \
        .solver(solver) \
        .initial_state(state) \
        .build()
```

**Step 4: Run test to verify it passes**

**Step 5: Commit**

```bash
git add jax_frc/simulation/presets/
git commit -m "feat(presets): add magnetic_diffusion factory function"
```

---

## Task 8: Remove apply_constraints from Model

**Files:**
- Modify: `jax_frc/models/base.py`
- Modify: All model implementations
- Update: Tests

**Step 1: Remove abstract method from PhysicsModel**

Remove `apply_constraints` from `PhysicsModel` ABC.

**Step 2: Update all model implementations**

Remove `apply_constraints` implementations from:
- `ResistiveMHD`
- `ExtendedMHD`
- `HybridKinetic`
- etc.

**Step 3: Update tests that call model.apply_constraints**

Change to call `solver._apply_constraints` instead.

**Step 4: Run full test suite**

**Step 5: Commit**

```bash
git add jax_frc/models/ tests/
git commit -m "refactor(model): remove apply_constraints from Model"
```

---

## Task 9: Deprecate NumericalRecipe and TimeController

**Files:**
- Modify: `jax_frc/solvers/recipe.py`
- Modify: `jax_frc/solvers/time_controller.py`

**Step 1: Add deprecation warnings**

```python
# jax_frc/solvers/recipe.py
import warnings

class NumericalRecipe:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "NumericalRecipe is deprecated. Use Solver directly.",
            DeprecationWarning,
            stacklevel=2
        )
        # ... existing code
```

**Step 2: Update imports to use new Solver API**

**Step 3: Run tests to ensure backward compatibility**

**Step 4: Commit**

```bash
git add jax_frc/solvers/
git commit -m "deprecate: add warnings to NumericalRecipe and TimeController"
```

---

## Task 10: Update Existing Tests

**Files:**
- Modify: `tests/test_numerical_recipe.py`
- Modify: `tests/test_simulation_integration.py`
- Modify: Other affected tests

**Step 1: Update tests to use new API**

**Step 2: Run full test suite**

**Step 3: Fix any failures**

**Step 4: Commit**

```bash
git add tests/
git commit -m "test: update tests for new Solver/Simulation API"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Add timestep attributes to Solver | `solvers/base.py` |
| 2 | Add `_compute_dt` method | `solvers/base.py` |
| 3 | Move `apply_constraints` to Solver | `solvers/base.py` |
| 4 | Update `Solver.step()` | `solvers/base.py` |
| 5 | Create simulation module | `simulation/` |
| 6 | Implement SimulationBuilder | `simulation/simulation.py` |
| 7 | Create presets module | `simulation/presets/` |
| 8 | Remove `apply_constraints` from Model | `models/base.py` |
| 9 | Deprecate old classes | `solvers/recipe.py` |
| 10 | Update tests | `tests/` |
