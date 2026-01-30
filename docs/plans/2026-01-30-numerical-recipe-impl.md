# NumericalRecipe Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce a NumericalRecipe runtime that owns stepping and constraint policy, and add normalization utilities, while keeping existing config and solver APIs backward compatible.

**Architecture:** Add a NumericalRecipe dataclass that composes `Solver` + `TimeController` + divergence strategy. Refactor solvers to expose a constraint-free `advance` path while preserving `step` behavior. Wire `Simulation` to delegate to a recipe when configured. Add a units normalization module and tests.

**Tech Stack:** Python 3.9+, JAX, dataclasses, pytest.

---

### Task 1: Add failing unit tests for NumericalRecipe stepping and divergence handling

**Files:**
- Create: `tests/test_numerical_recipe.py`
- Modify: `tests/test_simulation_integration.py:12-80` (add a recipe integration test)

**Step 1: Write the failing tests (NumericalRecipe unit tests)**

```python
import jax.numpy as jnp
import pytest

from dataclasses import dataclass
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel
from jax_frc.solvers import RK4Solver, TimeController
from jax_frc.solvers.recipe import NumericalRecipe
from jax_frc.operators import divergence_3d
from tests.utils.cartesian import make_geometry


@dataclass(frozen=True)
class DummyModel(PhysicsModel):
    def compute_rhs(self, state, geometry):
        # Zero RHS so only dt/step/constraints matter
        zeros_B = jnp.zeros_like(state.B)
        zeros_E = jnp.zeros_like(state.E)
        zeros_n = jnp.zeros_like(state.n)
        zeros_p = jnp.zeros_like(state.p)
        zeros_psi = jnp.zeros_like(state.psi) if state.psi is not None else None
        return state.replace(B=zeros_B, E=zeros_E, n=zeros_n, p=zeros_p, psi=zeros_psi)

    def compute_stable_dt(self, state, geometry):
        return 0.2

    def apply_constraints(self, state, geometry):
        # Marker for "called once"
        return state.replace(p=state.p + 1.0)


def test_recipe_step_uses_time_controller_and_constraints_once():
    geom = make_geometry(nx=4, ny=2, nz=4)
    model = DummyModel()
    solver = RK4Solver()
    tc = TimeController(cfl_safety=1.0, dt_min=0.0, dt_max=1.0)
    recipe = NumericalRecipe(solver=solver, time_controller=tc, divergence_strategy="none")

    state = State.zeros(geom.nx, geom.ny, geom.nz)
    next_state = recipe.step(state, model, geom)

    assert next_state.step == 1
    assert next_state.time == pytest.approx(0.2)
    assert jnp.allclose(next_state.p, state.p + 1.0)


def test_recipe_divergence_cleaning_reduces_div_b():
    geom = make_geometry(nx=8, ny=2, nz=8)
    model = DummyModel()
    solver = RK4Solver()
    tc = TimeController(cfl_safety=1.0, dt_min=0.0, dt_max=1.0)
    recipe = NumericalRecipe(solver=solver, time_controller=tc, divergence_strategy="clean")

    state = State.zeros(geom.nx, geom.ny, geom.nz)
    # Create a simple divergent field
    B = state.B.at[..., 0].set(geom.x_grid)
    state = state.replace(B=B)

    div_before = jnp.linalg.norm(divergence_3d(state.B, geom))
    next_state = recipe.step(state, model, geom)
    div_after = jnp.linalg.norm(divergence_3d(next_state.B, geom))

    assert div_after <= div_before
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_numerical_recipe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'jax_frc.solvers.recipe'` (or missing symbols).

**Step 3: Add a failing integration test for Simulation + recipe**

```python
def test_simulation_from_config_with_recipe():
    config = {
        "geometry": {
            "nx": 8, "ny": 2, "nz": 8,
            "x_min": -1.0, "x_max": 1.0,
            "y_min": -1.0, "y_max": 1.0,
            "z_min": -1.0, "z_max": 1.0,
            "bc_x": "neumann", "bc_y": "periodic", "bc_z": "neumann",
        },
        "model": {"type": "resistive_mhd", "eta": 1e-4},
        "solver": {"type": "rk4"},
        "time": {"cfl_safety": 0.25, "dt_max": 1e-4},
        "numerics": {"divergence_strategy": "none"},
    }
    sim = Simulation.from_config(config)
    sim.initialize()
    state = sim.step()
    assert state.step == 1
```

**Step 4: Run the integration test to verify it fails**

Run: `python -m pytest tests/test_simulation_integration.py::TestSimulationIntegration::test_simulation_from_config_with_recipe -v`
Expected: FAIL because `numerics` is ignored or `NumericalRecipe` is missing.

**Step 5: Commit**

```bash
git add tests/test_numerical_recipe.py tests/test_simulation_integration.py
git commit -m "test: add numerical recipe tests"
```

---

### Task 2: Refactor solvers to separate advance from constraint application

**Files:**
- Modify: `jax_frc/solvers/base.py:13-78`
- Modify: `jax_frc/solvers/explicit.py:19-183`
- Modify: `jax_frc/solvers/semi_implicit.py:30-227`
- Modify: `jax_frc/solvers/imex.py:42-74`

**Step 1: Update Solver base class with an abstract `advance` method**

```python
class Solver(ABC):
    @abstractmethod
    def advance(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance state by one timestep without applying constraints."""
        raise NotImplementedError

    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance state by one timestep and apply constraints."""
        new_state = self.advance(state, dt, model, geometry)
        return model.apply_constraints(new_state, geometry)
```

**Step 2: Run tests to ensure it fails due to missing `advance` implementations**

Run: `python -m pytest tests/test_simulation_integration.py::TestSimulationIntegration::test_simulation_run_steps -v`
Expected: FAIL with `TypeError: Can't instantiate abstract class ... advance`.

**Step 3: Move each solver's logic into `advance`**

Example (Euler/RK4/SemiLagrangian in `explicit.py`):
```python
class EulerSolver(Solver):
    def advance(...):
        ...
        return new_state
```

Do the same for:
- `RK4Solver.advance`
- `SemiLagrangianSolver.advance`
- `SemiImplicitSolver.advance`
- `HybridSolver.advance`
- `ImexSolver.advance`

**Step 4: Run targeted solver tests**

Run: `python -m pytest tests/test_simulation_integration.py::TestSimulationIntegration::test_simulation_run_steps -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/solvers/base.py jax_frc/solvers/explicit.py jax_frc/solvers/semi_implicit.py jax_frc/solvers/imex.py
git commit -m "refactor: separate solver advance from constraints"
```

---

### Task 3: Implement NumericalRecipe and export it

**Files:**
- Create: `jax_frc/solvers/recipe.py`
- Modify: `jax_frc/solvers/__init__.py:1-18`

**Step 1: Implement NumericalRecipe**

```python
from dataclasses import dataclass
from typing import Literal

from jax_frc.solvers.base import Solver
from jax_frc.solvers.time_controller import TimeController
from jax_frc.solvers.divergence_cleaning import clean_divergence

DivergenceStrategy = Literal["ct", "clean", "none"]

@dataclass(frozen=True)
class NumericalRecipe:
    solver: Solver
    time_controller: TimeController
    divergence_strategy: DivergenceStrategy = "none"
    use_checked_step: bool = False

    def validate(self, model, geometry) -> None:
        if self.divergence_strategy == "ct":
            # CT requires model-level CT paths; leave hard checks minimal for now
            return

    def apply_constraints(self, state, model, geometry):
        state = model.apply_constraints(state, geometry)
        if self.divergence_strategy == "clean":
            state = state.replace(B=clean_divergence(state.B, geometry))
        return state

    def step(self, state, model, geometry):
        self.validate(model, geometry)
        dt = self.time_controller.compute_dt(state, model, geometry)
        if self.use_checked_step:
            next_state = self.solver.step_checked(state, dt, model, geometry)
        else:
            next_state = self.solver.advance(state, dt, model, geometry)
        return self.apply_constraints(next_state, model, geometry)
```

**Step 2: Export NumericalRecipe from solvers package**

Add `NumericalRecipe` to `jax_frc/solvers/__init__.py`.

**Step 3: Run NumericalRecipe tests**

Run: `python -m pytest tests/test_numerical_recipe.py -v`
Expected: PASS

**Step 4: Commit**

```bash
git add jax_frc/solvers/recipe.py jax_frc/solvers/__init__.py
git commit -m "feat: add numerical recipe runtime"
```

---

### Task 4: Wire NumericalRecipe into Simulation and config

**Files:**
- Modify: `jax_frc/core/simulation.py:48-85`
- Modify: `jax_frc/__init__.py:1-57`

**Step 1: Add optional recipe field to Simulation and delegate step**

```python
from jax_frc.solvers.recipe import NumericalRecipe

@dataclass
class Simulation:
    ...
    recipe: Optional[NumericalRecipe] = None

    def step(self) -> State:
        if self.recipe is not None:
            self.state = self.recipe.step(self.state, self.model, self.geometry)
            return self.state
        dt = self.time_controller.compute_dt(self.state, self.model, self.geometry)
        self.state = self.solver.step(self.state, dt, self.model, self.geometry)
        return self.state
```

**Step 2: Teach from_config to build NumericalRecipe from `numerics`**

```python
        numerics_config = config.get("numerics")
        recipe = None
        if numerics_config is not None:
            recipe = NumericalRecipe(
                solver=solver,
                time_controller=time_controller,
                divergence_strategy=numerics_config.get("divergence_strategy", "none"),
                use_checked_step=bool(numerics_config.get("use_checked_step", False)),
            )
```

**Step 3: Export NumericalRecipe in package init (optional)**

Add `from jax_frc.solvers.recipe import NumericalRecipe` and include in `__all__`.

**Step 4: Run integration tests**

Run: `python -m pytest tests/test_simulation_integration.py::TestSimulationIntegration::test_simulation_from_config_with_recipe -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/core/simulation.py jax_frc/__init__.py
git commit -m "feat: wire numerical recipe into simulation config"
```

---

### Task 5: Add normalization utilities and tests

**Files:**
- Create: `jax_frc/units/__init__.py`
- Create: `jax_frc/units/normalization.py`
- Create: `tests/test_normalization.py`

**Step 1: Write failing normalization tests**

```python
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.units.normalization import NormScales, to_dimless_state, to_physical_state, scale_eta_nu


def test_state_round_trip():
    scales = NormScales(L0=1.0, rho0=2.0, B0=3.0)
    state = State.zeros(4, 2, 4)
    state = state.replace(
        n=state.n + 1.0,
        p=state.p + 2.0,
        v=jnp.ones_like(state.B),
    )
    dimless = to_dimless_state(state, scales)
    physical = to_physical_state(dimless, scales)
    assert jnp.allclose(physical.n, state.n)
    assert jnp.allclose(physical.p, state.p)
    assert jnp.allclose(physical.v, state.v)


def test_eta_nu_scaling():
    scales = NormScales(L0=2.0, rho0=1.0, B0=4.0)
    eta_star, nu_star = scale_eta_nu(eta=0.5, nu=0.25, scales=scales)
    v0 = scales.v0
    assert eta_star == pytest.approx(0.5 / (scales.L0 * v0))
    assert nu_star == pytest.approx(0.25 / (scales.L0 * v0))
```

**Step 2: Run tests to confirm failure**

Run: `python -m pytest tests/test_normalization.py -v`
Expected: FAIL with `ModuleNotFoundError: jax_frc.units.normalization`.

**Step 3: Implement normalization module**

```python
from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class NormScales:
    L0: float
    rho0: float
    B0: float

    @property
    def v0(self) -> float:
        return self.B0 / jnp.sqrt(self.rho0)

    @property
    def T0(self) -> float:
        return self.L0 / self.v0

    @property
    def p0(self) -> float:
        return self.rho0 * self.v0**2


def to_dimless_state(state, scales: NormScales):
    return state.replace(
        n=state.n / scales.rho0,
        p=state.p / scales.p0,
        v=state.v / scales.v0 if state.v is not None else state.v,
        B=state.B / scales.B0,
        time=state.time / scales.T0,
    )


def to_physical_state(state, scales: NormScales):
    return state.replace(
        n=state.n * scales.rho0,
        p=state.p * scales.p0,
        v=state.v * scales.v0 if state.v is not None else state.v,
        B=state.B * scales.B0,
        time=state.time * scales.T0,
    )


def scale_eta_nu(eta: float, nu: float, scales: NormScales) -> tuple[float, float]:
    factor = scales.L0 * scales.v0
    return eta / factor, nu / factor
```

**Step 4: Run normalization tests**

Run: `python -m pytest tests/test_normalization.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/units jax_frc/units/normalization.py tests/test_normalization.py
git commit -m "feat: add normalization utilities"
```

---

### Task 6: Full verification pass

**Files:**
- None

**Step 1: Run core fast tests**

Run: `python -m pytest tests/ -k "not slow" -v`
Expected: PASS

**Step 2: Commit any remaining fixes**

```bash
git status
git add -A
git commit -m "test: fix numerical recipe regressions"
```

---

**Plan complete and saved to `docs/plans/2026-01-30-numerical-recipe-impl.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
