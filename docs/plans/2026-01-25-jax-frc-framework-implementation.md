# JAX-FRC Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the OOP JAX-FRC simulation framework as designed.

**Architecture:** Modular component-based design with swappable physics plugins.

**Tech Stack:** JAX, Python 3.10+, PyYAML, h5py

---

## Task 1: Project Structure and Core Base Classes

**Files:**
- Create: `jax_frc/__init__.py`
- Create: `jax_frc/core/__init__.py`
- Create: `jax_frc/core/geometry.py`
- Create: `jax_frc/core/state.py`

**Step 1: Create package structure**

```bash
mkdir -p jax_frc/core jax_frc/models jax_frc/solvers jax_frc/boundaries
mkdir -p jax_frc/transport jax_frc/sources jax_frc/equilibrium jax_frc/diagnostics
mkdir -p jax_frc/config
```

**Step 2: Write jax_frc/__init__.py**

```python
"""JAX-FRC: GPU-accelerated FRC plasma simulation framework."""

__version__ = "0.1.0"

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State, ParticleState
from jax_frc.core.simulation import Simulation

__all__ = ["Geometry", "State", "ParticleState", "Simulation"]
```

**Step 3: Write jax_frc/core/geometry.py**

```python
"""Computational geometry and coordinate systems."""

from dataclasses import dataclass
from typing import Literal
import jax.numpy as jnp
from jax import Array

@dataclass(frozen=True)
class Geometry:
    """Defines the computational domain and coordinate system."""

    coord_system: Literal["cylindrical", "cartesian"]
    nr: int
    nz: int
    r_min: float
    r_max: float
    z_min: float
    z_max: float

    def __post_init__(self):
        # Validate inputs
        if self.r_min <= 0 and self.coord_system == "cylindrical":
            raise ValueError("r_min must be > 0 for cylindrical coordinates")
        if self.nr < 2 or self.nz < 2:
            raise ValueError("Grid must have at least 2 points in each dimension")

    @property
    def r(self) -> Array:
        """1D array of radial coordinates."""
        return jnp.linspace(self.r_min, self.r_max, self.nr)

    @property
    def z(self) -> Array:
        """1D array of axial coordinates."""
        return jnp.linspace(self.z_min, self.z_max, self.nz)

    @property
    def dr(self) -> float:
        """Radial grid spacing."""
        return (self.r_max - self.r_min) / (self.nr - 1)

    @property
    def dz(self) -> float:
        """Axial grid spacing."""
        return (self.z_max - self.z_min) / (self.nz - 1)

    @property
    def r_grid(self) -> Array:
        """2D array of radial coordinates (nr, nz)."""
        return self.r[:, None] * jnp.ones((1, self.nz))

    @property
    def z_grid(self) -> Array:
        """2D array of axial coordinates (nr, nz)."""
        return jnp.ones((self.nr, 1)) * self.z[None, :]

    @property
    def cell_volumes(self) -> Array:
        """2D array of cell volumes including 2πr factor for cylindrical."""
        if self.coord_system == "cylindrical":
            return 2 * jnp.pi * self.r_grid * self.dr * self.dz
        else:
            return jnp.ones((self.nr, self.nz)) * self.dr * self.dz

    @classmethod
    def from_config(cls, config: dict) -> "Geometry":
        """Create Geometry from configuration dictionary."""
        return cls(**config)
```

**Step 4: Write jax_frc/core/state.py**

```python
"""Simulation state containers."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jax import Array
import jax

@dataclass(frozen=True)
class ParticleState:
    """State container for kinetic particles."""

    x: Array          # Positions (n_particles, 3)
    v: Array          # Velocities (n_particles, 3)
    w: Array          # Delta-f weights (n_particles,)
    species: str      # "ion", "beam", etc.

    @property
    def n_particles(self) -> int:
        return self.x.shape[0]

@dataclass(frozen=True)
class State:
    """Complete simulation state at a single time."""

    # Scalar fields (nr, nz)
    psi: Array        # Poloidal flux function
    n: Array          # Number density
    p: Array          # Pressure

    # Vector fields (nr, nz, 3)
    B: Array          # Magnetic field
    E: Array          # Electric field
    v: Array          # Fluid velocity

    # Particles (optional, for hybrid)
    particles: Optional[ParticleState]

    # Metadata
    time: float
    step: int

    @classmethod
    def zeros(cls, nr: int, nz: int, with_particles: bool = False,
              n_particles: int = 0) -> "State":
        """Create a zero-initialized state."""
        particles = None
        if with_particles and n_particles > 0:
            particles = ParticleState(
                x=jnp.zeros((n_particles, 3)),
                v=jnp.zeros((n_particles, 3)),
                w=jnp.zeros(n_particles),
                species="ion"
            )

        return cls(
            psi=jnp.zeros((nr, nz)),
            n=jnp.zeros((nr, nz)),
            p=jnp.zeros((nr, nz)),
            B=jnp.zeros((nr, nz, 3)),
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=particles,
            time=0.0,
            step=0
        )

    def replace(self, **kwargs) -> "State":
        """Return new State with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)

# Register State as a JAX pytree for JIT compatibility
def _state_flatten(state):
    children = (state.psi, state.n, state.p, state.B, state.E, state.v,
                state.particles, state.time, state.step)
    aux_data = None
    return children, aux_data

def _state_unflatten(aux_data, children):
    psi, n, p, B, E, v, particles, time, step = children
    return State(psi=psi, n=n, p=p, B=B, E=E, v=v,
                 particles=particles, time=time, step=step)

jax.tree_util.register_pytree_node(State, _state_flatten, _state_unflatten)
```

**Step 5: Run test to verify imports**

```bash
cd jax_frc && python -c "from jax_frc import Geometry, State; print('Task 1: OK')"
```

**Step 6: Commit**

```bash
git add jax_frc/
git commit -m "feat: add core Geometry and State classes"
```

---

## Task 2: Physics Model Base Class and Resistive MHD

**Files:**
- Create: `jax_frc/models/__init__.py`
- Create: `jax_frc/models/base.py`
- Create: `jax_frc/models/resistive_mhd.py`
- Create: `jax_frc/models/resistivity.py`

**Step 1: Write jax_frc/models/base.py**

```python
"""Abstract base class for physics models."""

from abc import ABC, abstractmethod
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

class PhysicsModel(ABC):
    """Base class for all physics models."""

    @abstractmethod
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for all evolved quantities."""
        pass

    @abstractmethod
    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Return CFL-stable timestep for this model."""
        pass

    @abstractmethod
    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Enforce physical constraints (e.g., div(B)=0)."""
        pass

    @classmethod
    def create(cls, config: dict) -> "PhysicsModel":
        """Factory method to create model from config."""
        model_type = config.get("type", "resistive_mhd")
        if model_type == "resistive_mhd":
            from jax_frc.models.resistive_mhd import ResistiveMHD
            return ResistiveMHD.from_config(config)
        elif model_type == "extended_mhd":
            from jax_frc.models.extended_mhd import ExtendedMHD
            return ExtendedMHD.from_config(config)
        elif model_type == "hybrid_kinetic":
            from jax_frc.models.hybrid_kinetic import HybridKinetic
            return HybridKinetic.from_config(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
```

**Step 2: Write jax_frc/models/resistivity.py**

```python
"""Resistivity models for MHD simulations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array

class ResistivityModel(ABC):
    """Base class for resistivity models."""

    @abstractmethod
    def compute(self, j_phi: Array, **kwargs) -> Array:
        """Compute resistivity field."""
        pass

@dataclass
class SpitzerResistivity(ResistivityModel):
    """Classical Spitzer resistivity."""
    eta_0: float = 1e-6

    def compute(self, j_phi: Array, **kwargs) -> Array:
        return jnp.full_like(j_phi, self.eta_0)

@dataclass
class ChoduraResistivity(ResistivityModel):
    """Anomalous resistivity for reconnection."""
    eta_0: float = 1e-6
    eta_anom: float = 1e-3
    threshold: float = 1e4

    def compute(self, j_phi: Array, **kwargs) -> Array:
        j_mag = jnp.abs(j_phi)
        anomalous_factor = 0.5 * (1 + jnp.tanh((j_mag - self.threshold) / (self.threshold * 0.1)))
        return self.eta_0 + self.eta_anom * anomalous_factor
```

**Step 3: Write jax_frc/models/resistive_mhd.py**

```python
"""Resistive MHD physics model."""

from dataclasses import dataclass
from typing import Union
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.models.resistivity import ResistivityModel, SpitzerResistivity, ChoduraResistivity
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

MU0 = 1.2566e-6

@dataclass
class ResistiveMHD(PhysicsModel):
    """Single-fluid resistive MHD model.

    Solves: ∂ψ/∂t + v·∇ψ = (η/μ₀)Δ*ψ
    """

    resistivity: ResistivityModel

    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute dψ/dt from Grad-Shafranov evolution."""
        psi = state.psi
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Compute Δ*ψ
        delta_star_psi = self._laplace_star(psi, dr, dz, r)

        # Compute j_phi = -Δ*ψ / (μ₀ r)
        j_phi = -delta_star_psi / (MU0 * r)

        # Get resistivity
        eta = self.resistivity.compute(j_phi)

        # Diffusion: (η/μ₀)Δ*ψ
        d_psi = (eta / MU0) * delta_star_psi

        # Advection: -v·∇ψ (if velocity present)
        v_r = state.v[:, :, 0]
        v_z = state.v[:, :, 2]
        if jnp.any(v_r != 0) or jnp.any(v_z != 0):
            dpsi_dr = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dr)
            dpsi_dz = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) / (2 * dz)
            d_psi = d_psi - (v_r * dpsi_dr + v_z * dpsi_dz)

        # Return state with d_psi as the RHS (stored in psi temporarily)
        return state.replace(psi=d_psi)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Diffusion CFL: dt < dx² / (4D) where D = η_max/μ₀."""
        # Get maximum resistivity
        j_phi = self._compute_j_phi(state.psi, geometry)
        eta = self.resistivity.compute(j_phi)
        eta_max = jnp.max(eta)

        D = eta_max / MU0
        dx_min = jnp.minimum(geometry.dr, geometry.dz)
        return 0.25 * dx_min**2 / D

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply boundary conditions for conducting wall."""
        psi = state.psi

        # Inner boundary: Neumann (extrapolate)
        psi = psi.at[0, :].set(psi[1, :])
        # Outer boundaries: Dirichlet (psi = 0)
        psi = psi.at[-1, :].set(0)
        psi = psi.at[:, 0].set(0)
        psi = psi.at[:, -1].set(0)

        return state.replace(psi=psi)

    def _laplace_star(self, psi, dr, dz, r):
        """Compute Δ*ψ = ∂²ψ/∂r² - (1/r)∂ψ/∂r + ∂²ψ/∂z²."""
        psi_rr = (jnp.roll(psi, -1, axis=0) - 2*psi + jnp.roll(psi, 1, axis=0)) / dr**2
        psi_zz = (jnp.roll(psi, -1, axis=1) - 2*psi + jnp.roll(psi, 1, axis=1)) / dz**2
        psi_r = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dr)
        return psi_rr - (1.0 / r) * psi_r + psi_zz

    def _compute_j_phi(self, psi, geometry):
        """Compute toroidal current density."""
        delta_star = self._laplace_star(psi, geometry.dr, geometry.dz, geometry.r_grid)
        return -delta_star / (MU0 * geometry.r_grid)

    @classmethod
    def from_config(cls, config: dict) -> "ResistiveMHD":
        """Create from configuration dictionary."""
        res_config = config.get("resistivity", {"type": "spitzer"})
        res_type = res_config.get("type", "spitzer")

        if res_type == "spitzer":
            resistivity = SpitzerResistivity(eta_0=res_config.get("eta_0", 1e-6))
        elif res_type == "chodura":
            resistivity = ChoduraResistivity(
                eta_0=res_config.get("eta_0", 1e-6),
                eta_anom=res_config.get("eta_anom", 1e-3),
                threshold=res_config.get("threshold", 1e4)
            )
        else:
            raise ValueError(f"Unknown resistivity type: {res_type}")

        return cls(resistivity=resistivity)
```

**Step 4: Run test**

```bash
python -c "
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
import jax.numpy as jnp

geom = Geometry('cylindrical', 32, 64, 0.01, 1.0, -1.0, 1.0)
model = ResistiveMHD.from_config({'resistivity': {'type': 'chodura'}})
state = State.zeros(32, 64)
state = state.replace(psi=(1 - geom.r_grid**2) * jnp.exp(-geom.z_grid**2))
dt = model.compute_stable_dt(state, geom)
print(f'Task 2: OK, stable dt = {dt:.2e}')
"
```

**Step 5: Commit**

```bash
git add jax_frc/models/
git commit -m "feat: add PhysicsModel base and ResistiveMHD implementation"
```

---

## Task 3: Solver Infrastructure

**Files:**
- Create: `jax_frc/solvers/__init__.py`
- Create: `jax_frc/solvers/base.py`
- Create: `jax_frc/solvers/explicit.py`
- Create: `jax_frc/solvers/time_controller.py`

**Step 1: Write jax_frc/solvers/base.py**

```python
"""Abstract base class for time integrators."""

from abc import ABC, abstractmethod
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel

class Solver(ABC):
    """Base class for time integration solvers."""

    @abstractmethod
    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        """Advance state by one timestep."""
        pass

    @classmethod
    def create(cls, config: dict) -> "Solver":
        """Factory method to create solver from config."""
        solver_type = config.get("type", "euler")
        if solver_type == "euler":
            from jax_frc.solvers.explicit import EulerSolver
            return EulerSolver()
        elif solver_type == "rk4":
            from jax_frc.solvers.explicit import RK4Solver
            return RK4Solver()
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")
```

**Step 2: Write jax_frc/solvers/explicit.py**

```python
"""Explicit time integration schemes."""

from dataclasses import dataclass
import jax.numpy as jnp
from jax_frc.solvers.base import Solver
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel

@dataclass
class EulerSolver(Solver):
    """Simple forward Euler integration."""

    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        rhs = model.compute_rhs(state, geometry)
        new_psi = state.psi + dt * rhs.psi
        new_state = state.replace(psi=new_psi, time=state.time + dt, step=state.step + 1)
        return model.apply_constraints(new_state, geometry)

@dataclass
class RK4Solver(Solver):
    """4th-order Runge-Kutta integration."""

    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        # k1
        k1 = model.compute_rhs(state, geometry)

        # k2
        state_k2 = state.replace(psi=state.psi + 0.5*dt*k1.psi)
        k2 = model.compute_rhs(state_k2, geometry)

        # k3
        state_k3 = state.replace(psi=state.psi + 0.5*dt*k2.psi)
        k3 = model.compute_rhs(state_k3, geometry)

        # k4
        state_k4 = state.replace(psi=state.psi + dt*k3.psi)
        k4 = model.compute_rhs(state_k4, geometry)

        # Combine
        new_psi = state.psi + (dt/6) * (k1.psi + 2*k2.psi + 2*k3.psi + k4.psi)
        new_state = state.replace(psi=new_psi, time=state.time + dt, step=state.step + 1)
        return model.apply_constraints(new_state, geometry)
```

**Step 3: Write jax_frc/solvers/time_controller.py**

```python
"""Adaptive timestep control."""

from dataclasses import dataclass
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.base import PhysicsModel

@dataclass
class TimeController:
    """Manages adaptive timestepping."""

    cfl_safety: float = 0.5
    dt_min: float = 1e-12
    dt_max: float = 1e-3

    def compute_dt(self, state: State, model: PhysicsModel, geometry: Geometry) -> float:
        """Compute stable timestep from model constraints."""
        dt_model = model.compute_stable_dt(state, geometry)
        dt = self.cfl_safety * dt_model
        return float(jnp.clip(dt, self.dt_min, self.dt_max))
```

**Step 4: Run test**

```bash
python -c "
from jax_frc.solvers.explicit import EulerSolver, RK4Solver
from jax_frc.solvers.time_controller import TimeController
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
import jax.numpy as jnp

geom = Geometry('cylindrical', 32, 64, 0.01, 1.0, -1.0, 1.0)
model = ResistiveMHD.from_config({'resistivity': {'type': 'spitzer', 'eta_0': 1e-4}})
state = State.zeros(32, 64)
state = state.replace(psi=(1 - geom.r_grid**2) * jnp.exp(-geom.z_grid**2))

tc = TimeController(cfl_safety=0.25)
dt = tc.compute_dt(state, model, geom)

solver = RK4Solver()
new_state = solver.step(state, dt, model, geom)
print(f'Task 3: OK, stepped to t={new_state.time:.2e}')
"
```

**Step 5: Commit**

```bash
git add jax_frc/solvers/
git commit -m "feat: add Solver base class and explicit integrators"
```

---

## Task 4: Simulation Orchestrator

**Files:**
- Create: `jax_frc/core/simulation.py`

**Step 1: Write jax_frc/core/simulation.py**

```python
"""Main simulation orchestrator."""

from dataclasses import dataclass, field
from typing import Optional, List, Callable
import jax.numpy as jnp
from jax import lax

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel
from jax_frc.solvers.base import Solver
from jax_frc.solvers.time_controller import TimeController

@dataclass
class Simulation:
    """Main simulation class that orchestrates all components."""

    geometry: Geometry
    model: PhysicsModel
    solver: Solver
    time_controller: TimeController

    state: Optional[State] = None

    def initialize(self, initial_state: Optional[State] = None,
                   psi_init: Optional[Callable] = None) -> None:
        """Initialize simulation state."""
        if initial_state is not None:
            self.state = initial_state
        elif psi_init is not None:
            state = State.zeros(self.geometry.nr, self.geometry.nz)
            psi = psi_init(self.geometry.r_grid, self.geometry.z_grid)
            self.state = state.replace(psi=psi)
        else:
            self.state = State.zeros(self.geometry.nr, self.geometry.nz)

    def step(self) -> State:
        """Advance simulation by one timestep."""
        dt = self.time_controller.compute_dt(self.state, self.model, self.geometry)
        self.state = self.solver.step(self.state, dt, self.model, self.geometry)
        return self.state

    def run(self, t_end: float, callback: Optional[Callable] = None) -> State:
        """Run simulation until t_end."""
        while self.state.time < t_end:
            self.step()
            if callback is not None:
                callback(self.state)
        return self.state

    def run_steps(self, n_steps: int) -> State:
        """Run simulation for fixed number of steps."""
        for _ in range(n_steps):
            self.step()
        return self.state

    @classmethod
    def from_config(cls, config: dict) -> "Simulation":
        """Create Simulation from configuration dictionary."""
        geometry = Geometry.from_config(config["geometry"])
        model = PhysicsModel.create(config.get("model", {"type": "resistive_mhd"}))
        solver = Solver.create(config.get("solver", {"type": "euler"}))
        time_controller = TimeController(**config.get("time", {}))

        return cls(
            geometry=geometry,
            model=model,
            solver=solver,
            time_controller=time_controller
        )
```

**Step 2: Run test**

```bash
python -c "
from jax_frc.core.simulation import Simulation
import jax.numpy as jnp

config = {
    'geometry': {
        'coord_system': 'cylindrical',
        'nr': 32, 'nz': 64,
        'r_min': 0.01, 'r_max': 1.0,
        'z_min': -1.0, 'z_max': 1.0
    },
    'model': {'type': 'resistive_mhd', 'resistivity': {'type': 'spitzer', 'eta_0': 1e-4}},
    'solver': {'type': 'rk4'},
    'time': {'cfl_safety': 0.25, 'dt_max': 1e-4}
}

sim = Simulation.from_config(config)
sim.initialize(psi_init=lambda r, z: (1 - r**2) * jnp.exp(-z**2))
final = sim.run_steps(10)
print(f'Task 4: OK, ran 10 steps to t={final.time:.2e}, psi_max={jnp.max(final.psi):.4f}')
"
```

**Step 3: Commit**

```bash
git add jax_frc/core/simulation.py
git commit -m "feat: add Simulation orchestrator class"
```

---

## Task 5: YAML Configuration System

**Files:**
- Create: `jax_frc/config/__init__.py`
- Create: `jax_frc/config/loader.py`
- Create: `configs/example_frc.yaml`

**Step 1: Write jax_frc/config/loader.py**

```python
"""Configuration loading and validation."""

from pathlib import Path
from typing import Union
import yaml

def load_config(path: Union[str, Path]) -> dict:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    return config

def save_config(config: dict, path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
```

**Step 2: Write configs/example_frc.yaml**

```yaml
# Example FRC simulation configuration

geometry:
  coord_system: cylindrical
  nr: 64
  nz: 128
  r_min: 0.01
  r_max: 0.4
  z_min: -1.0
  z_max: 1.0

model:
  type: resistive_mhd
  resistivity:
    type: chodura
    eta_0: 1.0e-6
    eta_anom: 1.0e-3
    threshold: 1.0e4

solver:
  type: rk4

time:
  cfl_safety: 0.25
  dt_min: 1.0e-12
  dt_max: 1.0e-4
```

**Step 3: Update Simulation to load from YAML**

Add to `jax_frc/core/simulation.py`:

```python
@classmethod
def from_yaml(cls, path: str) -> "Simulation":
    """Create Simulation from YAML config file."""
    from jax_frc.config.loader import load_config
    config = load_config(path)
    return cls.from_config(config)
```

**Step 4: Run test**

```bash
python -c "
from jax_frc.core.simulation import Simulation
import jax.numpy as jnp

sim = Simulation.from_yaml('configs/example_frc.yaml')
sim.initialize(psi_init=lambda r, z: (1 - r**2) * jnp.exp(-z**2))
final = sim.run_steps(5)
print(f'Task 5: OK, loaded from YAML, t={final.time:.2e}')
"
```

**Step 5: Commit**

```bash
git add jax_frc/config/ configs/
git commit -m "feat: add YAML configuration system"
```

---

## Task 6: Boundary Condition System

**Files:**
- Create: `jax_frc/boundaries/__init__.py`
- Create: `jax_frc/boundaries/base.py`
- Create: `jax_frc/boundaries/conducting.py`
- Create: `jax_frc/boundaries/symmetry.py`

**Step 1: Write boundary base and implementations**

```python
# jax_frc/boundaries/base.py
from abc import ABC, abstractmethod
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

class BoundaryCondition(ABC):
    """Base class for boundary conditions."""

    @abstractmethod
    def apply(self, state: State, geometry: Geometry) -> State:
        """Apply boundary condition to state."""
        pass

# jax_frc/boundaries/conducting.py
from dataclasses import dataclass
from jax_frc.boundaries.base import BoundaryCondition
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

@dataclass
class ConductingWall(BoundaryCondition):
    """Perfect conductor: psi = 0 at wall."""
    location: str  # "r_max", "z_min", "z_max"

    def apply(self, state: State, geometry: Geometry) -> State:
        psi = state.psi
        if self.location == "r_max":
            psi = psi.at[-1, :].set(0)
        elif self.location == "z_min":
            psi = psi.at[:, 0].set(0)
        elif self.location == "z_max":
            psi = psi.at[:, -1].set(0)
        return state.replace(psi=psi)

# jax_frc/boundaries/symmetry.py
@dataclass
class SymmetryAxis(BoundaryCondition):
    """Handles r=0 axis with Neumann condition."""

    def apply(self, state: State, geometry: Geometry) -> State:
        psi = state.psi
        psi = psi.at[0, :].set(psi[1, :])
        return state.replace(psi=psi)
```

**Step 2: Run test and commit**

```bash
git add jax_frc/boundaries/
git commit -m "feat: add boundary condition system"
```

---

## Task 7: Extended MHD Model

**Files:**
- Create: `jax_frc/models/extended_mhd.py`

Implement Extended MHD with Hall term and electron pressure following the same pattern as ResistiveMHD, including:
- Hall term: (J×B)/(ne)
- Electron pressure: -∇pₑ/(ne)
- Whistler CFL computation
- Halo density model for vacuum regions

**Commit:**
```bash
git commit -m "feat: add ExtendedMHD model with Hall physics"
```

---

## Task 8: Hybrid Kinetic Model

**Files:**
- Create: `jax_frc/models/hybrid_kinetic.py`
- Create: `jax_frc/models/particle_pusher.py`

Implement Hybrid model with:
- Boris particle pusher
- Delta-f weight evolution
- Field interpolation to particles
- Current deposition from particles

**Commit:**
```bash
git commit -m "feat: add HybridKinetic model with delta-f PIC"
```

---

## Task 9: Equilibrium Solvers

**Files:**
- Create: `jax_frc/equilibrium/__init__.py`
- Create: `jax_frc/equilibrium/base.py`
- Create: `jax_frc/equilibrium/grad_shafranov.py`
- Create: `jax_frc/equilibrium/rigid_rotor.py`

Implement:
- Grad-Shafranov iterative solver
- Rigid rotor analytic equilibrium
- Equilibrium constraints dataclass

**Commit:**
```bash
git commit -m "feat: add equilibrium solvers including Grad-Shafranov"
```

---

## Task 10: Diagnostics System

**Files:**
- Create: `jax_frc/diagnostics/__init__.py`
- Create: `jax_frc/diagnostics/probes.py`
- Create: `jax_frc/diagnostics/output.py`

Implement:
- Probe base class and common probes (flux, energy, beta)
- HDF5 checkpoint saving
- CSV time history export

**Commit:**
```bash
git commit -m "feat: add diagnostics and output system"
```

---

## Task 11: Integration Tests

**Files:**
- Create: `tests/test_simulation.py`
- Create: `tests/test_models.py`
- Create: `tests/test_equilibrium.py`

Write comprehensive tests validating:
- Resistive decay rate matches analytic τ_R
- Conservation properties
- Equilibrium solver convergence

**Commit:**
```bash
git commit -m "test: add integration tests for simulation framework"
```

---

## Task 12: Migration Script and Documentation

**Files:**
- Create: `scripts/migrate_old_sims.py`
- Update: `README.md`

Create script to run old-style simulations using new framework for validation. Update documentation with usage examples.

**Commit:**
```bash
git commit -m "docs: add migration script and updated documentation"
```
