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

    @classmethod
    def from_yaml(cls, path: str) -> "Simulation":
        """Create Simulation from YAML config file."""
        from jax_frc.config.loader import load_config
        config = load_config(path)
        return cls.from_config(config)
