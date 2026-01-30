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
from jax_frc.solvers.recipe import NumericalRecipe

@dataclass
class Simulation:
    """Main simulation class that orchestrates all components."""

    geometry: Geometry
    model: PhysicsModel
    solver: Solver
    time_controller: TimeController
    recipe: Optional[NumericalRecipe] = None

    state: Optional[State] = None

    def initialize(self, initial_state: Optional[State] = None,
                   psi_init: Optional[Callable] = None,
                   B_init: Optional[Callable] = None) -> None:
        """Initialize simulation state."""
        if initial_state is not None:
            self.state = initial_state
            return

        state = State.zeros(self.geometry.nx, self.geometry.ny, self.geometry.nz)

        if B_init is not None:
            B = B_init(self.geometry.x_grid, self.geometry.y_grid, self.geometry.z_grid)
            self.state = state.replace(B=B)
            return

        if psi_init is not None:
            psi = psi_init(self.geometry.x_grid[:, 0, :], self.geometry.z_grid[:, 0, :])
            B = jnp.zeros((self.geometry.nx, self.geometry.ny, self.geometry.nz, 3))
            B = B.at[:, :, :, 2].set(jnp.repeat(psi[:, None, :], self.geometry.ny, axis=1))
            self.state = state.replace(B=B)
            return

        self.state = state

    def step(self) -> State:
        """Advance simulation by one timestep."""
        if self.recipe is not None:
            self.state = self.recipe.step(self.state, self.model, self.geometry)
            return self.state
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

        # Convert time config values to floats (YAML may load them as strings)
        time_config = config.get("time", {})
        time_kwargs = {k: float(v) for k, v in time_config.items()}
        time_controller = TimeController(**time_kwargs)

        numerics_config = config.get("numerics")
        recipe = None
        if numerics_config is not None:
            recipe = NumericalRecipe(
                solver=solver,
                time_controller=time_controller,
                divergence_strategy=numerics_config.get("divergence_strategy", "none"),
                use_checked_step=bool(numerics_config.get("use_checked_step", False)),
            )

        return cls(
            geometry=geometry,
            model=model,
            solver=solver,
            time_controller=time_controller,
            recipe=recipe,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Simulation":
        """Create Simulation from YAML config file."""
        from jax_frc.config.loader import load_config
        config = load_config(path)
        return cls.from_config(config)
