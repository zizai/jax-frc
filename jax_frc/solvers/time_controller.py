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
