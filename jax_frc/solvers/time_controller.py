"""Adaptive timestep control.

DEPRECATED: TimeController functionality has been absorbed into Solver.
Use Solver.cfl_safety, Solver.dt_min, Solver.dt_max instead.
"""
import warnings
from dataclasses import dataclass
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.base import PhysicsModel

@dataclass
class TimeController:
    """Manages adaptive timestepping.
    
    DEPRECATED: Use Solver directly. Solver now has cfl_safety, dt_min, dt_max
    attributes and _compute_dt() method.
    """

    cfl_safety: float = 0.5
    dt_min: float = 1e-12
    dt_max: float = 1e-3

    def __post_init__(self):
        warnings.warn(
            "TimeController is deprecated. Use Solver directly - it now has "
            "cfl_safety, dt_min, dt_max attributes and _compute_dt() method.",
            DeprecationWarning,
            stacklevel=2
        )

    def compute_dt(self, state: State, model: PhysicsModel, geometry: Geometry) -> float:
        """Compute stable timestep from model constraints."""
        dt_model = model.compute_stable_dt(state, geometry)
        dt = self.cfl_safety * dt_model
        return float(jnp.clip(dt, self.dt_min, self.dt_max))
