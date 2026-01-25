"""Conducting wall boundary conditions."""

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
