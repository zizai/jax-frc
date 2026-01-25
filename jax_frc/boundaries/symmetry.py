"""Symmetry boundary conditions."""

from dataclasses import dataclass
from jax_frc.boundaries.base import BoundaryCondition
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

@dataclass
class SymmetryAxis(BoundaryCondition):
    """Handles r=0 axis with Neumann condition."""

    def apply(self, state: State, geometry: Geometry) -> State:
        psi = state.psi
        psi = psi.at[0, :].set(psi[1, :])
        return state.replace(psi=psi)
