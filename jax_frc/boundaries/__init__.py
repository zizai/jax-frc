"""Boundary conditions for JAX-FRC simulation."""

from jax_frc.boundaries.base import BoundaryCondition
from jax_frc.boundaries.conducting import ConductingWall
from jax_frc.boundaries.symmetry import SymmetryAxis
from jax_frc.boundaries.time_dependent import TimeDependentMirrorBC

__all__ = [
    "BoundaryCondition",
    "ConductingWall",
    "SymmetryAxis",
    "TimeDependentMirrorBC",
]
