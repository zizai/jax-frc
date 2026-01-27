"""Equilibrium solvers for JAX-FRC simulation."""

from jax_frc.equilibrium.base import EquilibriumSolver, EquilibriumConstraints
from jax_frc.equilibrium.grad_shafranov import GradShafranovSolver, ForceBalanceSolver
from jax_frc.equilibrium.rigid_rotor import RigidRotorEquilibrium
from jax_frc.equilibrium.initializers import (
    harris_sheet_3d,
    uniform_field_3d,
    flux_rope_3d,
)

__all__ = [
    "EquilibriumSolver",
    "EquilibriumConstraints",
    "GradShafranovSolver",
    "ForceBalanceSolver",
    "RigidRotorEquilibrium",
    "harris_sheet_3d",
    "uniform_field_3d",
    "flux_rope_3d",
]
