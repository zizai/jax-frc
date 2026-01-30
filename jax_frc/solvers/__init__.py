"""Time integration solvers for JAX-FRC simulation."""

from jax_frc.solvers.base import Solver
from jax_frc.solvers.explicit import EulerSolver, RK4Solver
from jax_frc.solvers.semi_implicit import SemiImplicitSolver, HybridSolver
from jax_frc.solvers.time_controller import TimeController
from jax_frc.solvers.divergence_cleaning import clean_divergence, poisson_solve_jacobi
from jax_frc.solvers.recipe import NumericalRecipe

__all__ = [
    "Solver",
    "EulerSolver",
    "RK4Solver",
    "SemiImplicitSolver",
    "HybridSolver",
    "TimeController",
    "clean_divergence",
    "poisson_solve_jacobi",
    "NumericalRecipe",
]
