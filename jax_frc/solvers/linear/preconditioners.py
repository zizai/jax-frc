"""Preconditioners for iterative linear solvers."""

from typing import Callable
import jax.numpy as jnp
from jax import Array


def jacobi_preconditioner(diag: Array) -> Callable[[Array], Array]:
    """Create Jacobi (diagonal) preconditioner.

    Args:
        diag: Diagonal elements of the matrix A

    Returns:
        Function M^{-1}(r) = r / diag
    """
    raise NotImplementedError("Jacobi preconditioner not yet implemented")
