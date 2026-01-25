"""Preconditioners for iterative linear solvers."""

from typing import Callable
import jax.numpy as jnp
from jax import Array


def jacobi_preconditioner(diag: Array) -> Callable[[Array], Array]:
    """Create Jacobi (diagonal) preconditioner.

    The Jacobi preconditioner approximates A^{-1} as diag(A)^{-1}.
    For diffusion operators on uniform grids, this is effective because
    the diagonal dominates.

    Args:
        diag: Diagonal elements of the matrix A (same shape as solution)

    Returns:
        Function M^{-1}(r) = r / diag
    """
    # Avoid division by zero
    safe_diag = jnp.where(jnp.abs(diag) > 1e-14, diag, 1.0)

    def apply(r: Array) -> Array:
        return r / safe_diag

    return apply
