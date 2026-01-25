"""Conjugate Gradient solver for symmetric positive-definite systems."""

from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class CGResult:
    """Result of Conjugate Gradient solve."""
    x: Array           # Solution
    converged: bool    # Did it converge?
    iterations: int    # Iterations used
    residual: float    # Final |Ax - b|/|b|


def conjugate_gradient(
    operator: Callable[[Array], Array],
    b: Array,
    x0: Optional[Array] = None,
    preconditioner: Optional[Callable[[Array], Array]] = None,
    tol: float = 1e-6,
    max_iter: int = 1000,
) -> CGResult:
    """Solve Ax = b using Conjugate Gradient.

    Args:
        operator: Matrix-free operator A(x) returning A @ x
        b: Right-hand side vector
        x0: Initial guess (defaults to zeros)
        preconditioner: Optional M^{-1}(r) preconditioner
        tol: Convergence tolerance for relative residual
        max_iter: Maximum iterations

    Returns:
        CGResult with solution and convergence info
    """
    raise NotImplementedError("CG solver not yet implemented")
