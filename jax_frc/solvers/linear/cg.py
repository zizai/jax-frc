"""Conjugate Gradient solver for symmetric positive-definite systems."""

from dataclasses import dataclass
from typing import Callable, Optional
import jax.numpy as jnp
import jax.lax as lax
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

    Implements the standard CG algorithm for symmetric positive-definite A.
    Uses lax.while_loop for JIT compatibility.

    Args:
        operator: Matrix-free operator A(x) returning A @ x
        b: Right-hand side vector
        x0: Initial guess (defaults to zeros)
        preconditioner: Optional M^{-1}(r) preconditioner
        tol: Convergence tolerance for relative residual |r|/|b|
        max_iter: Maximum iterations

    Returns:
        CGResult with solution and convergence info
    """
    # Initial guess
    if x0 is None:
        x = jnp.zeros_like(b)
    else:
        x = x0

    # Identity preconditioner if none provided
    if preconditioner is None:
        preconditioner = lambda r: r

    # Initial residual r = b - Ax
    r = b - operator(x)

    # Preconditioned residual z = M^{-1} r
    z = preconditioner(r)

    # Initial search direction
    p = z

    # r^T z for update
    rz = jnp.sum(r * z)

    # Norm of b for relative residual
    b_norm = jnp.linalg.norm(b)
    b_norm = jnp.where(b_norm > 1e-14, b_norm, 1.0)  # Avoid div by zero

    # CG iteration state: (x, r, z, p, rz, iteration, converged)
    def cg_body(state):
        x, r, z, p, rz, iteration, _ = state

        # Ap = A @ p
        Ap = operator(p)

        # alpha = r^T z / p^T A p
        pAp = jnp.sum(p * Ap)
        alpha = rz / jnp.where(jnp.abs(pAp) > 1e-14, pAp, 1e-14)

        # Update solution: x = x + alpha * p
        x_new = x + alpha * p

        # Update residual: r = r - alpha * Ap
        r_new = r - alpha * Ap

        # Preconditioned residual
        z_new = preconditioner(r_new)

        # New r^T z
        rz_new = jnp.sum(r_new * z_new)

        # beta = r_new^T z_new / r^T z
        beta = rz_new / jnp.where(jnp.abs(rz) > 1e-14, rz, 1e-14)

        # Update search direction: p = z + beta * p
        p_new = z_new + beta * p

        # Check convergence
        r_norm = jnp.linalg.norm(r_new)
        converged = (r_norm / b_norm) < tol

        return (x_new, r_new, z_new, p_new, rz_new, iteration + 1, converged)

    def cg_cond(state):
        _, _, _, _, _, iteration, converged = state
        return jnp.logical_and(iteration < max_iter, jnp.logical_not(converged))

    # Initial state
    init_state = (x, r, z, p, rz, 0, False)

    # Run CG loop
    final_state = lax.while_loop(cg_cond, cg_body, init_state)

    x_final, r_final, _, _, _, iterations, converged = final_state

    # Compute final residual
    residual = jnp.linalg.norm(r_final) / b_norm

    return CGResult(
        x=x_final,
        converged=converged,
        iterations=iterations,
        residual=residual
    )
