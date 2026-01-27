"""Divergence cleaning methods for magnetic field evolution.

Numerical errors in field evolution accumulate div(B) != 0, which causes
unphysical parallel forces and energy conservation violations.

This module provides projection-based cleaning for 3D Cartesian coordinates:
    B_clean = B - grad(phi), where laplacian(phi) = div(B)

Uses periodic boundary conditions with Jacobi iteration for the Poisson solve.
"""

import jax.numpy as jnp
from jax import jit, lax
from jax import Array

from jax_frc.core.geometry import Geometry
from jax_frc.operators import divergence_3d, gradient_3d


@jit(static_argnums=(1, 2, 3))
def clean_divergence(B: Array, geometry: Geometry,
                     max_iter: int = 100, tol: float = 1e-6) -> Array:
    """Clean divergence from magnetic field using projection method.

    Solves: laplacian(phi) = div(B)
    Then:   B_clean = B - grad(phi)

    Uses periodic boundary conditions and Jacobi iteration.

    Args:
        B: Magnetic field array of shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry
        max_iter: Maximum iterations for Poisson solver
        tol: Convergence tolerance for Poisson solver (unused in fixed iteration)

    Returns:
        Cleaned magnetic field with same shape
    """
    # Compute divergence of B
    div_B = divergence_3d(B, geometry)

    # Solve laplacian(phi) = div_B using Jacobi iteration
    phi = poisson_solve_jacobi(div_B, geometry, max_iter, tol)

    # Compute gradient of phi
    grad_phi = gradient_3d(phi, geometry)

    # Subtract gradient from B to make it divergence-free
    return B - grad_phi


@jit(static_argnums=(1, 2, 3))
def poisson_solve_jacobi(rhs: Array, geometry: Geometry,
                         max_iter: int = 100, tol: float = 1e-6) -> Array:
    """Solve 3D Poisson equation laplacian(phi) = rhs using Jacobi iteration.

    Uses the 7-point stencil with periodic boundary conditions.
    The discretization is:
        (phi[i+1] - 2*phi[i] + phi[i-1])/dx^2 +
        (phi[j+1] - 2*phi[j] + phi[j-1])/dy^2 +
        (phi[k+1] - 2*phi[k] + phi[k-1])/dz^2 = rhs

    Rearranging for Jacobi update:
        phi_new = (sum of neighbors with coefficients - rhs) / center_coeff

    Args:
        rhs: Right-hand side array of shape (nx, ny, nz)
        geometry: 3D Cartesian geometry
        max_iter: Maximum number of iterations
        tol: Convergence tolerance (unused in this implementation)

    Returns:
        Solution phi with same shape as rhs
    """
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz

    # Coefficients for the 7-point stencil
    cx = 1.0 / dx**2
    cy = 1.0 / dy**2
    cz = 1.0 / dz**2
    center_coeff = 2.0 * (cx + cy + cz)

    def jacobi_step(phi: Array, _) -> tuple[Array, None]:
        """Single Jacobi iteration step."""
        # Get neighbors using periodic wrapping (jnp.roll)
        phi_xp = jnp.roll(phi, -1, axis=0)  # phi[i+1, j, k]
        phi_xm = jnp.roll(phi, 1, axis=0)   # phi[i-1, j, k]
        phi_yp = jnp.roll(phi, -1, axis=1)  # phi[i, j+1, k]
        phi_ym = jnp.roll(phi, 1, axis=1)   # phi[i, j-1, k]
        phi_zp = jnp.roll(phi, -1, axis=2)  # phi[i, j, k+1]
        phi_zm = jnp.roll(phi, 1, axis=2)   # phi[i, j, k-1]

        # Jacobi update
        phi_new = (
            cx * (phi_xp + phi_xm) +
            cy * (phi_yp + phi_ym) +
            cz * (phi_zp + phi_zm) -
            rhs
        ) / center_coeff

        return phi_new, None

    # Initialize phi to zeros
    phi_init = jnp.zeros_like(rhs)

    # Run fixed number of iterations using lax.scan (JAX pattern)
    phi_final, _ = lax.scan(jacobi_step, phi_init, jnp.arange(max_iter))

    return phi_final
