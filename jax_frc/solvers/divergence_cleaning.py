"""Divergence cleaning methods for magnetic field evolution.

Numerical errors in field evolution accumulate div(B) != 0, which causes
unphysical parallel forces and energy conservation violations.

This module provides projection-based cleaning for 3D Cartesian coordinates:
    B_clean = B - grad(phi), where laplacian(phi) = div(B)

The Poisson solve respects the geometry boundary conditions via ghost-cell
padding (periodic, Dirichlet, or Neumann).
"""

import jax.numpy as jnp
from jax import jit, lax
from jax import Array

from jax_frc.core.geometry import Geometry
from jax_frc.operators import (
    divergence_3d,
    _derivative_with_bc,
)
from jax.scipy.sparse.linalg import cg


@jit(static_argnums=(1, 2, 3))
def clean_divergence(B: Array, geometry: Geometry,
                     max_iter: int = 100, tol: float = 1e-6) -> Array:
    """Clean divergence from magnetic field using projection method.

    Solves: laplacian(phi) = div(B)
    Then:   B_clean = B - grad(phi)

    Uses Jacobi iteration with geometry boundary conditions.

    Args:
        B: Magnetic field array of shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry
        max_iter: Maximum iterations for Poisson solver
        tol: Convergence tolerance for Poisson solver (unused in fixed iteration)

    Returns:
        Cleaned magnetic field with same shape
    """
    # Use the same discrete divergence operator as diagnostics.
    if (
        geometry.bc_x == "periodic"
        and geometry.bc_y == "periodic"
        and geometry.bc_z == "periodic"
    ):
        order = 4
    else:
        order = 2
    # Compute divergence of B
    div_B = divergence_3d(B, geometry, order=order)
    if (
        geometry.bc_x == "neumann"
        or geometry.bc_y == "neumann"
        or geometry.bc_z == "neumann"
    ):
        # Neumann Poisson problems are solvable only for zero-mean RHS.
        div_B = div_B - jnp.mean(div_B)

    # Solve laplacian(phi) = div_B using a consistent discrete operator.
    effective_max_iter = max_iter
    if (
        order == 2
        and (
            geometry.bc_x != "periodic"
            or geometry.bc_y != "periodic"
            or geometry.bc_z != "periodic"
        )
        and max_iter == 100
    ):
        # Non-periodic cases converge slower with the DG operator; use more iters by default.
        effective_max_iter = 200
    phi = poisson_solve_jacobi(div_B, geometry, effective_max_iter, tol, order=order)

    # Compute gradient of phi
    grad_phi = _gradient_with_bc_pad(phi, geometry, order=order)

    # Subtract gradient from B to make it divergence-free
    return B - grad_phi


@jit(static_argnums=(1, 2, 3, 4))
def poisson_solve_jacobi(rhs: Array, geometry: Geometry,
                         max_iter: int = 100, tol: float = 1e-6,
                         order: int = 2) -> Array:
    """Solve 3D Poisson equation laplacian(phi) = rhs using Jacobi iteration.

    Uses the 7-point stencil with geometry boundary conditions.
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

    if order == 4:
        # Sum of squares of 4th-order central difference coefficients.
        # D4 stencil: [1, -8, 0, 8, -1] / (12 dx) -> sum(coeff^2) = 65/72 * 1/dx^2
        coeff_1d = 65.0 / 72.0
    elif order == 2:
        # D2 stencil: [-1, 0, 1] / (2 dx) -> sum(coeff^2) = 1/2 * 1/dx^2
        coeff_1d = 0.5
    else:
        raise ValueError(f"Unsupported order {order}")

    center_coeff = coeff_1d * (1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2)

    def jacobi_step(phi: Array, _) -> tuple[Array, None]:
        """Single Jacobi iteration step."""
        # Jacobi update using the same discrete operators as divergence_3d.
        lap_phi = divergence_3d(
            _gradient_with_bc_pad(phi, geometry, order=order),
            geometry,
            order=order,
        )
        phi_new = phi + (lap_phi - rhs) / center_coeff

        # Neumann problems are defined up to a constant; fix mean each step.
        if (
            geometry.bc_x == "neumann"
            or geometry.bc_y == "neumann"
            or geometry.bc_z == "neumann"
        ):
            phi_new = phi_new - jnp.mean(phi_new)

        return phi_new, None

    # Initialize phi to zeros
    phi_init = jnp.zeros_like(rhs)

    # Run fixed number of iterations using lax.scan (JAX pattern)
    phi_final, _ = lax.scan(jacobi_step, phi_init, jnp.arange(max_iter))

    return phi_final


@jit(static_argnums=(1, 2, 3, 4))
def poisson_solve_cg(rhs: Array, geometry: Geometry,
                     max_iter: int = 100, tol: float = 1e-6,
                     order: int = 2) -> Array:
    """Solve 3D Poisson equation using conjugate gradient."""
    def matvec(phi: Array) -> Array:
        grad_phi = _gradient_with_bc_pad(phi, geometry, order=order)
        return divergence_3d(grad_phi, geometry, order=order)

    # Use -matvec so the operator is symmetric positive definite for CG.
    def matvec_spd(phi: Array) -> Array:
        return -matvec(phi)

    phi, _ = cg(matvec_spd, -rhs, maxiter=max_iter, tol=tol)
    return phi


@jit(static_argnums=(1, 2))
def _gradient_with_bc_pad(f: Array, geometry: Geometry, order: int = 2) -> Array:
    """Gradient using ghost-cell padding to match divergence_3d discretization."""
    df_dx = _derivative_with_bc(f, geometry.dx, 0, geometry.bc_x, order=order)
    df_dy = _derivative_with_bc(f, geometry.dy, 1, geometry.bc_y, order=order)
    df_dz = _derivative_with_bc(f, geometry.dz, 2, geometry.bc_z, order=order)

    return jnp.stack([df_dx, df_dy, df_dz], axis=-1)
