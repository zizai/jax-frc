"""Divergence cleaning methods for magnetic field evolution.

Numerical errors in field evolution accumulate div(B) != 0, which causes
unphysical parallel forces and energy conservation violations.

This module provides projection-based cleaning:
    B_clean = B - grad(phi), where laplacian(phi) = div(B)

Note: Due to boundary condition handling, cleaning is most effective in the
interior of the domain. Near boundaries, some divergence error may remain.
"""

import jax.numpy as jnp
from jax import jit
from typing import Tuple

from jax_frc.core.geometry import Geometry
from jax_frc.operators import divergence_cylindrical, gradient_r, gradient_z


def clean_divergence_b(B: jnp.ndarray, geometry: Geometry,
                       tol: float = 1e-10, maxiter: int = 10000) -> jnp.ndarray:
    """Clean divergence from magnetic field using projection method.

    Solves: laplacian(phi) = div(B) in cylindrical coordinates
    Then:   B_clean = B - grad(phi)

    Uses iterative Jacobi solver with Dirichlet BCs (phi=0 on boundaries).

    Note: Cleaning is most effective in the interior. Near domain boundaries,
    the Dirichlet condition on phi creates gradient artifacts. For best results,
    use a domain larger than the physics region of interest.

    Args:
        B: Magnetic field array of shape (nr, nz, 3)
        geometry: Computational geometry
        tol: Convergence tolerance for Poisson solver
        maxiter: Maximum iterations

    Returns:
        Cleaned magnetic field with same shape
    """
    nr, nz = geometry.nr, geometry.nz
    dr, dz = geometry.dr, geometry.dz
    r = geometry.r_grid

    B_r = B[:, :, 0]
    B_phi = B[:, :, 1]
    B_z = B[:, :, 2]

    # Compute divergence using the existing cylindrical operator
    div_B = divergence_cylindrical(B_r, B_z, dr, dz, r)

    # Solve laplacian(phi) = div_B using Jacobi iteration
    phi = _solve_poisson_jacobi(div_B, dr, dz, r, tol, maxiter)

    # Compute gradient of phi using the same operators as divergence
    dphi_dr = gradient_r(phi, dr)
    dphi_dz = gradient_z(phi, dz)

    # Subtract gradient from B
    B_r_clean = B_r - dphi_dr
    B_z_clean = B_z - dphi_dz
    # B_phi unchanged (no theta dependence in axisymmetric)

    return jnp.stack([B_r_clean, B_phi, B_z_clean], axis=-1)


def _solve_poisson_jacobi(rhs: jnp.ndarray, dr: float, dz: float,
                          r: jnp.ndarray, tol: float, maxiter: int) -> jnp.ndarray:
    """Solve cylindrical Poisson equation laplacian(phi) = rhs using Jacobi iteration.

    The cylindrical Laplacian is:
        laplacian(phi) = d2phi/dr2 + (1/r)*dphi/dr + d2phi/dz2

    Uses standard 5-point finite difference discretization.
    Uses Dirichlet boundary conditions (phi=0 on all boundaries).
    """
    nr, nz = rhs.shape
    phi = jnp.zeros((nr, nz))

    # Handle r=0 singularity
    r_safe = jnp.maximum(r, 1e-10)

    # Standard 5-point stencil coefficients for cylindrical Laplacian
    # laplacian ~ (phi[i+1] - 2*phi[i] + phi[i-1])/dr^2
    #           + (1/r)*(phi[i+1] - phi[i-1])/(2*dr)
    #           + (phi[j+1] - 2*phi[j] + phi[j-1])/dz^2
    #
    # Rearranged: a_c*phi[i,j] = a_ip*phi[i+1,j] + a_im*phi[i-1,j] + a_jp*phi[i,j+1] + a_jm*phi[i,j-1] - rhs

    a_ip = 1.0/dr**2 + 1.0/(2*dr*r_safe)  # coefficient for phi[i+1,j]
    a_im = 1.0/dr**2 - 1.0/(2*dr*r_safe)  # coefficient for phi[i-1,j]
    # Ensure non-negative for numerical stability near axis
    a_im = jnp.maximum(a_im, 0.0)
    a_jp = 1.0/dz**2  # coefficient for phi[i,j+1]
    a_jm = 1.0/dz**2  # coefficient for phi[i,j-1]
    a_c = a_ip + a_im + a_jp + a_jm  # center coefficient

    for iteration in range(maxiter):
        phi_old = phi

        # Get neighbors - use explicit indexing for Dirichlet BC
        # Boundaries contribute 0
        phi_ip = jnp.zeros_like(phi)
        phi_im = jnp.zeros_like(phi)
        phi_jp = jnp.zeros_like(phi)
        phi_jm = jnp.zeros_like(phi)

        phi_ip = phi_ip.at[:-1, :].set(phi[1:, :])   # phi[i+1,j]
        phi_im = phi_im.at[1:, :].set(phi[:-1, :])   # phi[i-1,j]
        phi_jp = phi_jp.at[:, :-1].set(phi[:, 1:])   # phi[i,j+1]
        phi_jm = phi_jm.at[:, 1:].set(phi[:, :-1])   # phi[i,j-1]

        # Jacobi update
        phi_new = (a_ip * phi_ip + a_im * phi_im + a_jp * phi_jp + a_jm * phi_jm - rhs) / a_c

        # Enforce Dirichlet BC (phi=0) at boundaries
        phi_new = phi_new.at[0, :].set(0.0)
        phi_new = phi_new.at[-1, :].set(0.0)
        phi_new = phi_new.at[:, 0].set(0.0)
        phi_new = phi_new.at[:, -1].set(0.0)

        phi = phi_new

        # Check convergence on interior
        error = jnp.max(jnp.abs(phi[1:-1, 1:-1] - phi_old[1:-1, 1:-1]))
        if error < tol:
            break

    return phi
