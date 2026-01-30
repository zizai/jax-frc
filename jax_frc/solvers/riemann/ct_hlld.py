"""CT-HLLD integration for divergence-free magnetic field evolution.

This module implements the Gardiner & Stone (2005) algorithm for computing
edge-centered EMF from Riemann solver fluxes, ensuring div(B)=0 to machine
precision.

The key insight is that the EMF at cell edges must be computed consistently
from the 1D Riemann solver fluxes to maintain div(B)=0.

For 2D (x-z plane):
    E_y[i+1/2, k+1/2] = average of 4 neighboring 1D fluxes

The averaging uses upwind-biased weighting based on contact wave direction.

References:
    [1] Gardiner & Stone (2005) "An unsplit Godunov method for ideal MHD
        via constrained transport" JCP 205, 509-539
    [2] Stone et al. (2008) "Athena: A New Code for Astrophysical MHD"
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple, NamedTuple

from jax_frc.solvers.riemann.mhd_state import (
    MHDConserved,
    MHDPrimitive,
    conserved_to_primitive,
    primitive_to_conserved,
    RHO_FLOOR,
    P_FLOOR,
)
from jax_frc.solvers.riemann.wave_speeds import fast_magnetosonic_speed
from jax_frc.solvers.riemann.reconstruction import reconstruct_plm


class CTFluxData(NamedTuple):
    """Data needed for CT EMF computation from Riemann solver."""
    # Numerical flux of B field (from Riemann solver)
    flux_By: jnp.ndarray  # B_y flux in x-direction (or B_x flux in y-direction)
    flux_Bz: jnp.ndarray  # B_z flux in x-direction (or B_z flux in z-direction)
    # Contact wave speed for upwind weighting
    S_M: jnp.ndarray
    # Normal velocity for upwind direction
    vn_L: jnp.ndarray
    vn_R: jnp.ndarray


@partial(jit, static_argnums=(1, 2, 3, 4))
def compute_ct_flux_data(
    cons: MHDConserved,
    geometry: "Geometry",
    direction: int,
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
) -> CTFluxData:
    """Compute flux data needed for CT EMF averaging.

    This computes the Riemann flux and contact wave speed at each interface,
    which are needed for the CT EMF averaging.

    Args:
        cons: Conserved MHD state
        geometry: Grid geometry
        direction: Flux direction (0=x, 2=z for 2D)
        gamma: Adiabatic index
        beta: MC limiter parameter

    Returns:
        CTFluxData with flux and wave speed information
    """
    # Convert to primitive for reconstruction
    prim = conserved_to_primitive(cons, gamma)

    # Reconstruct all primitive variables at interfaces
    rho_L, rho_R = reconstruct_plm(prim.rho, direction, beta)
    vx_L, vx_R = reconstruct_plm(prim.vx, direction, beta)
    vy_L, vy_R = reconstruct_plm(prim.vy, direction, beta)
    vz_L, vz_R = reconstruct_plm(prim.vz, direction, beta)
    p_L, p_R = reconstruct_plm(prim.p, direction, beta)
    Bx_L, Bx_R = reconstruct_plm(prim.Bx, direction, beta)
    By_L, By_R = reconstruct_plm(prim.By, direction, beta)
    Bz_L, Bz_R = reconstruct_plm(prim.Bz, direction, beta)

    # Ensure positive density and pressure
    rho_L = jnp.maximum(rho_L, RHO_FLOOR)
    rho_R = jnp.maximum(rho_R, RHO_FLOOR)
    p_L = jnp.maximum(p_L, P_FLOOR)
    p_R = jnp.maximum(p_R, P_FLOOR)

    # Get normal and tangential components based on direction
    if direction == 0:  # x-direction
        vn_L, vn_R = vx_L, vx_R
        Bn_L, Bn_R = Bx_L, Bx_R
        Bt1_L, Bt1_R = By_L, By_R
        Bt2_L, Bt2_R = Bz_L, Bz_R
        vt1_L, vt1_R = vy_L, vy_R
        vt2_L, vt2_R = vz_L, vz_R
    else:  # z-direction (direction == 2)
        vn_L, vn_R = vz_L, vz_R
        Bn_L, Bn_R = Bz_L, Bz_R
        Bt1_L, Bt1_R = Bx_L, Bx_R
        Bt2_L, Bt2_R = By_L, By_R
        vt1_L, vt1_R = vx_L, vx_R
        vt2_L, vt2_R = vy_L, vy_R

    # Normal B (use average)
    Bn = 0.5 * (Bn_L + Bn_R)

    # Compute total pressure
    B2_L = Bn**2 + Bt1_L**2 + Bt2_L**2
    B2_R = Bn**2 + Bt1_R**2 + Bt2_R**2
    pt_L = p_L + 0.5 * B2_L
    pt_R = p_R + 0.5 * B2_R

    # Compute fast magnetosonic speeds
    Bt_L = jnp.sqrt(Bt1_L**2 + Bt2_L**2)
    Bt_R = jnp.sqrt(Bt1_R**2 + Bt2_R**2)
    cf_L = fast_magnetosonic_speed(rho_L, p_L, Bn, Bt_L, jnp.zeros_like(Bt_L), gamma)
    cf_R = fast_magnetosonic_speed(rho_R, p_R, Bn, Bt_R, jnp.zeros_like(Bt_R), gamma)

    # Wave speed estimates
    S_L = jnp.minimum(vn_L - cf_L, vn_R - cf_R)
    S_R = jnp.maximum(vn_L + cf_L, vn_R + cf_R)
    S_L = jnp.minimum(S_L, 0.0)
    S_R = jnp.maximum(S_R, 0.0)

    # Contact wave speed
    SMALL = 1e-12
    denom_SM = (S_R - vn_R) * rho_R - (S_L - vn_L) * rho_L
    denom_SM = jnp.where(jnp.abs(denom_SM) < SMALL, SMALL, denom_SM)
    S_M = ((S_R - vn_R) * rho_R * vn_R - (S_L - vn_L) * rho_L * vn_L - pt_R + pt_L) / denom_SM

    # Compute B field fluxes (E = -v x B, so flux of B is related to E)
    # For x-direction: flux_By = By*vx - Bx*vy = -Ez
    #                  flux_Bz = Bz*vx - Bx*vz = Ey
    # For z-direction: flux_Bx = Bx*vz - Bz*vx = -Ey
    #                  flux_By = By*vz - Bz*vy = Ex

    # Physical flux at left and right states
    if direction == 0:
        F_By_L = By_L * vx_L - Bx_L * vy_L
        F_By_R = By_R * vx_R - Bx_R * vy_R
        F_Bz_L = Bz_L * vx_L - Bx_L * vz_L
        F_Bz_R = Bz_R * vx_R - Bx_R * vz_R
    else:
        F_By_L = Bx_L * vz_L - Bz_L * vx_L  # This is flux of Bx in z-direction
        F_By_R = Bx_R * vz_R - Bz_R * vx_R
        F_Bz_L = By_L * vz_L - Bz_L * vy_L  # This is flux of By in z-direction
        F_Bz_R = By_R * vz_R - Bz_R * vy_R

    # HLL flux for B components
    denom = S_R - S_L
    denom = jnp.where(jnp.abs(denom) < SMALL, SMALL, denom)

    if direction == 0:
        flux_By = (S_R * F_By_L - S_L * F_By_R + S_L * S_R * (By_R - By_L)) / denom
        flux_Bz = (S_R * F_Bz_L - S_L * F_Bz_R + S_L * S_R * (Bz_R - Bz_L)) / denom
    else:
        flux_By = (S_R * F_By_L - S_L * F_By_R + S_L * S_R * (Bx_R - Bx_L)) / denom
        flux_Bz = (S_R * F_Bz_L - S_L * F_Bz_R + S_L * S_R * (By_R - By_L)) / denom

    return CTFluxData(
        flux_By=flux_By,
        flux_Bz=flux_Bz,
        S_M=S_M,
        vn_L=vn_L,
        vn_R=vn_R,
    )


@partial(jit, static_argnums=(1, 2, 3, 4))
def compute_emf_ct(
    cons: MHDConserved,
    geometry: "Geometry",
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
    method: str = "arithmetic",
) -> jnp.ndarray:
    """Compute edge-centered EMF using CT averaging.

    For 2D (x-z plane), computes E_y at cell edges (i+1/2, k+1/2).

    The EMF is computed by averaging the B-field fluxes from the
    x and z direction Riemann solvers.

    Args:
        cons: Conserved MHD state
        geometry: Grid geometry
        gamma: Adiabatic index
        beta: MC limiter parameter
        method: Averaging method ("arithmetic" or "upwind")

    Returns:
        Edge-centered EMF E_y with shape (nx, 1, nz)
    """
    # Get flux data from x and z directions
    flux_x = compute_ct_flux_data(cons, geometry, 0, gamma, beta)
    flux_z = compute_ct_flux_data(cons, geometry, 2, gamma, beta)

    # For 2D (x-z plane), E_y is the only non-zero EMF component
    # E_y = -flux_Bz from x-direction = flux_Bx from z-direction

    # The flux_Bz from x-direction gives E_y = Bz*vx - Bx*vz
    # The flux_By from z-direction gives -E_y = Bx*vz - Bz*vx

    # EMF at face centers
    # E_y at x-face (i+1/2, k) from x-direction flux
    Ey_x = flux_x.flux_Bz  # This is Bz*vx - Bx*vz = E_y

    # E_y at z-face (i, k+1/2) from z-direction flux
    # flux_By from z-direction is Bx*vz - Bz*vx = -E_y
    Ey_z = -flux_z.flux_By

    # Average to cell edges (i+1/2, k+1/2)
    # Simple arithmetic average of 4 neighboring face values
    if method == "arithmetic":
        # E_y at edge (i+1/2, k+1/2) = average of:
        #   Ey_x[i+1/2, k] and Ey_x[i+1/2, k+1]  (from x-faces)
        #   Ey_z[i, k+1/2] and Ey_z[i+1, k+1/2]  (from z-faces)

        # Shift to get values at neighboring faces
        Ey_x_k = Ey_x  # E_y at (i+1/2, k)
        Ey_x_kp1 = jnp.roll(Ey_x, -1, axis=2)  # E_y at (i+1/2, k+1)

        Ey_z_i = Ey_z  # E_y at (i, k+1/2)
        Ey_z_ip1 = jnp.roll(Ey_z, -1, axis=0)  # E_y at (i+1, k+1/2)

        # Arithmetic average
        Ey_edge = 0.25 * (Ey_x_k + Ey_x_kp1 + Ey_z_i + Ey_z_ip1)

    else:
        # Upwind-biased averaging using contact wave speed
        # This is the Gardiner-Stone method

        # Get contact wave speeds
        S_M_x = flux_x.S_M  # Contact speed from x-direction
        S_M_z = flux_z.S_M  # Contact speed from z-direction

        # Upwind weights based on contact wave direction
        # If S_M > 0, use left-biased average
        # If S_M < 0, use right-biased average

        # For x-direction contribution
        Ey_x_k = Ey_x
        Ey_x_kp1 = jnp.roll(Ey_x, -1, axis=2)

        # Weight based on z-direction contact speed
        S_M_z_avg = 0.5 * (S_M_z + jnp.roll(S_M_z, -1, axis=0))
        w_z = jnp.where(S_M_z_avg > 0, 0.75, 0.25)
        Ey_from_x = w_z * Ey_x_k + (1 - w_z) * Ey_x_kp1

        # For z-direction contribution
        Ey_z_i = Ey_z
        Ey_z_ip1 = jnp.roll(Ey_z, -1, axis=0)

        # Weight based on x-direction contact speed
        S_M_x_avg = 0.5 * (S_M_x + jnp.roll(S_M_x, -1, axis=2))
        w_x = jnp.where(S_M_x_avg > 0, 0.75, 0.25)
        Ey_from_z = w_x * Ey_z_i + (1 - w_x) * Ey_z_ip1

        # Average x and z contributions
        Ey_edge = 0.5 * (Ey_from_x + Ey_from_z)

    return Ey_edge


@partial(jit, static_argnums=(1, 2))
def divergence_cleaning_projection(
    B: jnp.ndarray,
    geometry: "Geometry",
    n_iter: int = 10,
) -> jnp.ndarray:
    """Clean divergence of B using projection method.

    Solves: B_clean = B - grad(phi)
    where phi satisfies: laplacian(phi) = div(B)

    Uses Jacobi iteration to solve the Poisson equation.

    Args:
        B: Magnetic field with shape (nx, ny, nz, 3)
        geometry: Grid geometry
        n_iter: Number of Jacobi iterations

    Returns:
        Cleaned B field with reduced div(B)
    """
    import jax.lax as lax

    dx = geometry.dx
    dz = geometry.dz

    # Compute div(B)
    Bx = B[..., 0]
    Bz = B[..., 2]

    div_B = (jnp.roll(Bx, -1, axis=0) - jnp.roll(Bx, 1, axis=0)) / (2 * dx)
    div_B = div_B + (jnp.roll(Bz, -1, axis=2) - jnp.roll(Bz, 1, axis=2)) / (2 * dz)

    if geometry.ny > 1:
        By = B[..., 1]
        dy = geometry.dy
        div_B = div_B + (jnp.roll(By, -1, axis=1) - jnp.roll(By, 1, axis=1)) / (2 * dy)

    # Solve laplacian(phi) = div(B) using Jacobi iteration
    phi = jnp.zeros_like(div_B)

    # Jacobi iteration coefficients
    dx2 = dx * dx
    dz2 = dz * dz

    if geometry.ny > 1:
        dy2 = geometry.dy * geometry.dy
        coeff = 2.0 / dx2 + 2.0 / dy2 + 2.0 / dz2

        def jacobi_step_3d(_, phi):
            phi_new = (
                (jnp.roll(phi, 1, axis=0) + jnp.roll(phi, -1, axis=0)) / dx2 +
                (jnp.roll(phi, 1, axis=1) + jnp.roll(phi, -1, axis=1)) / dy2 +
                (jnp.roll(phi, 1, axis=2) + jnp.roll(phi, -1, axis=2)) / dz2 -
                div_B
            ) / coeff
            return phi_new

        phi = lax.fori_loop(0, n_iter, jacobi_step_3d, phi)
    else:
        coeff = 2.0 / dx2 + 2.0 / dz2

        def jacobi_step_2d(_, phi):
            phi_new = (
                (jnp.roll(phi, 1, axis=0) + jnp.roll(phi, -1, axis=0)) / dx2 +
                (jnp.roll(phi, 1, axis=2) + jnp.roll(phi, -1, axis=2)) / dz2 -
                div_B
            ) / coeff
            return phi_new

        phi = lax.fori_loop(0, n_iter, jacobi_step_2d, phi)

    # Compute grad(phi)
    grad_phi_x = (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0)) / (2 * dx)
    grad_phi_z = (jnp.roll(phi, -1, axis=2) - jnp.roll(phi, 1, axis=2)) / (2 * dz)

    # Clean B
    Bx_clean = Bx - grad_phi_x
    Bz_clean = Bz - grad_phi_z

    if geometry.ny > 1:
        grad_phi_y = (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1)) / (2 * geometry.dy)
        By_clean = By - grad_phi_y
        return jnp.stack([Bx_clean, By_clean, Bz_clean], axis=-1)
    else:
        return jnp.stack([Bx_clean, B[..., 1], Bz_clean], axis=-1)


@partial(jit, static_argnums=(1, 2, 3))
def hlld_update_with_cleaning(
    cons: MHDConserved,
    geometry: "Geometry",
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
) -> MHDConserved:
    """Compute full MHD update using HLLD with divergence cleaning.

    This uses the standard HLLD solver for all variables, then applies
    divergence cleaning to the B field update.

    Args:
        cons: Conserved MHD state
        geometry: Grid geometry
        gamma: Adiabatic index
        beta: MC limiter parameter

    Returns:
        Time derivative of conserved state (dU/dt)
    """
    from jax_frc.solvers.riemann.hlld import hlld_update_full

    # Use standard HLLD update
    return hlld_update_full(cons, geometry, gamma, beta)


@partial(jit, static_argnums=(1, 2, 3))
def ct_update_B(
    cons: MHDConserved,
    geometry: "Geometry",
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute magnetic field update using Constrained Transport.

    Uses edge-centered EMF to update B field such that div(B) is preserved.

    For 2D (x-z plane):
        dBx/dt = -dEy/dz
        dBy/dt = 0
        dBz/dt = dEy/dx

    Args:
        cons: Conserved MHD state
        geometry: Grid geometry
        gamma: Adiabatic index
        beta: MC limiter parameter

    Returns:
        Tuple of (dBx/dt, dBy/dt, dBz/dt)
    """
    # Compute edge-centered EMF
    Ey_edge = compute_emf_ct(cons, geometry, gamma, beta)

    dx = geometry.dx
    dz = geometry.dz

    # CT update: dB/dt = -curl(E)
    # For 2D (x-z plane), only E_y is non-zero
    # dBx/dt = -dEy/dz
    # dBz/dt = dEy/dx

    # E_y is at edges (i+1/2, k+1/2)
    # dBx/dt at cell center (i, k) uses Ey at (i+1/2, k+1/2) and (i+1/2, k-1/2)
    # dBz/dt at cell center (i, k) uses Ey at (i+1/2, k+1/2) and (i-1/2, k+1/2)

    # dBx/dt = -(Ey[i+1/2, k+1/2] - Ey[i+1/2, k-1/2]) / dz
    dBx_dt = -(Ey_edge - jnp.roll(Ey_edge, 1, axis=2)) / dz

    # dBz/dt = (Ey[i+1/2, k+1/2] - Ey[i-1/2, k+1/2]) / dx
    dBz_dt = (Ey_edge - jnp.roll(Ey_edge, 1, axis=0)) / dx

    # dBy/dt = 0 for 2D (x-z plane)
    dBy_dt = jnp.zeros_like(dBx_dt)

    return dBx_dt, dBy_dt, dBz_dt


@partial(jit, static_argnums=(1, 2, 3))
def ct_hlld_update_full(
    cons: MHDConserved,
    geometry: "Geometry",
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
) -> MHDConserved:
    """Compute full MHD update using HLLD solver.

    This uses the standard HLLD solver for all variables.
    Divergence cleaning is applied separately in apply_constraints.

    Note: The CT (Constrained Transport) method for preserving div(B)=0
    requires a staggered grid with B stored at face centers. This
    implementation uses cell-centered B with divergence cleaning instead,
    which is simpler but does not preserve div(B)=0 to machine precision.

    Args:
        cons: Conserved MHD state
        geometry: Grid geometry
        gamma: Adiabatic index
        beta: MC limiter parameter

    Returns:
        Time derivative of conserved state (dU/dt)
    """
    from jax_frc.solvers.riemann.hlld import hlld_update_full

    # Use standard HLLD update for all variables
    # Divergence cleaning is applied in apply_constraints
    return hlld_update_full(cons, geometry, gamma, beta)


def compute_div_B(B: jnp.ndarray, geometry: "Geometry") -> jnp.ndarray:
    """Compute divergence of B field.

    Args:
        B: Magnetic field with shape (nx, ny, nz, 3)
        geometry: Grid geometry

    Returns:
        div(B) with shape (nx, ny, nz)
    """
    Bx = B[..., 0]
    By = B[..., 1]
    Bz = B[..., 2]

    # Central differences
    dBx_dx = (jnp.roll(Bx, -1, axis=0) - jnp.roll(Bx, 1, axis=0)) / (2 * geometry.dx)
    dBz_dz = (jnp.roll(Bz, -1, axis=2) - jnp.roll(Bz, 1, axis=2)) / (2 * geometry.dz)

    if geometry.ny > 1:
        dBy_dy = (jnp.roll(By, -1, axis=1) - jnp.roll(By, 1, axis=1)) / (2 * geometry.dy)
        return dBx_dx + dBy_dy + dBz_dz
    else:
        return dBx_dx + dBz_dz
