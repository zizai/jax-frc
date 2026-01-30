"""Full HLL Riemann solver for MHD with coupled evolution.

This module implements the HLL solver for the complete MHD system,
evolving all 8 conserved variables together.

The HLL flux formula:
    F = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)

References:
    [1] Harten, Lax, van Leer (1983) "On Upstream Differencing..."
    [2] AGATE: agate/agate/baseRiemann.py
"""

import jax.numpy as jnp
from jax import jit
from functools import partial
from typing import Tuple

from jax_frc.solvers.riemann.mhd_state import (
    MHDConserved,
    MHDPrimitive,
    conserved_to_primitive,
    primitive_to_conserved,
    compute_mhd_flux,
)
from jax_frc.solvers.riemann.wave_speeds import fast_magnetosonic_speed
from jax_frc.solvers.riemann.reconstruction import reconstruct_plm


@jit
def hll_flux_mhd(
    U_L: MHDConserved,
    U_R: MHDConserved,
    F_L: MHDConserved,
    F_R: MHDConserved,
    S_L: jnp.ndarray,
    S_R: jnp.ndarray,
) -> MHDConserved:
    """Compute HLL numerical flux for MHD.

    Args:
        U_L, U_R: Left/right conserved states
        F_L, F_R: Left/right physical fluxes
        S_L, S_R: Left/right wave speed estimates

    Returns:
        HLL numerical flux
    """
    # Ensure S_L <= 0 <= S_R for stability
    S_L = jnp.minimum(S_L, 0.0)
    S_R = jnp.maximum(S_R, 0.0)

    # Avoid division by zero
    denom = S_R - S_L
    denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)

    # HLL flux for each component
    def hll_component(u_l, u_r, f_l, f_r):
        return (S_R * f_l - S_L * f_r + S_L * S_R * (u_r - u_l)) / denom

    return MHDConserved(
        rho=hll_component(U_L.rho, U_R.rho, F_L.rho, F_R.rho),
        mom_x=hll_component(U_L.mom_x, U_R.mom_x, F_L.mom_x, F_R.mom_x),
        mom_y=hll_component(U_L.mom_y, U_R.mom_y, F_L.mom_y, F_R.mom_y),
        mom_z=hll_component(U_L.mom_z, U_R.mom_z, F_L.mom_z, F_R.mom_z),
        E=hll_component(U_L.E, U_R.E, F_L.E, F_R.E),
        Bx=hll_component(U_L.Bx, U_R.Bx, F_L.Bx, F_R.Bx),
        By=hll_component(U_L.By, U_R.By, F_L.By, F_R.By),
        Bz=hll_component(U_L.Bz, U_R.Bz, F_L.Bz, F_R.Bz),
    )


@partial(jit, static_argnums=(1, 2, 3, 4))
def hll_flux_direction(
    cons: MHDConserved,
    geometry: "Geometry",
    direction: int,
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
) -> MHDConserved:
    """Compute HLL flux in a single direction for full MHD system.

    Args:
        cons: Conserved MHD state
        geometry: Grid geometry
        direction: Flux direction (0=x, 1=y, 2=z)
        gamma: Adiabatic index
        beta: MC limiter parameter

    Returns:
        Numerical flux at interfaces
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
    rho_L = jnp.maximum(rho_L, 1e-12)
    rho_R = jnp.maximum(rho_R, 1e-12)
    p_L = jnp.maximum(p_L, 1e-12)
    p_R = jnp.maximum(p_R, 1e-12)

    # Create primitive states at interfaces
    prim_L = MHDPrimitive(rho=rho_L, vx=vx_L, vy=vy_L, vz=vz_L, p=p_L, Bx=Bx_L, By=By_L, Bz=Bz_L)
    prim_R = MHDPrimitive(rho=rho_R, vx=vx_R, vy=vy_R, vz=vz_R, p=p_R, Bx=Bx_R, By=By_R, Bz=Bz_R)

    # Convert to conserved
    cons_L = primitive_to_conserved(prim_L, gamma)
    cons_R = primitive_to_conserved(prim_R, gamma)

    # Compute physical fluxes
    flux_L = compute_mhd_flux(prim_L, cons_L, direction, gamma)
    flux_R = compute_mhd_flux(prim_R, cons_R, direction, gamma)

    # Compute wave speeds
    if direction == 0:
        vn_L, vn_R = vx_L, vx_R
        Bn = 0.5 * (Bx_L + Bx_R)
        Bt_L = jnp.sqrt(By_L**2 + Bz_L**2)
        Bt_R = jnp.sqrt(By_R**2 + Bz_R**2)
    elif direction == 1:
        vn_L, vn_R = vy_L, vy_R
        Bn = 0.5 * (By_L + By_R)
        Bt_L = jnp.sqrt(Bx_L**2 + Bz_L**2)
        Bt_R = jnp.sqrt(Bx_R**2 + Bz_R**2)
    else:
        vn_L, vn_R = vz_L, vz_R
        Bn = 0.5 * (Bz_L + Bz_R)
        Bt_L = jnp.sqrt(Bx_L**2 + By_L**2)
        Bt_R = jnp.sqrt(Bx_R**2 + By_R**2)

    cf_L = fast_magnetosonic_speed(rho_L, p_L, Bn, Bt_L, jnp.zeros_like(Bt_L), gamma)
    cf_R = fast_magnetosonic_speed(rho_R, p_R, Bn, Bt_R, jnp.zeros_like(Bt_R), gamma)

    # Davis wave speed estimates
    S_L = jnp.minimum(vn_L - cf_L, vn_R - cf_R)
    S_R = jnp.maximum(vn_L + cf_L, vn_R + cf_R)

    # Compute HLL flux
    return hll_flux_mhd(cons_L, cons_R, flux_L, flux_R, S_L, S_R)


@partial(jit, static_argnums=(1, 2, 3))
def hll_update_full(
    cons: MHDConserved,
    geometry: "Geometry",
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
) -> MHDConserved:
    """Compute full MHD update using HLL solver.

    This computes dU/dt = -div(F) for all conserved variables.

    Args:
        cons: Conserved MHD state
        geometry: Grid geometry
        gamma: Adiabatic index
        beta: MC limiter parameter

    Returns:
        Time derivative of conserved state (dU/dt)
    """
    # Compute fluxes in each direction
    flux_x = hll_flux_direction(cons, geometry, 0, gamma, beta)
    flux_z = hll_flux_direction(cons, geometry, 2, gamma, beta)

    # Compute divergence: dU/dt = -(F[i+1/2] - F[i-1/2]) / dx
    dx = geometry.dx
    dz = geometry.dz

    def flux_divergence_x(flux):
        return -(flux - jnp.roll(flux, 1, axis=0)) / dx

    def flux_divergence_z(flux):
        return -(flux - jnp.roll(flux, 1, axis=2)) / dz

    # X-direction contribution
    dU_x = MHDConserved(
        rho=flux_divergence_x(flux_x.rho),
        mom_x=flux_divergence_x(flux_x.mom_x),
        mom_y=flux_divergence_x(flux_x.mom_y),
        mom_z=flux_divergence_x(flux_x.mom_z),
        E=flux_divergence_x(flux_x.E),
        Bx=flux_divergence_x(flux_x.Bx),
        By=flux_divergence_x(flux_x.By),
        Bz=flux_divergence_x(flux_x.Bz),
    )

    # Z-direction contribution
    dU_z = MHDConserved(
        rho=flux_divergence_z(flux_z.rho),
        mom_x=flux_divergence_z(flux_z.mom_x),
        mom_y=flux_divergence_z(flux_z.mom_y),
        mom_z=flux_divergence_z(flux_z.mom_z),
        E=flux_divergence_z(flux_z.E),
        Bx=flux_divergence_z(flux_z.Bx),
        By=flux_divergence_z(flux_z.By),
        Bz=flux_divergence_z(flux_z.Bz),
    )

    # Y-direction (if 3D)
    if geometry.ny > 1:
        flux_y = hll_flux_direction(cons, geometry, 1, gamma, beta)
        dy = geometry.dy

        def flux_divergence_y(flux):
            return -(flux - jnp.roll(flux, 1, axis=1)) / dy

        dU_y = MHDConserved(
            rho=flux_divergence_y(flux_y.rho),
            mom_x=flux_divergence_y(flux_y.mom_x),
            mom_y=flux_divergence_y(flux_y.mom_y),
            mom_z=flux_divergence_y(flux_y.mom_z),
            E=flux_divergence_y(flux_y.E),
            Bx=flux_divergence_y(flux_y.Bx),
            By=flux_divergence_y(flux_y.By),
            Bz=flux_divergence_y(flux_y.Bz),
        )

        # Sum all directions
        return MHDConserved(
            rho=dU_x.rho + dU_y.rho + dU_z.rho,
            mom_x=dU_x.mom_x + dU_y.mom_x + dU_z.mom_x,
            mom_y=dU_x.mom_y + dU_y.mom_y + dU_z.mom_y,
            mom_z=dU_x.mom_z + dU_y.mom_z + dU_z.mom_z,
            E=dU_x.E + dU_y.E + dU_z.E,
            Bx=dU_x.Bx + dU_y.Bx + dU_z.Bx,
            By=dU_x.By + dU_y.By + dU_z.By,
            Bz=dU_x.Bz + dU_y.Bz + dU_z.Bz,
        )
    else:
        # 2D case (x-z plane)
        return MHDConserved(
            rho=dU_x.rho + dU_z.rho,
            mom_x=dU_x.mom_x + dU_z.mom_x,
            mom_y=dU_x.mom_y + dU_z.mom_y,
            mom_z=dU_x.mom_z + dU_z.mom_z,
            E=dU_x.E + dU_z.E,
            Bx=dU_x.Bx + dU_z.Bx,
            By=dU_x.By + dU_z.By,
            Bz=dU_x.Bz + dU_z.Bz,
        )
