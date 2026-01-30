"""HLLD (Harten-Lax-van Leer-Discontinuities) Riemann solver for MHD.

This module implements the HLLD approximate Riemann solver, which is the
"gold standard" for ideal MHD. It approximates the Riemann fan with 5 waves:
- Two fast magnetosonic waves (S_L, S_R)
- Two Alfven waves (S_L*, S_R*)
- One contact discontinuity (S_M)

HLLD resolves Alfven waves exactly and is more accurate than HLL while
remaining computationally efficient.

References:
    [1] Miyoshi & Kusano (2005) "A multi-state HLL approximate Riemann solver
        for ideal magnetohydrodynamics" JCP 208, 315-344
    [2] Stone et al. (2008) "Athena: A New Code for Astrophysical MHD"
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
    RHO_FLOOR,
    P_FLOOR,
)
from jax_frc.solvers.riemann.wave_speeds import fast_magnetosonic_speed
from jax_frc.solvers.riemann.reconstruction import reconstruct_plm


# Small number for numerical stability
SMALL = 1e-12


@partial(jit, static_argnums=(6, 7))
def hlld_flux_1d(
    prim_L: MHDPrimitive,
    prim_R: MHDPrimitive,
    cons_L: MHDConserved,
    cons_R: MHDConserved,
    flux_L: MHDConserved,
    flux_R: MHDConserved,
    direction: int,
    gamma: float,
) -> MHDConserved:
    """Compute HLLD numerical flux at a single interface.

    Implements the Miyoshi & Kusano (2005) HLLD algorithm.

    Args:
        prim_L, prim_R: Left/right primitive states
        cons_L, cons_R: Left/right conserved states
        flux_L, flux_R: Left/right physical fluxes
        direction: Normal direction (0=x, 1=y, 2=z)
        gamma: Adiabatic index

    Returns:
        HLLD numerical flux
    """
    # Extract primitive variables
    rho_L, rho_R = prim_L.rho, prim_R.rho
    p_L, p_R = prim_L.p, prim_R.p

    # Get normal and tangential velocities/fields based on direction
    if direction == 0:  # x-direction
        vn_L, vn_R = prim_L.vx, prim_R.vx
        vt1_L, vt1_R = prim_L.vy, prim_R.vy
        vt2_L, vt2_R = prim_L.vz, prim_R.vz
        Bn_L, Bn_R = prim_L.Bx, prim_R.Bx
        Bt1_L, Bt1_R = prim_L.By, prim_R.By
        Bt2_L, Bt2_R = prim_L.Bz, prim_R.Bz
    elif direction == 1:  # y-direction
        vn_L, vn_R = prim_L.vy, prim_R.vy
        vt1_L, vt1_R = prim_L.vz, prim_R.vz
        vt2_L, vt2_R = prim_L.vx, prim_R.vx
        Bn_L, Bn_R = prim_L.By, prim_R.By
        Bt1_L, Bt1_R = prim_L.Bz, prim_R.Bz
        Bt2_L, Bt2_R = prim_L.Bx, prim_R.Bx
    else:  # z-direction
        vn_L, vn_R = prim_L.vz, prim_R.vz
        vt1_L, vt1_R = prim_L.vx, prim_R.vx
        vt2_L, vt2_R = prim_L.vy, prim_R.vy
        Bn_L, Bn_R = prim_L.Bz, prim_R.Bz
        Bt1_L, Bt1_R = prim_L.Bx, prim_R.Bx
        Bt2_L, Bt2_R = prim_L.By, prim_R.By

    # Normal B should be continuous (use average)
    Bn = 0.5 * (Bn_L + Bn_R)

    # Compute total pressure: p_tot = p + B^2/2
    B2_L = Bn**2 + Bt1_L**2 + Bt2_L**2
    B2_R = Bn**2 + Bt1_R**2 + Bt2_R**2
    pt_L = p_L + 0.5 * B2_L
    pt_R = p_R + 0.5 * B2_R

    # Compute fast magnetosonic speeds
    Bt_L = jnp.sqrt(Bt1_L**2 + Bt2_L**2)
    Bt_R = jnp.sqrt(Bt1_R**2 + Bt2_R**2)
    cf_L = fast_magnetosonic_speed(rho_L, p_L, Bn, Bt_L, jnp.zeros_like(Bt_L), gamma)
    cf_R = fast_magnetosonic_speed(rho_R, p_R, Bn, Bt_R, jnp.zeros_like(Bt_R), gamma)

    # Wave speed estimates (Davis)
    S_L = jnp.minimum(vn_L - cf_L, vn_R - cf_R)
    S_R = jnp.maximum(vn_L + cf_L, vn_R + cf_R)

    # Ensure S_L < 0 < S_R for robustness
    S_L = jnp.minimum(S_L, 0.0)
    S_R = jnp.maximum(S_R, 0.0)

    # =========================================================================
    # Compute contact wave speed S_M (Eq. 38 in Miyoshi & Kusano)
    # =========================================================================
    # S_M = ((S_R - vn_R)*rho_R*vn_R - (S_L - vn_L)*rho_L*vn_L - pt_R + pt_L) /
    #       ((S_R - vn_R)*rho_R - (S_L - vn_L)*rho_L)

    denom_SM = (S_R - vn_R) * rho_R - (S_L - vn_L) * rho_L
    denom_SM = jnp.where(jnp.abs(denom_SM) < SMALL, SMALL * jnp.sign(denom_SM + SMALL), denom_SM)

    S_M = ((S_R - vn_R) * rho_R * vn_R - (S_L - vn_L) * rho_L * vn_L - pt_R + pt_L) / denom_SM

    # =========================================================================
    # Compute total pressure in star region (Eq. 41)
    # =========================================================================
    pt_star = pt_L + rho_L * (S_L - vn_L) * (S_M - vn_L)

    # =========================================================================
    # Compute star state densities (Eq. 43)
    # =========================================================================
    denom_L = S_L - S_M
    denom_R = S_R - S_M
    denom_L = jnp.where(jnp.abs(denom_L) < SMALL, SMALL, denom_L)
    denom_R = jnp.where(jnp.abs(denom_R) < SMALL, SMALL, denom_R)

    rho_L_star = rho_L * (S_L - vn_L) / denom_L
    rho_R_star = rho_R * (S_R - vn_R) / denom_R

    # Ensure positive density
    rho_L_star = jnp.maximum(rho_L_star, RHO_FLOOR)
    rho_R_star = jnp.maximum(rho_R_star, RHO_FLOOR)

    # =========================================================================
    # Compute star state tangential velocities and B fields (Eq. 44-47)
    # =========================================================================
    sqrt_rho_L_star = jnp.sqrt(rho_L_star)
    sqrt_rho_R_star = jnp.sqrt(rho_R_star)

    # Factor for tangential components
    factor_L = rho_L * (S_L - vn_L) * (S_L - S_M) - Bn**2
    factor_R = rho_R * (S_R - vn_R) * (S_R - S_M) - Bn**2

    # Handle degenerate case when Bn ~ 0
    factor_L = jnp.where(jnp.abs(factor_L) < SMALL, SMALL, factor_L)
    factor_R = jnp.where(jnp.abs(factor_R) < SMALL, SMALL, factor_R)

    # Tangential velocities in star region
    vt1_L_star = vt1_L - Bn * Bt1_L * (S_M - vn_L) / factor_L
    vt2_L_star = vt2_L - Bn * Bt2_L * (S_M - vn_L) / factor_L
    vt1_R_star = vt1_R - Bn * Bt1_R * (S_M - vn_R) / factor_R
    vt2_R_star = vt2_R - Bn * Bt2_R * (S_M - vn_R) / factor_R

    # Tangential B fields in star region
    Bt1_L_star = Bt1_L * (rho_L * (S_L - vn_L)**2 - Bn**2) / factor_L
    Bt2_L_star = Bt2_L * (rho_L * (S_L - vn_L)**2 - Bn**2) / factor_L
    Bt1_R_star = Bt1_R * (rho_R * (S_R - vn_R)**2 - Bn**2) / factor_R
    Bt2_R_star = Bt2_R * (rho_R * (S_R - vn_R)**2 - Bn**2) / factor_R

    # =========================================================================
    # Compute star state energies (Eq. 48)
    # =========================================================================
    vdotB_L = vn_L * Bn + vt1_L * Bt1_L + vt2_L * Bt2_L
    vdotB_L_star = S_M * Bn + vt1_L_star * Bt1_L_star + vt2_L_star * Bt2_L_star
    E_L_star = ((S_L - vn_L) * cons_L.E - pt_L * vn_L + pt_star * S_M +
                Bn * (vdotB_L - vdotB_L_star)) / denom_L

    vdotB_R = vn_R * Bn + vt1_R * Bt1_R + vt2_R * Bt2_R
    vdotB_R_star = S_M * Bn + vt1_R_star * Bt1_R_star + vt2_R_star * Bt2_R_star
    E_R_star = ((S_R - vn_R) * cons_R.E - pt_R * vn_R + pt_star * S_M +
                Bn * (vdotB_R - vdotB_R_star)) / denom_R

    # =========================================================================
    # Compute Alfven wave speeds (Eq. 51)
    # =========================================================================
    S_L_star = S_M - jnp.abs(Bn) / sqrt_rho_L_star
    S_R_star = S_M + jnp.abs(Bn) / sqrt_rho_R_star

    # =========================================================================
    # Compute double-star states (Eq. 59-62)
    # =========================================================================
    sqrt_sum = sqrt_rho_L_star + sqrt_rho_R_star
    sqrt_sum = jnp.where(jnp.abs(sqrt_sum) < SMALL, SMALL, sqrt_sum)

    sign_Bn = jnp.sign(Bn)

    # Double-star tangential velocities (average weighted by sqrt(rho))
    vt1_star_star = (sqrt_rho_L_star * vt1_L_star + sqrt_rho_R_star * vt1_R_star +
                    (Bt1_R_star - Bt1_L_star) * sign_Bn) / sqrt_sum
    vt2_star_star = (sqrt_rho_L_star * vt2_L_star + sqrt_rho_R_star * vt2_R_star +
                    (Bt2_R_star - Bt2_L_star) * sign_Bn) / sqrt_sum

    # Double-star tangential B fields
    Bt1_star_star = (sqrt_rho_L_star * Bt1_R_star + sqrt_rho_R_star * Bt1_L_star +
                    sqrt_rho_L_star * sqrt_rho_R_star * (vt1_R_star - vt1_L_star) * sign_Bn) / sqrt_sum
    Bt2_star_star = (sqrt_rho_L_star * Bt2_R_star + sqrt_rho_R_star * Bt2_L_star +
                    sqrt_rho_L_star * sqrt_rho_R_star * (vt2_R_star - vt2_L_star) * sign_Bn) / sqrt_sum

    # Double-star energies (Eq. 63)
    vdotB_L_star_star = S_M * Bn + vt1_star_star * Bt1_star_star + vt2_star_star * Bt2_star_star
    E_L_star_star = E_L_star - sqrt_rho_L_star * (vdotB_L_star - vdotB_L_star_star) * sign_Bn
    E_R_star_star = E_R_star + sqrt_rho_R_star * (vdotB_R_star - vdotB_L_star_star) * sign_Bn

    # =========================================================================
    # Construct flux based on wave structure
    # =========================================================================
    # F = F_L                           if S_L > 0
    # F = F_L* = F_L + S_L*(U_L* - U_L) if S_L* > 0 > S_L
    # F = F_L** = F_L* + S_L*(U_L** - U_L*) if S_M > 0 > S_L*
    # F = F_R** = F_R* + S_R*(U_R** - U_R*) if S_R* > 0 > S_M
    # F = F_R* = F_R + S_R*(U_R* - U_R) if S_R > 0 > S_R*
    # F = F_R                           if S_R < 0

    # Build conserved states for star and double-star regions
    # We need to convert back to (x,y,z) coordinates from (n,t1,t2)

    def build_flux_component(f_L, f_R, u_L, u_R, u_L_star, u_R_star, u_L_ss, u_R_ss):
        """Build flux component based on wave structure."""
        # F_L* = F_L + S_L * (U_L* - U_L)
        f_L_star = f_L + S_L * (u_L_star - u_L)
        # F_R* = F_R + S_R * (U_R* - U_R)
        f_R_star = f_R + S_R * (u_R_star - u_R)
        # F_L** = F_L* + S_L* * (U_L** - U_L*)
        f_L_ss = f_L_star + S_L_star * (u_L_ss - u_L_star)
        # F_R** = F_R* + S_R* * (U_R** - U_R*)
        f_R_ss = f_R_star + S_R_star * (u_R_ss - u_R_star)

        # Select based on wave position
        flux = jnp.where(S_L >= 0, f_L,
               jnp.where(S_L_star >= 0, f_L_star,
               jnp.where(S_M >= 0, f_L_ss,
               jnp.where(S_R_star >= 0, f_R_ss,
               jnp.where(S_R >= 0, f_R_star, f_R)))))
        return flux

    # Build star and double-star conserved states
    # Need to convert (vn, vt1, vt2) back to (vx, vy, vz)
    if direction == 0:
        # Star states
        mom_x_L_star = rho_L_star * S_M
        mom_y_L_star = rho_L_star * vt1_L_star
        mom_z_L_star = rho_L_star * vt2_L_star
        mom_x_R_star = rho_R_star * S_M
        mom_y_R_star = rho_R_star * vt1_R_star
        mom_z_R_star = rho_R_star * vt2_R_star
        Bx_star = Bn
        By_L_star, Bz_L_star = Bt1_L_star, Bt2_L_star
        By_R_star, Bz_R_star = Bt1_R_star, Bt2_R_star

        # Double-star states
        mom_x_L_ss = rho_L_star * S_M
        mom_y_L_ss = rho_L_star * vt1_star_star
        mom_z_L_ss = rho_L_star * vt2_star_star
        mom_x_R_ss = rho_R_star * S_M
        mom_y_R_ss = rho_R_star * vt1_star_star
        mom_z_R_ss = rho_R_star * vt2_star_star
        By_ss, Bz_ss = Bt1_star_star, Bt2_star_star

    elif direction == 1:
        mom_y_L_star = rho_L_star * S_M
        mom_z_L_star = rho_L_star * vt1_L_star
        mom_x_L_star = rho_L_star * vt2_L_star
        mom_y_R_star = rho_R_star * S_M
        mom_z_R_star = rho_R_star * vt1_R_star
        mom_x_R_star = rho_R_star * vt2_R_star
        By_star = Bn
        Bz_L_star, Bx_L_star = Bt1_L_star, Bt2_L_star
        Bz_R_star, Bx_R_star = Bt1_R_star, Bt2_R_star

        mom_y_L_ss = rho_L_star * S_M
        mom_z_L_ss = rho_L_star * vt1_star_star
        mom_x_L_ss = rho_L_star * vt2_star_star
        mom_y_R_ss = rho_R_star * S_M
        mom_z_R_ss = rho_R_star * vt1_star_star
        mom_x_R_ss = rho_R_star * vt2_star_star
        Bz_ss, Bx_ss = Bt1_star_star, Bt2_star_star
        Bx_star = Bx_L_star  # placeholder
        By_ss = Bn

    else:  # direction == 2
        mom_z_L_star = rho_L_star * S_M
        mom_x_L_star = rho_L_star * vt1_L_star
        mom_y_L_star = rho_L_star * vt2_L_star
        mom_z_R_star = rho_R_star * S_M
        mom_x_R_star = rho_R_star * vt1_R_star
        mom_y_R_star = rho_R_star * vt2_R_star
        Bz_star = Bn
        Bx_L_star, By_L_star = Bt1_L_star, Bt2_L_star
        Bx_R_star, By_R_star = Bt1_R_star, Bt2_R_star

        mom_z_L_ss = rho_L_star * S_M
        mom_x_L_ss = rho_L_star * vt1_star_star
        mom_y_L_ss = rho_L_star * vt2_star_star
        mom_z_R_ss = rho_R_star * S_M
        mom_x_R_ss = rho_R_star * vt1_star_star
        mom_y_R_ss = rho_R_star * vt2_star_star
        Bx_ss, By_ss = Bt1_star_star, Bt2_star_star
        Bx_star = Bx_L_star
        Bz_ss = Bn

    # Compute flux for each component
    F_rho = build_flux_component(
        flux_L.rho, flux_R.rho,
        cons_L.rho, cons_R.rho,
        rho_L_star, rho_R_star,
        rho_L_star, rho_R_star  # density same in ** region
    )

    F_mom_x = build_flux_component(
        flux_L.mom_x, flux_R.mom_x,
        cons_L.mom_x, cons_R.mom_x,
        mom_x_L_star, mom_x_R_star,
        mom_x_L_ss, mom_x_R_ss
    )

    F_mom_y = build_flux_component(
        flux_L.mom_y, flux_R.mom_y,
        cons_L.mom_y, cons_R.mom_y,
        mom_y_L_star, mom_y_R_star,
        mom_y_L_ss, mom_y_R_ss
    )

    F_mom_z = build_flux_component(
        flux_L.mom_z, flux_R.mom_z,
        cons_L.mom_z, cons_R.mom_z,
        mom_z_L_star, mom_z_R_star,
        mom_z_L_ss, mom_z_R_ss
    )

    F_E = build_flux_component(
        flux_L.E, flux_R.E,
        cons_L.E, cons_R.E,
        E_L_star, E_R_star,
        E_L_star_star, E_R_star_star
    )

    # For B fields, handle direction-specific
    if direction == 0:
        F_Bx = jnp.zeros_like(F_rho)  # div B = 0
        F_By = build_flux_component(
            flux_L.By, flux_R.By,
            cons_L.By, cons_R.By,
            By_L_star, By_R_star,
            By_ss, By_ss
        )
        F_Bz = build_flux_component(
            flux_L.Bz, flux_R.Bz,
            cons_L.Bz, cons_R.Bz,
            Bz_L_star, Bz_R_star,
            Bz_ss, Bz_ss
        )
    elif direction == 1:
        F_Bx = build_flux_component(
            flux_L.Bx, flux_R.Bx,
            cons_L.Bx, cons_R.Bx,
            Bx_L_star, Bx_R_star,
            Bx_ss, Bx_ss
        )
        F_By = jnp.zeros_like(F_rho)  # div B = 0
        F_Bz = build_flux_component(
            flux_L.Bz, flux_R.Bz,
            cons_L.Bz, cons_R.Bz,
            Bz_L_star, Bz_R_star,
            Bz_ss, Bz_ss
        )
    else:
        F_Bx = build_flux_component(
            flux_L.Bx, flux_R.Bx,
            cons_L.Bx, cons_R.Bx,
            Bx_L_star, Bx_R_star,
            Bx_ss, Bx_ss
        )
        F_By = build_flux_component(
            flux_L.By, flux_R.By,
            cons_L.By, cons_R.By,
            By_L_star, By_R_star,
            By_ss, By_ss
        )
        F_Bz = jnp.zeros_like(F_rho)  # div B = 0

    return MHDConserved(
        rho=F_rho,
        mom_x=F_mom_x,
        mom_y=F_mom_y,
        mom_z=F_mom_z,
        E=F_E,
        Bx=F_Bx,
        By=F_By,
        Bz=F_Bz,
    )


@partial(jit, static_argnums=(1, 2, 3, 4))
def hlld_flux_direction(
    cons: MHDConserved,
    geometry: "Geometry",
    direction: int,
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
) -> MHDConserved:
    """Compute HLLD flux in a single direction for full MHD system.

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
    rho_L = jnp.maximum(rho_L, RHO_FLOOR)
    rho_R = jnp.maximum(rho_R, RHO_FLOOR)
    p_L = jnp.maximum(p_L, P_FLOOR)
    p_R = jnp.maximum(p_R, P_FLOOR)

    # Create primitive states at interfaces
    prim_L = MHDPrimitive(rho=rho_L, vx=vx_L, vy=vy_L, vz=vz_L, p=p_L, Bx=Bx_L, By=By_L, Bz=Bz_L)
    prim_R = MHDPrimitive(rho=rho_R, vx=vx_R, vy=vy_R, vz=vz_R, p=p_R, Bx=Bx_R, By=By_R, Bz=Bz_R)

    # Convert to conserved
    cons_L = primitive_to_conserved(prim_L, gamma)
    cons_R = primitive_to_conserved(prim_R, gamma)

    # Compute physical fluxes
    flux_L = compute_mhd_flux(prim_L, cons_L, direction, gamma)
    flux_R = compute_mhd_flux(prim_R, cons_R, direction, gamma)

    # Compute HLLD flux
    return hlld_flux_1d(prim_L, prim_R, cons_L, cons_R, flux_L, flux_R, direction, gamma)


@partial(jit, static_argnums=(1, 2, 3))
def hlld_update_full(
    cons: MHDConserved,
    geometry: "Geometry",
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
) -> MHDConserved:
    """Compute full MHD update using HLLD solver.

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
    flux_x = hlld_flux_direction(cons, geometry, 0, gamma, beta)
    flux_z = hlld_flux_direction(cons, geometry, 2, gamma, beta)

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
        flux_y = hlld_flux_direction(cons, geometry, 1, gamma, beta)
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
