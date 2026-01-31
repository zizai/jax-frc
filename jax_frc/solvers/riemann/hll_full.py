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
from jax_frc.solvers.riemann.wave_speeds import fast_magnetosonic_speed, hall_signal_speed
import jax_frc.operators as ops
from jax_frc.solvers.riemann.reconstruction import reconstruct_plm_bc
from jax_frc.solvers.riemann.dedner import dedner_flux


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


@partial(jit, static_argnums=(1, 2, 3, 4, 5, 6))
def hll_flux_direction(
    cons: MHDConserved,
    geometry: "Geometry",
    direction: int,
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
    include_hall: bool = False,
    hall_scale: float = 1.0,
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

    if direction == 0:
        bc = geometry.bc_x
    elif direction == 1:
        bc = geometry.bc_y
    else:
        bc = geometry.bc_z

    if include_hall:
        # Reconstruct conserved variables (AGATE-style)
        rho_L, rho_R = reconstruct_plm_bc(cons.rho, direction, beta, bc)
        mom_x_L, mom_x_R = reconstruct_plm_bc(cons.mom_x, direction, beta, bc)
        mom_y_L, mom_y_R = reconstruct_plm_bc(cons.mom_y, direction, beta, bc)
        mom_z_L, mom_z_R = reconstruct_plm_bc(cons.mom_z, direction, beta, bc)
        E_L, E_R = reconstruct_plm_bc(cons.E, direction, beta, bc)
        Bx_L, Bx_R = reconstruct_plm_bc(cons.Bx, direction, beta, bc)
        By_L, By_R = reconstruct_plm_bc(cons.By, direction, beta, bc)
        Bz_L, Bz_R = reconstruct_plm_bc(cons.Bz, direction, beta, bc)

        cons_L = MHDConserved(
            rho=rho_L,
            mom_x=mom_x_L,
            mom_y=mom_y_L,
            mom_z=mom_z_L,
            E=E_L,
            Bx=Bx_L,
            By=By_L,
            Bz=Bz_L,
        )
        cons_R = MHDConserved(
            rho=rho_R,
            mom_x=mom_x_R,
            mom_y=mom_y_R,
            mom_z=mom_z_R,
            E=E_R,
            Bx=Bx_R,
            By=By_R,
            Bz=Bz_R,
        )
        prim_L = conserved_to_primitive(cons_L, gamma)
        prim_R = conserved_to_primitive(cons_R, gamma)

        # Current density at faces from edge-centered curl (AGATE-style)
        B_center = jnp.stack([cons.Bx, cons.By, cons.Bz], axis=-1)
        J_face = ops.edge_curl_3d(B_center, geometry, direction)
        Jx_L = Jx_R = J_face[..., 0]
        Jy_L = Jy_R = J_face[..., 1]
        Jz_L = Jz_R = J_face[..., 2]
    else:
        # Reconstruct all primitive variables at interfaces
        rho_L, rho_R = reconstruct_plm_bc(prim.rho, direction, beta, bc)
        vx_L, vx_R = reconstruct_plm_bc(prim.vx, direction, beta, bc)
        vy_L, vy_R = reconstruct_plm_bc(prim.vy, direction, beta, bc)
        vz_L, vz_R = reconstruct_plm_bc(prim.vz, direction, beta, bc)
        p_L, p_R = reconstruct_plm_bc(prim.p, direction, beta, bc)
        Bx_L, Bx_R = reconstruct_plm_bc(prim.Bx, direction, beta, bc)
        By_L, By_R = reconstruct_plm_bc(prim.By, direction, beta, bc)
        Bz_L, Bz_R = reconstruct_plm_bc(prim.Bz, direction, beta, bc)

    if include_hall:
        rho_L, rho_R = prim_L.rho, prim_R.rho
        vx_L, vx_R = prim_L.vx, prim_R.vx
        vy_L, vy_R = prim_L.vy, prim_R.vy
        vz_L, vz_R = prim_L.vz, prim_R.vz
        p_L, p_R = prim_L.p, prim_R.p
    else:
        # Ensure positive density and pressure
        rho_L = jnp.maximum(rho_L, 1e-12)
        rho_R = jnp.maximum(rho_R, 1e-12)
        p_L = jnp.maximum(p_L, 1e-12)
        p_R = jnp.maximum(p_R, 1e-12)

        # Create primitive states at interfaces
        prim_L = MHDPrimitive(
            rho=rho_L, vx=vx_L, vy=vy_L, vz=vz_L, p=p_L, Bx=Bx_L, By=By_L, Bz=Bz_L
        )
        prim_R = MHDPrimitive(
            rho=rho_R, vx=vx_R, vy=vy_R, vz=vz_R, p=p_R, Bx=Bx_R, By=By_R, Bz=Bz_R
        )

        # Convert to conserved
        cons_L = primitive_to_conserved(prim_L, gamma)
        cons_R = primitive_to_conserved(prim_R, gamma)

    def _hall_flux(
        rho, vx, vy, vz, p, Bx, By, Bz, Jx, Jy, Jz
    ) -> MHDConserved:
        if direction == 0:
            vn, vt1, vt2 = vx, vy, vz
            Bn, Bt1, Bt2 = Bx, By, Bz
            Jn, Jt1, Jt2 = Jx, Jy, Jz
        elif direction == 1:
            vn, vt1, vt2 = vy, vx, vz
            Bn, Bt1, Bt2 = By, Bx, Bz
            Jn, Jt1, Jt2 = Jy, Jx, Jz
        else:
            vn, vt1, vt2 = vz, vx, vy
            Bn, Bt1, Bt2 = Bz, Bx, By
            Jn, Jt1, Jt2 = Jz, Jx, Jy

        inv_rho = 1.0 / rho
        vn_e = vn - hall_scale * Jn * inv_rho
        vt1_e = vt1 - hall_scale * Jt1 * inv_rho
        vt2_e = vt2 - hall_scale * Jt2 * inv_rho

        bmag_2 = Bn**2 + Bt1**2 + Bt2**2

        F_rho = rho * vn
        F_mn = rho * vn * vn + 0.5 * bmag_2 - Bn**2 + p
        F_mt1 = rho * vn * vt1 - Bn * Bt1
        F_mt2 = rho * vn * vt2 - Bn * Bt2

        rho_vmag_2 = rho * (vn * vn + vt1 * vt1 + vt2 * vt2)
        vt_dot_b = vn_e * Bn + vt1_e * Bt1 + vt2_e * Bt2
        F_E = (
            (gamma / (gamma - 1.0) * p + 0.5 * rho_vmag_2) * vn
            + bmag_2 * vn_e
            - Bn * vt_dot_b
        )

        F_Bn = jnp.zeros_like(F_rho)
        F_Bt1 = vn_e * Bt1 - Bn * vt1_e
        F_Bt2 = vn_e * Bt2 - Bn * vt2_e

        if direction == 0:
            return MHDConserved(
                rho=F_rho,
                mom_x=F_mn,
                mom_y=F_mt1,
                mom_z=F_mt2,
                E=F_E,
                Bx=F_Bn,
                By=F_Bt1,
                Bz=F_Bt2,
            )
        if direction == 1:
            return MHDConserved(
                rho=F_rho,
                mom_x=F_mt1,
                mom_y=F_mn,
                mom_z=F_mt2,
                E=F_E,
                Bx=F_Bt1,
                By=F_Bn,
                Bz=F_Bt2,
            )
        return MHDConserved(
            rho=F_rho,
            mom_x=F_mt1,
            mom_y=F_mt2,
            mom_z=F_mn,
            E=F_E,
            Bx=F_Bt1,
            By=F_Bt2,
            Bz=F_Bn,
        )

    if include_hall:
        flux_L = _hall_flux(rho_L, vx_L, vy_L, vz_L, p_L, Bx_L, By_L, Bz_L, Jx_L, Jy_L, Jz_L)
        flux_R = _hall_flux(rho_R, vx_R, vy_R, vz_R, p_R, Bx_R, By_R, Bz_R, Jx_R, Jy_R, Jz_R)
    else:
        # Compute physical fluxes
        flux_L = compute_mhd_flux(prim_L, cons_L, direction, gamma)
        flux_R = compute_mhd_flux(prim_R, cons_R, direction, gamma)

    # Compute wave speeds
    if direction == 0:
        vn_L, vn_R = vx_L, vx_R
        Bn_L, Bn_R = Bx_L, Bx_R
        Bt_L = jnp.sqrt(By_L**2 + Bz_L**2)
        Bt_R = jnp.sqrt(By_R**2 + Bz_R**2)
    elif direction == 1:
        vn_L, vn_R = vy_L, vy_R
        Bn_L, Bn_R = By_L, By_R
        Bt_L = jnp.sqrt(Bx_L**2 + Bz_L**2)
        Bt_R = jnp.sqrt(Bx_R**2 + Bz_R**2)
    else:
        vn_L, vn_R = vz_L, vz_R
        Bn_L, Bn_R = Bz_L, Bz_R
        Bt_L = jnp.sqrt(Bx_L**2 + By_L**2)
        Bt_R = jnp.sqrt(Bx_R**2 + By_R**2)

    if include_hall:
        cell_size = geometry.dx if direction == 0 else geometry.dy if direction == 1 else geometry.dz
        if direction == 0:
            bn_l, bt1_l, bt2_l = Bx_L, By_L, Bz_L
            bn_r, bt1_r, bt2_r = Bx_R, By_R, Bz_R
        elif direction == 1:
            bn_l, bt1_l, bt2_l = By_L, Bx_L, Bz_L
            bn_r, bt1_r, bt2_r = By_R, Bx_R, Bz_R
        else:
            bn_l, bt1_l, bt2_l = Bz_L, Bx_L, By_L
            bn_r, bt1_r, bt2_r = Bz_R, Bx_R, By_R
        cf_L = hall_signal_speed(
            rho_L, p_L, bn_l, bt1_l, bt2_l, cell_size, gamma, hall_scale
        )
        cf_R = hall_signal_speed(
            rho_R, p_R, bn_r, bt1_r, bt2_r, cell_size, gamma, hall_scale
        )
    else:
        cf_L = fast_magnetosonic_speed(
            rho_L, p_L, Bn_L, Bt_L, jnp.zeros_like(Bt_L), gamma
        )
        cf_R = fast_magnetosonic_speed(
            rho_R, p_R, Bn_R, Bt_R, jnp.zeros_like(Bt_R), gamma
        )

    # Davis wave speed estimates
    S_L = jnp.minimum(vn_L - cf_L, vn_R - cf_R)
    S_R = jnp.maximum(vn_L + cf_L, vn_R + cf_R)

    # Compute HLL flux
    return hll_flux_mhd(cons_L, cons_R, flux_L, flux_R, S_L, S_R)


@partial(jit, static_argnums=(1, 2, 3, 4, 5))
def hll_update_full(
    cons: MHDConserved,
    geometry: "Geometry",
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
    include_hall: bool = False,
    hall_scale: float = 1.0,
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
    flux_x = hll_flux_direction(cons, geometry, 0, gamma, beta, include_hall, hall_scale)
    flux_z = hll_flux_direction(cons, geometry, 2, gamma, beta, include_hall, hall_scale)

    # Compute divergence: dU/dt = -(F[i+1/2] - F[i-1/2]) / dx
    dx = geometry.dx
    dz = geometry.dz

    def flux_divergence(flux, axis: int, spacing: float, bc: str):
        if bc == "periodic":
            return -(flux - jnp.roll(flux, 1, axis=axis)) / spacing
        # Neumann / outflow: zero-gradient padding at boundaries.
        pad_width = [(0, 0)] * flux.ndim
        pad_width[axis] = (1, 1)
        flux_pad = jnp.pad(flux, pad_width, mode="edge")
        slicer_hi = [slice(None)] * flux.ndim
        slicer_lo = [slice(None)] * flux.ndim
        slicer_hi[axis] = slice(1, -1)
        slicer_lo[axis] = slice(0, -2)
        return -(flux_pad[tuple(slicer_hi)] - flux_pad[tuple(slicer_lo)]) / spacing

    # X-direction contribution
    dU_x = MHDConserved(
        rho=flux_divergence(flux_x.rho, 0, dx, geometry.bc_x),
        mom_x=flux_divergence(flux_x.mom_x, 0, dx, geometry.bc_x),
        mom_y=flux_divergence(flux_x.mom_y, 0, dx, geometry.bc_x),
        mom_z=flux_divergence(flux_x.mom_z, 0, dx, geometry.bc_x),
        E=flux_divergence(flux_x.E, 0, dx, geometry.bc_x),
        Bx=flux_divergence(flux_x.Bx, 0, dx, geometry.bc_x),
        By=flux_divergence(flux_x.By, 0, dx, geometry.bc_x),
        Bz=flux_divergence(flux_x.Bz, 0, dx, geometry.bc_x),
    )

    # Z-direction contribution
    dU_z = MHDConserved(
        rho=flux_divergence(flux_z.rho, 2, dz, geometry.bc_z),
        mom_x=flux_divergence(flux_z.mom_x, 2, dz, geometry.bc_z),
        mom_y=flux_divergence(flux_z.mom_y, 2, dz, geometry.bc_z),
        mom_z=flux_divergence(flux_z.mom_z, 2, dz, geometry.bc_z),
        E=flux_divergence(flux_z.E, 2, dz, geometry.bc_z),
        Bx=flux_divergence(flux_z.Bx, 2, dz, geometry.bc_z),
        By=flux_divergence(flux_z.By, 2, dz, geometry.bc_z),
        Bz=flux_divergence(flux_z.Bz, 2, dz, geometry.bc_z),
    )

    # Y-direction (if 3D)
    if geometry.ny > 1:
        flux_y = hll_flux_direction(cons, geometry, 1, gamma, beta, include_hall, hall_scale)
        dy = geometry.dy

        def flux_divergence_y(flux):
            return flux_divergence(flux, 1, dy, geometry.bc_y)

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


@partial(jit, static_argnums=(2, 3, 4, 5, 6))
def hll_update_full_with_dedner(
    cons: MHDConserved,
    psi: jnp.ndarray,
    geometry: "Geometry",
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
    include_hall: bool = False,
    hall_scale: float = 1.0,
    ch: float = 1.0,
) -> tuple[MHDConserved, jnp.ndarray]:
    """Compute full MHD update with Dedner (HLLDiv-style) Bn/psi fluxes."""
    flux_x = hll_flux_direction(cons, geometry, 0, gamma, beta, include_hall, hall_scale)
    flux_z = hll_flux_direction(cons, geometry, 2, gamma, beta, include_hall, hall_scale)

    def flux_divergence(flux, axis: int, spacing: float, bc: str):
        if bc == "periodic":
            return -(flux - jnp.roll(flux, 1, axis=axis)) / spacing
        pad_width = [(0, 0)] * flux.ndim
        pad_width[axis] = (1, 1)
        flux_pad = jnp.pad(flux, pad_width, mode="edge")
        slicer_hi = [slice(None)] * flux.ndim
        slicer_lo = [slice(None)] * flux.ndim
        slicer_hi[axis] = slice(1, -1)
        slicer_lo[axis] = slice(0, -2)
        return -(flux_pad[tuple(slicer_hi)] - flux_pad[tuple(slicer_lo)]) / spacing

    # Dedner fluxes for Bn and psi
    Bx_L, Bx_R = reconstruct_plm_bc(cons.Bx, 0, beta, geometry.bc_x)
    psi_Lx, psi_Rx = reconstruct_plm_bc(psi, 0, beta, geometry.bc_x)
    flux_Bx, flux_psi_x = dedner_flux(Bx_L, Bx_R, psi_Lx, psi_Rx, ch)

    Bz_L, Bz_R = reconstruct_plm_bc(cons.Bz, 2, beta, geometry.bc_z)
    psi_Lz, psi_Rz = reconstruct_plm_bc(psi, 2, beta, geometry.bc_z)
    flux_Bz, flux_psi_z = dedner_flux(Bz_L, Bz_R, psi_Lz, psi_Rz, ch)

    flux_x = MHDConserved(
        rho=flux_x.rho,
        mom_x=flux_x.mom_x,
        mom_y=flux_x.mom_y,
        mom_z=flux_x.mom_z,
        E=flux_x.E,
        Bx=flux_Bx,
        By=flux_x.By,
        Bz=flux_x.Bz,
    )
    flux_z = MHDConserved(
        rho=flux_z.rho,
        mom_x=flux_z.mom_x,
        mom_y=flux_z.mom_y,
        mom_z=flux_z.mom_z,
        E=flux_z.E,
        Bx=flux_z.Bx,
        By=flux_z.By,
        Bz=flux_Bz,
    )

    dx = geometry.dx
    dz = geometry.dz

    dU_x = MHDConserved(
        rho=flux_divergence(flux_x.rho, 0, dx, geometry.bc_x),
        mom_x=flux_divergence(flux_x.mom_x, 0, dx, geometry.bc_x),
        mom_y=flux_divergence(flux_x.mom_y, 0, dx, geometry.bc_x),
        mom_z=flux_divergence(flux_x.mom_z, 0, dx, geometry.bc_x),
        E=flux_divergence(flux_x.E, 0, dx, geometry.bc_x),
        Bx=flux_divergence(flux_x.Bx, 0, dx, geometry.bc_x),
        By=flux_divergence(flux_x.By, 0, dx, geometry.bc_x),
        Bz=flux_divergence(flux_x.Bz, 0, dx, geometry.bc_x),
    )

    dU_z = MHDConserved(
        rho=flux_divergence(flux_z.rho, 2, dz, geometry.bc_z),
        mom_x=flux_divergence(flux_z.mom_x, 2, dz, geometry.bc_z),
        mom_y=flux_divergence(flux_z.mom_y, 2, dz, geometry.bc_z),
        mom_z=flux_divergence(flux_z.mom_z, 2, dz, geometry.bc_z),
        E=flux_divergence(flux_z.E, 2, dz, geometry.bc_z),
        Bx=flux_divergence(flux_z.Bx, 2, dz, geometry.bc_z),
        By=flux_divergence(flux_z.By, 2, dz, geometry.bc_z),
        Bz=flux_divergence(flux_z.Bz, 2, dz, geometry.bc_z),
    )

    dpsi = flux_divergence(flux_psi_x, 0, dx, geometry.bc_x)
    dpsi = dpsi + flux_divergence(flux_psi_z, 2, dz, geometry.bc_z)

    if geometry.ny > 1:
        flux_y = hll_flux_direction(cons, geometry, 1, gamma, beta, include_hall, hall_scale)
        By_L, By_R = reconstruct_plm_bc(cons.By, 1, beta, geometry.bc_y)
        psi_Ly, psi_Ry = reconstruct_plm_bc(psi, 1, beta, geometry.bc_y)
        flux_By, flux_psi_y = dedner_flux(By_L, By_R, psi_Ly, psi_Ry, ch)
        flux_y = MHDConserved(
            rho=flux_y.rho,
            mom_x=flux_y.mom_x,
            mom_y=flux_y.mom_y,
            mom_z=flux_y.mom_z,
            E=flux_y.E,
            Bx=flux_y.Bx,
            By=flux_By,
            Bz=flux_y.Bz,
        )
        dy = geometry.dy

        def flux_divergence_y(flux):
            return flux_divergence(flux, 1, dy, geometry.bc_y)

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
        dpsi = dpsi + flux_divergence_y(flux_psi_y)

        return (
            MHDConserved(
                rho=dU_x.rho + dU_y.rho + dU_z.rho,
                mom_x=dU_x.mom_x + dU_y.mom_x + dU_z.mom_x,
                mom_y=dU_x.mom_y + dU_y.mom_y + dU_z.mom_y,
                mom_z=dU_x.mom_z + dU_y.mom_z + dU_z.mom_z,
                E=dU_x.E + dU_y.E + dU_z.E,
                Bx=dU_x.Bx + dU_y.Bx + dU_z.Bx,
                By=dU_x.By + dU_y.By + dU_z.By,
                Bz=dU_x.Bz + dU_y.Bz + dU_z.Bz,
            ),
            dpsi,
        )

    return (
        MHDConserved(
            rho=dU_x.rho + dU_z.rho,
            mom_x=dU_x.mom_x + dU_z.mom_x,
            mom_y=dU_x.mom_y + dU_z.mom_y,
            mom_z=dU_x.mom_z + dU_z.mom_z,
            E=dU_x.E + dU_z.E,
            Bx=dU_x.Bx + dU_z.Bx,
            By=dU_x.By + dU_z.By,
            Bz=dU_x.Bz + dU_z.Bz,
        ),
        dpsi,
    )
