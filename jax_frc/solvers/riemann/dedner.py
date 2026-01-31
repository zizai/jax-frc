"""
Dedner/GLM divergence cleaning utilities for MHD.

Implements the hyperbolic cleaning fluxes:
  F_Bn = 0.5*(psi_L + psi_R) - 0.5*ch*(Bn_R - Bn_L)
  F_psi = 0.5*ch^2*(Bn_L + Bn_R) - 0.5*ch*(psi_R - psi_L)

These match the linear Riemann solution used by AGATE's HLLDiv.
"""

from functools import partial
import jax.numpy as jnp
from jax import jit

from jax_frc.solvers.riemann.reconstruction import reconstruct_plm, reconstruct_plm_bc


@jit
def dedner_flux(
    Bn_L: jnp.ndarray,
    Bn_R: jnp.ndarray,
    psi_L: jnp.ndarray,
    psi_R: jnp.ndarray,
    ch: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Dedner cleaning fluxes for Bn and psi."""
    flux_Bn = 0.5 * (psi_L + psi_R) - 0.5 * ch * (Bn_R - Bn_L)
    flux_psi = 0.5 * ch * ch * (Bn_L + Bn_R) - 0.5 * ch * (psi_R - psi_L)
    return flux_Bn, flux_psi


@partial(jit, static_argnums=(1, 4))
def glm_cleaning_update(
    B: jnp.ndarray,
    geometry: "Geometry",
    psi: jnp.ndarray,
    ch: float,
    beta: float = 1.3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GLM cleaning contributions for B and psi.

    Returns:
        dB_clean: (nx, ny, nz, 3) contribution to dB/dt from cleaning.
        dpsi: (nx, ny, nz) contribution to dpsi/dt from cleaning fluxes.
    """
    ch = jnp.asarray(ch)
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]

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

    # X-direction cleaning fluxes
    Bx_L, Bx_R = reconstruct_plm_bc(Bx, 0, beta, geometry.bc_x)
    psi_L, psi_R = reconstruct_plm_bc(psi, 0, beta, geometry.bc_x)
    flux_Bx, flux_psi_x = dedner_flux(Bx_L, Bx_R, psi_L, psi_R, ch)

    # Z-direction cleaning fluxes
    Bz_L, Bz_R = reconstruct_plm_bc(Bz, 2, beta, geometry.bc_z)
    psi_Lz, psi_Rz = reconstruct_plm_bc(psi, 2, beta, geometry.bc_z)
    flux_Bz, flux_psi_z = dedner_flux(Bz_L, Bz_R, psi_Lz, psi_Rz, ch)

    dx = geometry.dx
    dz = geometry.dz

    dBx = flux_divergence(flux_Bx, 0, dx, geometry.bc_x)
    dBz = flux_divergence(flux_Bz, 2, dz, geometry.bc_z)
    dpsi = flux_divergence(flux_psi_x, 0, dx, geometry.bc_x)
    dpsi = dpsi + flux_divergence(flux_psi_z, 2, dz, geometry.bc_z)

    if geometry.ny > 1:
        By_L, By_R = reconstruct_plm_bc(By, 1, beta, geometry.bc_y)
        psi_Ly, psi_Ry = reconstruct_plm_bc(psi, 1, beta, geometry.bc_y)
        flux_By, flux_psi_y = dedner_flux(By_L, By_R, psi_Ly, psi_Ry, ch)
        dy = geometry.dy
        dBy = flux_divergence(flux_By, 1, dy, geometry.bc_y)
        dpsi = dpsi + flux_divergence(flux_psi_y, 1, dy, geometry.bc_y)
    else:
        dBy = jnp.zeros_like(By)

    dB_clean = jnp.stack([dBx, dBy, dBz], axis=-1)
    return dB_clean, dpsi
