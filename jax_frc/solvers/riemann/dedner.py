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

from jax_frc.solvers.riemann.reconstruction import reconstruct_plm


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

    # X-direction cleaning fluxes
    Bx_L, Bx_R = reconstruct_plm(Bx, 0, beta)
    psi_L, psi_R = reconstruct_plm(psi, 0, beta)
    flux_Bx, flux_psi_x = dedner_flux(Bx_L, Bx_R, psi_L, psi_R, ch)

    # Z-direction cleaning fluxes
    Bz_L, Bz_R = reconstruct_plm(Bz, 2, beta)
    psi_Lz, psi_Rz = reconstruct_plm(psi, 2, beta)
    flux_Bz, flux_psi_z = dedner_flux(Bz_L, Bz_R, psi_Lz, psi_Rz, ch)

    dx = geometry.dx
    dz = geometry.dz

    dBx = -(flux_Bx - jnp.roll(flux_Bx, 1, axis=0)) / dx
    dBz = -(flux_Bz - jnp.roll(flux_Bz, 1, axis=2)) / dz
    dpsi = -(flux_psi_x - jnp.roll(flux_psi_x, 1, axis=0)) / dx
    dpsi = dpsi + (-(flux_psi_z - jnp.roll(flux_psi_z, 1, axis=2)) / dz)

    if geometry.ny > 1:
        By_L, By_R = reconstruct_plm(By, 1, beta)
        psi_Ly, psi_Ry = reconstruct_plm(psi, 1, beta)
        flux_By, flux_psi_y = dedner_flux(By_L, By_R, psi_Ly, psi_Ry, ch)
        dy = geometry.dy
        dBy = -(flux_By - jnp.roll(flux_By, 1, axis=1)) / dy
        dpsi = dpsi + (-(flux_psi_y - jnp.roll(flux_psi_y, 1, axis=1)) / dy)
    else:
        dBy = jnp.zeros_like(By)

    dB_clean = jnp.stack([dBx, dBy, dBz], axis=-1)
    return dB_clean, dpsi
