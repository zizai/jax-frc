"""HLL (Harten-Lax-van Leer) approximate Riemann solver for MHD.

Implements the two-wave HLL solver which is robust but diffusive.
Uses the formula from AGATE:
    flux = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)

References:
    [1] Harten, Lax, van Leer (1983) "On Upstream Differencing..."
    [2] AGATE: agate/agate/baseRiemann.py
"""

import jax.numpy as jnp
from jax import jit
from typing import Tuple, NamedTuple
from functools import partial

from jax_frc.solvers.riemann.wave_speeds import fast_magnetosonic_speed


class MHDState(NamedTuple):
    """Conserved MHD state vector."""

    rho: jnp.ndarray  # Density
    mom_x: jnp.ndarray  # Momentum x
    mom_y: jnp.ndarray  # Momentum y
    mom_z: jnp.ndarray  # Momentum z
    E: jnp.ndarray  # Total energy
    Bx: jnp.ndarray  # Magnetic field x
    By: jnp.ndarray  # Magnetic field y
    Bz: jnp.ndarray  # Magnetic field z


class MHDFlux(NamedTuple):
    """MHD flux vector in a given direction."""

    rho: jnp.ndarray  # Mass flux
    mom_x: jnp.ndarray  # Momentum flux x
    mom_y: jnp.ndarray  # Momentum flux y
    mom_z: jnp.ndarray  # Momentum flux z
    E: jnp.ndarray  # Energy flux
    Bx: jnp.ndarray  # B flux x
    By: jnp.ndarray  # B flux y
    Bz: jnp.ndarray  # B flux z


@jit
def compute_mhd_flux_x(
    rho: jnp.ndarray,
    vx: jnp.ndarray,
    vy: jnp.ndarray,
    vz: jnp.ndarray,
    p: jnp.ndarray,
    Bx: jnp.ndarray,
    By: jnp.ndarray,
    Bz: jnp.ndarray,
    E: jnp.ndarray,
) -> MHDFlux:
    """Compute physical MHD flux in x-direction.

    Args:
        rho: Density
        vx, vy, vz: Velocity components
        p: Thermal pressure
        Bx, By, Bz: Magnetic field components
        E: Total energy density

    Returns:
        MHDFlux in x-direction
    """
    # Magnetic pressure
    B2 = Bx**2 + By**2 + Bz**2
    p_mag = 0.5 * B2

    # Total pressure
    p_tot = p + p_mag

    # v dot B
    vdotB = vx * Bx + vy * By + vz * Bz

    return MHDFlux(
        rho=rho * vx,
        mom_x=rho * vx * vx + p_tot - Bx * Bx,
        mom_y=rho * vx * vy - Bx * By,
        mom_z=rho * vx * vz - Bx * Bz,
        E=(E + p_tot) * vx - Bx * vdotB,
        Bx=jnp.zeros_like(Bx),  # dBx/dt from x-flux is 0 (div B = 0)
        By=By * vx - Bx * vy,
        Bz=Bz * vx - Bx * vz,
    )


@jit
def estimate_wave_speeds(
    rho_L: jnp.ndarray,
    rho_R: jnp.ndarray,
    vn_L: jnp.ndarray,
    vn_R: jnp.ndarray,
    p_L: jnp.ndarray,
    p_R: jnp.ndarray,
    Bn: jnp.ndarray,
    Bt_L: jnp.ndarray,
    Bt_R: jnp.ndarray,
    gamma: float = 5.0 / 3.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate left and right wave speeds for HLL solver.

    Uses Davis wave speed estimates which are simple and robust.

    Args:
        rho_L, rho_R: Left/right densities
        vn_L, vn_R: Left/right normal velocities
        p_L, p_R: Left/right pressures
        Bn: Normal magnetic field (constant across interface)
        Bt_L, Bt_R: Left/right tangential B magnitude
        gamma: Adiabatic index

    Returns:
        Tuple of (S_L, S_R) wave speed estimates
    """
    # Compute fast magnetosonic speeds on each side
    # For simplicity, use Bt as By and 0 as Bz
    cf_L = fast_magnetosonic_speed(rho_L, p_L, Bn, Bt_L, jnp.zeros_like(Bt_L), gamma)
    cf_R = fast_magnetosonic_speed(rho_R, p_R, Bn, Bt_R, jnp.zeros_like(Bt_R), gamma)

    # Davis estimates: S_L = min(vn_L - cf_L, vn_R - cf_R)
    #                  S_R = max(vn_L + cf_L, vn_R + cf_R)
    S_L = jnp.minimum(vn_L - cf_L, vn_R - cf_R)
    S_R = jnp.maximum(vn_L + cf_L, vn_R + cf_R)

    return S_L, S_R


@jit
def hll_flux_1d(
    U_L: jnp.ndarray,
    U_R: jnp.ndarray,
    F_L: jnp.ndarray,
    F_R: jnp.ndarray,
    rho_L: jnp.ndarray,
    rho_R: jnp.ndarray,
    vn_L: jnp.ndarray,
    vn_R: jnp.ndarray,
    cf_L: jnp.ndarray,
    cf_R: jnp.ndarray,
) -> jnp.ndarray:
    """HLL approximate Riemann solver for 1D interface.

    Implements the standard HLL formula from AGATE:
        flux = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / (S_R - S_L)

    Args:
        U_L, U_R: Left/right conserved states (nvar, ...)
        F_L, F_R: Left/right physical fluxes (nvar, ...)
        rho_L, rho_R: Left/right densities (for wave speed)
        vn_L, vn_R: Left/right normal velocities
        cf_L, cf_R: Left/right fast magnetosonic speeds

    Returns:
        Numerical flux at interface (nvar, ...)
    """
    # Wave speed estimates (Davis)
    S_L = jnp.minimum(vn_L - cf_L, vn_R - cf_R)
    S_R = jnp.maximum(vn_L + cf_L, vn_R + cf_R)

    # Ensure S_L <= 0 <= S_R for stability
    S_L = jnp.minimum(S_L, 0.0)
    S_R = jnp.maximum(S_R, 0.0)

    # HLL flux formula
    denom = S_R - S_L
    # Avoid division by zero
    denom = jnp.where(jnp.abs(denom) < 1e-12, 1e-12, denom)

    flux = (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / denom

    return flux


@partial(jit, static_argnums=(1, 2, 3, 4))
def hll_flux_3d(
    state: "State",
    geometry: "Geometry",
    gamma: float = 5.0 / 3.0,
    beta: float = 1.3,
    direction: int = 0,
) -> jnp.ndarray:
    """Compute HLL flux for 3D MHD in a given direction.

    This is the main entry point for computing numerical fluxes
    using the HLL Riemann solver with PLM reconstruction.

    Args:
        state: JAX-FRC State object with n, v, B, p fields
        geometry: JAX-FRC Geometry object
        gamma: Adiabatic index
        beta: MC limiter parameter for reconstruction
        direction: Flux direction (0=x, 1=y, 2=z)

    Returns:
        dB/dt contribution from this direction
    """
    from jax_frc.solvers.riemann.reconstruction import reconstruct_plm

    # Extract fields
    rho = state.n
    vx, vy, vz = state.v[..., 0], state.v[..., 1], state.v[..., 2]
    Bx, By, Bz = state.B[..., 0], state.B[..., 1], state.B[..., 2]
    p = state.p

    # Compute total energy
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2
    E = p / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * B2

    # Reconstruct at interfaces
    rho_L, rho_R = reconstruct_plm(rho, direction, beta)
    vx_L, vx_R = reconstruct_plm(vx, direction, beta)
    vy_L, vy_R = reconstruct_plm(vy, direction, beta)
    vz_L, vz_R = reconstruct_plm(vz, direction, beta)
    Bx_L, Bx_R = reconstruct_plm(Bx, direction, beta)
    By_L, By_R = reconstruct_plm(By, direction, beta)
    Bz_L, Bz_R = reconstruct_plm(Bz, direction, beta)
    p_L, p_R = reconstruct_plm(p, direction, beta)
    E_L, E_R = reconstruct_plm(E, direction, beta)

    # Select normal and tangential components based on direction
    if direction == 0:  # x-direction
        vn_L, vn_R = vx_L, vx_R
        Bn_L, Bn_R = Bx_L, Bx_R
        Bt_L = jnp.sqrt(By_L**2 + Bz_L**2)
        Bt_R = jnp.sqrt(By_R**2 + Bz_R**2)
    elif direction == 1:  # y-direction
        vn_L, vn_R = vy_L, vy_R
        Bn_L, Bn_R = By_L, By_R
        Bt_L = jnp.sqrt(Bx_L**2 + Bz_L**2)
        Bt_R = jnp.sqrt(Bx_R**2 + Bz_R**2)
    else:  # z-direction
        vn_L, vn_R = vz_L, vz_R
        Bn_L, Bn_R = Bz_L, Bz_R
        Bt_L = jnp.sqrt(Bx_L**2 + By_L**2)
        Bt_R = jnp.sqrt(Bx_R**2 + By_R**2)

    # Average normal B (should be constant for div B = 0)
    Bn = 0.5 * (Bn_L + Bn_R)

    # Compute fast magnetosonic speeds
    cf_L = fast_magnetosonic_speed(rho_L, p_L, Bn, Bt_L, jnp.zeros_like(Bt_L), gamma)
    cf_R = fast_magnetosonic_speed(rho_R, p_R, Bn, Bt_R, jnp.zeros_like(Bt_R), gamma)

    # Compute physical fluxes (for B field only, since we only evolve B)
    # E = -v x B, so dB/dt = -curl(E) = curl(v x B)
    # The flux of By in x-direction is: By*vx - Bx*vy
    # The flux of Bz in x-direction is: Bz*vx - Bx*vz

    if direction == 0:  # x-direction fluxes
        # Flux of By: By*vx - Bx*vy
        F_By_L = By_L * vx_L - Bx_L * vy_L
        F_By_R = By_R * vx_R - Bx_R * vy_R
        # Flux of Bz: Bz*vx - Bx*vz
        F_Bz_L = Bz_L * vx_L - Bx_L * vz_L
        F_Bz_R = Bz_R * vx_R - Bx_R * vz_R

        # HLL flux for By and Bz
        flux_By = hll_flux_1d(By_L, By_R, F_By_L, F_By_R, rho_L, rho_R, vn_L, vn_R, cf_L, cf_R)
        flux_Bz = hll_flux_1d(Bz_L, Bz_R, F_Bz_L, F_Bz_R, rho_L, rho_R, vn_L, vn_R, cf_L, cf_R)

        # dB/dt = -d(flux)/dx = -(flux[i+1/2] - flux[i-1/2]) / dx
        # flux is at interface i+1/2, so flux[i-1/2] = roll(flux, 1)
        dx = geometry.dx
        dBy_dt = -(flux_By - jnp.roll(flux_By, 1, axis=0)) / dx
        dBz_dt = -(flux_Bz - jnp.roll(flux_Bz, 1, axis=0)) / dx
        dBx_dt = jnp.zeros_like(Bx)

    elif direction == 1:  # y-direction fluxes
        # Flux of Bx: Bx*vy - By*vx
        F_Bx_L = Bx_L * vy_L - By_L * vx_L
        F_Bx_R = Bx_R * vy_R - By_R * vx_R
        # Flux of Bz: Bz*vy - By*vz
        F_Bz_L = Bz_L * vy_L - By_L * vz_L
        F_Bz_R = Bz_R * vy_R - By_R * vz_R

        flux_Bx = hll_flux_1d(Bx_L, Bx_R, F_Bx_L, F_Bx_R, rho_L, rho_R, vn_L, vn_R, cf_L, cf_R)
        flux_Bz = hll_flux_1d(Bz_L, Bz_R, F_Bz_L, F_Bz_R, rho_L, rho_R, vn_L, vn_R, cf_L, cf_R)

        dy = geometry.dy
        dBx_dt = -(flux_Bx - jnp.roll(flux_Bx, 1, axis=1)) / dy
        dBz_dt = -(flux_Bz - jnp.roll(flux_Bz, 1, axis=1)) / dy
        dBy_dt = jnp.zeros_like(By)

    else:  # z-direction fluxes
        # Flux of Bx: Bx*vz - Bz*vx
        F_Bx_L = Bx_L * vz_L - Bz_L * vx_L
        F_Bx_R = Bx_R * vz_R - Bz_R * vx_R
        # Flux of By: By*vz - Bz*vy
        F_By_L = By_L * vz_L - Bz_L * vy_L
        F_By_R = By_R * vz_R - Bz_R * vy_R

        flux_Bx = hll_flux_1d(Bx_L, Bx_R, F_Bx_L, F_Bx_R, rho_L, rho_R, vn_L, vn_R, cf_L, cf_R)
        flux_By = hll_flux_1d(By_L, By_R, F_By_L, F_By_R, rho_L, rho_R, vn_L, vn_R, cf_L, cf_R)

        dz = geometry.dz
        dBx_dt = -(flux_Bx - jnp.roll(flux_Bx, 1, axis=2)) / dz
        dBy_dt = -(flux_By - jnp.roll(flux_By, 1, axis=2)) / dz
        dBz_dt = jnp.zeros_like(Bz)

    return jnp.stack([dBx_dt, dBy_dt, dBz_dt], axis=-1)
