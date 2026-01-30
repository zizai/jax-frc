"""MHD conserved variable system for finite volume methods.

This module defines the conserved MHD state vector and provides
conversion functions between conserved and primitive variables.

Conserved Variables (8):
    - rho: Mass density
    - mom_x, mom_y, mom_z: Momentum density (rho * v)
    - E: Total energy density
    - Bx, By, Bz: Magnetic field components

Primitive Variables (8):
    - rho: Mass density
    - vx, vy, vz: Velocity components
    - p: Thermal pressure
    - Bx, By, Bz: Magnetic field components

References:
    [1] Toro (2009) "Riemann Solvers and Numerical Methods for Fluid Dynamics"
    [2] Stone et al. (2008) "Athena: A New Code for Astrophysical MHD"
"""

import jax.numpy as jnp
from jax import jit
from typing import NamedTuple, Tuple
from functools import partial


class MHDConserved(NamedTuple):
    """Conserved MHD state vector (8 variables).

    All fields have shape (nx, ny, nz).
    """
    rho: jnp.ndarray    # Mass density
    mom_x: jnp.ndarray  # Momentum x (rho * vx)
    mom_y: jnp.ndarray  # Momentum y (rho * vy)
    mom_z: jnp.ndarray  # Momentum z (rho * vz)
    E: jnp.ndarray      # Total energy density
    Bx: jnp.ndarray     # Magnetic field x
    By: jnp.ndarray     # Magnetic field y
    Bz: jnp.ndarray     # Magnetic field z


class MHDPrimitive(NamedTuple):
    """Primitive MHD state vector (8 variables).

    All fields have shape (nx, ny, nz).
    """
    rho: jnp.ndarray  # Mass density
    vx: jnp.ndarray   # Velocity x
    vy: jnp.ndarray   # Velocity y
    vz: jnp.ndarray   # Velocity z
    p: jnp.ndarray    # Thermal pressure
    Bx: jnp.ndarray   # Magnetic field x
    By: jnp.ndarray   # Magnetic field y
    Bz: jnp.ndarray   # Magnetic field z


# Density and pressure floors for numerical stability
RHO_FLOOR = 1e-12
P_FLOOR = 1e-12


@jit
def primitive_to_conserved(
    prim: MHDPrimitive,
    gamma: float = 5.0 / 3.0,
) -> MHDConserved:
    """Convert primitive variables to conserved variables.

    Args:
        prim: Primitive state (rho, v, p, B)
        gamma: Adiabatic index

    Returns:
        Conserved state (rho, mom, E, B)
    """
    rho = prim.rho
    vx, vy, vz = prim.vx, prim.vy, prim.vz
    p = prim.p
    Bx, By, Bz = prim.Bx, prim.By, prim.Bz

    # Momentum: mom = rho * v
    mom_x = rho * vx
    mom_y = rho * vy
    mom_z = rho * vz

    # Total energy: E = p/(gamma-1) + 0.5*rho*v^2 + 0.5*B^2
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2
    E = p / (gamma - 1.0) + 0.5 * rho * v2 + 0.5 * B2

    return MHDConserved(
        rho=rho,
        mom_x=mom_x,
        mom_y=mom_y,
        mom_z=mom_z,
        E=E,
        Bx=Bx,
        By=By,
        Bz=Bz,
    )


@jit
def conserved_to_primitive(
    cons: MHDConserved,
    gamma: float = 5.0 / 3.0,
) -> MHDPrimitive:
    """Convert conserved variables to primitive variables.

    Args:
        cons: Conserved state (rho, mom, E, B)
        gamma: Adiabatic index

    Returns:
        Primitive state (rho, v, p, B)
    """
    rho = jnp.maximum(cons.rho, RHO_FLOOR)
    mom_x, mom_y, mom_z = cons.mom_x, cons.mom_y, cons.mom_z
    E = cons.E
    Bx, By, Bz = cons.Bx, cons.By, cons.Bz

    # Velocity: v = mom / rho
    vx = mom_x / rho
    vy = mom_y / rho
    vz = mom_z / rho

    # Pressure: p = (gamma-1) * (E - 0.5*rho*v^2 - 0.5*B^2)
    v2 = vx**2 + vy**2 + vz**2
    B2 = Bx**2 + By**2 + Bz**2
    p = (gamma - 1.0) * (E - 0.5 * rho * v2 - 0.5 * B2)
    p = jnp.maximum(p, P_FLOOR)

    return MHDPrimitive(
        rho=rho,
        vx=vx,
        vy=vy,
        vz=vz,
        p=p,
        Bx=Bx,
        By=By,
        Bz=Bz,
    )


@partial(jit, static_argnums=(2,))
def compute_mhd_flux(
    prim: MHDPrimitive,
    cons: MHDConserved,
    direction: int,
    gamma: float = 5.0 / 3.0,
) -> MHDConserved:
    """Compute physical MHD flux in a given direction.

    The MHD flux in direction d is:
        F_rho = rho * v_d
        F_mom_i = rho * v_i * v_d + (p + B^2/2) * delta_id - B_i * B_d
        F_E = (E + p + B^2/2) * v_d - B_d * (v . B)
        F_B_i = B_i * v_d - B_d * v_i  (for i != d)
        F_B_d = 0  (div B = 0 constraint)

    Args:
        prim: Primitive state
        cons: Conserved state
        direction: Flux direction (0=x, 1=y, 2=z)
        gamma: Adiabatic index

    Returns:
        MHD flux as MHDConserved tuple
    """
    rho = prim.rho
    vx, vy, vz = prim.vx, prim.vy, prim.vz
    p = prim.p
    Bx, By, Bz = prim.Bx, prim.By, prim.Bz
    E = cons.E

    # Magnetic pressure
    B2 = Bx**2 + By**2 + Bz**2
    p_mag = 0.5 * B2

    # Total pressure
    p_tot = p + p_mag

    # v dot B
    vdotB = vx * Bx + vy * By + vz * Bz

    if direction == 0:  # x-direction
        vn = vx
        Bn = Bx
        F_rho = rho * vn
        F_mom_x = rho * vx * vn + p_tot - Bx * Bn
        F_mom_y = rho * vy * vn - By * Bn
        F_mom_z = rho * vz * vn - Bz * Bn
        F_E = (E + p_tot) * vn - Bn * vdotB
        F_Bx = jnp.zeros_like(Bx)  # div B = 0
        F_By = By * vn - Bn * vy
        F_Bz = Bz * vn - Bn * vz

    elif direction == 1:  # y-direction
        vn = vy
        Bn = By
        F_rho = rho * vn
        F_mom_x = rho * vx * vn - Bx * Bn
        F_mom_y = rho * vy * vn + p_tot - By * Bn
        F_mom_z = rho * vz * vn - Bz * Bn
        F_E = (E + p_tot) * vn - Bn * vdotB
        F_Bx = Bx * vn - Bn * vx
        F_By = jnp.zeros_like(By)  # div B = 0
        F_Bz = Bz * vn - Bn * vz

    else:  # z-direction
        vn = vz
        Bn = Bz
        F_rho = rho * vn
        F_mom_x = rho * vx * vn - Bx * Bn
        F_mom_y = rho * vy * vn - By * Bn
        F_mom_z = rho * vz * vn + p_tot - Bz * Bn
        F_E = (E + p_tot) * vn - Bn * vdotB
        F_Bx = Bx * vn - Bn * vx
        F_By = By * vn - Bn * vy
        F_Bz = jnp.zeros_like(Bz)  # div B = 0

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


def state_to_conserved(state: "State", gamma: float = 5.0 / 3.0) -> MHDConserved:
    """Convert JAX-FRC State to MHDConserved.

    Args:
        state: JAX-FRC State object
        gamma: Adiabatic index

    Returns:
        MHDConserved tuple
    """
    rho = state.n  # Assuming normalized units where n = rho
    vx = state.v[..., 0]
    vy = state.v[..., 1]
    vz = state.v[..., 2]
    p = state.p
    Bx = state.B[..., 0]
    By = state.B[..., 1]
    Bz = state.B[..., 2]

    prim = MHDPrimitive(rho=rho, vx=vx, vy=vy, vz=vz, p=p, Bx=Bx, By=By, Bz=Bz)
    return primitive_to_conserved(prim, gamma)


def conserved_to_state(
    cons: MHDConserved,
    state: "State",
    gamma: float = 5.0 / 3.0,
) -> "State":
    """Convert MHDConserved back to JAX-FRC State.

    Args:
        cons: MHDConserved tuple
        state: Original State (for structure)
        gamma: Adiabatic index

    Returns:
        Updated JAX-FRC State
    """
    prim = conserved_to_primitive(cons, gamma)

    v = jnp.stack([prim.vx, prim.vy, prim.vz], axis=-1)
    B = jnp.stack([prim.Bx, prim.By, prim.Bz], axis=-1)

    return state.replace(n=prim.rho, v=v, p=prim.p, B=B)
