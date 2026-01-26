"""Neutral fluid model for plasma-neutral coupling.

Implements Euler equations for neutral gas with atomic source terms.
"""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit

from jax_frc.constants import MI

# Adiabatic index for monatomic gas
GAMMA = 5.0 / 3.0


@dataclass(frozen=True)
class NeutralState:
    """Neutral fluid state variables.

    All fields use SI units.
    """

    rho_n: Array  # Mass density [kg/m³], shape (nr, nz)
    mom_n: Array  # Momentum density [kg/m²/s], shape (nr, nz, 3)
    E_n: Array    # Total energy density [J/m³], shape (nr, nz)

    @property
    def v_n(self) -> Array:
        """Velocity [m/s], shape (nr, nz, 3)."""
        rho_safe = jnp.maximum(self.rho_n[..., None], 1e-20)
        return self.mom_n / rho_safe

    @property
    def p_n(self) -> Array:
        """Pressure [Pa] from ideal gas EOS, shape (nr, nz)."""
        rho_safe = jnp.maximum(self.rho_n, 1e-20)
        ke = 0.5 * jnp.sum(self.mom_n**2, axis=-1) / rho_safe
        internal_energy = self.E_n - ke
        return (GAMMA - 1) * jnp.maximum(internal_energy, 0.0)

    @property
    def T_n(self) -> Array:
        """Temperature [J], shape (nr, nz)."""
        n_n = self.rho_n / MI
        n_safe = jnp.maximum(n_n, 1e-10)
        return self.p_n / n_safe

    def replace(self, **kwargs) -> "NeutralState":
        """Return new NeutralState with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)


# Register NeutralState as JAX pytree
def _neutral_state_flatten(state):
    children = (state.rho_n, state.mom_n, state.E_n)
    aux_data = None
    return children, aux_data


def _neutral_state_unflatten(aux_data, children):
    rho_n, mom_n, E_n = children
    return NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)


jax.tree_util.register_pytree_node(
    NeutralState, _neutral_state_flatten, _neutral_state_unflatten
)


@jit
def euler_flux_1d(rho: Array, v: Array, p: Array, E: Array) -> Tuple[Array, Array, Array]:
    """Compute 1D Euler fluxes.

    Args:
        rho: Mass density [kg/m³]
        v: Velocity component in flux direction [m/s]
        p: Pressure [Pa]
        E: Total energy density [J/m³]

    Returns:
        F_rho: Mass flux [kg/m²/s]
        F_mom: Momentum flux [Pa]
        F_E: Energy flux [W/m²]
    """
    F_rho = rho * v
    F_mom = rho * v**2 + p
    F_E = (E + p) * v
    return F_rho, F_mom, F_E


@jit
def hlle_flux_1d(
    rho_L: Array, rho_R: Array,
    v_L: Array, v_R: Array,
    p_L: Array, p_R: Array,
    E_L: Array, E_R: Array,
    gamma: float = GAMMA
) -> Tuple[Array, Array, Array]:
    """HLLE approximate Riemann solver for 1D Euler equations.

    Args:
        rho_L, rho_R: Left/right densities
        v_L, v_R: Left/right velocities (normal component)
        p_L, p_R: Left/right pressures
        E_L, E_R: Left/right total energies
        gamma: Adiabatic index

    Returns:
        F_rho, F_mom, F_E: Numerical fluxes at interface
    """
    # Sound speeds
    rho_L_safe = jnp.maximum(rho_L, 1e-20)
    rho_R_safe = jnp.maximum(rho_R, 1e-20)
    p_L_safe = jnp.maximum(p_L, 1e-10)
    p_R_safe = jnp.maximum(p_R, 1e-10)

    c_L = jnp.sqrt(gamma * p_L_safe / rho_L_safe)
    c_R = jnp.sqrt(gamma * p_R_safe / rho_R_safe)

    # Wave speed estimates (Davis)
    S_L = jnp.minimum(v_L - c_L, v_R - c_R)
    S_R = jnp.maximum(v_L + c_L, v_R + c_R)

    # Physical fluxes
    F_rho_L, F_mom_L, F_E_L = euler_flux_1d(rho_L, v_L, p_L, E_L)
    F_rho_R, F_mom_R, F_E_R = euler_flux_1d(rho_R, v_R, p_R, E_R)

    # Conserved variables
    U_rho_L, U_rho_R = rho_L, rho_R
    U_mom_L, U_mom_R = rho_L * v_L, rho_R * v_R
    U_E_L, U_E_R = E_L, E_R

    # Avoid division by zero
    dS = S_R - S_L
    dS_safe = jnp.where(jnp.abs(dS) < 1e-10, 1e-10, dS)

    # HLLE flux formula
    def hlle_component(F_L, F_R, U_L, U_R):
        return jnp.where(
            S_L >= 0, F_L,
            jnp.where(
                S_R <= 0, F_R,
                (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / dS_safe
            )
        )

    F_rho = hlle_component(F_rho_L, F_rho_R, U_rho_L, U_rho_R)
    F_mom = hlle_component(F_mom_L, F_mom_R, U_mom_L, U_mom_R)
    F_E = hlle_component(F_E_L, F_E_R, U_E_L, U_E_R)

    return F_rho, F_mom, F_E


@dataclass
class NeutralFluid:
    """Hydrodynamic neutral fluid model.

    Solves Euler equations with optional atomic source terms.
    """

    gamma: float = GAMMA

    def compute_flux_divergence(
        self, state: NeutralState, geometry: "Geometry"
    ) -> Tuple[Array, Array, Array]:
        """Compute -div(F) for Euler equations using HLLE.

        Args:
            state: Current neutral state
            geometry: Grid geometry

        Returns:
            d_rho: Mass density RHS [kg/m³/s]
            d_mom: Momentum density RHS [kg/m²/s²]
            d_E: Energy density RHS [W/m³]
        """
        dr, dz = geometry.dr, geometry.dz
        rho = state.rho_n
        v = state.v_n
        p = state.p_n
        E = state.E_n

        # Radial fluxes (r-direction, axis=0)
        F_r = self._compute_radial_flux(rho, v, p, E)

        # Axial fluxes (z-direction, axis=1)
        F_z = self._compute_axial_flux(rho, v, p, E)

        # Flux divergence: -d(F_r)/dr - d(F_z)/dz
        # Using central differences for divergence
        d_rho = -(
            (jnp.roll(F_r[0], -1, axis=0) - jnp.roll(F_r[0], 1, axis=0)) / (2 * dr) +
            (jnp.roll(F_z[0], -1, axis=1) - jnp.roll(F_z[0], 1, axis=1)) / (2 * dz)
        )

        # Momentum: handle each component
        d_mom_r = -(
            (jnp.roll(F_r[1], -1, axis=0) - jnp.roll(F_r[1], 1, axis=0)) / (2 * dr) +
            (jnp.roll(F_z[1], -1, axis=1) - jnp.roll(F_z[1], 1, axis=1)) / (2 * dz)
        )
        d_mom_theta = jnp.zeros_like(d_mom_r)  # No theta flux in axisymmetric
        d_mom_z = -(
            (jnp.roll(F_r[2], -1, axis=0) - jnp.roll(F_r[2], 1, axis=0)) / (2 * dr) +
            (jnp.roll(F_z[2], -1, axis=1) - jnp.roll(F_z[2], 1, axis=1)) / (2 * dz)
        )
        d_mom = jnp.stack([d_mom_r, d_mom_theta, d_mom_z], axis=-1)

        d_E = -(
            (jnp.roll(F_r[3], -1, axis=0) - jnp.roll(F_r[3], 1, axis=0)) / (2 * dr) +
            (jnp.roll(F_z[3], -1, axis=1) - jnp.roll(F_z[3], 1, axis=1)) / (2 * dz)
        )

        return d_rho, d_mom, d_E

    def _compute_radial_flux(self, rho, v, p, E):
        """Compute HLLE flux in r-direction at cell faces."""
        v_r = v[..., 0]  # Radial velocity

        # Left and right states (i-1/2 interface uses i-1 and i)
        rho_L = jnp.roll(rho, 1, axis=0)
        rho_R = rho
        v_L = jnp.roll(v_r, 1, axis=0)
        v_R = v_r
        p_L = jnp.roll(p, 1, axis=0)
        p_R = p
        E_L = jnp.roll(E, 1, axis=0)
        E_R = E

        F_rho, F_mom_r, F_E = hlle_flux_1d(
            rho_L, rho_R, v_L, v_R, p_L, p_R, E_L, E_R, self.gamma
        )

        # For momentum components perpendicular to flux direction,
        # flux is rho * v_r * v_perp
        mom_theta_L = jnp.roll(rho * v[..., 1], 1, axis=0)
        mom_theta_R = rho * v[..., 1]
        F_mom_theta = jnp.where(
            v_L + v_R > 0,
            v_L * mom_theta_L / jnp.maximum(rho_L, 1e-20),
            v_R * mom_theta_R / jnp.maximum(rho_R, 1e-20)
        ) * 0.5 * (rho_L + rho_R)

        mom_z_L = jnp.roll(rho * v[..., 2], 1, axis=0)
        mom_z_R = rho * v[..., 2]
        F_mom_z = jnp.where(
            v_L + v_R > 0,
            v_L * mom_z_L / jnp.maximum(rho_L, 1e-20),
            v_R * mom_z_R / jnp.maximum(rho_R, 1e-20)
        ) * 0.5 * (rho_L + rho_R)

        return (F_rho, F_mom_r, F_mom_z, F_E)

    def _compute_axial_flux(self, rho, v, p, E):
        """Compute HLLE flux in z-direction at cell faces."""
        v_z = v[..., 2]  # Axial velocity

        # Left and right states
        rho_L = jnp.roll(rho, 1, axis=1)
        rho_R = rho
        v_L = jnp.roll(v_z, 1, axis=1)
        v_R = v_z
        p_L = jnp.roll(p, 1, axis=1)
        p_R = p
        E_L = jnp.roll(E, 1, axis=1)
        E_R = E

        F_rho, F_mom_z, F_E = hlle_flux_1d(
            rho_L, rho_R, v_L, v_R, p_L, p_R, E_L, E_R, self.gamma
        )

        # Perpendicular momentum fluxes
        mom_r_L = jnp.roll(rho * v[..., 0], 1, axis=1)
        mom_r_R = rho * v[..., 0]
        F_mom_r = jnp.where(
            v_L + v_R > 0,
            v_L * mom_r_L / jnp.maximum(rho_L, 1e-20),
            v_R * mom_r_R / jnp.maximum(rho_R, 1e-20)
        ) * 0.5 * (rho_L + rho_R)

        return (F_rho, F_mom_r, F_mom_z, F_E)
