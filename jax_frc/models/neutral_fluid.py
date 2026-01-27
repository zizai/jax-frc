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

    All fields use SI units and 3D Cartesian coordinates.
    """

    rho_n: Array  # Mass density [kg/m³], shape (nx, ny, nz)
    mom_n: Array  # Momentum density [kg/m²/s], shape (nx, ny, nz, 3)
    E_n: Array    # Total energy density [J/m³], shape (nx, ny, nz)

    @property
    def v_n(self) -> Array:
        """Velocity [m/s], shape (nx, ny, nz, 3)."""
        rho_safe = jnp.maximum(self.rho_n[..., None], 1e-20)
        return self.mom_n / rho_safe

    @property
    def p_n(self) -> Array:
        """Pressure [Pa] from ideal gas EOS, shape (nx, ny, nz)."""
        rho_safe = jnp.maximum(self.rho_n, 1e-20)
        ke = 0.5 * jnp.sum(self.mom_n**2, axis=-1) / rho_safe
        internal_energy = self.E_n - ke
        return (GAMMA - 1) * jnp.maximum(internal_energy, 0.0)

    @property
    def T_n(self) -> Array:
        """Temperature [J], shape (nx, ny, nz)."""
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
    Uses 3D Cartesian coordinates.
    """

    gamma: float = GAMMA

    def compute_flux_divergence(
        self, state: NeutralState, geometry: "Geometry"
    ) -> Tuple[Array, Array, Array]:
        """Compute -div(F) for 3D Euler equations using dimension-by-dimension HLLE.

        Args:
            state: Current neutral state
            geometry: Grid geometry with dx, dy, dz

        Returns:
            d_rho: Mass density RHS [kg/m³/s], shape (nx, ny, nz)
            d_mom: Momentum density RHS [kg/m²/s²], shape (nx, ny, nz, 3)
            d_E: Energy density RHS [W/m³], shape (nx, ny, nz)
        """
        dx, dy, dz = geometry.dx, geometry.dy, geometry.dz
        rho = state.rho_n
        v = state.v_n
        p = state.p_n
        E = state.E_n

        # X-direction flux (axis=0, vel_idx=0)
        d_rho_x, d_mom_x, d_E_x = self._compute_flux_dir(rho, v, p, E, dx, axis=0, vel_idx=0)

        # Y-direction flux (axis=1, vel_idx=1)
        d_rho_y, d_mom_y, d_E_y = self._compute_flux_dir(rho, v, p, E, dy, axis=1, vel_idx=1)

        # Z-direction flux (axis=2, vel_idx=2)
        d_rho_z, d_mom_z, d_E_z = self._compute_flux_dir(rho, v, p, E, dz, axis=2, vel_idx=2)

        # Sum contributions from all directions
        d_rho = d_rho_x + d_rho_y + d_rho_z
        d_mom = d_mom_x + d_mom_y + d_mom_z
        d_E = d_E_x + d_E_y + d_E_z

        return d_rho, d_mom, d_E

    def _compute_flux_dir(self, rho, v, p, E, dx, axis, vel_idx):
        """Compute flux divergence in one direction using HLLE.

        Args:
            rho: Mass density, shape (nx, ny, nz)
            v: Velocity, shape (nx, ny, nz, 3)
            p: Pressure, shape (nx, ny, nz)
            E: Total energy, shape (nx, ny, nz)
            dx: Grid spacing in this direction
            axis: Array axis for this direction (0=x, 1=y, 2=z)
            vel_idx: Velocity component index (0=vx, 1=vy, 2=vz)

        Returns:
            d_rho: Mass density change, shape (nx, ny, nz)
            d_mom: Momentum density change, shape (nx, ny, nz, 3)
            d_E: Energy density change, shape (nx, ny, nz)
        """
        # Velocity component in flux direction
        v_dir = v[..., vel_idx]

        # Left and right states at cell interfaces (i-1/2)
        rho_L = jnp.roll(rho, 1, axis=axis)
        rho_R = rho
        v_L = jnp.roll(v_dir, 1, axis=axis)
        v_R = v_dir
        p_L = jnp.roll(p, 1, axis=axis)
        p_R = p
        E_L = jnp.roll(E, 1, axis=axis)
        E_R = E

        # HLLE flux at i-1/2 interfaces
        F_rho, F_mom, F_E = hlle_flux_1d(
            rho_L, rho_R, v_L, v_R, p_L, p_R, E_L, E_R, self.gamma
        )

        # Flux divergence: -(F_{i+1/2} - F_{i-1/2}) / dx
        # F at i+1/2 is jnp.roll(F, -1, axis) since F is stored at i-1/2
        d_rho = -(jnp.roll(F_rho, -1, axis=axis) - F_rho) / dx
        d_E = -(jnp.roll(F_E, -1, axis=axis) - F_E) / dx

        # Momentum: only the component in flux direction gets the HLLE flux
        d_mom = jnp.zeros_like(v)
        d_mom = d_mom.at[..., vel_idx].set(-(jnp.roll(F_mom, -1, axis=axis) - F_mom) / dx)

        # Transverse momentum advection: F_perp = rho * v_dir * v_perp
        # Use upwind for transverse components
        for perp_idx in range(3):
            if perp_idx != vel_idx:
                v_perp = v[..., perp_idx]
                mom_perp_L = jnp.roll(rho * v_perp, 1, axis=axis)
                mom_perp_R = rho * v_perp

                # Simple upwind flux for transverse momentum
                F_perp = jnp.where(
                    v_L + v_R > 0,
                    mom_perp_L * v_L,
                    mom_perp_R * v_R
                )

                d_mom = d_mom.at[..., perp_idx].add(
                    -(jnp.roll(F_perp, -1, axis=axis) - F_perp) / dx
                )

        return d_rho, d_mom, d_E

    def apply_boundary_conditions(
        self, state: NeutralState, geometry: "Geometry", bc_type: str = "periodic"
    ) -> NeutralState:
        """Apply boundary conditions to neutral state.

        Args:
            state: Current neutral state
            geometry: Grid geometry
            bc_type: "periodic" (default) - just returns state unchanged

        Returns:
            State with boundary conditions applied
        """
        # For periodic BCs (default in 3D Cartesian), nothing to do
        # The jnp.roll operations in flux computation naturally handle periodic BC
        return state
