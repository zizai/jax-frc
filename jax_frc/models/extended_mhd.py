"""3D Extended MHD model with Hall physics."""

from dataclasses import dataclass
from functools import partial
from typing import Optional
import jax
import jax.numpy as jnp

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.operators import curl_3d, gradient_3d, laplacian_3d
from jax_frc.constants import MU0, QE


@dataclass(frozen=True)
class HaloDensityModel:
    """Halo density model for vacuum region handling (3D version).

    For 3D Cartesian geometry, the halo region is applied based on
    distance from domain center rather than radial coordinate.

    Attributes:
        halo_density: Low density in vacuum region [m^-3]
        core_density: High density in plasma core [m^-3]
        r_cutoff: Transition radius (normalized to domain size)
        transition_width: Width of transition region
    """
    halo_density: float = 1e16
    core_density: float = 1e19
    r_cutoff: float = 0.8
    transition_width: float = 0.05

    def apply(self, n: jnp.ndarray, geometry: Geometry) -> jnp.ndarray:
        """Apply halo density model in 3D.

        Uses distance from domain center normalized by domain size.
        """
        # Compute normalized distance from center
        x_center = (geometry.x_max + geometry.x_min) / 2
        y_center = (geometry.y_max + geometry.y_min) / 2
        Lx = geometry.x_max - geometry.x_min
        Ly = geometry.y_max - geometry.y_min

        x = geometry.x_grid
        y = geometry.y_grid

        # Normalized radius in x-y plane
        r_norm = jnp.sqrt(((x - x_center) / (Lx / 2))**2 +
                         ((y - y_center) / (Ly / 2))**2)

        halo_mask = 0.5 * (1 + jnp.tanh((r_norm - self.r_cutoff) / self.transition_width))
        return halo_mask * self.halo_density + (1 - halo_mask) * jnp.maximum(n, self.halo_density)


@dataclass(frozen=True)
class TemperatureBoundaryCondition:
    """Temperature boundary condition settings for 3D.

    Supports:
    - Dirichlet: T = T_wall (fixed wall temperature)
    - Neumann: dT/dn = 0 (insulating wall, zero heat flux)

    Attributes:
        bc_type: "dirichlet" or "neumann"
        T_wall: Wall temperature [eV] for Dirichlet BC
        apply_axis_symmetry: Legacy parameter (ignored in 3D Cartesian)
    """
    bc_type: str = "neumann"
    T_wall: float = 10.0
    apply_axis_symmetry: bool = True  # Kept for compatibility, ignored in 3D


@dataclass(frozen=True)
class ExtendedMHD(PhysicsModel):
    """Extended MHD model with Hall and electron pressure terms.

    Generalized Ohm's law:
    E = -v x B + eta*J + (J x B)/(ne) - grad(p_e)/(ne)

    Attributes:
        eta: Resistivity [Ohm*m]
        include_hall: Include Hall term (J x B)/(ne)
        include_electron_pressure: Include grad(p_e)/(ne) term
        kappa_parallel: Parallel thermal conductivity
        kappa_perp: Perpendicular thermal conductivity
    """
    eta: float = 1e-4
    include_hall: bool = True
    include_electron_pressure: bool = True
    kappa_parallel: float = 1e20
    kappa_perp: float = 1e18

    @partial(jax.jit, static_argnums=(0, 2))  # self and geometry are static
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for extended MHD."""
        B = state.B
        n = jnp.maximum(state.n, 1e16)  # Avoid division by zero

        # Current density: J = curl(B) / mu_0
        J = curl_3d(B, geometry) / MU0

        # Start with resistive term: E = eta*J
        E = self.eta * J

        # Hall term: (J x B) / (ne)
        if self.include_hall:
            J_cross_B = jnp.stack([
                J[..., 1] * B[..., 2] - J[..., 2] * B[..., 1],
                J[..., 2] * B[..., 0] - J[..., 0] * B[..., 2],
                J[..., 0] * B[..., 1] - J[..., 1] * B[..., 0],
            ], axis=-1)
            E = E + J_cross_B / (n[..., None] * QE)

        # Electron pressure term: -grad(p_e) / (ne)
        if self.include_electron_pressure and state.Te is not None:
            p_e = n * state.Te  # Electron pressure
            grad_pe = gradient_3d(p_e, geometry)
            E = E - grad_pe / (n[..., None] * QE)

        # Convective term: -v x B
        if state.v is not None:
            v = state.v
            v_cross_B = jnp.stack([
                v[..., 1] * B[..., 2] - v[..., 2] * B[..., 1],
                v[..., 2] * B[..., 0] - v[..., 0] * B[..., 2],
                v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0],
            ], axis=-1)
            E = E - v_cross_B

        # Faraday's law: dB/dt = -curl(E)
        dB_dt = -curl_3d(E, geometry)

        # Temperature evolution with thermal conduction
        dTe_dt = None
        if state.Te is not None:
            # Simplified isotropic conduction for now
            lap_Te = laplacian_3d(state.Te, geometry)
            dTe_dt = self.kappa_perp * lap_Te / (n * 1.5)  # 3/2 * n * dT/dt

        return state.replace(B=dB_dt, Te=dTe_dt)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """CFL constraint including Hall term."""
        dx_min = min(geometry.dx, geometry.dy, geometry.dz)

        # Resistive diffusion
        dt_resistive = 0.25 * dx_min**2 * MU0 / self.eta if self.eta > 0 else jnp.inf

        # Hall wave (whistler): omega ~ k^2 * B / (mu0 * n * e)
        if self.include_hall:
            B_max = jnp.max(jnp.sqrt(jnp.sum(state.B**2, axis=-1)))
            n_min = jnp.maximum(jnp.min(state.n), 1e16)
            whistler_speed = B_max / (MU0 * n_min * QE) * (2 * jnp.pi / dx_min)
            dt_hall = 0.1 * dx_min / jnp.maximum(whistler_speed, 1e-10)
        else:
            dt_hall = jnp.inf

        return float(jnp.minimum(dt_resistive, dt_hall))

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply boundary conditions."""
        return state
