"""Resistive MHD physics model."""

from dataclasses import dataclass
from functools import partial
from typing import Union, Optional
import jax
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.models.resistivity import ResistivityModel, SpitzerResistivity, ChoduraResistivity
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.fields import CoilField

MU0 = 1.2566e-6

@dataclass(frozen=True)
class ResistiveMHD(PhysicsModel):
    """Single-fluid resistive MHD model.

    Solves: d(psi)/dt + v*grad(psi) = (eta/mu_0)*Delta*psi

    Args:
        resistivity: Model for plasma resistivity
        external_field: Optional external magnetic field from coils
    """

    resistivity: ResistivityModel
    external_field: Optional[CoilField] = None

    def get_total_B(self, state: State, geometry: Geometry, t: float = 0.0) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get total B field including external field contribution.

        The total magnetic field is the sum of:
        - B from psi (equilibrium field from plasma currents)
        - External field from coils (if present)

        In cylindrical coordinates, B from psi is:
        - B_r = -(1/r) * dpsi/dz
        - B_z = (1/r) * dpsi/dr

        Args:
            state: Current simulation state containing psi
            geometry: Computational geometry
            t: Time for time-dependent external fields

        Returns:
            (B_r, B_z): Radial and axial field components on the grid
        """
        psi = state.psi
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Compute B from psi using central differences
        # B_r = -(1/r) * dpsi/dz
        dpsi_dz = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) / (2 * dz)
        B_r_psi = -dpsi_dz / r

        # B_z = (1/r) * dpsi/dr
        dpsi_dr = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dr)
        B_z_psi = dpsi_dr / r

        # Add external field if present
        if self.external_field is not None:
            r_grid = geometry.r_grid
            z_grid = geometry.z_grid
            B_r_ext, B_z_ext = self.external_field.B_field(r_grid, z_grid, t)
            B_r = B_r_psi + B_r_ext
            B_z = B_z_psi + B_z_ext
        else:
            B_r = B_r_psi
            B_z = B_z_psi

        return B_r, B_z

    @partial(jax.jit, static_argnums=(0, 2))  # self and geometry are static
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute d(psi)/dt from Grad-Shafranov evolution."""
        psi = state.psi
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Compute Delta*psi
        delta_star_psi = self._laplace_star(psi, dr, dz, r)

        # Compute j_phi = -Delta*psi / (mu_0 * r)
        j_phi = -delta_star_psi / (MU0 * r)

        # Get resistivity
        eta = self.resistivity.compute(j_phi)

        # Diffusion: (eta/mu_0)*Delta*psi
        d_psi = (eta / MU0) * delta_star_psi

        # Advection: -v*grad(psi)
        # Always compute advection term (costs nothing when v=0, avoids tracing issues)
        v_r = state.v[:, :, 0]
        v_z = state.v[:, :, 2]
        dpsi_dr = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dr)
        dpsi_dz = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) / (2 * dz)
        d_psi = d_psi - (v_r * dpsi_dr + v_z * dpsi_dz)

        # Return state with d_psi as the RHS, zeros for all other fields
        return State(
            psi=d_psi,
            n=jnp.zeros_like(state.n),
            p=jnp.zeros_like(state.p),
            T=jnp.zeros_like(state.T),
            B=jnp.zeros_like(state.B),
            E=jnp.zeros_like(state.E),
            v=jnp.zeros_like(state.v),
            particles=None,
            time=0.0,
            step=0,
        )

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Diffusion CFL: dt < dx^2 / (4D) where D = eta_max/mu_0."""
        # Get maximum resistivity
        j_phi = self._compute_j_phi(state.psi, geometry)
        eta = self.resistivity.compute(j_phi)
        eta_max = jnp.max(eta)

        D = eta_max / MU0
        dx_min = jnp.minimum(geometry.dr, geometry.dz)
        return 0.25 * dx_min**2 / D

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply boundary conditions for conducting wall."""
        psi = state.psi

        # Inner boundary: Neumann (extrapolate)
        psi = psi.at[0, :].set(psi[1, :])
        # Outer boundaries: Dirichlet (psi = 0)
        psi = psi.at[-1, :].set(0)
        psi = psi.at[:, 0].set(0)
        psi = psi.at[:, -1].set(0)

        return state.replace(psi=psi)

    def _laplace_star(self, psi, dr, dz, r):
        """Compute Delta*psi = d^2(psi)/dr^2 - (1/r)*d(psi)/dr + d^2(psi)/dz^2."""
        psi_rr = (jnp.roll(psi, -1, axis=0) - 2*psi + jnp.roll(psi, 1, axis=0)) / dr**2
        psi_zz = (jnp.roll(psi, -1, axis=1) - 2*psi + jnp.roll(psi, 1, axis=1)) / dz**2
        psi_r = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dr)
        return psi_rr - (1.0 / r) * psi_r + psi_zz

    def _compute_j_phi(self, psi, geometry):
        """Compute toroidal current density."""
        delta_star = self._laplace_star(psi, geometry.dr, geometry.dz, geometry.r_grid)
        return -delta_star / (MU0 * geometry.r_grid)

    def explicit_rhs(self, state: State, geometry: Geometry, t: float = 0.0) -> State:
        """Advection term only: -v . grad(psi).

        This is the explicit part for IMEX splitting.
        """
        psi = state.psi
        v_r = state.v[:, :, 0]
        v_z = state.v[:, :, 2]
        dr, dz = geometry.dr, geometry.dz

        # Compute gradient of psi using central differences (consistent with compute_rhs)
        dpsi_dr = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dr)
        dpsi_dz = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) / (2 * dz)

        # Advection: -v . grad(psi)
        advection = -(v_r * dpsi_dr + v_z * dpsi_dz)

        # Return state with zeros for all other fields
        return State(
            psi=advection,
            n=jnp.zeros_like(state.n),
            p=jnp.zeros_like(state.p),
            T=jnp.zeros_like(state.T),
            B=jnp.zeros_like(state.B),
            E=jnp.zeros_like(state.E),
            v=jnp.zeros_like(state.v),
            particles=None,
            time=0.0,
            step=0,
        )

    def implicit_rhs(self, state: State, geometry: Geometry, t: float = 0.0) -> State:
        """Diffusion term only: (eta/mu0) * Delta*psi.

        This is the implicit part for IMEX splitting.
        """
        psi = state.psi
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Compute Delta*psi
        delta_star_psi = self._laplace_star(psi, dr, dz, r)

        # Get resistivity
        j_phi = -delta_star_psi / (MU0 * r)
        eta = self.resistivity.compute(j_phi)

        # Diffusion: (eta/mu_0)*Delta*psi
        diffusion = (eta / MU0) * delta_star_psi

        # Return state with zeros for all other fields
        return State(
            psi=diffusion,
            n=jnp.zeros_like(state.n),
            p=jnp.zeros_like(state.p),
            T=jnp.zeros_like(state.T),
            B=jnp.zeros_like(state.B),
            E=jnp.zeros_like(state.E),
            v=jnp.zeros_like(state.v),
            particles=None,
            time=0.0,
            step=0,
        )

    def apply_implicit_operator(
        self, state: State, geometry: Geometry, dt: float, theta: float
    ) -> State:
        """Apply (I - theta*dt*L) where L is diffusion operator.

        Used for matrix-free CG solve.
        """
        psi = state.psi
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Compute diffusion operator L*psi = (eta/mu0) * Delta*psi
        delta_star_psi = self._laplace_star(psi, dr, dz, r)
        j_phi = -delta_star_psi / (MU0 * r)
        eta = self.resistivity.compute(j_phi)
        L_psi = (eta / MU0) * delta_star_psi

        # Apply (I - theta*dt*L)
        new_psi = psi - theta * dt * L_psi

        return state.replace(psi=new_psi)

    @classmethod
    def from_config(cls, config: dict) -> "ResistiveMHD":
        """Create from configuration dictionary."""
        res_config = config.get("resistivity", {"type": "spitzer"})
        res_type = res_config.get("type", "spitzer")

        if res_type == "spitzer":
            resistivity = SpitzerResistivity(eta_0=float(res_config.get("eta_0", 1e-6)))
        elif res_type == "chodura":
            resistivity = ChoduraResistivity(
                eta_0=float(res_config.get("eta_0", 1e-6)),
                eta_anom=float(res_config.get("eta_anom", 1e-3)),
                threshold=float(res_config.get("threshold", 1e4))
            )
        else:
            raise ValueError(f"Unknown resistivity type: {res_type}")

        return cls(resistivity=resistivity)
