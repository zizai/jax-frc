"""Resistive MHD physics model."""

from dataclasses import dataclass
from typing import Union
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.models.resistivity import ResistivityModel, SpitzerResistivity, ChoduraResistivity
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

MU0 = 1.2566e-6

@dataclass
class ResistiveMHD(PhysicsModel):
    """Single-fluid resistive MHD model.

    Solves: d(psi)/dt + v*grad(psi) = (eta/mu_0)*Delta*psi
    """

    resistivity: ResistivityModel

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

        # Advection: -v*grad(psi) (if velocity present)
        v_r = state.v[:, :, 0]
        v_z = state.v[:, :, 2]
        if jnp.any(v_r != 0) or jnp.any(v_z != 0):
            dpsi_dr = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dr)
            dpsi_dz = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) / (2 * dz)
            d_psi = d_psi - (v_r * dpsi_dr + v_z * dpsi_dz)

        # Return state with d_psi as the RHS (stored in psi temporarily)
        return state.replace(psi=d_psi)

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
