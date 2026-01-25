"""Extended MHD physics model with Hall effect."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.models.resistivity import ResistivityModel, SpitzerResistivity
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

MU0 = 1.2566e-6
QE = 1.602e-19
ME = 9.109e-31

@dataclass
class HaloDensityModel:
    """Halo density model for vacuum region handling."""
    halo_density: float = 1e16      # Low density in vacuum region
    core_density: float = 1e19      # High density in plasma core
    r_cutoff: float = 0.8           # Transition radius (normalized)
    transition_width: float = 0.05  # Width of transition region

    def apply(self, n: jnp.ndarray, geometry: Geometry) -> jnp.ndarray:
        """Apply halo density model."""
        r_norm = (geometry.r_grid - geometry.r_min) / (geometry.r_max - geometry.r_min)
        halo_mask = 0.5 * (1 + jnp.tanh((r_norm - self.r_cutoff) / self.transition_width))
        return halo_mask * self.halo_density + (1 - halo_mask) * jnp.maximum(n, self.halo_density)


@dataclass
class ExtendedMHD(PhysicsModel):
    """Two-fluid extended MHD model with Hall effect.

    Includes:
    - Hall term: (J x B) / (ne)
    - Electron pressure gradient: -grad(p_e) / (ne)
    - Semi-implicit stepping for Whistler wave stability
    """

    resistivity: ResistivityModel
    halo_model: HaloDensityModel

    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute dB/dt from extended Ohm's law."""
        dr, dz = geometry.dr, geometry.dz

        # Apply halo density model
        n = self.halo_model.apply(state.n, geometry)
        n = jnp.maximum(n, self.halo_model.halo_density)  # Ensure positive density

        # Extract B components
        B_r = state.B[:, :, 0]
        B_phi = state.B[:, :, 1]
        B_z = state.B[:, :, 2]

        # Compute current density: J = curl(B) / mu_0
        r = geometry.r_grid
        J_r, J_phi, J_z = self._compute_current(B_r, B_phi, B_z, dr, dz, r)

        # Compute electric field from extended Ohm's law
        E_r, E_phi, E_z = self._extended_ohm_law(
            state.v, state.B, J_r, J_phi, J_z, n, state.p, dr, dz
        )

        # Faraday's law: dB/dt = -curl(E)
        dB_r, dB_phi, dB_z = self._compute_curl_E(E_r, E_phi, E_z, dr, dz)
        dB_r = -dB_r
        dB_phi = -dB_phi
        dB_z = -dB_z

        # Stack into (nr, nz, 3) array
        dB = jnp.stack([dB_r, dB_phi, dB_z], axis=-1)

        return state.replace(B=dB)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Whistler CFL constraint: dt < dx^2 * n * e * mu_0 / B."""
        n = self.halo_model.apply(state.n, geometry)
        n = jnp.maximum(n, self.halo_model.halo_density)

        B_mag = jnp.sqrt(jnp.sum(state.B**2, axis=-1))
        B_max = jnp.maximum(jnp.max(B_mag), 1e-10)

        dx_min = jnp.minimum(geometry.dr, geometry.dz)
        n_min = jnp.max(n)  # Use max density for stability

        # Whistler CFL: dt ~ dx^2 * n * e * mu_0 / B
        dt_whistler = dx_min**2 * n_min * QE * MU0 / B_max

        # Also check resistive diffusion
        eta_max = jnp.max(self.resistivity.compute(jnp.zeros_like(state.psi)))
        dt_resistive = 0.25 * dx_min**2 * MU0 / eta_max

        return jnp.minimum(dt_whistler, dt_resistive)

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply boundary conditions."""
        B = state.B

        # Conducting wall: B_tangential = 0 at boundaries
        # r boundaries
        B = B.at[0, :, :].set(0)
        B = B.at[-1, :, :].set(0)
        # z boundaries
        B = B.at[:, 0, :].set(0)
        B = B.at[:, -1, :].set(0)

        return state.replace(B=B)

    def _compute_current(self, B_r, B_phi, B_z, dr, dz, r):
        """Compute J = curl(B) / mu_0 in cylindrical coordinates.

        Curl in cylindrical (r, theta, z) with axisymmetry (d/dtheta = 0):
            J_r = -(1/mu_0) * dB_phi/dz
            J_phi = (1/mu_0) * (dB_r/dz - dB_z/dr)
            J_z = (1/mu_0) * (1/r) * d(r*B_phi)/dr
                = (1/mu_0) * (B_phi/r + dB_phi/dr)

        At r=0, uses L'Hopital's rule:
            J_z[0,:] = (1/mu_0) * 2 * dB_phi/dr[0,:]
        """
        # J_r = -(1/mu_0) * dB_phi/dz
        dB_phi_dz = (jnp.roll(B_phi, -1, axis=1) - jnp.roll(B_phi, 1, axis=1)) / (2 * dz)
        J_r = -(1.0 / MU0) * dB_phi_dz

        # J_phi = (1/mu_0) * (dB_r/dz - dB_z/dr)
        dB_r_dz = (jnp.roll(B_r, -1, axis=1) - jnp.roll(B_r, 1, axis=1)) / (2 * dz)
        dB_z_dr = (jnp.roll(B_z, -1, axis=0) - jnp.roll(B_z, 1, axis=0)) / (2 * dr)
        J_phi = (1.0 / MU0) * (dB_r_dz - dB_z_dr)

        # J_z = (1/mu_0) * (B_phi/r + dB_phi/dr)
        dB_phi_dr = (jnp.roll(B_phi, -1, axis=0) - jnp.roll(B_phi, 1, axis=0)) / (2 * dr)

        # Handle r=0 singularity
        r_safe = jnp.where(r > 1e-10, r, 1.0)
        J_z = (1.0 / MU0) * jnp.where(
            r > 1e-10,
            B_phi / r_safe + dB_phi_dr,
            2.0 * dB_phi_dr  # L'Hopital at r=0
        )

        return J_r, J_phi, J_z

    def _extended_ohm_law(self, v, B, J_r, J_phi, J_z, n, p_e, dr, dz):
        """Compute E from extended Ohm's law.

        E = -v x B + eta*J + (J x B)/(ne) - grad(p_e)/(ne)
        """
        v_r, v_phi, v_z = v[:, :, 0], v[:, :, 1], v[:, :, 2]
        B_r, B_phi, B_z = B[:, :, 0], B[:, :, 1], B[:, :, 2]

        # Term 1: -v x B
        vxB_r = v_phi * B_z - v_z * B_phi
        vxB_phi = v_z * B_r - v_r * B_z
        vxB_z = v_r * B_phi - v_phi * B_r

        # Term 2: eta * J
        eta = self.resistivity.compute(J_phi)
        etaJ_r = eta * J_r
        etaJ_phi = eta * J_phi
        etaJ_z = eta * J_z

        # Term 3: (J x B) / (ne) - Hall term
        JxB_r = J_phi * B_z - J_z * B_phi
        JxB_phi = J_z * B_r - J_r * B_z
        JxB_z = J_r * B_phi - J_phi * B_r

        ne = n * QE
        hall_r = JxB_r / ne
        hall_phi = JxB_phi / ne
        hall_z = JxB_z / ne

        # Term 4: -grad(p_e) / (ne) - Electron pressure
        dp_dr = (jnp.roll(p_e, -1, axis=0) - jnp.roll(p_e, 1, axis=0)) / (2 * dr)
        dp_dz = (jnp.roll(p_e, -1, axis=1) - jnp.roll(p_e, 1, axis=1)) / (2 * dz)

        pe_r = -dp_dr / ne
        pe_phi = jnp.zeros_like(p_e)  # No phi gradient (axisymmetric)
        pe_z = -dp_dz / ne

        # Sum all terms
        E_r = -vxB_r + etaJ_r + hall_r + pe_r
        E_phi = -vxB_phi + etaJ_phi + hall_phi + pe_phi
        E_z = -vxB_z + etaJ_z + hall_z + pe_z

        return E_r, E_phi, E_z

    def _compute_curl_E(self, E_r, E_phi, E_z, dr, dz):
        """Compute curl(E) for Faraday's law."""
        # curl_r = dE_z/dphi/r - dE_phi/dz ~ -dE_phi/dz (axisymmetric)
        dE_phi_dz = (jnp.roll(E_phi, -1, axis=1) - jnp.roll(E_phi, 1, axis=1)) / (2 * dz)
        curl_r = -dE_phi_dz

        # curl_phi = dE_r/dz - dE_z/dr
        dE_r_dz = (jnp.roll(E_r, -1, axis=1) - jnp.roll(E_r, 1, axis=1)) / (2 * dz)
        dE_z_dr = (jnp.roll(E_z, -1, axis=0) - jnp.roll(E_z, 1, axis=0)) / (2 * dr)
        curl_phi = dE_r_dz - dE_z_dr

        # curl_z = (1/r)*d(r*E_phi)/dr ~ dE_phi/dr (simplified)
        dE_phi_dr = (jnp.roll(E_phi, -1, axis=0) - jnp.roll(E_phi, 1, axis=0)) / (2 * dr)
        curl_z = dE_phi_dr

        return curl_r, curl_phi, curl_z

    @classmethod
    def from_config(cls, config: dict) -> "ExtendedMHD":
        """Create from configuration dictionary."""
        # Resistivity
        res_config = config.get("resistivity", {"type": "spitzer"})
        res_type = res_config.get("type", "spitzer")

        if res_type == "spitzer":
            resistivity = SpitzerResistivity(eta_0=float(res_config.get("eta_0", 1e-6)))
        else:
            resistivity = SpitzerResistivity(eta_0=float(res_config.get("eta_0", 1e-6)))

        # Halo model
        halo_config = config.get("halo", {})
        halo_model = HaloDensityModel(
            halo_density=float(halo_config.get("halo_density", 1e16)),
            core_density=float(halo_config.get("core_density", 1e19)),
            r_cutoff=float(halo_config.get("r_cutoff", 0.8)),
            transition_width=float(halo_config.get("transition_width", 0.05))
        )

        return cls(resistivity=resistivity, halo_model=halo_model)
