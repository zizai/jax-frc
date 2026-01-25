"""Rigid rotor analytic equilibrium for FRC."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jax import Array

from jax_frc.equilibrium.base import EquilibriumSolver, EquilibriumConstraints
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State

MU0 = 1.2566e-6
QE = 1.602e-19
MI = 1.673e-27


@dataclass
class RigidRotorEquilibrium(EquilibriumSolver):
    """Analytic rigid rotor FRC equilibrium.

    The rigid rotor model assumes all ions rotate at constant angular
    velocity Omega, leading to an analytic equilibrium solution.

    Key parameters:
    - r_s: Separatrix radius
    - z_s: Separatrix half-length
    - B_e: External axial field
    - Omega: Rotation frequency
    """

    r_s: float = 0.2          # Separatrix radius (m)
    z_s: float = 0.5          # Separatrix half-length (m)
    B_e: float = 0.1          # External field (T)
    Omega: float = 1e5        # Rotation frequency (rad/s)
    n0: float = 1e19          # Peak density (m^-3)
    T0: float = 1000.0        # Temperature (eV)

    def solve(self, geometry: Geometry, constraints: EquilibriumConstraints,
              initial_guess: Optional[Array] = None) -> State:
        """Generate analytic rigid rotor equilibrium."""
        nr, nz = geometry.nr, geometry.nz
        r = geometry.r_grid
        z = geometry.z_grid

        # Compute equilibrium psi
        psi = self._compute_psi(r, z)

        # Compute all profiles
        profiles = self.compute_profiles(psi, geometry, constraints)

        # Build state
        state = State.zeros(nr, nz)
        state = state.replace(
            psi=psi,
            n=profiles["density"],
            p=profiles["pressure"],
            B=profiles["B"],
            v=profiles["velocity"]
        )

        return state

    def _compute_psi(self, r: Array, z: Array) -> Array:
        """Compute poloidal flux function for rigid rotor.

        For a rigid rotor FRC, the flux function has the form:
        psi = psi_0 * (1 - (r/r_s)^2)^2 * exp(-(z/z_s)^2)

        inside the separatrix, and transitions to external field outside.
        """
        # Normalized coordinates
        r_norm = r / self.r_s
        z_norm = z / self.z_s

        # Inside separatrix: FRC flux
        psi_frc = (1 - r_norm**2)**2 * jnp.exp(-z_norm**2)

        # Outside separatrix: external field psi = B_e * r^2 / 2
        psi_ext = self.B_e * r**2 / 2

        # Smooth transition at separatrix
        r_sep = self.r_s * jnp.sqrt(1 - jnp.exp(-z_norm**2/2))  # Approximate separatrix
        inside = r < self.r_s * jnp.exp(-z_norm**2/4)

        # Scale FRC flux to match external field at separatrix
        psi_scale = self.B_e * self.r_s**2 / 4
        psi = jnp.where(inside, psi_scale * psi_frc, psi_ext * 0)

        return psi

    def compute_profiles(self, psi: Array, geometry: Geometry,
                        constraints: EquilibriumConstraints) -> dict:
        """Compute all equilibrium profiles for rigid rotor."""
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid
        z = geometry.z_grid

        # Magnetic field from psi
        dpsi_dr = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2*dr)
        dpsi_dz = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) / (2*dz)

        B_r = -dpsi_dz / jnp.maximum(r, 1e-10)
        B_z = dpsi_dr / jnp.maximum(r, 1e-10)
        B_phi = jnp.zeros_like(psi)  # No toroidal field

        B = jnp.stack([B_r, B_phi, B_z], axis=-1)

        # Density profile: peaked on axis
        r_norm = r / self.r_s
        z_norm = z / self.z_s
        density = self.n0 * jnp.exp(-(r_norm**2 + z_norm**2))

        # Pressure: p = n * k * T
        T_joules = self.T0 * QE
        pressure = density * T_joules

        # Velocity: rigid rotation v_phi = Omega * r
        v_r = jnp.zeros_like(psi)
        v_phi = self.Omega * r
        v_z = jnp.zeros_like(psi)
        velocity = jnp.stack([v_r, v_phi, v_z], axis=-1)

        # Current density
        delta_star_psi = self._laplace_star(psi, dr, dz, r)
        j_phi = -delta_star_psi / (MU0 * jnp.maximum(r, 1e-10))

        # Plasma beta
        B_mag = jnp.sqrt(B_r**2 + B_z**2 + 1e-20)
        beta = 2 * MU0 * pressure / B_mag**2

        return {
            "B": B,
            "B_r": B_r,
            "B_z": B_z,
            "B_phi": B_phi,
            "j_phi": j_phi,
            "density": density,
            "pressure": pressure,
            "velocity": velocity,
            "beta": beta,
        }

    @staticmethod
    def _laplace_star(psi, dr, dz, r):
        """Compute Delta* operator."""
        psi_rr = (jnp.roll(psi, -1, axis=0) - 2*psi + jnp.roll(psi, 1, axis=0)) / dr**2
        psi_zz = (jnp.roll(psi, -1, axis=1) - 2*psi + jnp.roll(psi, 1, axis=1)) / dz**2
        psi_r = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2*dr)
        return psi_rr - (1.0/jnp.maximum(r, 1e-10)) * psi_r + psi_zz

    def get_separatrix(self, n_points: int = 100) -> tuple:
        """Return (r, z) coordinates of the separatrix.

        Useful for plotting and diagnostics.
        """
        # Parametric representation of FRC separatrix
        theta = jnp.linspace(0, 2*jnp.pi, n_points)

        # Simple elliptical approximation
        r_sep = self.r_s * jnp.abs(jnp.sin(theta))
        z_sep = self.z_s * jnp.cos(theta)

        return r_sep, z_sep

    @classmethod
    def from_config(cls, config: dict) -> "RigidRotorEquilibrium":
        """Create from configuration dictionary."""
        return cls(
            r_s=float(config.get("r_s", 0.2)),
            z_s=float(config.get("z_s", 0.5)),
            B_e=float(config.get("B_e", 0.1)),
            Omega=float(config.get("Omega", 1e5)),
            n0=float(config.get("n0", 1e19)),
            T0=float(config.get("T0", 1000.0)),
        )
