"""Validation benchmark configurations (non-FRC-specific)."""
import jax.numpy as jnp
from dataclasses import dataclass

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity
from .base import AbstractConfiguration


@dataclass
class CylindricalShockConfiguration(AbstractConfiguration):
    """Z-directed MHD shock tube (Brio-Wu adapted to cylindrical).

    Tests shock-capturing numerics in cylindrical coordinates.
    Initial conditions are r-independent (1D physics in z).
    """

    name: str = "cylindrical_shock"
    description: str = "Brio-Wu shock tube in cylindrical coordinates"

    # Grid parameters
    nr: int = 16  # Minimal r resolution (r-uniform problem)
    nz: int = 512
    r_min: float = 0.01
    r_max: float = 0.5
    z_min: float = -1.0
    z_max: float = 1.0

    # Left state (z < 0)
    rho_L: float = 1.0
    p_L: float = 1.0
    Br_L: float = 1.0

    # Right state (z > 0)
    rho_R: float = 0.125
    p_R: float = 0.1
    Br_R: float = -1.0

    # Common
    Bz: float = 0.75  # Guide field
    gamma: float = 2.0  # Adiabatic index

    def build_geometry(self) -> Geometry:
        return Geometry(
            coord_system="cylindrical",
            r_min=self.r_min, r_max=self.r_max,
            z_min=self.z_min, z_max=self.z_max,
            nr=self.nr, nz=self.nz
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        z = geometry.z_grid

        # Left/right states based on z
        rho = jnp.where(z < 0, self.rho_L, self.rho_R)
        p = jnp.where(z < 0, self.p_L, self.p_R)

        # Magnetic field: Br reverses, Bz constant
        Br = jnp.where(z < 0, self.Br_L, self.Br_R)
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 0].set(Br)
        B = B.at[:, :, 2].set(self.Bz)

        # Temperature from ideal gas: p = n * T (normalized)
        T = p / rho

        return State(
            psi=jnp.zeros((geometry.nr, geometry.nz)),
            n=rho,
            p=p,
            T=T,
            B=B,
            E=jnp.zeros((geometry.nr, geometry.nz, 3)),
            v=jnp.zeros((geometry.nr, geometry.nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

    def build_model(self) -> ResistiveMHD:
        return ResistiveMHD(
            resistivity=SpitzerResistivity(eta_0=1e-8)
        )

    def build_boundary_conditions(self) -> list:
        return []  # Dirichlet at z boundaries (fixed states)

    def default_runtime(self) -> dict:
        # t=0.1 in Alfven time units
        return {"t_end": 0.1, "dt": 1e-4}
