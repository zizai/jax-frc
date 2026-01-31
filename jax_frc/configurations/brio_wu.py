"""Brio-Wu MHD shock tube configuration."""
import jax.numpy as jnp
from dataclasses import dataclass

from jax_frc.simulation import Geometry, State
from jax_frc.models.resistive_mhd import ResistiveMHD
from .base import AbstractConfiguration


@dataclass
class BrioWuConfiguration(AbstractConfiguration):
    """Brio-Wu MHD shock tube in Cartesian slab geometry.

    Z-directed shock tube for testing shock-capturing numerics.
    Initial conditions are x/y-independent (1D physics in z).
    Uses pseudo-1D domain (nx=1, ny=1).
    """

    name: str = "brio_wu"
    description: str = "Brio-Wu MHD shock tube"

    # Grid parameters (Cartesian naming)
    nx: int = 1  # Pseudo-1D
    ny: int = 1  # Pseudo-1D
    nz: int = 512
    x_min: float = 0.0
    x_max: float = 1.0
    y_min: float = 0.0
    y_max: float = 1.0
    z_min: float = 0.0
    z_max: float = 1.0

    # Left state (z < 0)
    rho_L: float = 1.0
    p_L: float = 1.0
    Bx_L: float = 1.0

    # Right state (z > 0)
    rho_R: float = 0.125
    p_R: float = 0.1
    Bx_R: float = -1.0

    # Common
    Bz: float = 0.75  # Constant normal component (mapped to z-direction)
    gamma: float = 2.0
    eta: float = 1e-8

    def build_geometry(self) -> Geometry:
        return Geometry(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=float(self.y_max),
            z_min=self.z_min,
            z_max=self.z_max,
            bc_x="neumann",
            bc_y="periodic",
            bc_z="neumann",
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        z = geometry.z_grid

        z_mid = 0.5 * (geometry.z_min + geometry.z_max)
        rho = jnp.where(z < z_mid, self.rho_L, self.rho_R)
        p = jnp.where(z < z_mid, self.p_L, self.p_R)

        # Magnetic field: tangential component varies across z, normal component constant.
        # Map Brio-Wu x-directed normal field to z-directed setup.
        By = jnp.where(z < z_mid, self.Bx_L, self.Bx_R)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 1].set(By)
        B = B.at[:, :, :, 2].set(self.Bz)

        return State(
            n=rho,
            p=p,
            B=B,
            E=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
            v=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
        )

    def build_model(self) -> ResistiveMHD:
        return ResistiveMHD(eta=self.eta)

    def build_boundary_conditions(self) -> list:
        return []

    def default_runtime(self) -> dict:
        return {"t_end": 0.1, "dt": 1e-4}
