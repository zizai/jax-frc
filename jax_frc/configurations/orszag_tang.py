"""Orszag-Tang vortex configuration."""
import jax.numpy as jnp
from dataclasses import dataclass

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from .base import AbstractConfiguration


@dataclass
class OrszagTangConfiguration(AbstractConfiguration):
    """Orszag-Tang vortex in Cartesian slab geometry.

    Standard 2D MHD turbulence test case. Uses pseudo-2D domain
    (ny=1) to simulate 2D behavior in 3D Cartesian space.
    Tests nonlinear MHD dynamics and current sheet formation.
    """

    name: str = "orszag_tang"
    description: str = "Orszag-Tang vortex for MHD turbulence validation"

    # Grid parameters (Cartesian naming)
    nx: int = 256
    ny: int = 1  # Pseudo-2D
    nz: int = 256
    x_min: float = 0.0
    x_max: float = 2 * jnp.pi
    y_min: float = 0.0
    y_max: float = 2 * jnp.pi
    z_min: float = 0.0
    z_max: float = 2 * jnp.pi

    # Physics parameters
    v0: float = 1.0
    B0: float = 1.0
    rho0: float = 25.0 / (36.0 * jnp.pi)
    p0: float = 5.0 / (12.0 * jnp.pi)
    gamma: float = 5.0 / 3.0
    eta: float = 1e-6

    def build_geometry(self) -> Geometry:
        return Geometry(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            x_min=self.x_min,
            x_max=float(self.x_max),
            y_min=self.y_min,
            y_max=float(self.y_max),
            z_min=self.z_min,
            z_max=float(self.z_max),
            bc_x="periodic",
            bc_y="periodic",
            bc_z="periodic",
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        x = geometry.x_grid
        z = geometry.z_grid

        # Normalized coordinates for patterns
        x_norm = (x - self.x_min) / (self.x_max - self.x_min)

        # Velocity: vx = -v0*sin(z), vz = v0*sin(2*pi*x_norm)
        vx = -self.v0 * jnp.sin(z)
        vz = self.v0 * jnp.sin(2 * jnp.pi * x_norm)
        v = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        v = v.at[:, :, :, 0].set(vx)
        v = v.at[:, :, :, 2].set(vz)

        # Magnetic field: Bx = -B0*sin(z), Bz = B0*sin(4*pi*x_norm)
        Bx = -self.B0 * jnp.sin(z)
        Bz = self.B0 * jnp.sin(4 * jnp.pi * x_norm)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 0].set(Bx)
        B = B.at[:, :, :, 2].set(Bz)

        # Uniform density and pressure
        rho = jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * self.rho0
        p = jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * self.p0

        return State(
            n=rho,
            p=p,
            B=B,
            E=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
            v=v,
        )

    def build_model(self) -> ResistiveMHD:
        return ResistiveMHD(eta=self.eta)

    def build_boundary_conditions(self) -> list:
        return []

    def default_runtime(self) -> dict:
        return {"t_end": 0.5, "dt": 1e-4}
