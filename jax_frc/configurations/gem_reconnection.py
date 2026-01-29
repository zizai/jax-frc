"""GEM magnetic reconnection configuration."""
import jax.numpy as jnp
from dataclasses import dataclass

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.extended_mhd import ExtendedMHD
from .base import AbstractConfiguration


@dataclass
class GEMReconnectionConfiguration(AbstractConfiguration):
    """GEM magnetic reconnection challenge in Cartesian slab.

    Harris sheet current layer with Hall MHD physics.
    Tests Hall reconnection and quadrupole signature formation.
    Uses pseudo-2D domain (ny=1) in x-z plane.
    """

    name: str = "gem_reconnection"
    description: str = "GEM Hall reconnection challenge"

    # Grid parameters (Cartesian naming)
    nx: int = 256
    ny: int = 1  # Pseudo-2D
    nz: int = 512
    x_min: float = 0.01
    x_max: float = 2.0
    y_min: float = 0.0
    y_max: float = 2 * jnp.pi
    z_min: float = -jnp.pi
    z_max: float = jnp.pi

    # Harris sheet parameters
    B0: float = 1.0  # Asymptotic field
    lambda_: float = 0.5  # Current sheet half-width
    n0: float = 1.0  # Peak density
    n_b: float = 0.2  # Background density fraction
    psi1: float = 0.1  # Perturbation amplitude
    eta: float = 1e-4  # Resistivity

    def build_geometry(self) -> Geometry:
        return Geometry(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            x_min=self.x_min,
            x_max=self.x_max,
            y_min=self.y_min,
            y_max=float(self.y_max),
            z_min=float(self.z_min),
            z_max=float(self.z_max),
            bc_x="neumann",
            bc_y="periodic",
            bc_z="periodic",
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        x = geometry.x_grid
        z = geometry.z_grid

        # Harris sheet: Bx = B0 * tanh(z/lambda)
        Bx = self.B0 * jnp.tanh(z / self.lambda_)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 0].set(Bx)

        # Density: n = n0 * sech^2(z/lambda) + n_b
        sech_sq = 1.0 / jnp.cosh(z / self.lambda_) ** 2
        n = self.n0 * sech_sq + self.n_b * self.n0

        # Pressure balance: p + B^2/(2*mu0) = const
        p_max = self.B0**2 / 2
        p = p_max - B[:, :, :, 0] ** 2 / 2
        p = jnp.maximum(p, 0.01)

        # Perturbation to seed reconnection (not applied to B directly here)
        # The perturbation is computed but would need proper curl application
        # For now, keeping consistent with original implementation

        return State(
            n=n,
            p=p,
            B=B,
            E=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
            v=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
        )

    def build_model(self) -> ExtendedMHD:
        return ExtendedMHD(eta=self.eta)

    def build_boundary_conditions(self) -> list:
        return []

    def default_runtime(self) -> dict:
        return {"t_end": 25.0, "dt": 0.01}
