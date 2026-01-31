"""GEM magnetic reconnection configuration."""
import jax.numpy as jnp
from dataclasses import dataclass

from jax_frc.simulation import Geometry, State
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
    description: str = "GEM reconnection challenge (ideal MHD fallback)"

    # Grid parameters (Cartesian naming)
    nx: int = 256
    ny: int = 1  # Pseudo-2D
    nz: int = 128
    # Match GEM reference domain in AGATE (x-z plane maps to AGATE x-y)
    x_min: float = -12.8
    x_max: float = 12.8
    y_min: float = 0.0
    y_max: float = 1.0
    z_min: float = -6.4
    z_max: float = 6.4

    # Harris sheet parameters
    B0: float = 1.0  # Asymptotic field
    lambda_: float = 0.5  # Current sheet half-width (tanh(2z) when lambda=0.5)
    n0: float = 1.0  # Peak density
    n_b: float = 0.2  # Background density fraction
    psi1: float = 0.1  # Perturbation amplitude (dB in AGATE)
    guide_field: float = 0.0  # Out-of-plane guide field
    eta: float = 0.0  # Resistivity (ideal Hall MHD)

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
            bc_x="periodic",
            bc_y="periodic",
            bc_z="neumann",
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        x = geometry.x_grid
        z = geometry.z_grid

        Lx = self.x_max - self.x_min
        Lz = self.z_max - self.z_min
        dB = self.psi1

        # AGATE GEM initial condition (x-z plane mapping of AGATE x-y plane)
        Bx = self.B0 * jnp.tanh(z / self.lambda_) - dB * (
            jnp.pi / Lz
        ) * jnp.cos(2.0 * jnp.pi * x / Lx) * jnp.sin(jnp.pi * z / Lz)
        Bz = dB * (2.0 * jnp.pi / Lx) * jnp.sin(2.0 * jnp.pi * x / Lx) * jnp.cos(
            jnp.pi * z / Lz
        )

        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 0].set(Bx)
        B = B.at[:, :, :, 2].set(Bz)
        if self.guide_field != 0.0:
            B = B.at[:, :, :, 1].set(self.guide_field)

        # Density: n = n_b + n0 * sech^2(z/lambda)
        sech_sq = 1.0 / jnp.cosh(z / self.lambda_) ** 2
        n = self.n_b + self.n0 * sech_sq

        # AGATE uses p = 0.5 * rho
        p = 0.5 * n

        return State(
            n=n,
            p=p,
            B=B,
            E=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
            v=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
        )

    def build_model(self) -> ExtendedMHD:
        return ExtendedMHD(
            eta=self.eta,
            include_hall=False,
            include_electron_pressure=False,
            apply_divergence_cleaning=True,
            normalized_units=True,
        )

    def build_boundary_conditions(self) -> list:
        return []

    def default_runtime(self) -> dict:
        return {"t_end": 25.0, "dt": 0.01}
