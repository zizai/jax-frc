"""Validation benchmark configurations (non-FRC-specific)."""
import jax.numpy as jnp
from dataclasses import dataclass

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.extended_mhd import ExtendedMHD
from .base import AbstractConfiguration


@dataclass
class CylindricalShockConfiguration(AbstractConfiguration):
    """Z-directed MHD shock tube (Brio-Wu adapted to Cartesian).

    Tests shock-capturing numerics in 3D Cartesian coordinates.
    Initial conditions are x/y-independent (1D physics in z).
    """

    name: str = "cylindrical_shock"
    description: str = "Brio-Wu shock tube in cylindrical coordinates"

    # Grid parameters
    nr: int = 16  # X resolution (legacy name for radial coordinate)
    ny: int = 4   # Thin periodic y direction
    nz: int = 512
    r_min: float = 0.01
    r_max: float = 0.5
    y_min: float = 0.0
    y_max: float = 2 * jnp.pi
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
            nx=self.nr,
            ny=self.ny,
            nz=self.nz,
            x_min=self.r_min,
            x_max=self.r_max,
            y_min=self.y_min,
            y_max=self.y_max,
            z_min=self.z_min,
            z_max=self.z_max,
            bc_x="neumann",
            bc_y="periodic",
            bc_z="neumann",
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        z = geometry.z_grid

        # Left/right states based on z
        rho = jnp.where(z < 0, self.rho_L, self.rho_R)
        p = jnp.where(z < 0, self.p_L, self.p_R)

        # Magnetic field: Bx reverses, Bz constant
        Bx = jnp.where(z < 0, self.Br_L, self.Br_R)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 0].set(Bx)
        B = B.at[:, :, :, 2].set(self.Bz)

        return State(
            n=rho,
            p=p,
            B=B,
            E=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
            v=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
        )

    def build_model(self) -> ResistiveMHD:
        return ResistiveMHD(eta=1e-8)

    def build_boundary_conditions(self) -> list:
        return []  # Dirichlet at z boundaries (fixed states)

    def default_runtime(self) -> dict:
        # t=0.1 in Alfven time units
        return {"t_end": 0.1, "dt": 1e-4}


@dataclass
class CylindricalVortexConfiguration(AbstractConfiguration):
    """Orszag-Tang vortex adapted to Cartesian slab.

    Tests nonlinear MHD dynamics, current sheet formation.
    Domain is a thin-y slab (periodic) to mimic 2D behavior.
    """

    name: str = "cylindrical_vortex"
    description: str = "Orszag-Tang vortex in cylindrical annulus"

    # Grid parameters
    nr: int = 256
    ny: int = 4
    nz: int = 256
    r_min: float = 0.2
    r_max: float = 1.2
    y_min: float = 0.0
    y_max: float = 2 * jnp.pi
    z_min: float = 0.0
    z_max: float = 2 * jnp.pi

    # Physics parameters (Orszag-Tang standard)
    v0: float = 1.0
    B0: float = 1.0
    rho0: float = 25.0 / (36.0 * jnp.pi)
    p0: float = 5.0 / (12.0 * jnp.pi)
    gamma: float = 5.0 / 3.0

    def build_geometry(self) -> Geometry:
        return Geometry(
            nx=self.nr,
            ny=self.ny,
            nz=self.nz,
            x_min=self.r_min,
            x_max=self.r_max,
            y_min=self.y_min,
            y_max=self.y_max,
            z_min=self.z_min,
            z_max=float(self.z_max),
            bc_x="neumann",
            bc_y="periodic",
            bc_z="periodic",
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        x = geometry.x_grid
        z = geometry.z_grid

        # Normalized radial coordinate for patterns
        r_norm = (x - self.r_min) / (self.r_max - self.r_min)

        # Velocity: vr = -v0*sin(z), vz = v0*sin(2*pi*r_norm)
        vr = -self.v0 * jnp.sin(z)
        vz = self.v0 * jnp.sin(2 * jnp.pi * r_norm)
        v = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        v = v.at[:, :, :, 0].set(vr)
        v = v.at[:, :, :, 2].set(vz)

        # Magnetic field: Br = -B0*sin(z), Bz = B0*sin(4*pi*r_norm)
        Br = -self.B0 * jnp.sin(z)
        Bz = self.B0 * jnp.sin(4 * jnp.pi * r_norm)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 0].set(Br)
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
        return ResistiveMHD(eta=1e-6)

    def build_boundary_conditions(self) -> list:
        # Periodic in z handled by geometry, conducting at r boundaries
        return []

    def default_runtime(self) -> dict:
        return {"t_end": 0.5, "dt": 1e-4}


@dataclass
class CylindricalGEMConfiguration(AbstractConfiguration):
    """GEM reconnection challenge adapted to Cartesian slab.

    Harris sheet current layer with Hall MHD.
    Tests Hall reconnection physics, quadrupole signature.
    """

    name: str = "cylindrical_gem"
    description: str = "GEM magnetic reconnection in cylindrical coordinates"

    # Grid parameters
    nr: int = 256
    ny: int = 4
    nz: int = 512
    r_min: float = 0.01
    r_max: float = 2.0
    y_min: float = 0.0
    y_max: float = 2 * jnp.pi
    z_min: float = -jnp.pi
    z_max: float = jnp.pi

    # Harris sheet parameters
    B0: float = 1.0  # Asymptotic field
    lambda_: float = 0.5  # Current sheet half-width (in d_i units)
    n0: float = 1.0  # Peak density
    n_b: float = 0.2  # Background density (fraction of n0)

    # Perturbation
    psi1: float = 0.1  # Perturbation amplitude (fraction of B0*lambda)

    def build_geometry(self) -> Geometry:
        return Geometry(
            nx=self.nr,
            ny=self.ny,
            nz=self.nz,
            x_min=self.r_min,
            x_max=self.r_max,
            y_min=self.y_min,
            y_max=self.y_max,
            z_min=float(self.z_min),
            z_max=float(self.z_max),
            bc_x="neumann",
            bc_y="periodic",
            bc_z="periodic",
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        x = geometry.x_grid
        z = geometry.z_grid

        # Harris sheet: Br = B0 * tanh(z/lambda)
        Bx = self.B0 * jnp.tanh(z / self.lambda_)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 0].set(Bx)

        # Density: n = n0 * sech^2(z/lambda) + n_b
        sech_sq = 1.0 / jnp.cosh(z / self.lambda_)**2
        n = self.n0 * sech_sq + self.n_b * self.n0

        # Pressure balance: p + B^2/(2*mu0) = const
        # At z=0: p_max, B=0
        # At z->inf: p_min, B=B0
        p_max = self.B0**2 / 2  # Total pressure (normalized)
        p = p_max - B[:, :, :, 0]**2 / 2
        p = jnp.maximum(p, 0.01)  # Floor to avoid negative pressure

        # Add perturbation to seed reconnection
        Lr = self.r_max - self.r_min
        psi_pert = self.psi1 * self.B0 * self.lambda_ * (
            jnp.cos(2 * jnp.pi * (x - self.r_min) / Lr) *
            jnp.cos(z / self.lambda_)
        )

        return State(
            n=n,
            p=p,
            B=B,
            E=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
            v=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
        )

    def build_model(self) -> ExtendedMHD:
        return ExtendedMHD(eta=1e-4)

    def build_boundary_conditions(self) -> list:
        return []

    def default_runtime(self) -> dict:
        return {"t_end": 25.0, "dt": 0.01}  # In Alfven time units
