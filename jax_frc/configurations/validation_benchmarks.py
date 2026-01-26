"""Validation benchmark configurations (non-FRC-specific)."""
import jax.numpy as jnp
from dataclasses import dataclass

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.models.extended_mhd import ExtendedMHD, HaloDensityModel
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


@dataclass
class CylindricalVortexConfiguration(AbstractConfiguration):
    """Orszag-Tang vortex adapted to cylindrical annulus.

    Tests nonlinear MHD dynamics, current sheet formation.
    Domain is an annulus to avoid axis singularity.
    """

    name: str = "cylindrical_vortex"
    description: str = "Orszag-Tang vortex in cylindrical annulus"

    # Grid parameters
    nr: int = 256
    nz: int = 256
    r_min: float = 0.2
    r_max: float = 1.2
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
            coord_system="cylindrical",
            r_min=self.r_min, r_max=self.r_max,
            z_min=self.z_min, z_max=float(self.z_max),
            nr=self.nr, nz=self.nz
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        r = geometry.r_grid
        z = geometry.z_grid

        # Normalized radial coordinate for patterns
        r_norm = (r - self.r_min) / (self.r_max - self.r_min)

        # Velocity: vr = -v0*sin(z), vz = v0*sin(2*pi*r_norm)
        vr = -self.v0 * jnp.sin(z)
        vz = self.v0 * jnp.sin(2 * jnp.pi * r_norm)
        v = jnp.zeros((geometry.nr, geometry.nz, 3))
        v = v.at[:, :, 0].set(vr)
        v = v.at[:, :, 2].set(vz)

        # Magnetic field: Br = -B0*sin(z), Bz = B0*sin(4*pi*r_norm)
        Br = -self.B0 * jnp.sin(z)
        Bz = self.B0 * jnp.sin(4 * jnp.pi * r_norm)
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 0].set(Br)
        B = B.at[:, :, 2].set(Bz)

        # Uniform density and pressure
        rho = jnp.ones((geometry.nr, geometry.nz)) * self.rho0
        p = jnp.ones((geometry.nr, geometry.nz)) * self.p0
        T = p / rho

        return State(
            psi=jnp.zeros((geometry.nr, geometry.nz)),
            n=rho,
            p=p,
            T=T,
            B=B,
            E=jnp.zeros((geometry.nr, geometry.nz, 3)),
            v=v,
            particles=None,
            time=0.0,
            step=0
        )

    def build_model(self) -> ResistiveMHD:
        return ResistiveMHD(
            resistivity=SpitzerResistivity(eta_0=1e-6)
        )

    def build_boundary_conditions(self) -> list:
        # Periodic in z handled by geometry, conducting at r boundaries
        return []

    def default_runtime(self) -> dict:
        return {"t_end": 0.5, "dt": 1e-4}


@dataclass
class CylindricalGEMConfiguration(AbstractConfiguration):
    """GEM reconnection challenge adapted to cylindrical coordinates.

    Harris sheet current layer with Hall MHD.
    Tests Hall reconnection physics, quadrupole signature.
    """

    name: str = "cylindrical_gem"
    description: str = "GEM magnetic reconnection in cylindrical coordinates"

    # Grid parameters
    nr: int = 256
    nz: int = 512
    r_min: float = 0.01
    r_max: float = 2.0
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
            coord_system="cylindrical",
            r_min=self.r_min, r_max=self.r_max,
            z_min=float(self.z_min), z_max=float(self.z_max),
            nr=self.nr, nz=self.nz
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        r = geometry.r_grid
        z = geometry.z_grid

        # Harris sheet: Br = B0 * tanh(z/lambda)
        Br = self.B0 * jnp.tanh(z / self.lambda_)
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 0].set(Br)

        # Density: n = n0 * sech^2(z/lambda) + n_b
        sech_sq = 1.0 / jnp.cosh(z / self.lambda_)**2
        n = self.n0 * sech_sq + self.n_b * self.n0

        # Pressure balance: p + B^2/(2*mu0) = const
        # At z=0: p_max, B=0
        # At z->inf: p_min, B=B0
        p_max = self.B0**2 / 2  # Total pressure (normalized)
        p = p_max - B[:, :, 0]**2 / 2
        p = jnp.maximum(p, 0.01)  # Floor to avoid negative pressure

        T = p / n

        # Add perturbation to seed reconnection
        Lr = self.r_max - self.r_min
        psi_pert = self.psi1 * self.B0 * self.lambda_ * (
            jnp.cos(2 * jnp.pi * (r - self.r_min) / Lr) *
            jnp.cos(z / self.lambda_)
        )

        return State(
            psi=psi_pert,
            n=n,
            p=p,
            T=T,
            B=B,
            E=jnp.zeros((geometry.nr, geometry.nz, 3)),
            v=jnp.zeros((geometry.nr, geometry.nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

    def build_model(self) -> ExtendedMHD:
        return ExtendedMHD(
            resistivity=SpitzerResistivity(eta_0=1e-4),
            halo_model=HaloDensityModel(
                halo_density=self.n_b * self.n0,
                core_density=self.n0
            ),
            # Hall term enabled by default in ExtendedMHD
        )

    def build_boundary_conditions(self) -> list:
        return []

    def default_runtime(self) -> dict:
        return {"t_end": 25.0, "dt": 0.01}  # In Alfven time units
