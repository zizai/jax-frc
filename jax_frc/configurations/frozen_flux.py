"""Frozen-in flux validation configuration (Rm >> 1)."""
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Literal

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.models.hybrid_kinetic import HybridKinetic, RigidRotorEquilibrium
from jax_frc.models.coupled import CoupledModel, CoupledModelConfig
from jax_frc.models.neutral_fluid import NeutralFluid
from jax_frc.models.atomic_coupling import AtomicCoupling
from .base import AbstractConfiguration


ModelType = Literal["resistive_mhd", "extended_mhd", "plasma_neutral", "hybrid_kinetic"]


@dataclass
class FrozenFluxConfiguration(AbstractConfiguration):
    """Frozen-in flux test (Rm >> 1) in 3D Cartesian geometry.

    Physics (Cartesian induction equation):
        dB/dt = curl(v x B) + (1 / (mu0 * sigma)) * laplacian(B)

    Ideal-MHD limit (Rm >> 1, eta ~ 1 / (mu0 * sigma) -> 0):
        dB/dt = curl(v x B)

    Tests the advection-dominated regime where magnetic Reynolds number
    is much greater than 1. In 3D Cartesian coordinates with uniform
    velocity and uniform B, the induction equation predicts no change
    in B because curl(v x B) = 0.

    Setup: Cartesian slab with uniform expansion v_x = v0
    Initial: Uniform B_y field (maps to B_phi in legacy naming)
    Analytic: B_y(t) = constant (uniform field advects without distortion)

    Supports: resistive_mhd, extended_mhd, plasma_neutral, hybrid_kinetic
    """

    name: str = "frozen_flux"
    description: str = "Frozen-in magnetic flux advection (Rm >> 1)"

    # Grid parameters (thin-y slab for inexpensive 3D)
    nx: int = 64         # X resolution
    ny: int = 1          # Y resolution (thin periodic)
    nz: int = 64         # Z resolution
    r_min: float = 0.2   # X minimum [m] (legacy name for radial coordinate)
    r_max: float = 1.0   # X maximum [m]
    z_extent: float = 0.5  # Domain: y,z ∈ [-z_extent, z_extent]

    # Physics parameters
    B_phi_0: float = 1.0     # Initial B_phi [T]
    v_r: float = 0.1         # Radial expansion velocity [m/s]
    eta: float = 1e-8        # Very small resistivity (Rm >> 1)

    # Plasma parameters (for models that need them)
    n0: float = 1e19         # Density [m^-3]
    T0: float = 100.0        # Temperature [eV]

    # Model selection
    model_type: ModelType = "resistive_mhd"

    def build_geometry(self) -> Geometry:
        """3D Cartesian geometry."""
        return Geometry(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            x_min=self.r_min,
            x_max=self.r_max,
            y_min=-self.z_extent,
            y_max=self.z_extent,
            z_min=-self.z_extent,
            z_max=self.z_extent,
            bc_x="neumann",
            bc_y="periodic",
            bc_z="neumann",
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        """Uniform B_y with prescribed x-velocity v_x."""
        # Uniform toroidal field mapped to B_y
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 1].set(self.B_phi_0)

        # Uniform expansion velocity mapped to v_x
        v = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        v = v.at[:, :, :, 0].set(self.v_r)

        return State(
            B=B,
            E=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
            n=jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * self.n0,
            p=jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * self.n0 * self.T0,
            v=v,
        )

    def build_model(self):
        """Build physics model with minimal resistivity (Rm >> 1)."""
        if self.model_type == "resistive_mhd":
            return ResistiveMHD(eta=self.eta)

        elif self.model_type == "extended_mhd":
            return ExtendedMHD(eta=self.eta)

        elif self.model_type == "plasma_neutral":
            plasma_model = ResistiveMHD(eta=self.eta)
            neutral_model = NeutralFluid()
            coupling = AtomicCoupling()
            return CoupledModel(
                plasma_model=plasma_model,
                neutral_model=neutral_model,
                coupling=coupling,
                config=CoupledModelConfig(),
            )

        elif self.model_type == "hybrid_kinetic":
            equilibrium = RigidRotorEquilibrium(
                n0=self.n0,
                T0=self.T0,
                Omega=0.0,  # No rotation for frozen flux test
            )
            return HybridKinetic(
                equilibrium=equilibrium,
                eta=self.eta,
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def build_boundary_conditions(self) -> list:
        """Boundary conditions for expanding plasma."""
        return []

    def default_runtime(self) -> dict:
        """Runtime parameters based on advection timescale."""
        tau_adv = self.advection_timescale()
        return {
            "t_end": 0.5 * tau_adv,   # Run for half advection time
            "dt": 1e-3 * tau_adv,     # CFL-like timestep
        }

    def analytic_solution(self, r: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute analytic B_phi at time t for uniform advection.

        In 3D Cartesian coordinates with uniform velocity and uniform B,
        curl(v x B) = 0, so B remains constant in time.
        """
        return jnp.ones_like(r) * self.B_phi_0

    def analytic_solution_lagrangian(self, r0: jnp.ndarray, t: float) -> tuple:
        """Return unchanged positions and uniform B for this Cartesian setup."""
        r_t = r0 + self.v_r * t
        B_phi_t = jnp.ones_like(r0) * self.B_phi_0
        return r_t, B_phi_t

    def advection_timescale(self) -> float:
        """Characteristic advection time tau = L/v_r."""
        L = self.r_max - self.r_min
        return L / self.v_r

    def magnetic_reynolds_number(self) -> float:
        """Compute magnetic Reynolds number Rm = v*L/eta.

        For this configuration, Rm >> 1 by design.

        Returns:
            Magnetic Reynolds number (should be >> 1)
        """
        L = self.r_max - self.r_min
        return self.v_r * L / self.eta
