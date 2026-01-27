"""Frozen-in flux validation configuration (Rm >> 1)."""
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Literal

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.extended_mhd import ExtendedMHD, HaloDensityModel
from jax_frc.models.hybrid_kinetic import HybridKinetic, RigidRotorEquilibrium
from jax_frc.models.coupled import CoupledModel, CoupledModelConfig
from jax_frc.models.neutral_fluid import NeutralFluid
from jax_frc.models.atomic_coupling import AtomicCoupling
from jax_frc.models.resistivity import SpitzerResistivity
from .base import AbstractConfiguration


ModelType = Literal["resistive_mhd", "extended_mhd", "plasma_neutral", "hybrid_kinetic"]


@dataclass
class FrozenFluxConfiguration(AbstractConfiguration):
    """Frozen-in flux test (Rm >> 1).

    Physics: ∂B/∂t ≈ ∇×(v×B) (ideal MHD, flux frozen to plasma)

    Tests the advection-dominated regime where magnetic Reynolds number
    is much greater than 1. The magnetic field is "frozen" into the plasma
    and moves with it.

    Setup: Radial geometry with uniform expansion v_r = v₀
    Initial: Uniform B_φ in annular region
    Analytic: B_φ(t) = B₀ · r₀/(r₀ + v₀t) from flux conservation

    The toroidal flux Φ = ∫B_φ dr is conserved. As the plasma expands
    radially, B_φ decreases to maintain constant flux.

    Supports: resistive_mhd, extended_mhd, plasma_neutral, hybrid_kinetic
    """

    name: str = "frozen_flux"
    description: str = "Frozen-in magnetic flux advection (Rm >> 1)"

    # Grid parameters (annular to avoid axis singularity)
    nr: int = 64         # High radial resolution for flux advection
    nz: int = 8          # Minimal axial (uniform in z)
    r_min: float = 0.2   # Inner radius [m]
    r_max: float = 1.0   # Outer radius [m]
    z_extent: float = 0.5  # Domain: z ∈ [-z_extent, z_extent]

    # Physics parameters
    B_phi_0: float = 1.0     # Initial B_φ [T]
    v_r: float = 0.1         # Radial expansion velocity [m/s]
    eta: float = 1e-8        # Very small resistivity (Rm >> 1)

    # Plasma parameters (for models that need them)
    n0: float = 1e19         # Density [m^-3]
    T0: float = 100.0        # Temperature [eV]

    # Model selection
    model_type: ModelType = "resistive_mhd"

    def build_geometry(self) -> Geometry:
        """Annular cylindrical geometry, high r resolution."""
        return Geometry(
            coord_system="cylindrical",
            r_min=self.r_min,
            r_max=self.r_max,
            z_min=-self.z_extent,
            z_max=self.z_extent,
            nr=self.nr,
            nz=self.nz,
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        """Uniform B_φ with prescribed radial velocity v_r."""
        # Uniform toroidal field B_φ
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 1].set(self.B_phi_0)  # B_φ component (index 1 in cylindrical)

        # Uniform radial expansion velocity
        v = jnp.zeros((geometry.nr, geometry.nz, 3))
        v = v.at[:, :, 0].set(self.v_r)  # v_r component

        return State(
            psi=jnp.zeros((geometry.nr, geometry.nz)),
            n=jnp.ones((geometry.nr, geometry.nz)) * self.n0,
            p=jnp.ones((geometry.nr, geometry.nz)) * self.n0 * self.T0,  # p = nT
            T=jnp.ones((geometry.nr, geometry.nz)) * self.T0,
            B=B,
            E=jnp.zeros((geometry.nr, geometry.nz, 3)),
            v=v,
            particles=None,
            time=0.0,
            step=0,
        )

    def build_model(self):
        """Build physics model with minimal resistivity (Rm >> 1)."""
        resistivity = SpitzerResistivity(eta_0=self.eta)

        if self.model_type == "resistive_mhd":
            return ResistiveMHD(resistivity=resistivity)

        elif self.model_type == "extended_mhd":
            return ExtendedMHD(
                resistivity=resistivity,
                halo_model=HaloDensityModel(
                    halo_density=self.n0,
                    core_density=self.n0,
                ),
                thermal=None,  # No thermal transport for this test
            )

        elif self.model_type == "plasma_neutral":
            plasma_model = ResistiveMHD(resistivity=resistivity)
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
        """Compute analytic B_φ at time t from flux conservation.

        For uniform radial expansion with v_r = const, the toroidal flux
        Φ = ∫B_φ dr is conserved. This gives:

            B_φ(t) = B₀ · r₀ / (r₀ + v_r·t)

        where r₀ is the initial inner radius.

        Note: This simplified solution assumes the field decreases uniformly.
        A more accurate treatment would track the Lagrangian displacement
        of each fluid element.

        Args:
            r: Radial coordinates [m] (unused in simplified model)
            t: Time [s]

        Returns:
            B_φ field value at time t
        """
        # Simplified: uniform decrease based on inner radius expansion
        r_inner_t = self.r_min + self.v_r * t
        return self.B_phi_0 * self.r_min / r_inner_t

    def analytic_solution_lagrangian(self, r0: jnp.ndarray, t: float) -> tuple:
        """Compute B_φ tracking Lagrangian fluid elements.

        For a fluid element initially at r₀, its position at time t is:
            r(t) = r₀ + v_r·t

        Flux conservation for that element gives:
            B_φ(r,t) · r = B_φ₀ · r₀
            B_φ(r,t) = B_φ₀ · r₀ / r(t)

        Args:
            r0: Initial radial coordinates [m]
            t: Time [s]

        Returns:
            (r_t, B_phi_t): New positions and B_φ values
        """
        r_t = r0 + self.v_r * t
        B_phi_t = self.B_phi_0 * r0 / r_t
        return r_t, B_phi_t

    def advection_timescale(self) -> float:
        """Characteristic advection time τ = L/v_r."""
        L = self.r_max - self.r_min
        return L / self.v_r

    def magnetic_reynolds_number(self) -> float:
        """Compute magnetic Reynolds number Rm = v·L/η.

        For this configuration, Rm >> 1 by design.

        Returns:
            Magnetic Reynolds number (should be >> 1)
        """
        L = self.r_max - self.r_min
        return self.v_r * L / self.eta
