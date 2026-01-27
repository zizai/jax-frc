"""Magnetic diffusion validation configuration (Rm << 1)."""
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
class MagneticDiffusionConfiguration(AbstractConfiguration):
    """1D magnetic diffusion test (Rm << 1).

    Physics: ∂B/∂t = η∇²B (resistive diffusion, no flow)

    Tests the diffusion-dominated regime where magnetic Reynolds number
    is much less than 1. The magnetic field diffuses through the plasma
    like heat through a conductor.

    Initial condition: Gaussian B_z profile along z
    Analytic solution: Spreading Gaussian
        B_z(z,t) = B_peak * sqrt(σ₀²/(σ₀² + 2ηt)) * exp(-z²/(2(σ₀² + 2ηt)))

    Supports: resistive_mhd, extended_mhd, plasma_neutral, hybrid_kinetic
    """

    name: str = "magnetic_diffusion"
    description: str = "1D magnetic field diffusion (Rm << 1)"

    # Grid parameters
    nr: int = 8          # Minimal radial resolution (1D in z)
    nz: int = 128        # Axial resolution
    r_min: float = 0.1
    r_max: float = 0.5
    z_extent: float = 2.0  # Domain: z ∈ [-z_extent, z_extent]

    # Physics parameters
    B_peak: float = 1.0      # Peak B_z [T]
    sigma: float = 0.3       # Initial Gaussian width [m]
    eta: float = 1e-2        # Magnetic diffusivity [m²/s] (η = 1/(μ₀σ))

    # Plasma parameters (for models that need them)
    n0: float = 1e19         # Density [m^-3]
    T0: float = 100.0        # Temperature [eV]

    # Model selection
    model_type: ModelType = "resistive_mhd"

    def build_geometry(self) -> Geometry:
        """Cylindrical geometry, minimal r resolution (1D in z)."""
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
        """Gaussian B_z profile, v=0."""
        z = geometry.z_grid

        # Gaussian B_z profile along z
        B_z = self.B_peak * jnp.exp(-z**2 / (2 * self.sigma**2))

        # Build B field array (only B_z component)
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(B_z)

        # v = 0 for pure diffusion (Rm << 1)
        v = jnp.zeros((geometry.nr, geometry.nz, 3))

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
        """Build physics model with resistivity configured for diffusion test."""
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
                Omega=0.0,  # No rotation for diffusion test
            )
            return HybridKinetic(
                equilibrium=equilibrium,
                eta=self.eta,
            )

        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def build_boundary_conditions(self) -> list:
        """Neumann BCs (zero gradient at boundaries)."""
        return []

    def default_runtime(self) -> dict:
        """Runtime parameters based on diffusion timescale."""
        tau_diff = self.diffusion_timescale()
        return {
            "t_end": 0.1 * tau_diff,  # Run for 0.1 diffusion times
            "dt": 1e-4 * tau_diff,    # Small timestep for stability
        }

    def analytic_solution(self, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute analytic B_z at time t.

        The solution is a spreading Gaussian:
            B_z(z,t) = B_peak * sqrt(σ₀²/(σ₀² + 2ηt)) * exp(-z²/(2(σ₀² + 2ηt)))

        Args:
            z: Axial coordinates [m]
            t: Time [s]

        Returns:
            B_z field values at the given coordinates and time
        """
        sigma_eff_sq = self.sigma**2 + 2 * self.eta * t
        amplitude = self.B_peak * jnp.sqrt(self.sigma**2 / sigma_eff_sq)
        return amplitude * jnp.exp(-z**2 / (2 * sigma_eff_sq))

    def diffusion_timescale(self) -> float:
        """Characteristic diffusion time τ = σ²/(2η)."""
        return self.sigma**2 / (2 * self.eta)

    def magnetic_reynolds_number(self, v: float = 0.0) -> float:
        """Compute magnetic Reynolds number Rm = vL/η.

        For this configuration, v=0 by design, so Rm ≈ 0 << 1.

        Args:
            v: Characteristic velocity (default 0 for pure diffusion)

        Returns:
            Magnetic Reynolds number (should be << 1)
        """
        L = 2 * self.z_extent  # Characteristic length
        return v * L / self.eta
