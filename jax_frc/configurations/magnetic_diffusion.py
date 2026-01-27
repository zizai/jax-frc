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
    """2D magnetic diffusion test in slab geometry (Rm << 1).

    Physics: ∂B/∂t = η∇²B (resistive diffusion, no flow)

    Tests the diffusion-dominated regime where magnetic Reynolds number
    is much less than 1. The magnetic field diffuses through the plasma
    like heat through a conductor.

    Coordinate Transformation:
        The MHD code uses cylindrical 2D (r,z) coordinates, while the analytic
        solution is for 2D Cartesian diffusion. By centering the domain at
        large r (r_center >> sigma), the cylindrical Laplacian approximates
        Cartesian: ∇²B_z ≈ ∂²B_z/∂r² + ∂²B_z/∂z² (the (1/r)∂B_z/∂r term
        becomes negligible when r >> sigma).

    Initial condition: 2D Gaussian B_z profile centered at (r_center, z_center)
    Analytic solution: 2D spreading Gaussian (Cartesian approximation)
        σ_t² = σ₀² + 2ηt
        B_z(r,z,t) = B_peak * (σ₀²/σ_t²) * exp(-((r-r₀)² + (z-z₀)²)/(2σ_t²))

    Supports: resistive_mhd, extended_mhd, plasma_neutral, hybrid_kinetic
    """

    name: str = "magnetic_diffusion"
    description: str = "2D magnetic field diffusion in slab geometry (Rm << 1)"

    # Grid parameters - slab geometry for Cartesian approximation
    nr: int = 64          # Radial resolution
    nz: int = 64          # Axial resolution
    r_min: float = 1.0    # Inner radius [m] - far from axis
    r_max: float = 2.0    # Outer radius [m]
    z_extent: float = 1.0 # Domain: z ∈ [-z_extent, z_extent]

    # Physics parameters
    B_peak: float = 1.0        # Peak B_z [T]
    sigma: float = 0.1         # Initial Gaussian width [m] (should be << r_min)
    # Note: eta is resistivity [Ω·m], diffusivity = eta/mu_0
    # For diffusivity D = 1e-4 m²/s, eta = D * mu_0 ≈ 1.26e-10 Ω·m
    eta: float = 1.26e-10      # Magnetic resistivity [Ω·m]

    # Center of Gaussian in slab
    r_center: float = 1.5    # Radial center [m] = (r_min + r_max) / 2
    z_center: float = 0.0    # Axial center [m]

    # Plasma parameters (for models that need them)
    # Use very high density to suppress Hall term for pure diffusion test
    n0: float = 1e23         # Density [m^-3] (high to suppress Hall term)
    T0: float = 100.0        # Temperature [eV]

    # Model selection
    model_type: ModelType = "extended_mhd"

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
        """2D Gaussian B_z profile in slab geometry, v=0."""
        r = geometry.r_grid
        z = geometry.z_grid

        # 2D Gaussian centered at (r_center, z_center)
        # Using (r - r_center) as the "x" coordinate in Cartesian approximation
        x_sq = (r - self.r_center)**2
        z_sq = (z - self.z_center)**2
        B_z = self.B_peak * jnp.exp(-(x_sq + z_sq) / (2 * self.sigma**2))

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
                include_hall=False,  # Disable Hall term for pure diffusion test
                include_electron_pressure=False,  # Disable electron pressure
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

    def analytic_solution(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """2D Cartesian analytic solution mapped to cylindrical coordinates.

        In 2D Cartesian, the spreading Gaussian solution is:
            B_z(x,z,t) = B_peak * (σ₀²/σ_t²) * exp(-((x-x₀)² + (z-z₀)²)/(2σ_t²))

        Mapping: x = r - r_center (slab approximation when r >> σ)

        where σ_t² = σ₀² + 2Dt, D = eta/mu_0 is the magnetic diffusivity

        Args:
            r: Radial coordinates [m]
            z: Axial coordinates [m]
            t: Time [s]

        Returns:
            B_z field values at the given coordinates and time
        """
        # Convert resistivity to diffusivity: D = eta / mu_0
        MU0 = 1.2566e-6
        diffusivity = self.eta / MU0

        sigma_eff_sq = self.sigma**2 + 2 * diffusivity * t

        # Map cylindrical r to Cartesian x relative to center
        x = r - self.r_center
        z_rel = z - self.z_center

        # 2D Cartesian amplitude decay: σ₀²/σ_t² (not sqrt for 2D)
        amplitude = self.B_peak * (self.sigma**2 / sigma_eff_sq)
        return amplitude * jnp.exp(-(x**2 + z_rel**2) / (2 * sigma_eff_sq))

    def diffusion_timescale(self) -> float:
        """Characteristic diffusion time τ = σ²/(2D) where D = eta/mu_0."""
        MU0 = 1.2566e-6
        diffusivity = self.eta / MU0
        return self.sigma**2 / (2 * diffusivity)

    def magnetic_reynolds_number(self, v: float = 0.0) -> float:
        """Compute magnetic Reynolds number Rm = vL/D where D = eta/mu_0.

        For this configuration, v=0 by design, so Rm ≈ 0 << 1.

        Args:
            v: Characteristic velocity (default 0 for pure diffusion)

        Returns:
            Magnetic Reynolds number (should be << 1)
        """
        MU0 = 1.2566e-6
        diffusivity = self.eta / MU0
        L = 2 * self.z_extent  # Characteristic length
        return v * L / diffusivity
