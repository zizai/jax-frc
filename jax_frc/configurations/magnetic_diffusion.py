"""Magnetic diffusion validation configuration (Rm << 1)."""
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
class MagneticDiffusionConfiguration(AbstractConfiguration):
    """3D magnetic diffusion test (Rm << 1).

    Physics: ∂B/∂t = η∇²B (resistive diffusion, no flow)

    Tests the diffusion-dominated regime where magnetic Reynolds number
    is much less than 1. The magnetic field diffuses through the plasma
    like heat through a conductor.

    Initial condition: 2D Gaussian B_z(x,y) profile in x-y plane
    Analytic solution: 2D spreading Gaussian
        σ_t² = σ₀² + 2Dt
        B_z(x,y,t) = B_peak * (σ₀/σ_t)² * exp(-(x² + y²)/(2σ_t²))

    The Gaussian is in the x-y plane (uniform in z) to ensure div(B) = 0,
    since ∂Bz/∂z = 0. Set nz=1 for 2D-like behavior.

    Supports: resistive_mhd, extended_mhd, plasma_neutral, hybrid_kinetic
    """

    name: str = "magnetic_diffusion"
    description: str = "3D magnetic field diffusion (Rm << 1)"

    # Grid parameters
    nx: int = 64          # X resolution
    ny: int = 64          # Y resolution
    nz: int = 1           # Z resolution (pseudo-dimension for 2D-like behavior)
    extent: float = 1.0   # Domain: [-extent, extent] in each direction
    bc_x: str = "neumann"
    bc_y: str = "neumann"
    bc_z: str = "neumann"

    # Physics parameters
    B_peak: float = 1.0        # Peak B_z [T]
    sigma: float = 0.1         # Initial Gaussian width [m]
    # Note: eta is resistivity [Ω·m], diffusivity = eta/mu_0
    # For diffusivity D = 1e-4 m²/s, eta = D * mu_0 ≈ 1.26e-10 Ω·m
    eta: float = 1.26e-10      # Magnetic resistivity [Ω·m]

    # Plasma parameters (for models that need them)
    # Use very high density to suppress Hall term for pure diffusion test
    n0: float = 1e23         # Density [m^-3] (high to suppress Hall term)
    T0: float = 100.0        # Temperature [eV]

    # Model selection
    model_type: ModelType = "extended_mhd"

    # Freeze density/velocity/pressure for pure diffusion test
    evolve_density: bool = False
    evolve_velocity: bool = False
    evolve_pressure: bool = False

    def build_geometry(self) -> Geometry:
        """3D Cartesian geometry."""
        return Geometry(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            x_min=-self.extent,
            x_max=self.extent,
            y_min=-self.extent,
            y_max=self.extent,
            z_min=-self.extent,
            z_max=self.extent,
            bc_x=self.bc_x,
            bc_y=self.bc_y,
            bc_z=self.bc_z,
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        """2D Gaussian B_z profile in x-y plane (uniform in z), v=0.

        Using x-y plane ensures div(B) = 0 since ∂Bz/∂z = 0.
        """
        x = geometry.x_grid
        y = geometry.y_grid

        # 2D Gaussian centered at origin (uniform in z for div(B) = 0)
        r_sq = x**2 + y**2
        B_z = self.B_peak * jnp.exp(-r_sq / (2 * self.sigma**2))

        # Build B field array (only B_z component)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 2].set(B_z)

        # v = 0 for pure diffusion (Rm << 1)
        v = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))

        return State(
            B=B,
            E=jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3)),
            n=jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * self.n0,
            p=jnp.ones((geometry.nx, geometry.ny, geometry.nz)) * self.n0 * self.T0,
            v=v,
        )

    def build_model(self):
        """Build physics model with resistivity configured for diffusion test."""
        if self.model_type == "resistive_mhd":
            return ResistiveMHD(
                eta=self.eta,
                evolve_density=self.evolve_density,
                evolve_velocity=self.evolve_velocity,
                evolve_pressure=self.evolve_pressure,
            )

        elif self.model_type == "extended_mhd":
            return ExtendedMHD(
                eta=self.eta,
                include_hall=False,  # Disable Hall term for pure diffusion test
                include_electron_pressure=False,  # Disable electron pressure
                evolve_density=self.evolve_density,
                evolve_velocity=self.evolve_velocity,
                evolve_pressure=self.evolve_pressure,
            )

        elif self.model_type == "plasma_neutral":
            plasma_model = ResistiveMHD(
                eta=self.eta,
                evolve_density=self.evolve_density,
                evolve_velocity=self.evolve_velocity,
                evolve_pressure=self.evolve_pressure,
            )
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

    def analytic_solution(
        self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """3D analytic solution for spreading Gaussian.

        The 3D spreading Gaussian solution is:
            B_z(x,y,z,t) = B_peak * (σ₀/σ_t)³ * exp(-(x² + y² + z²)/(2σ_t²))

        where σ_t² = σ₀² + 2Dt, D = eta/mu_0 is the magnetic diffusivity.

        Note: For thin-y domain (ny=4, periodic), the solution is effectively
        2D in the x-z plane. Use analytic_solution_2d() for that case.

        Args:
            x: X coordinates [m]
            y: Y coordinates [m]
            z: Z coordinates [m]
            t: Time [s]

        Returns:
            B_z field values at the given coordinates and time
        """
        MU0 = 1.2566e-6
        diffusivity = self.eta / MU0

        sigma_eff_sq = self.sigma**2 + 2 * diffusivity * t
        sigma_eff = jnp.sqrt(sigma_eff_sq)

        r_sq = x**2 + y**2 + z**2

        # 3D amplitude decay: (σ₀/σ_t)³
        amplitude = self.B_peak * (self.sigma / sigma_eff) ** 3
        return amplitude * jnp.exp(-r_sq / (2 * sigma_eff_sq))

    def analytic_solution_2d(
        self, x: jnp.ndarray, y: jnp.ndarray, t: float
    ) -> jnp.ndarray:
        """2D analytic solution for x-y plane (thin-z approximation).

        The 2D spreading Gaussian solution is:
            B_z(x,y,t) = B_peak * (σ₀/σ_t)² * exp(-(x² + y²)/(2σ_t²))

        Args:
            x: X coordinates [m]
            y: Y coordinates [m]
            t: Time [s]

        Returns:
            B_z field values at the given coordinates and time
        """
        MU0 = 1.2566e-6
        diffusivity = self.eta / MU0

        sigma_eff_sq = self.sigma**2 + 2 * diffusivity * t

        r_sq = x**2 + y**2

        # 2D amplitude decay: (σ₀²/σ_t²)
        amplitude = self.B_peak * (self.sigma**2 / sigma_eff_sq)
        return amplitude * jnp.exp(-r_sq / (2 * sigma_eff_sq))

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
        L = 2 * self.extent  # Characteristic length
        return v * L / diffusivity
