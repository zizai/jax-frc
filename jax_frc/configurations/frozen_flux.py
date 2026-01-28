"""Frozen-in flux validation configuration (Rm >> 1).

Circular advection of a magnetic loop benchmark for validating ideal MHD solvers.
A localized magnetic structure is advected by rigid body rotation and should
return to its initial position after one full period.
"""
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
    """Frozen-in flux test via circular advection of a magnetic loop.

    Physics (Cartesian induction equation):
        dB/dt = curl(v x B) + eta * laplacian(B)

    Ideal-MHD limit (Rm >> 1, eta -> 0):
        dB/dt = curl(v x B)

    Benchmark: A localized magnetic loop is advected by a rigid body rotation
    velocity field. After one full rotation period, the loop should return to
    its initial position. This tests:
    - Numerical diffusion (amplitude preservation)
    - Dispersion (shape preservation)
    - div(B) = 0 constraint maintenance

    Setup:
        - Domain: x,y ∈ [-1, 1], thin z (pseudo-2D)
        - Magnetic loop: B = curl(A_z z_hat) with Gaussian profile
        - Velocity: Rigid rotation v = (-ωy, ωx, 0)
        - Period: T = 2π/ω

    Note: Uses Gaussian profile (smooth) instead of compact support to avoid
    numerical issues with discontinuities in spectral methods.

    Supports: resistive_mhd, extended_mhd, plasma_neutral, hybrid_kinetic
    """

    name: str = "frozen_flux"
    description: str = "Circular advection of magnetic loop (Rm >> 1)"

    # Grid parameters (thin-z for pseudo-2D in x-y plane)
    nx: int = 64         # X resolution
    ny: int = 64         # Y resolution
    nz: int = 1          # Z resolution (thin periodic)

    # Domain parameters
    domain_extent: float = 1.0   # x,y ∈ [-extent, extent]
    z_extent: float = 0.1        # z ∈ [-z_extent, z_extent]

    # Magnetic loop parameters
    loop_x0: float = 0.3         # Loop center x [m] (off-center for advection test)
    loop_y0: float = 0.0         # Loop center y [m]
    loop_radius: float = 0.2     # Gaussian width sigma = loop_radius/2 [m]
    loop_amplitude: float = 1.0  # Vector potential amplitude A_0

    # Rotation parameters
    omega: float = 2 * 3.141592653589793  # Angular velocity [rad/s] (one rotation per second)

    # Physics parameters
    eta: float = 0.0         # Zero resistivity for ideal MHD (frozen flux)

    # Plasma parameters (for models that need them)
    n0: float = 1e19         # Density [m^-3]
    T0: float = 100.0        # Temperature [eV]

    # Model selection
    model_type: ModelType = "resistive_mhd"

    # Advection scheme for resistive_mhd model
    # "ct" = Constrained Transport (preserves div(B)=0, less diffusive)
    # "central" = Standard central differences
    advection_scheme: str = "ct"

    # ExtendedMHD options
    include_hall: bool = True  # Include Hall term (J x B)/(ne)
    include_electron_pressure: bool = True  # Include grad(p_e)/(ne) term

    def build_geometry(self) -> Geometry:
        """3D Cartesian geometry with thin z (pseudo-2D in x-y plane)."""
        return Geometry(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            x_min=-self.domain_extent,
            x_max=self.domain_extent,
            y_min=-self.domain_extent,
            y_max=self.domain_extent,
            z_min=-self.z_extent,
            z_max=self.z_extent,
            bc_x="periodic",
            bc_y="periodic",
            bc_z="periodic",
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        """Magnetic loop with rigid body rotation velocity.

        Uses Gaussian vector potential: A_z = A_0 * exp(-r²/(2σ²))
        where σ = loop_radius/2 for smooth, well-resolved profile.

        B = curl(A_z z_hat) = (∂A_z/∂y, -∂A_z/∂x, 0)
        """
        x = geometry.x_grid
        y = geometry.y_grid

        dx = x - self.loop_x0
        dy = y - self.loop_y0
        r_sq = dx**2 + dy**2
        sigma = self.loop_radius / 2  # Gaussian width
        A0 = self.loop_amplitude

        # Gaussian vector potential: A_z = A0 * exp(-r²/(2σ²))
        gaussian = A0 * jnp.exp(-r_sq / (2 * sigma**2))

        # B = curl(A_z z_hat):
        # B_x = ∂A_z/∂y = A0 * exp(-r²/(2σ²)) * (-dy/σ²)
        # B_y = -∂A_z/∂x = A0 * exp(-r²/(2σ²)) * (dx/σ²)
        Bx = gaussian * (-dy / sigma**2)
        By = gaussian * (dx / sigma**2)

        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[..., 0].set(Bx)
        B = B.at[..., 1].set(By)

        # Rigid body rotation velocity: v = (-ωy, ωx, 0)
        v = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        v = v.at[..., 0].set(-self.omega * y)  # v_x = -ω*y
        v = v.at[..., 1].set(self.omega * x)   # v_y = ω*x

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
            return ResistiveMHD(eta=self.eta, advection_scheme=self.advection_scheme)

        elif self.model_type == "extended_mhd":
            return ExtendedMHD(
                eta=self.eta,
                include_hall=self.include_hall,
                include_electron_pressure=self.include_electron_pressure,
            )

        elif self.model_type == "plasma_neutral":
            plasma_model = ResistiveMHD(eta=self.eta, advection_scheme=self.advection_scheme)
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
        """Boundary conditions (periodic, handled by geometry)."""
        return []

    def default_runtime(self) -> dict:
        """Runtime parameters for quarter rotation (less numerical diffusion)."""
        period = 2 * 3.141592653589793 / self.omega  # T = 2π/ω
        return {
            "t_end": period / 4,      # Quarter rotation (less diffusion)
            "dt": period / 1000,      # 250 steps for quarter rotation
        }

    def analytic_solution(self, geometry: Geometry, t: float) -> jnp.ndarray:
        """Compute analytic B field at time t.

        For rigid body rotation, the magnetic loop rotates with the flow.
        After one full period (t = 2π/ω), it returns to initial position.

        Uses Gaussian profile matching build_initial_state.
        """
        # Rotation angle
        theta = self.omega * t

        # Rotate coordinates backward to get initial position
        x = geometry.x_grid
        y = geometry.y_grid
        x_rot = x * jnp.cos(theta) + y * jnp.sin(theta)
        y_rot = -x * jnp.sin(theta) + y * jnp.cos(theta)

        # Compute B analytically in rotated frame using Gaussian profile
        dx = x_rot - self.loop_x0
        dy = y_rot - self.loop_y0
        r_sq = dx**2 + dy**2
        sigma = self.loop_radius / 2
        A0 = self.loop_amplitude

        gaussian = A0 * jnp.exp(-r_sq / (2 * sigma**2))
        Bx_rot = gaussian * (-dy / sigma**2)
        By_rot = gaussian * (dx / sigma**2)

        # Rotate B back to lab frame
        Bx = Bx_rot * jnp.cos(theta) - By_rot * jnp.sin(theta)
        By = Bx_rot * jnp.sin(theta) + By_rot * jnp.cos(theta)

        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[..., 0].set(Bx)
        B = B.at[..., 1].set(By)

        return B

    def rotation_period(self) -> float:
        """Return the rotation period T = 2π/ω."""
        return 2 * 3.141592653589793 / self.omega

    def magnetic_reynolds_number(self) -> float:
        """Compute magnetic Reynolds number Rm = v*L/eta.

        For rigid rotation, v_max = ω * domain_extent.

        Returns:
            Magnetic Reynolds number (should be >> 1)
        """
        v_max = self.omega * self.domain_extent
        L = 2 * self.domain_extent
        return v_max * L / self.eta
