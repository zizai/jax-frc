"""Analytic test case configurations."""
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.extended_mhd import ExtendedMHD, HaloDensityModel, TemperatureBoundaryCondition
from jax_frc.models.energy import ThermalTransport
from jax_frc.models.resistivity import SpitzerResistivity
from .base import AbstractConfiguration


@dataclass
class SlabDiffusionConfiguration(AbstractConfiguration):
    """1D heat diffusion test case with analytic solution.

    Gaussian temperature profile diffusing along z with uniform B_z.
    Analytic solution: T(z,t) = T0 * sqrt(sigma^2 / (sigma^2 + 2*D*t)) * exp(...)
    """

    name: str = "slab_diffusion"
    description: str = "1D heat conduction test with analytic solution"

    # Grid parameters
    nr: int = 8
    nz: int = 64
    z_extent: float = 2.0  # [-z_extent, z_extent]

    # Physics parameters
    T_peak: float = 200.0  # eV
    T_base: float = 50.0   # eV
    sigma: float = 0.3     # Initial Gaussian width
    kappa: float = 1e-3    # Thermal diffusivity coefficient
    n0: float = 1e19       # Density

    def build_geometry(self) -> Geometry:
        return Geometry(
            coord_system="cylindrical",
            r_min=0.1, r_max=0.9,
            z_min=-self.z_extent, z_max=self.z_extent,
            nr=self.nr, nz=self.nz
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        z = geometry.z_grid

        # Gaussian temperature profile in z
        T_init = self.T_peak * jnp.exp(-z**2 / (2 * self.sigma**2)) + self.T_base

        # B = 0 for pure diffusion test (no MHD dynamics)
        # Note: A non-zero B field with gradients causes MHD instability
        B = jnp.zeros((geometry.nr, geometry.nz, 3))

        return State(
            psi=jnp.zeros((geometry.nr, geometry.nz)),
            n=jnp.ones((geometry.nr, geometry.nz)) * self.n0,
            p=jnp.ones((geometry.nr, geometry.nz)) * 1e3,
            T=T_init,
            B=B,
            E=jnp.zeros((geometry.nr, geometry.nz, 3)),
            v=jnp.zeros((geometry.nr, geometry.nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

    def build_model(self) -> ExtendedMHD:
        # kappa_parallel = D * (3/2 * n)
        kappa_parallel = self.kappa * 1.5 * self.n0

        return ExtendedMHD(
            resistivity=SpitzerResistivity(eta_0=1e-10),
            halo_model=HaloDensityModel(halo_density=self.n0, core_density=self.n0),
            thermal=ThermalTransport(
                kappa_parallel_0=kappa_parallel,
                use_spitzer=False
            ),
            temperature_bc=TemperatureBoundaryCondition(bc_type="neumann")
        )

    def build_boundary_conditions(self) -> list:
        return []  # Neumann BCs built into model

    def default_runtime(self) -> dict:
        return {"t_end": 1e-3, "dt": 1e-5}

    def analytic_solution(self, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute analytic temperature at time t."""
        D = self.kappa
        sigma_eff_sq = self.sigma**2 + 2 * D * t
        amplitude = self.T_peak * jnp.sqrt(self.sigma**2 / sigma_eff_sq)
        return amplitude * jnp.exp(-z**2 / (2 * sigma_eff_sq)) + self.T_base
