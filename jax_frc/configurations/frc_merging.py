# jax_frc/configurations/frc_merging.py
"""FRC merging configurations based on Belova et al. paper."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import jax.numpy as jnp

from jax_frc.simulation import Geometry, State
from jax_frc.models.base import PhysicsModel
from jax_frc.configurations.linear_configuration import (
    LinearConfiguration,
    PhaseSpec,
    TransitionSpec,
)


@dataclass
class BelovaMergingConfiguration(LinearConfiguration):
    """Configuration for FRC merging simulations based on Belova et al.

    This configuration builds a single-FRC equilibrium state. The MergingPhase
    handles the mirror-flip transformation to create the two-FRC merging setup.

    Attributes:
        flux_conserver_radius: Flux conserver radius Rc (length units)
        domain_half_length: Half-length of domain zc (domain is -zc to +zc)
        nr: Number of radial grid points
        nz: Number of axial grid points
        s_star: Kinetic parameter S* = Rs/di (separatrix radius / ion skin depth)
        elongation: FRC elongation E = Zs/Rs
        xs: Normalized separatrix radius xs = Rs/Rc
        beta_s: Separatrix beta
        separation: Initial separation between FRC nulls
        initial_velocity: Initial axial velocity toward midplane (in vA units)
        compression: Optional compression BC configuration dict
        model_type: Physics model type ("resistive_mhd", "extended_mhd", "hybrid_kinetic")
        eta_0: Base resistivity
        eta_anom: Anomalous resistivity (Chodura model)
    """

    name: str = "belova_merging"
    description: str = "FRC merging simulation based on Belova et al."

    # Geometry parameters
    flux_conserver_radius: float = 1.0
    domain_half_length: float = 4.0
    nr: int = 64
    nz: int = 256

    # FRC equilibrium parameters
    s_star: float = 20.0
    elongation: float = 2.0
    xs: float = 0.6
    beta_s: float = 0.2

    # Merging parameters
    separation: float = 1.5
    initial_velocity: float = 0.1
    compression: Optional[Dict[str, Any]] = None

    # Physics model parameters
    model_type: str = "resistive_mhd"
    eta_0: float = 1e-6
    eta_anom: float = 1e-3

    # Runtime (inherited from LinearConfiguration, provide defaults)
    dt: float = 0.001
    output_interval: int = 100

    # Transition timeout
    timeout: float = 15.0
    separation_threshold: float = 0.3

    def build_geometry(self) -> Geometry:
        """Create 3D Cartesian geometry for merging simulation."""
        extent = self.flux_conserver_radius
        return Geometry(
            nx=self.nr,
            ny=1,
            nz=self.nz,
            x_min=-extent,
            x_max=extent,
            y_min=-extent,
            y_max=extent,
            z_min=-self.domain_half_length,
            z_max=self.domain_half_length,
            bc_x="neumann",
            bc_y="periodic",
            bc_z="neumann",
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        """Create initial single-FRC equilibrium.

        Uses simplified Gaussian profile. The MergingPhase.setup() will
        mirror-flip this to create the two-FRC configuration.

        Args:
            geometry: Computational geometry

        Returns:
            Single FRC equilibrium state (will be transformed by MergingPhase)
        """
        state = State.zeros(geometry.nx, geometry.ny, geometry.nz)

        r = jnp.abs(geometry.x_grid)
        z = geometry.z_grid

        # Compute FRC dimensions from parameters
        Rc = self.flux_conserver_radius
        Rs = self.xs * Rc
        Zs = self.elongation * Rs

        # Create Gaussian FRC profile centered at z=0
        B_z = jnp.exp(-((r - 0.5 * Rs) ** 2 / (0.3 * Rs) ** 2 + z ** 2 / Zs ** 2))

        # Pressure proportional to psi, with separatrix beta
        p = self.beta_s * B_z

        # Uniform density
        n = jnp.ones_like(B_z)

        # Temperature from T = p/n
        T = p / jnp.maximum(n, 1e-10)

        # Initialize B field (will be computed from psi by model)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 2].set(B_z)

        return state.replace(p=p, n=n, Te=T, B=B)

    def build_model(self) -> PhysicsModel:
        """Create physics model with Chodura resistivity."""
        config = {
            "type": self.model_type,
            "resistivity": {
                "type": "chodura",
                "eta_0": self.eta_0,
                "eta_anom": self.eta_anom,
            },
        }
        return PhysicsModel.create(config)

    def build_boundary_conditions(self) -> list:
        """Create conducting wall boundary conditions.

        Returns:
            List of BCs at r_max, z_min, z_max (conducting walls)
        """
        # Boundary conditions are typically handled within the model
        # Return empty list as BCs are implicit in the model formulation
        return []

    def build_phase_specs(self) -> List[PhaseSpec]:
        """Define merging phase with appropriate transitions."""
        # Build transition spec (any_of separation_below or timeout)
        transition = TransitionSpec(
            type="any_of",
            children=[
                TransitionSpec(type="separation_below", value=self.separation_threshold),
                TransitionSpec(type="timeout", value=self.timeout),
            ],
        )

        return [
            PhaseSpec(
                name="merging",
                transition=transition,
                phase_class="MergingPhase",
                config=self.merging_phase_config(),
            ),
        ]

    def merging_phase_config(self) -> Dict[str, Any]:
        """Return configuration dict for MergingPhase.setup().

        Provides parameters needed by MergingPhase to create the two-FRC
        initial condition from the single-FRC state.

        Returns:
            Dict with separation, initial_velocity, and compression config
        """
        return {
            "separation": self.separation,
            "initial_velocity": self.initial_velocity,
            "compression": self.compression,
        }

    def available_phases(self) -> list[str]:
        """List valid phases for this configuration."""
        return ["merging"]

    def default_runtime(self) -> dict:
        """Return suggested runtime parameters."""
        return {"t_end": self.timeout, "dt": self.dt}


@dataclass
class BelovaCase1Configuration(BelovaMergingConfiguration):
    """Large FRC merging without compression (Belova et al. Fig. 1-2).

    Parameters:
        S* = 25.6, E = 2.9, xs = 0.69, beta_s = 0.2
        Initial separation: dZ = 180 (normalized) -> separation = 3.0
        Initial velocity: Vz = 0.2 vA

    Expected outcome: Partial merge, doublet configuration
    """

    name: str = "belova_case1_large_frc"
    description: str = "Large FRC merging without compression (Belova Fig. 1-2)"

    # Geometry
    flux_conserver_radius: float = 1.0
    domain_half_length: float = 5.0
    nr: int = 64
    nz: int = 512

    # FRC parameters
    s_star: float = 25.6
    elongation: float = 2.9
    xs: float = 0.69
    beta_s: float = 0.2

    # Merging parameters
    separation: float = 3.0
    initial_velocity: float = 0.2
    compression: Optional[Dict[str, Any]] = None

    # Runtime
    timeout: float = 30.0
    separation_threshold: float = 0.5


@dataclass
class BelovaCase2Configuration(BelovaMergingConfiguration):
    """Small FRC merging without compression (Belova et al. Fig. 3-4).

    Parameters:
        S* = 20, E = 1.5, xs = 0.53, beta_s = 0.2
        Initial separation: dZ = 75 (normalized) -> separation = 1.5
        Initial velocity: Vz = 0.1 vA

    Expected outcome: Complete merge by ~5-7 tA
    """

    name: str = "belova_case2_small_frc"
    description: str = "Small FRC merging without compression (Belova Fig. 3-4)"

    # Geometry
    flux_conserver_radius: float = 1.0
    domain_half_length: float = 3.0
    nr: int = 64
    nz: int = 256

    # FRC parameters
    s_star: float = 20.0
    elongation: float = 1.5
    xs: float = 0.53
    beta_s: float = 0.2

    # Merging parameters
    separation: float = 1.5
    initial_velocity: float = 0.1
    compression: Optional[Dict[str, Any]] = None

    # Runtime
    timeout: float = 15.0
    separation_threshold: float = 0.3


@dataclass
class BelovaCase4Configuration(BelovaMergingConfiguration):
    """Large FRC with compression (Belova et al. Fig. 6-7).

    Parameters: Same as case1 but with compression
        Mirror ratio: 1.5
        Ramp time: 19 tA

    Expected outcome: Complete merge by ~20-25 tA
    """

    name: str = "belova_case4_compression"
    description: str = "Large FRC with compression (Belova Fig. 6-7)"

    # Geometry (same as case 1)
    flux_conserver_radius: float = 1.0
    domain_half_length: float = 5.0
    nr: int = 64
    nz: int = 512

    # FRC parameters (same as case 1)
    s_star: float = 25.6
    elongation: float = 2.9
    xs: float = 0.69
    beta_s: float = 0.2

    # Merging parameters - compression drives merging
    separation: float = 3.0
    initial_velocity: float = 0.0  # Compression drives merging, not initial velocity
    compression: Optional[Dict[str, Any]] = field(
        default_factory=lambda: {
            "base_field": 1.0,
            "mirror_ratio": 1.5,
            "ramp_time": 19.0,
            "profile": "cosine",
        }
    )

    # Runtime
    timeout: float = 40.0
    separation_threshold: float = 0.3
