"""Merging phase for two-FRC collision."""

from dataclasses import dataclass, field
from typing import Optional
import jax.numpy as jnp

from jax_frc.simulation import State, Geometry
from jax_frc.configurations.phase import Phase
from jax_frc.configurations.transitions import Transition
from jax_frc.boundaries.time_dependent import TimeDependentMirrorBC


@dataclass
class MergingPhase(Phase):
    """Phase for FRC merging simulation.

    Sets up two-FRC initial condition by mirror-flipping a single FRC,
    applies initial velocities, and optionally enables compression.

    Attributes:
        name: Phase name
        transition: Completion condition
        separation: Initial separation between FRC nulls (in length units)
        initial_velocity: Initial axial velocity (positive = toward midplane)
        compression: Optional compression BC configuration
    """

    separation: float = 1.0
    initial_velocity: float = 0.0
    compression: Optional[dict] = None

    _compression_bc: Optional[TimeDependentMirrorBC] = field(
        default=None, init=False, repr=False
    )

    def setup(self, state: State, geometry: Geometry, config: dict) -> State:
        """Create two-FRC configuration from single FRC."""
        # Override from config if provided
        separation = config.get("separation", self.separation)
        velocity = config.get("initial_velocity", self.initial_velocity)

        # Create two-FRC fields by mirroring and shifting
        n = self._create_two_frc_scalar(state.n, geometry, separation)
        p = self._create_two_frc_scalar(state.p, geometry, separation)
        B = self._create_two_frc_vector(state.B, geometry, separation)

        # Create antisymmetric velocity field
        v = self._create_velocity_field(state.v, geometry, velocity)
        E = state.E  # Keep E as is initially

        # Setup compression BC if configured
        compression_config = config.get("compression", self.compression)
        if compression_config:
            self._compression_bc = TimeDependentMirrorBC(
                base_field=compression_config.get("base_field", 1.0),
                mirror_ratio_final=compression_config.get("mirror_ratio", 1.5),
                ramp_time=compression_config.get("ramp_time", 10.0),
                profile=compression_config.get("profile", "cosine"),
            )

        return state.replace(n=n, p=p, B=B, E=E, v=v)

    def step_hook(self, state: State, geometry: Geometry, t: float) -> State:
        """Apply compression BC if configured."""
        if self._compression_bc is not None:
            state = self._compression_bc.apply(state, geometry)
        return state

    def _create_two_frc_scalar(
        self, field: jnp.ndarray, geometry: Geometry, separation: float
    ) -> jnp.ndarray:
        """Create two-FRC scalar field by mirror-flipping and shifting."""
        nz = field.shape[2]
        dz = geometry.dz

        # Compute shift in grid points (at least 1)
        shift_points = int(separation / (2 * dz))
        shift_points = max(1, min(shift_points, nz // 4))  # At least 1, no more than nz//4

        # Create shifted versions
        left = jnp.roll(field, -shift_points, axis=2)
        mirrored = jnp.flip(field, axis=2)
        right = jnp.roll(mirrored, shift_points, axis=2)

        return left + right

    def _create_two_frc_vector(
        self, field: jnp.ndarray, geometry: Geometry, separation: float
    ) -> jnp.ndarray:
        """Create two-FRC vector field by mirror-flipping and shifting."""
        nz = field.shape[2]
        dz = geometry.dz

        shift_points = int(separation / (2 * dz))
        shift_points = max(1, min(shift_points, nz // 4))

        left = jnp.roll(field, -shift_points, axis=2)
        mirrored = jnp.flip(field, axis=2)
        mirrored = mirrored.at[:, :, :, 2].set(-mirrored[:, :, :, 2])
        right = jnp.roll(mirrored, shift_points, axis=2)

        return left + right

    def _create_velocity_field(self, v: jnp.ndarray, geometry: Geometry,
                              velocity: float) -> jnp.ndarray:
        """Create antisymmetric velocity field for merging."""
        z = geometry.z_grid

        # Create antisymmetric Vz: positive for z<0, negative for z>0
        z_normalized = z / (geometry.z_max * 0.5)
        vz_profile = -velocity * jnp.tanh(z_normalized * 2)

        # Set Vz component
        if v is None:
            v = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        v_new = v.at[:, :, :, 2].set(vz_profile)

        return v_new
