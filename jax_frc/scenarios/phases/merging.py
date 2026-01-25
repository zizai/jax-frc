"""Merging phase for two-FRC collision."""

from dataclasses import dataclass, field
from typing import Optional
import jax.numpy as jnp

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.scenarios.phase import Phase
from jax_frc.scenarios.transitions import Transition
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

        # Mirror-flip to create two FRCs
        psi = self._create_two_frc_psi(state.psi, geometry, separation)

        # Create antisymmetric velocity field
        v = self._create_velocity_field(state.v, geometry, velocity)

        # Mirror other fields
        n = self._mirror_flip(state.n)
        p = self._mirror_flip(state.p)
        B = self._mirror_flip_vector(state.B)
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

        return state.replace(psi=psi, n=n, p=p, B=B, E=E, v=v)

    def step_hook(self, state: State, geometry: Geometry, t: float) -> State:
        """Apply compression BC if configured."""
        if self._compression_bc is not None:
            state = self._compression_bc.apply(state, geometry)
        return state

    def _create_two_frc_psi(self, psi: jnp.ndarray, geometry: Geometry,
                           separation: float) -> jnp.ndarray:
        """Create two-FRC psi by mirror-flipping and shifting."""
        nz = psi.shape[1]
        dz = geometry.dz

        # Compute shift in grid points (at least 1)
        shift_points = int(separation / (2 * dz))
        shift_points = max(1, min(shift_points, nz // 4))  # At least 1, no more than nz//4

        # Create shifted versions
        psi_left = jnp.roll(psi, -shift_points, axis=1)
        psi_mirrored = jnp.flip(psi, axis=1)
        psi_right = jnp.roll(psi_mirrored, shift_points, axis=1)

        # Combine
        psi_combined = psi_left + psi_right

        return psi_combined

    def _create_velocity_field(self, v: jnp.ndarray, geometry: Geometry,
                              velocity: float) -> jnp.ndarray:
        """Create antisymmetric velocity field for merging."""
        z = geometry.z_grid

        # Create antisymmetric Vz: positive for z<0, negative for z>0
        z_normalized = z / (geometry.z_max * 0.5)
        vz_profile = -velocity * jnp.tanh(z_normalized * 2)

        # Set Vz component
        v_new = v.at[:, :, 2].set(vz_profile)

        return v_new

    def _mirror_flip(self, field: jnp.ndarray) -> jnp.ndarray:
        """Mirror-flip a scalar field and add to original."""
        return field + jnp.flip(field, axis=1)

    def _mirror_flip_vector(self, field: jnp.ndarray) -> jnp.ndarray:
        """Mirror-flip a vector field (flip z-component sign)."""
        flipped = jnp.flip(field, axis=1)
        flipped = flipped.at[:, :, 2].set(-flipped[:, :, 2])
        return field + flipped
