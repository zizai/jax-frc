"""Time-dependent boundary conditions."""

from dataclasses import dataclass
from typing import Literal
import jax.numpy as jnp

from jax_frc.boundaries.base import BoundaryCondition
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


@dataclass
class TimeDependentMirrorBC(BoundaryCondition):
    """Mirror field with time-varying strength at z boundaries.

    Implements compression profile from Belova et al.:
    - Spatial: delta_Aphi(z) ~ 0.5(1 - cos(pi*z/Zc))
    - Temporal: f(t) ~ (1 - cos(pi*t/T)) for cosine profile

    Attributes:
        base_field: B0 at t=0
        mirror_ratio_final: B_end/B0 (e.g., 1.5)
        ramp_time: T in Alfven times
        profile: "cosine" or "linear"
    """

    base_field: float
    mirror_ratio_final: float
    ramp_time: float
    profile: Literal["cosine", "linear"] = "cosine"

    def apply(self, state: State, geometry: Geometry, t: float = 0.0) -> State:
        """Apply time-dependent mirror field at z boundaries.

        Args:
            state: Current state
            geometry: Computational geometry
            t: Current simulation time

        Returns:
            State with modified boundary values
        """
        psi = state.psi

        # Compute current mirror ratio
        ratio = self._compute_mirror_ratio(t)

        delta_ratio = ratio - 1.0

        # Spatial profile: strongest at ends, zero at midplane
        z = geometry.z
        Zc = geometry.z_max
        spatial_profile = 0.5 * (1 - jnp.cos(jnp.pi * jnp.abs(z) / Zc))

        # Apply to psi at z boundaries
        r = geometry.r

        # Compute boundary modification
        delta_psi_boundary = delta_ratio * self.base_field * r * spatial_profile[-1]

        # Apply to end boundaries
        psi = psi.at[:, 0].set(psi[:, 0] + delta_psi_boundary)
        psi = psi.at[:, -1].set(psi[:, -1] + delta_psi_boundary)

        return state.replace(psi=psi)

    def _compute_mirror_ratio(self, t: float) -> float:
        """Compute current mirror ratio based on time and profile.

        Args:
            t: Current time

        Returns:
            Current mirror ratio (1.0 to mirror_ratio_final)
        """
        if t <= 0:
            return 1.0
        if t >= self.ramp_time:
            return self.mirror_ratio_final

        # Normalized time
        tau = t / self.ramp_time

        if self.profile == "cosine":
            # Smooth cosine ramp: f(tau) = 0.5 * (1 - cos(pi * tau))
            f = 0.5 * (1 - jnp.cos(jnp.pi * tau))
        else:  # linear
            f = tau

        # Interpolate between 1.0 and final ratio
        return float(1.0 + (self.mirror_ratio_final - 1.0) * f)

    def get_current_field(self, t: float) -> float:
        """Get current boundary field strength.

        Args:
            t: Current time

        Returns:
            Current field strength at boundary
        """
        return self.base_field * self._compute_mirror_ratio(t)
