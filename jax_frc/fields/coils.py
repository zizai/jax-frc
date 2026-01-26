"""Analytical electromagnetic coil field models.

All models are JIT-compatible and return fields in cylindrical coordinates (r, z).
"""

from typing import Protocol, Callable, Union, Optional
from dataclasses import dataclass
import jax.numpy as jnp
from jax import jit
from jax_frc.constants import MU0


class CoilField(Protocol):
    """Protocol for external field sources."""

    def B_field(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute magnetic field components.

        Args:
            r: Radial coordinates [m]
            z: Axial coordinates [m]
            t: Time [s]

        Returns:
            (B_r, B_z): Radial and axial field components [T]
        """
        ...

    def A_phi(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute azimuthal vector potential.

        Args:
            r: Radial coordinates [m]
            z: Axial coordinates [m]
            t: Time [s]

        Returns:
            A_phi: Azimuthal vector potential [T*m]
        """
        ...


@dataclass
class Solenoid:
    """Ideal finite solenoid with analytical field.

    Uses the exact solution for a finite solenoid in terms of
    the on-axis field with end corrections.

    Args:
        length: Solenoid length [m]
        radius: Solenoid radius [m]
        n_turns: Total number of turns
        current: Current [A] or callable(t) -> current
        z_center: Axial position of solenoid center [m]
    """
    length: float
    radius: float
    n_turns: int
    current: Union[float, Callable[[float], float]]
    z_center: float = 0.0

    def _get_current(self, t: float) -> float:
        """Get current at time t."""
        if callable(self.current):
            return self.current(t)
        return self.current

    def B_field(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute magnetic field of finite solenoid.

        Uses the analytical approximation valid for r < radius.
        For points inside, B_z is approximately uniform with end corrections.
        B_r is small inside and computed from div(B)=0.
        """
        I = self._get_current(t)
        n = self.n_turns / self.length  # turns per meter
        B0 = MU0 * n * I  # infinite solenoid field

        # Relative positions from solenoid ends
        z_rel = z - self.z_center
        z_plus = z_rel + self.length / 2  # distance from bottom end
        z_minus = z_rel - self.length / 2  # distance from top end

        # End correction factors using geometry
        # cos(theta) where theta is angle to end from point
        denom_plus = jnp.sqrt(self.radius**2 + z_plus**2)
        denom_minus = jnp.sqrt(self.radius**2 + z_minus**2)

        cos_plus = z_plus / jnp.maximum(denom_plus, 1e-10)
        cos_minus = z_minus / jnp.maximum(denom_minus, 1e-10)

        # Axial field with end corrections
        B_z = 0.5 * B0 * (cos_plus - cos_minus)

        # Radial field from div(B) = 0: (1/r) d(r*B_r)/dr + dB_z/dz = 0
        # For small r: B_r ≈ -(r/2) * dB_z/dz
        dBz_dz_plus = -0.5 * B0 * self.radius**2 / jnp.maximum(denom_plus**3, 1e-30)
        dBz_dz_minus = -0.5 * B0 * self.radius**2 / jnp.maximum(denom_minus**3, 1e-30)
        dBz_dz = dBz_dz_plus - dBz_dz_minus

        B_r = -0.5 * r * dBz_dz

        return B_r, B_z

    def A_phi(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute vector potential from B_z = (1/r) d(r*A_phi)/dr.

        For uniform B_z: A_phi = B_z * r / 2
        """
        _, B_z = self.B_field(r, z, t)
        return 0.5 * B_z * r


@dataclass
class MirrorCoil:
    """Placeholder for magnetic mirror coil pair model."""
    pass


@dataclass
class ThetaPinchArray:
    """Placeholder for theta-pinch coil array model."""
    pass
