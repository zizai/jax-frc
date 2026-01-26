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
    """Placeholder for infinite solenoid field model."""
    pass


@dataclass
class MirrorCoil:
    """Placeholder for magnetic mirror coil pair model."""
    pass


@dataclass
class ThetaPinchArray:
    """Placeholder for theta-pinch coil array model."""
    pass
