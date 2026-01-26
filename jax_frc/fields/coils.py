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


def _elliptic_k(m: jnp.ndarray) -> jnp.ndarray:
    """Complete elliptic integral of first kind K(m).

    Uses polynomial approximation valid for 0 <= m < 1.
    """
    # Abramowitz & Stegun approximation
    m1 = 1.0 - m
    m1 = jnp.maximum(m1, 1e-10)  # avoid log(0)

    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212

    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    K = (a0 + a1*m1 + a2*m1**2 + a3*m1**3 + a4*m1**4 +
         (b0 + b1*m1 + b2*m1**2 + b3*m1**3 + b4*m1**4) * (-jnp.log(m1)))

    return K


def _elliptic_e(m: jnp.ndarray) -> jnp.ndarray:
    """Complete elliptic integral of second kind E(m).

    Uses polynomial approximation valid for 0 <= m < 1.
    """
    m1 = 1.0 - m
    m1 = jnp.maximum(m1, 1e-10)

    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451

    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639

    E = (1.0 + a1*m1 + a2*m1**2 + a3*m1**3 + a4*m1**4 +
         (b1*m1 + b2*m1**2 + b3*m1**3 + b4*m1**4) * (-jnp.log(m1)))

    return E


@dataclass
class MirrorCoil:
    """Single circular current loop (mirror coil).

    Computes exact field using elliptic integrals.

    Args:
        z_position: Axial position of coil center [m]
        radius: Coil radius [m]
        current: Current [A] or callable(t) -> current
    """
    z_position: float
    radius: float
    current: Union[float, Callable[[float], float]]

    def _get_current(self, t: float) -> float:
        """Get current at time t."""
        if callable(self.current):
            return self.current(t)
        return self.current

    def B_field(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute magnetic field of current loop using elliptic integrals.

        Uses exact expressions from Jackson, Classical Electrodynamics.
        """
        I = self._get_current(t)
        a = self.radius
        z_rel = z - self.z_position

        # Handle on-axis case separately for numerical stability
        r_safe = jnp.maximum(r, 1e-10)

        # Elliptic integral parameter
        alpha2 = a**2 + r_safe**2 + z_rel**2 - 2*a*r_safe
        beta2 = a**2 + r_safe**2 + z_rel**2 + 2*a*r_safe
        beta = jnp.sqrt(jnp.maximum(beta2, 1e-20))

        m = 1.0 - alpha2 / beta2
        m = jnp.clip(m, 0.0, 1.0 - 1e-10)

        K = _elliptic_k(m)
        E = _elliptic_e(m)

        # Prefactor
        C = MU0 * I / jnp.pi

        # Axial field B_z
        B_z = C / (2.0 * beta) * (K + (a**2 - r_safe**2 - z_rel**2) / alpha2 * E)

        # Radial field B_r
        B_r = C * z_rel / (2.0 * beta * r_safe) * (-K + (a**2 + r_safe**2 + z_rel**2) / alpha2 * E)

        # On axis, B_r = 0 by symmetry
        on_axis = r < 1e-10
        B_r = jnp.where(on_axis, 0.0, B_r)

        # On axis, use simple formula for B_z
        B_z_axis = MU0 * I * a**2 / (2.0 * (a**2 + z_rel**2)**1.5)
        B_z = jnp.where(on_axis, B_z_axis, B_z)

        return B_r, B_z

    def A_phi(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute vector potential using elliptic integrals."""
        I = self._get_current(t)
        a = self.radius
        z_rel = z - self.z_position

        r_safe = jnp.maximum(r, 1e-10)

        beta2 = a**2 + r_safe**2 + z_rel**2 + 2*a*r_safe
        beta = jnp.sqrt(jnp.maximum(beta2, 1e-20))
        alpha2 = a**2 + r_safe**2 + z_rel**2 - 2*a*r_safe

        m = 1.0 - alpha2 / beta2
        m = jnp.clip(m, 0.0, 1.0 - 1e-10)

        K = _elliptic_k(m)
        E = _elliptic_e(m)

        A = MU0 * I * a / (jnp.pi * beta) * ((1.0 - m/2.0) * K - E)

        # On axis A_phi = 0
        on_axis = r < 1e-10
        A = jnp.where(on_axis, 0.0, A)

        return A


@dataclass
class ThetaPinchArray:
    """Placeholder for theta-pinch coil array model."""
    pass
