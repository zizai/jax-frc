"""External circuits with coils and drivers."""

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp
from jax import Array

from jax_frc.circuits.state import CircuitParams
from jax_frc.core.geometry import Geometry
from jax_frc.constants import MU0


@dataclass(frozen=True)
class CoilGeometry:
    """Physical geometry of an external coil.

    Attributes:
        z_center: Axial position of coil center [m]
        radius: Coil radius [m]
        length: Coil length [m] (for solenoid model)
        n_turns: Number of turns
    """

    z_center: float
    radius: float
    length: float
    n_turns: int


@dataclass(frozen=True)
class CircuitDriver:
    """Voltage or current source for external circuit.

    Supports three modes:
    - "voltage": Apply voltage waveform directly
    - "current": Target current waveform (requires high-bandwidth control)
    - "feedback": PID control based on plasma state

    Attributes:
        mode: "voltage", "current", or "feedback"
        waveform: For voltage/current mode, callable(t) -> value
        feedback_gains: For feedback mode, (Kp, Ki, Kd) gains
        feedback_target: For feedback mode, callable(state) -> target value
        feedback_measure: For feedback mode, callable(state) -> measured value
    """

    mode: str
    waveform: Optional[Callable[[float], float]] = None
    feedback_gains: Optional[tuple[float, float, float]] = None
    feedback_target: Optional[Callable] = None
    feedback_measure: Optional[Callable] = None

    def get_voltage(self, t: float, state, error_integral: float) -> float:
        """Get applied voltage at time t.

        Args:
            t: Current time [s]
            state: Current plasma/circuit state (for feedback)
            error_integral: Accumulated error for integral term

        Returns:
            Voltage to apply [V]
        """
        if self.mode == "voltage":
            return self.waveform(t)
        elif self.mode == "feedback":
            Kp, Ki, Kd = self.feedback_gains
            target = self.feedback_target(state)
            measured = self.feedback_measure(state)
            error = target - measured
            # Simple PI control (no derivative for now)
            return Kp * error + Ki * error_integral
        else:
            # Current mode doesn't directly provide voltage
            return 0.0

    def get_target_current(self, t: float, state) -> Optional[float]:
        """Get target current for current-controlled mode.

        Args:
            t: Current time [s]
            state: Current state (unused for waveform mode)

        Returns:
            Target current [A] or None if not in current mode
        """
        if self.mode == "current":
            return self.waveform(t)
        return None


@dataclass(frozen=True)
class ExternalCircuit:
    """Single external circuit with coil and driver.

    Attributes:
        name: Circuit identifier
        coil: Physical coil geometry
        params: Circuit parameters (L, R, C)
        driver: Voltage/current source
    """

    name: str
    coil: CoilGeometry
    params: CircuitParams
    driver: CircuitDriver


@dataclass
class ExternalCircuits:
    """Collection of external circuits.

    Manages multiple external coils and their circuits.
    """

    circuits: list[ExternalCircuit]

    @property
    def n_circuits(self) -> int:
        """Number of external circuits."""
        return len(self.circuits)

    def get_combined_params(self) -> CircuitParams:
        """Get combined circuit parameters as arrays."""
        L = jnp.array([c.params.L[0] for c in self.circuits])
        R = jnp.array([c.params.R[0] for c in self.circuits])
        C = jnp.array([c.params.C[0] for c in self.circuits])
        return CircuitParams(L=L, R=R, C=C)

    def compute_b_field(self, I: Array, geometry: Geometry) -> Array:
        """Compute magnetic field from all external coil currents.

        Uses a finite solenoid model for each coil.

        Args:
            I: Current in each coil [A], shape (n_circuits,)
            geometry: Computational geometry

        Returns:
            B: Magnetic field contribution (nr, nz, 3)
        """
        B_total = jnp.zeros((geometry.nr, geometry.nz, 3))

        for i, circuit in enumerate(self.circuits):
            coil = circuit.coil
            current = I[i]

            B_coil = self._solenoid_field(
                current=current,
                z_center=coil.z_center,
                radius=coil.radius,
                length=coil.length,
                n_turns=coil.n_turns,
                geometry=geometry,
            )
            B_total = B_total + B_coil

        return B_total

    def _solenoid_field(
        self,
        current: float,
        z_center: float,
        radius: float,
        length: float,
        n_turns: int,
        geometry: Geometry,
    ) -> Array:
        """Compute B-field of a finite solenoid.

        Uses analytical approximation valid for points inside the solenoid.

        Args:
            current: Coil current [A]
            z_center: Solenoid center position [m]
            radius: Solenoid radius [m]
            length: Solenoid length [m]
            n_turns: Number of turns
            geometry: Computational geometry

        Returns:
            B: Magnetic field (nr, nz, 3)
        """
        n = n_turns / length  # Turns per meter
        B0 = MU0 * n * current  # Infinite solenoid field

        r_grid = geometry.r_grid
        z_grid = geometry.z_grid

        # Relative positions from solenoid ends
        z_rel = z_grid - z_center
        z_plus = z_rel + length / 2
        z_minus = z_rel - length / 2

        # End correction factors
        denom_plus = jnp.sqrt(radius**2 + z_plus**2)
        denom_minus = jnp.sqrt(radius**2 + z_minus**2)

        cos_plus = z_plus / jnp.maximum(denom_plus, 1e-10)
        cos_minus = z_minus / jnp.maximum(denom_minus, 1e-10)

        # Axial field with end corrections
        Bz = 0.5 * B0 * (cos_plus - cos_minus)

        # Radial field from div(B) = 0
        dBz_dz_plus = -0.5 * B0 * radius**2 / jnp.maximum(denom_plus**3, 1e-30)
        dBz_dz_minus = -0.5 * B0 * radius**2 / jnp.maximum(denom_minus**3, 1e-30)
        dBz_dz = dBz_dz_plus - dBz_dz_minus
        Br = -0.5 * r_grid * dBz_dz

        # Assemble B vector
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 0].set(Br)
        B = B.at[:, :, 2].set(Bz)

        return B
