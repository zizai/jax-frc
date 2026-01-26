"""Tests for external circuit components."""

import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.8,
        z_min=-1.0,
        z_max=1.0,
    )


class TestCoilGeometry:
    """Tests for CoilGeometry dataclass."""

    def test_creation(self):
        """Can create CoilGeometry."""
        from jax_frc.circuits.external import CoilGeometry

        coil = CoilGeometry(
            z_center=0.0,
            radius=0.6,
            length=0.5,
            n_turns=50,
        )
        assert coil.radius == 0.6


class TestCircuitDriver:
    """Tests for CircuitDriver."""

    def test_voltage_mode(self):
        """Voltage mode driver returns waveform value."""
        from jax_frc.circuits.external import CircuitDriver
        from jax_frc.circuits.waveforms import make_ramp

        driver = CircuitDriver(
            mode="voltage",
            waveform=make_ramp(0.0, 1000.0, 1e-4),
        )
        assert driver.get_voltage(t=1e-4, state=None, error_integral=0.0) == 1000.0

    def test_current_mode(self):
        """Current mode driver returns waveform value."""
        from jax_frc.circuits.external import CircuitDriver
        from jax_frc.circuits.waveforms import make_constant

        driver = CircuitDriver(
            mode="current",
            waveform=make_constant(100.0),
        )
        assert driver.get_target_current(t=0.0, state=None) == 100.0

    def test_feedback_mode(self):
        """Feedback mode computes PID control voltage."""
        from jax_frc.circuits.external import CircuitDriver

        # Simple proportional control
        driver = CircuitDriver(
            mode="feedback",
            feedback_gains=(100.0, 0.0, 0.0),  # Kp only
            feedback_target=lambda state: 1.0,  # Target value
            feedback_measure=lambda state: 0.5,  # Measured value
        )
        # Error = target - measured = 1.0 - 0.5 = 0.5
        # V = Kp * error = 100 * 0.5 = 50
        V = driver.get_voltage(t=0.0, state=None, error_integral=0.0)
        assert jnp.isclose(V, 50.0)


class TestExternalCircuit:
    """Tests for ExternalCircuit."""

    def test_creation(self):
        """Can create ExternalCircuit."""
        from jax_frc.circuits.external import ExternalCircuit, CoilGeometry, CircuitDriver
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.waveforms import make_constant

        circuit = ExternalCircuit(
            name="test_coil",
            coil=CoilGeometry(z_center=0.0, radius=0.6, length=0.5, n_turns=50),
            params=CircuitParams(
                L=jnp.array([5e-3]),
                R=jnp.array([0.05]),
                C=jnp.array([jnp.inf]),
            ),
            driver=CircuitDriver(mode="voltage", waveform=make_constant(0.0)),
        )
        assert circuit.name == "test_coil"


class TestExternalCircuits:
    """Tests for ExternalCircuits collection."""

    def test_creation(self):
        """Can create ExternalCircuits."""
        from jax_frc.circuits.external import (
            ExternalCircuits,
            ExternalCircuit,
            CoilGeometry,
            CircuitDriver,
        )
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.waveforms import make_constant

        circuit1 = ExternalCircuit(
            name="coil1",
            coil=CoilGeometry(z_center=-0.5, radius=0.6, length=0.5, n_turns=50),
            params=CircuitParams(
                L=jnp.array([5e-3]),
                R=jnp.array([0.05]),
                C=jnp.array([jnp.inf]),
            ),
            driver=CircuitDriver(mode="voltage", waveform=make_constant(0.0)),
        )
        circuit2 = ExternalCircuit(
            name="coil2",
            coil=CoilGeometry(z_center=0.5, radius=0.6, length=0.5, n_turns=50),
            params=CircuitParams(
                L=jnp.array([5e-3]),
                R=jnp.array([0.05]),
                C=jnp.array([jnp.inf]),
            ),
            driver=CircuitDriver(mode="voltage", waveform=make_constant(0.0)),
        )

        external = ExternalCircuits(circuits=[circuit1, circuit2])
        assert external.n_circuits == 2

    def test_compute_b_field(self, geometry):
        """External coils contribute B-field."""
        from jax_frc.circuits.external import (
            ExternalCircuits,
            ExternalCircuit,
            CoilGeometry,
            CircuitDriver,
        )
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.waveforms import make_constant

        circuit = ExternalCircuit(
            name="solenoid",
            coil=CoilGeometry(z_center=0.0, radius=0.6, length=1.0, n_turns=100),
            params=CircuitParams(
                L=jnp.array([5e-3]),
                R=jnp.array([0.05]),
                C=jnp.array([jnp.inf]),
            ),
            driver=CircuitDriver(mode="voltage", waveform=make_constant(0.0)),
        )
        external = ExternalCircuits(circuits=[circuit])

        I = jnp.array([1000.0])  # 1000 A
        B = external.compute_b_field(I, geometry)

        assert B.shape == (geometry.nr, geometry.nz, 3)
        # Should have significant Bz on axis
        Bz_center = B[0, geometry.nz // 2, 2]
        assert Bz_center > 0.05  # Should be order 0.1 T for 1000 A
