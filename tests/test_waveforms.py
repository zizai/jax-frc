"""Tests for circuit waveform functions."""

import jax.numpy as jnp
import pytest


class TestRampWaveform:
    """Tests for ramp voltage waveform."""

    def test_ramp_start(self):
        """Ramp starts at V0."""
        from jax_frc.circuits.waveforms import make_ramp

        ramp = make_ramp(V0=0.0, V1=1000.0, t_ramp=1e-4)
        assert ramp(0.0) == 0.0

    def test_ramp_end(self):
        """Ramp reaches V1 at t_ramp."""
        from jax_frc.circuits.waveforms import make_ramp

        ramp = make_ramp(V0=0.0, V1=1000.0, t_ramp=1e-4)
        assert ramp(1e-4) == 1000.0

    def test_ramp_midpoint(self):
        """Ramp is linear."""
        from jax_frc.circuits.waveforms import make_ramp

        ramp = make_ramp(V0=0.0, V1=1000.0, t_ramp=1e-4)
        assert jnp.isclose(ramp(5e-5), 500.0)

    def test_ramp_saturates(self):
        """Ramp holds at V1 after t_ramp."""
        from jax_frc.circuits.waveforms import make_ramp

        ramp = make_ramp(V0=0.0, V1=1000.0, t_ramp=1e-4)
        assert ramp(2e-4) == 1000.0


class TestSinusoidWaveform:
    """Tests for sinusoidal waveform."""

    def test_sinusoid_amplitude(self):
        """Sinusoid has correct amplitude."""
        from jax_frc.circuits.waveforms import make_sinusoid

        sin_wave = make_sinusoid(amplitude=100.0, frequency=1e3, phase=0.0)
        # At t = 1/(4*f), sin(pi/2) = 1
        t_quarter = 1.0 / (4 * 1e3)
        assert jnp.isclose(sin_wave(t_quarter), 100.0, rtol=1e-5)

    def test_sinusoid_frequency(self):
        """Sinusoid has correct period."""
        from jax_frc.circuits.waveforms import make_sinusoid

        sin_wave = make_sinusoid(amplitude=100.0, frequency=1e3, phase=0.0)
        # At t = 1/f (one period), should return to 0
        # Use atol=1e-3 to account for float32 precision in sin(2*pi)
        assert jnp.isclose(sin_wave(1e-3), 0.0, atol=1e-3)


class TestCrowbarWaveform:
    """Tests for crowbar (step-down) waveform."""

    def test_crowbar_before(self):
        """Crowbar is at V_initial before t_crowbar."""
        from jax_frc.circuits.waveforms import make_crowbar

        crowbar = make_crowbar(V_initial=1000.0, t_crowbar=1e-4)
        assert crowbar(5e-5) == 1000.0

    def test_crowbar_after(self):
        """Crowbar drops to 0 after t_crowbar."""
        from jax_frc.circuits.waveforms import make_crowbar

        crowbar = make_crowbar(V_initial=1000.0, t_crowbar=1e-4)
        assert crowbar(1.5e-4) == 0.0


class TestPulseWaveform:
    """Tests for pulse waveform."""

    def test_pulse_on(self):
        """Pulse is at amplitude during pulse."""
        from jax_frc.circuits.waveforms import make_pulse

        pulse = make_pulse(amplitude=500.0, t_start=1e-5, t_end=5e-5)
        assert pulse(3e-5) == 500.0

    def test_pulse_off_before(self):
        """Pulse is 0 before t_start."""
        from jax_frc.circuits.waveforms import make_pulse

        pulse = make_pulse(amplitude=500.0, t_start=1e-5, t_end=5e-5)
        assert pulse(0.5e-5) == 0.0

    def test_pulse_off_after(self):
        """Pulse is 0 after t_end."""
        from jax_frc.circuits.waveforms import make_pulse

        pulse = make_pulse(amplitude=500.0, t_start=1e-5, t_end=5e-5)
        assert pulse(6e-5) == 0.0


class TestWaveformFromConfig:
    """Tests for creating waveforms from config dicts."""

    def test_ramp_from_config(self):
        """Can create ramp from config."""
        from jax_frc.circuits.waveforms import waveform_from_config

        config = {"type": "ramp", "V0": 0, "V1": 1000, "t_ramp": 1e-4}
        waveform = waveform_from_config(config)
        assert waveform(1e-4) == 1000.0

    def test_constant_from_config(self):
        """Can create constant voltage from config."""
        from jax_frc.circuits.waveforms import waveform_from_config

        config = {"type": "constant", "value": 500.0}
        waveform = waveform_from_config(config)
        assert waveform(0.0) == 500.0
        assert waveform(1.0) == 500.0

    def test_sinusoid_from_config(self):
        """Can create sinusoid from config."""
        from jax_frc.circuits.waveforms import waveform_from_config

        config = {"type": "sinusoid", "amplitude": 100.0, "frequency": 1e3}
        waveform = waveform_from_config(config)
        t_quarter = 1.0 / (4 * 1e3)
        assert jnp.isclose(waveform(t_quarter), 100.0, rtol=1e-5)

    def test_crowbar_from_config(self):
        """Can create crowbar from config."""
        from jax_frc.circuits.waveforms import waveform_from_config

        config = {"type": "crowbar", "V_initial": 1000.0, "t_crowbar": 1e-4}
        waveform = waveform_from_config(config)
        assert waveform(5e-5) == 1000.0
        assert waveform(1.5e-4) == 0.0

    def test_pulse_from_config(self):
        """Can create pulse from config."""
        from jax_frc.circuits.waveforms import waveform_from_config

        config = {"type": "pulse", "amplitude": 500.0, "t_start": 1e-5, "t_end": 5e-5}
        waveform = waveform_from_config(config)
        assert waveform(3e-5) == 500.0

    def test_unknown_type_raises(self):
        """Unknown waveform type raises ValueError."""
        from jax_frc.circuits.waveforms import waveform_from_config

        config = {"type": "unknown"}
        with pytest.raises(ValueError, match="Unknown waveform type"):
            waveform_from_config(config)
