"""Circuit coupling for burning plasma simulations."""

from jax_frc.circuits.state import CircuitState, CircuitParams
from jax_frc.circuits.waveforms import (
    make_ramp,
    make_sinusoid,
    make_crowbar,
    make_pulse,
    make_constant,
    waveform_from_config,
)

__all__ = [
    "CircuitState",
    "CircuitParams",
    "make_ramp",
    "make_sinusoid",
    "make_crowbar",
    "make_pulse",
    "make_constant",
    "waveform_from_config",
]
