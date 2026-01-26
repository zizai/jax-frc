"""Circuit coupling for burning plasma simulations."""

from jax_frc.circuits.state import CircuitState, CircuitParams
from jax_frc.circuits.pickup import PickupCoilArray
from jax_frc.circuits.waveforms import (
    make_ramp,
    make_sinusoid,
    make_crowbar,
    make_pulse,
    make_constant,
    waveform_from_config,
)
from jax_frc.circuits.external import (
    CoilGeometry,
    CircuitDriver,
    ExternalCircuit,
    ExternalCircuits,
)
from jax_frc.circuits.coupling import FluxCoupling
from jax_frc.circuits.system import CircuitSystem

__all__ = [
    "CircuitState",
    "CircuitParams",
    "PickupCoilArray",
    "make_ramp",
    "make_sinusoid",
    "make_crowbar",
    "make_pulse",
    "make_constant",
    "waveform_from_config",
    "CoilGeometry",
    "CircuitDriver",
    "ExternalCircuit",
    "ExternalCircuits",
    "FluxCoupling",
    "CircuitSystem",
]
