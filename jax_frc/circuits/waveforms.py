"""Waveform generators for circuit voltage/current sources.

All waveforms are JAX-compatible pure functions suitable for JIT compilation.
"""

from typing import Callable

import jax.numpy as jnp


def make_ramp(V0: float, V1: float, t_ramp: float) -> Callable[[float], float]:
    """Create linear ramp waveform.

    Args:
        V0: Initial value
        V1: Final value
        t_ramp: Time to reach V1 [s]

    Returns:
        Function t -> V that ramps from V0 to V1 over t_ramp, then holds at V1
    """

    def ramp(t: float) -> float:
        fraction = jnp.clip(t / t_ramp, 0.0, 1.0)
        return V0 + (V1 - V0) * fraction

    return ramp


def make_sinusoid(
    amplitude: float, frequency: float, phase: float = 0.0
) -> Callable[[float], float]:
    """Create sinusoidal waveform.

    Args:
        amplitude: Peak amplitude
        frequency: Frequency [Hz]
        phase: Phase offset [rad]

    Returns:
        Function t -> A * sin(2*pi*f*t + phase)
    """

    def sinusoid(t: float) -> float:
        return amplitude * jnp.sin(2 * jnp.pi * frequency * t + phase)

    return sinusoid


def make_crowbar(V_initial: float, t_crowbar: float) -> Callable[[float], float]:
    """Create crowbar (step-down) waveform.

    Args:
        V_initial: Voltage before crowbar
        t_crowbar: Time at which voltage drops to zero [s]

    Returns:
        Function t -> V that is V_initial before t_crowbar, 0 after
    """

    def crowbar(t: float) -> float:
        return jnp.where(t < t_crowbar, V_initial, 0.0)

    return crowbar


def make_pulse(
    amplitude: float, t_start: float, t_end: float
) -> Callable[[float], float]:
    """Create square pulse waveform.

    Args:
        amplitude: Pulse amplitude
        t_start: Pulse start time [s]
        t_end: Pulse end time [s]

    Returns:
        Function t -> V that is amplitude during [t_start, t_end), 0 otherwise
    """

    def pulse(t: float) -> float:
        in_pulse = (t >= t_start) & (t < t_end)
        return jnp.where(in_pulse, amplitude, 0.0)

    return pulse


def make_constant(value: float) -> Callable[[float], float]:
    """Create constant waveform.

    Args:
        value: Constant value

    Returns:
        Function t -> value
    """

    def constant(t: float) -> float:
        # Return as JAX array for consistency with other waveforms
        return value + 0.0 * t  # Ensures output is same type as input

    return constant


def waveform_from_config(config: dict) -> Callable[[float], float]:
    """Create waveform from configuration dictionary.

    Args:
        config: Dictionary with 'type' key and type-specific parameters:
            - type: "ramp" -> V0, V1, t_ramp
            - type: "sinusoid" -> amplitude, frequency, phase (optional)
            - type: "crowbar" -> V_initial, t_crowbar
            - type: "pulse" -> amplitude, t_start, t_end
            - type: "constant" -> value

    Returns:
        Waveform function t -> V
    """
    waveform_type = config["type"]

    if waveform_type == "ramp":
        return make_ramp(
            V0=config["V0"],
            V1=config["V1"],
            t_ramp=config["t_ramp"],
        )
    elif waveform_type == "sinusoid":
        return make_sinusoid(
            amplitude=config["amplitude"],
            frequency=config["frequency"],
            phase=config.get("phase", 0.0),
        )
    elif waveform_type == "crowbar":
        return make_crowbar(
            V_initial=config["V_initial"],
            t_crowbar=config["t_crowbar"],
        )
    elif waveform_type == "pulse":
        return make_pulse(
            amplitude=config["amplitude"],
            t_start=config["t_start"],
            t_end=config["t_end"],
        )
    elif waveform_type == "constant":
        return make_constant(value=config["value"])
    else:
        raise ValueError(f"Unknown waveform type: {waveform_type}")
