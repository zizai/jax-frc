"""Scenario framework for multi-phase FRC simulations."""

from jax_frc.scenarios.phase import Phase, PhaseResult, PHASE_REGISTRY, register_phase
from jax_frc.scenarios.transitions import (
    Transition, timeout, condition, any_of, all_of,
    separation_below, temperature_above, flux_below, velocity_below,
    transition_from_spec,
)
# Import phases module to register MergingPhase in PHASE_REGISTRY
from jax_frc.scenarios import phases

__all__ = [
    # Phase framework
    "Phase",
    "PhaseResult",
    "PHASE_REGISTRY",
    "register_phase",
    # Transitions
    "Transition",
    "timeout",
    "condition",
    "any_of",
    "all_of",
    "separation_below",
    "temperature_above",
    "flux_below",
    "velocity_below",
    "transition_from_spec",
]
