"""Scenario framework for multi-phase FRC simulations."""

from jax_frc.scenarios.phase import Phase, PhaseResult
from jax_frc.scenarios.transitions import (
    Transition, timeout, condition, any_of, all_of,
    separation_below, temperature_above, flux_below, velocity_below
)
from jax_frc.scenarios.scenario import Scenario, ScenarioResult

__all__ = [
    "Phase",
    "PhaseResult",
    "Transition",
    "timeout",
    "condition",
    "any_of",
    "all_of",
    "separation_below",
    "temperature_above",
    "flux_below",
    "velocity_below",
    "Scenario",
    "ScenarioResult",
]
