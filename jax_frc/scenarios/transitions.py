"""Transition conditions for phase completion."""

from dataclasses import dataclass
from typing import Callable, Optional
from jax_frc.core.state import State


@dataclass
class Transition:
    """Evaluates whether a phase should complete.

    Combines optional condition-based and time-based triggers.
    Condition is checked first; timeout acts as fallback.
    """

    _condition: Optional[Callable[[State, float], bool]] = None
    _timeout: Optional[float] = None
    _name: str = ""

    def evaluate(self, state: State, t: float) -> tuple[bool, str]:
        """Check if transition should trigger.

        Args:
            state: Current simulation state
            t: Current simulation time

        Returns:
            (triggered, reason) where reason is "condition_met", "timeout", or ""
        """
        if self._condition is not None and self._condition(state, t):
            return True, "condition_met"
        if self._timeout is not None and t >= self._timeout:
            return True, "timeout"
        return False, ""


def timeout(t: float) -> Transition:
    """Create a time-based transition.

    Args:
        t: Time at which transition triggers

    Returns:
        Transition that triggers when simulation time >= t
    """
    return Transition(_timeout=t, _name=f"timeout({t})")


def condition(fn: Callable[[State, float], bool]) -> Transition:
    """Create a condition-based transition.

    Args:
        fn: Function (state, t) -> bool that returns True to trigger

    Returns:
        Transition that triggers when fn returns True
    """
    return Transition(_condition=fn, _name="condition")


def any_of(*transitions: Transition) -> Transition:
    """Create transition that triggers when any sub-transition triggers.

    Args:
        *transitions: Sub-transitions to check

    Returns:
        Transition that triggers on first matching sub-transition
    """
    def evaluate_combined(state: State, t: float) -> tuple[bool, str]:
        for trans in transitions:
            triggered, reason = trans.evaluate(state, t)
            if triggered:
                return True, reason
        return False, ""

    result = Transition(_name="any_of")
    result.evaluate = evaluate_combined  # Override method
    return result


def all_of(*transitions: Transition) -> Transition:
    """Create transition that triggers when all sub-transitions trigger.

    Args:
        *transitions: Sub-transitions that must all trigger

    Returns:
        Transition that triggers only when all sub-transitions match
    """
    def evaluate_combined(state: State, t: float) -> tuple[bool, str]:
        for trans in transitions:
            triggered, _ = trans.evaluate(state, t)
            if not triggered:
                return False, ""
        return True, "all_conditions_met"

    result = Transition(_name="all_of")
    result.evaluate = evaluate_combined
    return result
