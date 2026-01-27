"""Transition conditions for phase completion."""

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING
import jax.numpy as jnp
from jax_frc.core.state import State

if TYPE_CHECKING:
    from jax_frc.core.geometry import Geometry


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


def separation_below(threshold: float, geometry: "Geometry") -> Transition:
    """Transition when FRC null separation drops below threshold.

    Finds magnetic nulls (local maxima of psi) and computes their axial separation.

    Args:
        threshold: Separation threshold in length units
        geometry: Geometry for coordinate mapping

    Returns:
        Transition that triggers when dZ < threshold
    """
    def check(state: State, t: float) -> bool:
        B_z = state.B[:, state.B.shape[1] // 2, :, 2]

        # Find null positions (local maxima of psi along z at each r)
        # Simplified: find z indices of max psi in each half of domain
        nz = B_z.shape[1]
        mid = nz // 2

        # Find max in each half
        left_half = B_z[:, :mid]
        right_half = B_z[:, mid:]

        left_max_z = jnp.argmax(jnp.max(left_half, axis=0))
        right_max_z = mid + jnp.argmax(jnp.max(right_half, axis=0))

        # Convert to physical coordinates
        z1 = geometry.z_min + left_max_z * geometry.dz
        z2 = geometry.z_min + right_max_z * geometry.dz

        separation = jnp.abs(z2 - z1)
        return float(separation) < threshold

    return condition(check)


def temperature_above(threshold: float) -> Transition:
    """Transition when peak temperature exceeds threshold.

    Temperature computed as T = p/n (in appropriate units).

    Args:
        threshold: Temperature threshold

    Returns:
        Transition that triggers when max(T) > threshold
    """
    def check(state: State, t: float) -> bool:
        # T = p / n, avoid division by zero
        n_safe = jnp.maximum(state.n, 1e-10)
        T = state.p / n_safe
        return float(jnp.max(T)) > threshold

    return condition(check)


def flux_below(threshold: float) -> Transition:
    """Transition when peak poloidal flux drops below threshold.

    Indicates loss of FRC confinement.

    Args:
        threshold: Flux threshold

    Returns:
        Transition that triggers when max(psi) < threshold
    """
    def check(state: State, t: float) -> bool:
        return float(jnp.max(jnp.abs(state.B[:, :, :, 2]))) < threshold

    return condition(check)


def velocity_below(threshold: float) -> Transition:
    """Transition when peak velocity drops below threshold.

    Used for translation phase completion.

    Args:
        threshold: Velocity threshold

    Returns:
        Transition that triggers when max(|v|) < threshold
    """
    def check(state: State, t: float) -> bool:
        if state.v is None:
            return True
        v_mag = jnp.sqrt(jnp.sum(state.v**2, axis=-1))
        return float(jnp.max(v_mag)) < threshold

    return condition(check)


def transition_from_spec(spec: "TransitionSpec", geometry: "Geometry" = None) -> Transition:
    """Create Transition from declarative TransitionSpec.

    Factory function that converts a TransitionSpec dataclass into a
    concrete Transition object. Enables declarative/serializable phase
    definitions.

    Args:
        spec: TransitionSpec with type, value, and optional children
        geometry: Geometry for transitions that need spatial info
                 (e.g., separation_below)

    Returns:
        Transition object ready for evaluation

    Raises:
        ValueError: If spec.type is unknown

    Example::

        spec = TransitionSpec(type="timeout", value=1e-3)
        trans = transition_from_spec(spec)

        # Composite transition
        spec = TransitionSpec(
            type="any_of",
            children=[
                TransitionSpec(type="timeout", value=1e-3),
                TransitionSpec(type="separation_below", value=0.1),
            ]
        )
        trans = transition_from_spec(spec, geometry)
    """
    # Import here to avoid circular import
    from jax_frc.configurations.linear_configuration import TransitionSpec

    trans_type = spec.type

    if trans_type == "timeout":
        return timeout(spec.value)

    elif trans_type == "separation_below":
        if geometry is None:
            raise ValueError("separation_below requires geometry")
        return separation_below(spec.value, geometry)

    elif trans_type == "temperature_above":
        return temperature_above(spec.value)

    elif trans_type == "flux_below":
        return flux_below(spec.value)

    elif trans_type == "velocity_below":
        return velocity_below(spec.value)

    elif trans_type == "any_of":
        if spec.children is None:
            raise ValueError("any_of requires children")
        child_transitions = [transition_from_spec(c, geometry) for c in spec.children]
        return any_of(*child_transitions)

    elif trans_type == "all_of":
        if spec.children is None:
            raise ValueError("all_of requires children")
        child_transitions = [transition_from_spec(c, geometry) for c in spec.children]
        return all_of(*child_transitions)

    elif trans_type == "condition":
        # For condition type, spec.value should be a callable
        if spec.value is None or not callable(spec.value):
            raise ValueError("condition requires a callable value")
        return condition(spec.value)

    else:
        raise ValueError(f"Unknown transition type: {trans_type}. "
                        f"Valid types: timeout, separation_below, temperature_above, "
                        f"flux_below, velocity_below, any_of, all_of, condition")
