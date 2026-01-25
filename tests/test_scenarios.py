"""Tests for scenario framework."""

import pytest
from jax_frc.scenarios.transitions import Transition, timeout, condition, any_of
from jax_frc.scenarios.phase import Phase, PhaseResult
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


class TestTransition:
    """Tests for Transition class."""

    def test_timeout_triggers_at_time(self):
        """Timeout transition triggers when t >= timeout."""
        trans = timeout(10.0)

        triggered, reason = trans.evaluate(None, t=5.0)
        assert not triggered

        triggered, reason = trans.evaluate(None, t=10.0)
        assert triggered
        assert reason == "timeout"

    def test_condition_triggers_when_true(self):
        """Condition transition triggers when condition returns True."""
        trans = condition(lambda state, t: t > 5.0)

        triggered, reason = trans.evaluate(None, t=3.0)
        assert not triggered

        triggered, reason = trans.evaluate(None, t=6.0)
        assert triggered
        assert reason == "condition_met"

    def test_any_of_triggers_on_first_match(self):
        """any_of triggers when any sub-transition triggers."""
        trans = any_of(
            timeout(10.0),
            condition(lambda s, t: t > 5.0)
        )

        triggered, reason = trans.evaluate(None, t=6.0)
        assert triggered
        assert reason == "condition_met"


class TestPhase:
    """Tests for Phase base class."""

    def test_phase_setup_returns_state(self):
        """Phase.setup returns modified state."""
        phase = Phase(name="test", transition=timeout(10.0))
        geometry = Geometry(
            coord_system="cylindrical",
            nr=10, nz=20,
            r_min=0.1, r_max=1.0,
            z_min=-1.0, z_max=1.0
        )
        state = State.zeros(nr=10, nz=20)

        result = phase.setup(state, geometry, {})
        assert isinstance(result, State)

    def test_phase_is_complete_delegates_to_transition(self):
        """Phase.is_complete uses transition.evaluate."""
        phase = Phase(name="test", transition=timeout(5.0))
        state = State.zeros(nr=10, nz=20)

        complete, reason = phase.is_complete(state, t=3.0)
        assert not complete

        complete, reason = phase.is_complete(state, t=5.0)
        assert complete
        assert reason == "timeout"