"""Tests for scenario framework."""

import pytest
from jax_frc.scenarios.transitions import Transition, timeout, condition, any_of, all_of
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

    def test_all_of_triggers_when_all_match(self):
        """all_of triggers only when all sub-transitions trigger."""
        trans = all_of(
            timeout(5.0),
            condition(lambda s, t: t > 3.0)
        )

        # At t=4, timeout not met (need >= 5.0)
        triggered, reason = trans.evaluate(None, t=4.0)
        assert not triggered

        # At t=5, both met
        triggered, reason = trans.evaluate(None, t=5.0)
        assert triggered
        assert reason == "all_conditions_met"


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


class TestScenario:
    """Tests for Scenario runner."""

    @pytest.fixture
    def simple_geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=10, nz=20,
            r_min=0.1, r_max=1.0,
            z_min=-1.0, z_max=1.0
        )

    @pytest.fixture
    def initial_state(self):
        return State.zeros(nr=10, nz=20)

    def test_scenario_runs_single_phase(self, simple_geometry, initial_state, mock_physics_model, mock_solver):
        """Scenario runs a single phase to completion."""
        from jax_frc.scenarios.scenario import Scenario, ScenarioResult

        phase = Phase(name="test", transition=timeout(1.0))

        scenario = Scenario(
            name="test_scenario",
            phases=[phase],
            geometry=simple_geometry,
            initial_state=initial_state,
            physics_model=mock_physics_model,
            solver=mock_solver,
            dt=0.1,
        )

        result = scenario.run()

        assert isinstance(result, ScenarioResult)
        assert len(result.phase_results) == 1
        assert result.phase_results[0].name == "test"
        assert result.phase_results[0].termination == "timeout"

    def test_scenario_chains_phases(self, simple_geometry, initial_state, mock_physics_model, mock_solver):
        """Scenario passes state between phases."""
        from jax_frc.scenarios.scenario import Scenario, ScenarioResult

        phase1 = Phase(name="phase1", transition=timeout(1.0))
        phase2 = Phase(name="phase2", transition=timeout(2.0))

        scenario = Scenario(
            name="test_scenario",
            phases=[phase1, phase2],
            geometry=simple_geometry,
            initial_state=initial_state,
            physics_model=mock_physics_model,
            solver=mock_solver,
            dt=0.1,
        )

        result = scenario.run()

        assert len(result.phase_results) == 2
        assert result.phase_results[0].name == "phase1"
        assert result.phase_results[1].name == "phase2"


class TestPhysicsConditions:
    """Tests for physics-based transition conditions."""

    def test_separation_below_threshold(self):
        """separation_below triggers when dZ < threshold."""
        from jax_frc.scenarios.transitions import separation_below
        import jax.numpy as jnp

        # Create state with two magnetic nulls
        state = State.zeros(nr=10, nz=40)
        # Set psi with two peaks (simplified)
        psi = jnp.zeros((10, 40))
        psi = psi.at[5, 10].set(1.0)  # Null 1 at z_idx=10
        psi = psi.at[5, 30].set(1.0)  # Null 2 at z_idx=30
        state = state.replace(psi=psi)

        geometry = Geometry(
            coord_system="cylindrical",
            nr=10, nz=40,
            r_min=0.1, r_max=1.0,
            z_min=-2.0, z_max=2.0  # dz = 0.1, separation = 20*0.1 = 2.0
        )

        trans = separation_below(3.0, geometry)  # Should trigger (2.0 < 3.0)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert triggered

        trans = separation_below(1.0, geometry)  # Should not trigger (2.0 > 1.0)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert not triggered

    def test_temperature_above_threshold(self):
        """temperature_above triggers when T > threshold."""
        from jax_frc.scenarios.transitions import temperature_above
        import jax.numpy as jnp

        state = State.zeros(nr=10, nz=20)
        # Set pressure and density to give T = p/n
        state = state.replace(
            p=jnp.ones((10, 20)) * 100.0,
            n=jnp.ones((10, 20)) * 10.0  # T = 100/10 = 10
        )

        trans = temperature_above(5.0)  # Should trigger (10 > 5)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert triggered

        trans = temperature_above(15.0)  # Should not trigger (10 < 15)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert not triggered

    def test_flux_below_threshold(self):
        """flux_below triggers when max(psi) < threshold."""
        from jax_frc.scenarios.transitions import flux_below
        import jax.numpy as jnp

        state = State.zeros(nr=10, nz=20)
        psi = jnp.ones((10, 20)) * 0.5
        psi = psi.at[5, 10].set(2.0)  # Peak at 2.0
        state = state.replace(psi=psi)

        trans = flux_below(3.0)  # Should trigger (2.0 < 3.0)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert triggered

        trans = flux_below(1.0)  # Should not trigger (2.0 > 1.0)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert not triggered

    def test_velocity_below_threshold(self):
        """velocity_below triggers when max(|v|) < threshold."""
        from jax_frc.scenarios.transitions import velocity_below
        import jax.numpy as jnp

        state = State.zeros(nr=10, nz=20)
        # Set velocity field with peak magnitude
        v = jnp.zeros((10, 20, 3))
        v = v.at[5, 10, 0].set(3.0)  # vr = 3
        v = v.at[5, 10, 2].set(4.0)  # vz = 4, |v| = 5
        state = state.replace(v=v)

        trans = velocity_below(10.0)  # Should trigger (5.0 < 10.0)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert triggered

        trans = velocity_below(3.0)  # Should not trigger (5.0 > 3.0)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert not triggered


def test_scenario_requires_physics_model_and_solver():
    """Scenario should require physics_model and solver fields."""
    from jax_frc.scenarios.scenario import Scenario
    from jax_frc.models.base import PhysicsModel
    from jax_frc.solvers.base import Solver

    # Check that Scenario has these attributes in its signature
    import inspect
    sig = inspect.signature(Scenario)
    params = list(sig.parameters.keys())

    assert "physics_model" in params
    assert "solver" in params


def test_scenario_calls_solver_step():
    """Scenario._run_phase should call solver.step() for physics evolution."""
    from unittest.mock import MagicMock, patch
    from jax_frc.core.geometry import Geometry
    from jax_frc.core.state import State
    from jax_frc.scenarios import Scenario
    from jax_frc.scenarios.phase import Phase
    from jax_frc.scenarios.transitions import timeout

    geometry = Geometry(
        coord_system="cylindrical", nr=8, nz=16,
        r_min=0.01, r_max=1.0, z_min=-1.0, z_max=1.0
    )
    state = State.zeros(nr=8, nz=16)

    # Mock physics model and solver
    mock_model = MagicMock()
    mock_solver = MagicMock()

    # Solver.step returns updated state with incremented time
    def fake_step(s, dt, model, geom):
        return s.replace(time=s.time + dt, step=s.step + 1)
    mock_solver.step.side_effect = fake_step

    phase = Phase(name="test", transition=timeout(0.05))

    scenario = Scenario(
        name="test",
        phases=[phase],
        geometry=geometry,
        initial_state=state,
        physics_model=mock_model,
        solver=mock_solver,
        dt=0.01,
    )

    result = scenario.run()

    # Solver should have been called multiple times
    assert mock_solver.step.call_count >= 5