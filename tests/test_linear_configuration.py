"""Tests for LinearConfiguration and related classes."""

import pytest
from dataclasses import dataclass
from typing import List
import jax.numpy as jnp

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.configurations import (
    LinearConfiguration,
    ConfigurationResult,
    TransitionSpec,
    PhaseSpec,
)
from jax_frc.configurations import (
    Phase, PhaseResult, PHASE_REGISTRY, timeout, transition_from_spec,
)
from jax_frc.models.base import PhysicsModel


class TestTransitionSpec:
    """Tests for TransitionSpec dataclass."""

    def test_timeout_spec(self):
        """TransitionSpec can represent a timeout."""
        spec = TransitionSpec(type="timeout", value=1.0)
        assert spec.type == "timeout"
        assert spec.value == 1.0
        assert spec.children is None

    def test_composite_spec(self):
        """TransitionSpec can represent composite transitions."""
        spec = TransitionSpec(
            type="any_of",
            children=[
                TransitionSpec(type="timeout", value=1.0),
                TransitionSpec(type="temperature_above", value=100.0),
            ]
        )
        assert spec.type == "any_of"
        assert len(spec.children) == 2


class TestPhaseSpec:
    """Tests for PhaseSpec dataclass."""

    def test_basic_spec(self):
        """PhaseSpec can define a phase declaratively."""
        spec = PhaseSpec(
            name="test_phase",
            transition=TransitionSpec(type="timeout", value=1.0),
        )
        assert spec.name == "test_phase"
        assert spec.phase_class == "Phase"  # default
        assert spec.config == {}

    def test_spec_with_config(self):
        """PhaseSpec can include configuration."""
        spec = PhaseSpec(
            name="merging",
            transition=TransitionSpec(type="timeout", value=1.0),
            phase_class="MergingPhase",
            config={"separation": 0.5},
        )
        assert spec.phase_class == "MergingPhase"
        assert spec.config["separation"] == 0.5


class TestTransitionFromSpec:
    """Tests for transition_from_spec factory function."""

    def test_timeout_transition(self):
        """transition_from_spec creates timeout transitions."""
        spec = TransitionSpec(type="timeout", value=1.0)
        trans = transition_from_spec(spec)

        triggered, reason = trans.evaluate(None, t=0.5)
        assert not triggered

        triggered, reason = trans.evaluate(None, t=1.0)
        assert triggered
        assert reason == "timeout"

    def test_temperature_above_transition(self):
        """transition_from_spec creates temperature_above transitions."""
        spec = TransitionSpec(type="temperature_above", value=50.0)
        trans = transition_from_spec(spec)

        # Low temperature - should not trigger
        state = State.zeros(nr=10, nz=20)
        state = state.replace(
            p=jnp.ones((10, 20)) * 10.0,
            n=jnp.ones((10, 20)) * 1.0  # T = 10
        )
        triggered, _ = trans.evaluate(state, t=0.0)
        assert not triggered

        # High temperature - should trigger
        state = state.replace(p=jnp.ones((10, 20)) * 100.0)  # T = 100
        triggered, _ = trans.evaluate(state, t=0.0)
        assert triggered

    def test_any_of_composite(self):
        """transition_from_spec creates composite any_of transitions."""
        spec = TransitionSpec(
            type="any_of",
            children=[
                TransitionSpec(type="timeout", value=10.0),
                TransitionSpec(type="timeout", value=5.0),
            ]
        )
        trans = transition_from_spec(spec)

        triggered, _ = trans.evaluate(None, t=5.0)
        assert triggered  # Second timeout triggers

    def test_all_of_composite(self):
        """transition_from_spec creates composite all_of transitions."""
        spec = TransitionSpec(
            type="all_of",
            children=[
                TransitionSpec(type="timeout", value=3.0),
                TransitionSpec(type="timeout", value=5.0),
            ]
        )
        trans = transition_from_spec(spec)

        triggered, _ = trans.evaluate(None, t=4.0)
        assert not triggered  # First met but not second

        triggered, _ = trans.evaluate(None, t=5.0)
        assert triggered  # Both met

    def test_unknown_type_raises(self):
        """transition_from_spec raises on unknown type."""
        spec = TransitionSpec(type="unknown_type", value=1.0)
        with pytest.raises(ValueError, match="Unknown transition type"):
            transition_from_spec(spec)


class TestPhaseRegistry:
    """Tests for PHASE_REGISTRY."""

    def test_phase_registered(self):
        """Base Phase is in registry."""
        assert "Phase" in PHASE_REGISTRY
        assert PHASE_REGISTRY["Phase"] is Phase

    def test_merging_phase_registered(self):
        """MergingPhase is registered when phases module imported."""
        # Import should have happened via scenarios.__init__
        assert "MergingPhase" in PHASE_REGISTRY


class TestConfigurationResult:
    """Tests for ConfigurationResult."""

    def test_final_state_property(self):
        """ConfigurationResult.final_state returns last phase state."""
        state1 = State.zeros(nr=8, nz=16)
        state2 = State.zeros(nr=8, nz=16).replace(time=1.0)

        result = ConfigurationResult(
            name="test",
            phase_results=[
                PhaseResult(
                    name="phase1",
                    initial_state=state1,
                    final_state=state1,
                    start_time=0.0,
                    end_time=0.5,
                    termination="timeout",
                ),
                PhaseResult(
                    name="phase2",
                    initial_state=state1,
                    final_state=state2,
                    start_time=0.5,
                    end_time=1.0,
                    termination="timeout",
                ),
            ],
            total_time=1.0,
            success=True,
        )

        assert result.final_state is state2


class TestLinearConfiguration:
    """Tests for LinearConfiguration execution."""

    @pytest.fixture
    def simple_geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=8, nz=16,
            r_min=0.1, r_max=1.0,
            z_min=-1.0, z_max=1.0
        )

    def test_run_executes_phases(self, mock_physics_model, mock_solver):
        """LinearConfiguration.run() executes all phases."""

        @dataclass
        class TestConfiguration(LinearConfiguration):
            name: str = "test_config"

            def build_geometry(self) -> Geometry:
                return Geometry(
                    coord_system="cylindrical",
                    nr=8, nz=16,
                    r_min=0.1, r_max=1.0,
                    z_min=-1.0, z_max=1.0
                )

            def build_initial_state(self, geometry: Geometry) -> State:
                return State.zeros(nr=8, nz=16)

            def build_model(self) -> PhysicsModel:
                return mock_physics_model

            def build_boundary_conditions(self) -> list:
                return []

            def build_phase_specs(self) -> List[PhaseSpec]:
                return [
                    PhaseSpec(
                        name="phase1",
                        transition=TransitionSpec(type="timeout", value=0.05),
                    ),
                ]

            def build_solver(self):
                return mock_solver

        config = TestConfiguration(dt=0.01)
        result = config.run()

        assert isinstance(result, ConfigurationResult)
        assert result.name == "test_config"
        assert len(result.phase_results) == 1
        assert result.phase_results[0].name == "phase1"
        assert result.phase_results[0].termination == "timeout"
        assert result.success is True

    def test_run_chains_phases(self, mock_physics_model, mock_solver):
        """LinearConfiguration.run() chains phases and passes state."""

        @dataclass
        class TwoPhaseConfig(LinearConfiguration):
            name: str = "two_phase"

            def build_geometry(self) -> Geometry:
                return Geometry(
                    coord_system="cylindrical",
                    nr=8, nz=16,
                    r_min=0.1, r_max=1.0,
                    z_min=-1.0, z_max=1.0
                )

            def build_initial_state(self, geometry: Geometry) -> State:
                return State.zeros(nr=8, nz=16)

            def build_model(self) -> PhysicsModel:
                return mock_physics_model

            def build_boundary_conditions(self) -> list:
                return []

            def build_phase_specs(self) -> List[PhaseSpec]:
                return [
                    PhaseSpec(
                        name="phase1",
                        transition=TransitionSpec(type="timeout", value=0.05),
                    ),
                    PhaseSpec(
                        name="phase2",
                        transition=TransitionSpec(type="timeout", value=0.1),
                    ),
                ]

            def build_solver(self):
                return mock_solver

        config = TwoPhaseConfig(dt=0.01)
        result = config.run()

        assert len(result.phase_results) == 2
        assert result.phase_results[0].name == "phase1"
        assert result.phase_results[1].name == "phase2"

        # Phase 2 should start where phase 1 ended
        phase1_end = result.phase_results[0].end_time
        phase2_start = result.phase_results[1].start_time
        assert phase2_start >= phase1_end

    def test_run_with_condition_transition(self, mock_physics_model, mock_solver):
        """LinearConfiguration works with condition-based transitions."""

        @dataclass
        class ConditionConfig(LinearConfiguration):
            name: str = "condition_test"

            def build_geometry(self) -> Geometry:
                return Geometry(
                    coord_system="cylindrical",
                    nr=8, nz=16,
                    r_min=0.1, r_max=1.0,
                    z_min=-1.0, z_max=1.0
                )

            def build_initial_state(self, geometry: Geometry) -> State:
                return State.zeros(nr=8, nz=16)

            def build_model(self) -> PhysicsModel:
                return mock_physics_model

            def build_boundary_conditions(self) -> list:
                return []

            def build_phase_specs(self) -> List[PhaseSpec]:
                # Use velocity_below which should trigger immediately
                # since initial state has zero velocity
                return [
                    PhaseSpec(
                        name="moving",
                        transition=TransitionSpec(type="velocity_below", value=1.0),
                    ),
                ]

            def build_solver(self):
                return mock_solver

        config = ConditionConfig(dt=0.01)
        result = config.run()

        assert result.success is True
        assert result.phase_results[0].termination == "condition_met"

    def test_build_phases_creates_from_specs(self, mock_physics_model, mock_solver):
        """build_phases() creates Phase objects from PhaseSpecs."""

        @dataclass
        class SpecConfig(LinearConfiguration):
            name: str = "spec_test"

            def build_geometry(self) -> Geometry:
                return Geometry(
                    coord_system="cylindrical",
                    nr=8, nz=16,
                    r_min=0.1, r_max=1.0,
                    z_min=-1.0, z_max=1.0
                )

            def build_initial_state(self, geometry: Geometry) -> State:
                return State.zeros(nr=8, nz=16)

            def build_model(self) -> PhysicsModel:
                return mock_physics_model

            def build_boundary_conditions(self) -> list:
                return []

            def build_phase_specs(self) -> List[PhaseSpec]:
                return [
                    PhaseSpec(
                        name="test",
                        transition=TransitionSpec(type="timeout", value=1.0),
                        phase_class="Phase",
                    ),
                ]

        config = SpecConfig()
        geometry = config.build_geometry()
        phases = config.build_phases(geometry)

        assert len(phases) == 1
        assert isinstance(phases[0], Phase)
        assert phases[0].name == "test"

    def test_unknown_phase_class_raises(self, mock_physics_model):
        """build_phases() raises for unknown phase class."""

        @dataclass
        class BadConfig(LinearConfiguration):
            name: str = "bad_config"

            def build_geometry(self) -> Geometry:
                return Geometry(
                    coord_system="cylindrical",
                    nr=8, nz=16,
                    r_min=0.1, r_max=1.0,
                    z_min=-1.0, z_max=1.0
                )

            def build_initial_state(self, geometry: Geometry) -> State:
                return State.zeros(nr=8, nz=16)

            def build_model(self) -> PhysicsModel:
                return mock_physics_model

            def build_boundary_conditions(self) -> list:
                return []

            def build_phase_specs(self) -> List[PhaseSpec]:
                return [
                    PhaseSpec(
                        name="test",
                        transition=TransitionSpec(type="timeout", value=1.0),
                        phase_class="NonExistentPhase",
                    ),
                ]

        config = BadConfig()
        geometry = config.build_geometry()

        with pytest.raises(ValueError, match="Unknown phase class"):
            config.build_phases(geometry)

    def test_total_time_computed(self, mock_physics_model, mock_solver):
        """ConfigurationResult has correct total_time."""

        @dataclass
        class TimingConfig(LinearConfiguration):
            name: str = "timing"

            def build_geometry(self) -> Geometry:
                return Geometry(
                    coord_system="cylindrical",
                    nr=8, nz=16,
                    r_min=0.1, r_max=1.0,
                    z_min=-1.0, z_max=1.0
                )

            def build_initial_state(self, geometry: Geometry) -> State:
                return State.zeros(nr=8, nz=16)

            def build_model(self) -> PhysicsModel:
                return mock_physics_model

            def build_boundary_conditions(self) -> list:
                return []

            def build_phase_specs(self) -> List[PhaseSpec]:
                return [
                    PhaseSpec(
                        name="phase1",
                        transition=TransitionSpec(type="timeout", value=0.05),
                    ),
                    PhaseSpec(
                        name="phase2",
                        transition=TransitionSpec(type="timeout", value=0.1),
                    ),
                ]

            def build_solver(self):
                return mock_solver

        config = TimingConfig(dt=0.01)
        result = config.run()

        # total_time should be sum of phase durations
        expected_total = sum(
            r.end_time - r.start_time for r in result.phase_results
        )
        assert abs(result.total_time - expected_total) < 1e-10
