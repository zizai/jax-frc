"""Tests for progress reporting in LinearConfiguration."""

import pytest
from io import StringIO
from dataclasses import dataclass, field
from typing import List

import jax.numpy as jnp

from jax_frc.configurations.linear_configuration import (
    LinearConfiguration, PhaseSpec, TransitionSpec, ConfigurationResult
)
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from tests.utils.cartesian import make_geometry
from jax_frc.diagnostics.progress import ProgressReporter


@dataclass
class SimpleTestConfiguration(LinearConfiguration):
    """Minimal configuration for testing progress."""

    name: str = "test"
    timeout: float = 1e-5
    dt: float = 1e-6
    model_type: str = "resistive_mhd"

    def build_geometry(self):
        return make_geometry(nx=8, ny=1, nz=16, extent=1.0)

    def build_initial_state(self, geometry) -> State:
        state = State.zeros(geometry.nx, geometry.ny, geometry.nz)
        B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
        B = B.at[:, :, :, 2].set(jnp.exp(-geometry.x_grid**2))
        return state.replace(B=B)

    def build_model(self):
        return ResistiveMHD(eta=1e-6)

    def build_boundary_conditions(self):
        return []

    def build_phase_specs(self) -> List[PhaseSpec]:
        return [
            PhaseSpec(
                name="evolve",
                transition=TransitionSpec(type="timeout", value=self.timeout),
            )
        ]


class TestLinearConfigurationProgress:
    """Test progress reporting integration."""

    def test_run_with_progress_reporter(self):
        """Configuration can run with progress reporter."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(
            t_end=1e-5,
            stream=stderr_capture,
            enabled=True,
        )

        config = SimpleTestConfiguration()
        config.progress_reporter = reporter
        result = config.run()

        assert result.success
        output = stderr_capture.getvalue()
        # Should have some progress output
        assert len(output) > 0

    def test_run_without_progress_reporter(self):
        """Configuration runs fine without progress reporter."""
        config = SimpleTestConfiguration()
        result = config.run()
        assert result.success
