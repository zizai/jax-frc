"""Tests for progress reporting in LinearConfiguration."""

import pytest
from io import StringIO
from dataclasses import dataclass, field
from typing import List

import jax.numpy as jnp

from jax_frc.configurations.linear_configuration import (
    LinearConfiguration, PhaseSpec, TransitionSpec, ConfigurationResult
)
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.diagnostics.progress import ProgressReporter


@dataclass
class SimpleTestConfiguration(LinearConfiguration):
    """Minimal configuration for testing progress."""

    name: str = "test"
    timeout: float = 1e-5
    dt: float = 1e-6
    model_type: str = "resistive_mhd"

    def build_geometry(self) -> Geometry:
        return Geometry(
            coord_system="cylindrical",
            nr=8, nz=16,
            r_min=0.1, r_max=1.0,
            z_min=-1.0, z_max=1.0,
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        state = State.zeros(geometry.nr, geometry.nz)
        r = geometry.r_grid
        return state.replace(psi=jnp.exp(-r**2))

    def build_model(self):
        return ResistiveMHD(resistivity=SpitzerResistivity())

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
