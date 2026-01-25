"""Pytest fixtures for invariant testing."""
import pytest
import jax
import jax.numpy as jnp
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.invariants import Invariant, InvariantResult

@pytest.fixture
def invariant_checker():
    """Returns a function that checks all invariants and collects failures."""
    def check_all(
        invariants: list[Invariant],
        state_before,
        state_after,
        step: int
    ) -> tuple[list[InvariantResult], list[tuple[int, InvariantResult]]]:
        results = [inv.check(state_before, state_after) for inv in invariants]
        failures = [(step, r) for r in results if not r.passed]
        return results, failures
    return check_all


@pytest.fixture
def mock_physics_model():
    """Mock physics model that returns unchanged state."""
    model = MagicMock()
    model.compute_rhs.return_value = None
    model.apply_constraints.side_effect = lambda s, g: s
    return model


@pytest.fixture
def mock_solver():
    """Mock solver that just increments time."""
    solver = MagicMock()
    def fake_step(state, dt, model, geometry):
        return state.replace(time=state.time + dt, step=state.step + 1)
    solver.step.side_effect = fake_step
    return solver
