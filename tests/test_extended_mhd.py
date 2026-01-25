"""Invariant tests for Extended MHD simulation."""
import pytest
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from extended_mhd import step, apply_halo_density
from tests.invariants import format_failures
from tests.invariants.boundedness import FiniteValues, NoExponentialGrowth, PositiveValues
from tests.invariants.consistency import DivergenceFreeB

@pytest.fixture
def extended_mhd_state():
    """Initialize extended MHD state for testing."""
    nx, ny = 32, 32
    dx, dy = 1.0/nx, 1.0/ny
    dt = 1e-6
    eta = 1e-4

    r = jnp.linspace(0, 1, nx)[:, None]
    z = jnp.linspace(-1, 1, ny)[None, :]

    b_x_init = jnp.zeros((nx, ny))
    b_y_init = jnp.zeros((nx, ny))
    b_z_init = 1.0 * jnp.exp(-r**2 - z**2)

    v_x_init = jnp.zeros((nx, ny))
    v_y_init = jnp.zeros((nx, ny))
    v_z_init = jnp.zeros((nx, ny))

    n_init = jnp.ones((nx, ny)) * 1e19
    p_e_init = jnp.ones((nx, ny)) * 1e3

    state = (b_x_init, b_y_init, b_z_init, v_x_init, v_y_init, v_z_init,
             n_init, p_e_init, 0.0, dt, dx, dy, eta)
    step_fn = jax.jit(step)

    return state, step_fn, dx, dy

class TestExtendedMHDBoundedness:
    """Boundedness tests for Extended MHD."""

    def test_magnetic_field_finite(self, extended_mhd_state, invariant_checker):
        """Magnetic field components should stay finite."""
        state, step_fn, dx, dy = extended_mhd_state
        invariants = [
            FiniteValues("B_x"),
            FiniteValues("B_y"),
            FiniteValues("B_z"),
        ]

        all_failures = []
        for i in range(50):
            new_state, _ = step_fn(state, None)
            # Check each B component
            for j, name in enumerate(["B_x", "B_y", "B_z"]):
                inv = FiniteValues(name)
                result = inv.check(state[j], new_state[j])
                if not result.passed:
                    all_failures.append((i, result))
            state = new_state

        assert not all_failures, format_failures(all_failures)

    def test_no_hall_instability(self, extended_mhd_state, invariant_checker):
        """Hall term should not cause exponential growth."""
        state, step_fn, dx, dy = extended_mhd_state

        all_failures = []
        for i in range(50):
            new_state, _ = step_fn(state, None)
            # Check B_z growth (most susceptible to Hall instability)
            inv = NoExponentialGrowth("B_z", growth_factor=1.5)
            result = inv.check(state[2], new_state[2])
            if not result.passed:
                all_failures.append((i, result))
            state = new_state

        assert not all_failures, format_failures(all_failures)

    def test_density_positive(self, extended_mhd_state):
        """Density should remain positive."""
        state, step_fn, dx, dy = extended_mhd_state
        n = state[6]  # density field

        inv = PositiveValues("density", allow_zero=False)
        result = inv.check(None, n)

        assert result.passed, result.message

class TestExtendedMHDConsistency:
    """Consistency tests for Extended MHD."""

    def test_halo_density_applied(self, extended_mhd_state):
        """Halo density model should produce positive density everywhere."""
        state, step_fn, dx, dy = extended_mhd_state
        n = state[6]

        n_with_halo = apply_halo_density(n)

        inv = PositiveValues("n_with_halo", allow_zero=False)
        result = inv.check(None, n_with_halo)

        assert result.passed, result.message

class TestExtendedMHDIntegration:
    """Full simulation integration tests."""

    def test_50_steps_stable(self, extended_mhd_state, invariant_checker):
        """Simulation should remain stable for 50 steps."""
        state, step_fn, dx, dy = extended_mhd_state

        all_failures = []
        for i in range(50):
            new_state, _ = step_fn(state, None)

            # Check all B components are finite
            for j, name in enumerate(["B_x", "B_y", "B_z"]):
                inv = FiniteValues(name)
                result = inv.check(state[j], new_state[j])
                if not result.passed:
                    all_failures.append((i, result))

            state = new_state

        assert not all_failures, format_failures(all_failures)
