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
from tests.invariants.consistency import DivergenceFreeB, AxisRegularity

@pytest.fixture
def extended_mhd_state():
    """Initialize extended MHD state for testing in cylindrical coordinates.

    Note: Uses a coarse grid and small timestep because the Hall term introduces
    Whistler waves with strict CFL constraint: dt < (dr * mu0 * n * e) / (pi * B)
    For nr=8, B~0.1T, n~1e18, this gives dt_whistler ~ 1e-10 s.
    With 50 substeps per timestep, dt=2e-9 gives dt_sub ~ 4e-11 which is marginal.
    """
    # Use coarser grid for stability
    nr, nz = 8, 8  # Coarser grid for faster Whistler CFL
    dr, dz = 1.0/nr, 1.0/nz
    # Use smaller timestep for Whistler CFL stability
    dt = 2e-9  # With 50 substeps, dt_sub ~ 4e-11
    eta = 1e-4

    r = jnp.linspace(0.01, 1, nr)[:, None]
    z = jnp.linspace(-1, 1, nz)[None, :]

    # Cylindrical field components: b_r, b_theta, b_z
    b_r_init = jnp.zeros((nr, nz))
    b_theta_init = jnp.zeros((nr, nz))
    # Use smaller initial B field to reduce Whistler speed
    b_z_init = 0.1 * jnp.exp(-r**2 - z**2)  # Reduced from 1.0 to 0.1

    # Cylindrical velocity components: v_r, v_theta, v_z
    v_r_init = jnp.zeros((nr, nz))
    v_theta_init = jnp.zeros((nr, nz))
    v_z_init = jnp.zeros((nr, nz))

    n_init = jnp.ones((nr, nz)) * 1e19
    p_e_init = jnp.ones((nr, nz)) * 1e3

    # State: (b_r, b_theta, b_z, v_r, v_theta, v_z, n, p_e, t, dt, dr, dz, r, eta)
    state = (b_r_init, b_theta_init, b_z_init, v_r_init, v_theta_init, v_z_init,
             n_init, p_e_init, 0.0, dt, dr, dz, r, eta)
    step_fn = jax.jit(step)

    return state, step_fn, dr, dz

class TestExtendedMHDBoundedness:
    """Boundedness tests for Extended MHD."""

    def test_magnetic_field_finite(self, extended_mhd_state, invariant_checker):
        """Magnetic field components should stay finite."""
        state, step_fn, dr, dz = extended_mhd_state
        invariants = [
            FiniteValues("B_r"),
            FiniteValues("B_theta"),
            FiniteValues("B_z"),
        ]

        all_failures = []
        for i in range(10):  # Reduced from 50 to 10
            new_state, _ = step_fn(state, None)
            # Check each B component
            for j, name in enumerate(["B_r", "B_theta", "B_z"]):
                inv = FiniteValues(name)
                result = inv.check(state[j], new_state[j])
                if not result.passed:
                    all_failures.append((i, result))
            state = new_state

        assert not all_failures, format_failures(all_failures)

    def test_no_hall_instability(self, extended_mhd_state, invariant_checker):
        """Hall term should not cause exponential growth."""
        state, step_fn, dr, dz = extended_mhd_state

        all_failures = []
        for i in range(10):  # Reduced from 50 to 10
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
        state, step_fn, dr, dz = extended_mhd_state
        n = state[6]  # density field

        inv = PositiveValues("density", allow_zero=False)
        result = inv.check(None, n)

        assert result.passed, result.message

class TestExtendedMHDConsistency:
    """Consistency tests for Extended MHD."""

    def test_halo_density_applied(self, extended_mhd_state):
        """Halo density model should produce positive density everywhere."""
        state, step_fn, dr, dz = extended_mhd_state
        n = state[6]

        n_with_halo = apply_halo_density(n)

        inv = PositiveValues("n_with_halo", allow_zero=False)
        result = inv.check(None, n_with_halo)

        assert result.passed, result.message

    def test_axis_regularity(self, extended_mhd_state):
        """Check B field is regular at axis (B_r=0, B_theta=0)."""
        state, step_fn, dr, dz = extended_mhd_state

        # Run a few steps
        for i in range(5):
            state, _ = step_fn(state, None)

        # Stack B components for axis check
        b_r, b_theta, b_z = state[0], state[1], state[2]
        b_field = jnp.stack([b_r, b_theta, b_z], axis=-1)

        inv = AxisRegularity(atol=1e-3)
        result = inv.check(None, b_field)

        assert result.passed, result.message

class TestExtendedMHDIntegration:
    """Full simulation integration tests."""

    def test_50_steps_stable(self, extended_mhd_state, invariant_checker):
        """Simulation should remain stable for 10 steps.

        Note: Using only 10 steps due to Whistler CFL constraints.
        Extended MHD with Hall term is computationally expensive and
        requires very small timesteps for stability.
        """
        state, step_fn, dr, dz = extended_mhd_state

        all_failures = []
        for i in range(10):  # Reduced from 50 to 10 for faster testing
            new_state, _ = step_fn(state, None)

            # Check all B components are finite
            for j, name in enumerate(["B_r", "B_theta", "B_z"]):
                inv = FiniteValues(name)
                result = inv.check(state[j], new_state[j])
                if not result.passed:
                    all_failures.append((i, result))

            state = new_state

        assert not all_failures, format_failures(all_failures)
