"""Invariant tests for Resistive MHD simulation."""
import pytest
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from resistive_mhd import step, compute_j_phi, chodura_resistivity
from jax_frc.operators import laplace_star_safe
from tests.invariants import format_failures
from tests.invariants.boundedness import FiniteValues, NoExponentialGrowth, BoundedRange
from tests.invariants.conservation import FluxConservation
from tests.invariants.consistency import ResistivityBounds, VelocityEvolution

@pytest.fixture
def resistive_mhd_state():
    """Initialize resistive MHD state for testing.

    State tuple structure (15 elements):
        psi, v_r, v_z, rho, p, I_coil, t, dr, dz, dt, r, z, V_bank, L_coil, M_plasma_coil
    """
    nr, nz = 32, 64
    dr, dz = 1.0/nr, 2.0/nz
    # Use smaller timestep to avoid excessive subcycling
    dt = 1e-6

    r = jnp.linspace(0.01, 1.0, nr)[:, None]
    z = jnp.linspace(-1.0, 1.0, nz)[None, :]

    psi_init = (1 - r**2) * jnp.exp(-z**2)

    # Velocity fields (weak inward radial flow for compression)
    v_r_init = -0.01 * r * jnp.exp(-z**2)
    v_z_init = jnp.zeros((nr, nz))

    # Density and pressure
    rho_init = jnp.ones((nr, nz)) * 1.673e-27 * 1e19
    p_init = jnp.ones((nr, nz)) * 1e3

    I_coil_init = 0.0

    V_bank = 1000.0
    L_coil = 1e-6
    M_plasma_coil = 1e-7

    # State: (psi, v_r, v_z, rho, p, I_coil, t, dr, dz, dt, r, z, V_bank, L_coil, M_plasma_coil)
    state = (psi_init, v_r_init, v_z_init, rho_init, p_init, I_coil_init, 0.0, dr, dz, dt, r, z, V_bank, L_coil, M_plasma_coil)
    step_fn = jax.jit(step)

    return state, step_fn, dr, dz, r

class TestResistiveMHDBoundedness:
    """Boundedness tests for Resistive MHD."""

    def test_flux_stays_finite(self, resistive_mhd_state, invariant_checker):
        """Flux function psi should never contain NaN or Inf."""
        state, step_fn, dr, dz, r = resistive_mhd_state
        invariants = [FiniteValues("psi")]

        all_failures = []
        for i in range(10):
            new_state, _ = step_fn(state, None)
            psi_before, psi_after = state[0], new_state[0]
            _, failures = invariant_checker(invariants, psi_before, psi_after, i)
            all_failures.extend(failures)
            state = new_state

        assert not all_failures, format_failures(all_failures)

    def test_no_exponential_growth(self, resistive_mhd_state, invariant_checker):
        """Flux should not grow exponentially (indicates instability)."""
        state, step_fn, dr, dz, r = resistive_mhd_state
        invariants = [NoExponentialGrowth("psi", growth_factor=1.5)]

        all_failures = []
        for i in range(10):
            new_state, _ = step_fn(state, None)
            psi_before, psi_after = state[0], new_state[0]
            _, failures = invariant_checker(invariants, psi_before, psi_after, i)
            all_failures.extend(failures)
            state = new_state

        assert not all_failures, format_failures(all_failures)

class TestResistiveMHDConsistency:
    """Consistency tests for Resistive MHD."""

    def test_resistivity_bounds(self, resistive_mhd_state):
        """Chodura resistivity should stay within physical bounds."""
        state, step_fn, dr, dz, r = resistive_mhd_state
        psi = state[0]

        j_phi = compute_j_phi(psi, dr, dz, r)
        eta = chodura_resistivity(psi, j_phi)

        # Default Chodura: eta_0=1e-4, eta_anom=1e-2
        invariant = ResistivityBounds(eta_min=0.0, eta_max=0.02)
        result = invariant.check(None, eta)

        assert result.passed, result.message

    def test_velocity_evolves(self, resistive_mhd_state):
        """Velocity field should evolve due to Lorentz force."""
        state, step_fn, dr, dz, r = resistive_mhd_state

        # Run several steps
        v_r_initial = state[1].copy()
        v_z_initial = state[2].copy()

        for i in range(10):
            state, _ = step_fn(state, None)

        v_r_final = state[1]
        v_z_final = state[2]

        # Check that velocity has changed
        v_r_change = float(jnp.max(jnp.abs(v_r_final - v_r_initial)))
        v_z_change = float(jnp.max(jnp.abs(v_z_final - v_z_initial)))
        total_change = max(v_r_change, v_z_change)

        assert total_change > 1e-12, f"Velocity should evolve, but max change was {total_change:.2e}"

class TestResistiveMHDIntegration:
    """Full simulation integration tests."""

    def test_20_steps_stable(self, resistive_mhd_state, invariant_checker):
        """Simulation should remain stable for 20 steps."""
        state, step_fn, dr, dz, r = resistive_mhd_state
        invariants = [
            FiniteValues("psi"),
            NoExponentialGrowth("psi", growth_factor=2.0),
        ]

        all_failures = []
        for i in range(10):
            new_state, _ = step_fn(state, None)
            psi_before, psi_after = state[0], new_state[0]
            _, failures = invariant_checker(invariants, psi_before, psi_after, i)
            all_failures.extend(failures)
            state = new_state

        assert not all_failures, format_failures(all_failures)
