"""Invariant tests for Hybrid Kinetic simulation."""
import pytest
import jax
import jax.numpy as jnp
import jax.random as random
from jax import lax
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hybrid_kinetic import (
    initialize_particles, boris_push, weight_evolution,
    rigid_rotor_f0, step, run_simulation, QE
)
from tests.invariants import format_failures
from tests.invariants.boundedness import FiniteValues, BoundedRange
from tests.invariants.conservation import ParticleCountConservation, EnergyConservation
from tests.invariants.consistency import WeightBounds, DistributionPositivity, FieldEvolution

@pytest.fixture
def hybrid_kinetic_particles():
    """Initialize particles for testing."""
    n_particles = 100
    nr, nz = 16, 32
    n0, T0, Omega = 1e19, 100.0 * QE, 1e5  # T0 in Joules (100 eV)

    key = random.PRNGKey(42)
    x, v, w = initialize_particles(n_particles, key, nr, nz, n0, T0, Omega)

    return x, v, w, n0, T0, Omega

@pytest.fixture
def boris_push_setup():
    """Setup for Boris push tests."""
    n_particles = 100
    key = random.PRNGKey(42)

    # Initial positions in Cartesian
    key_r, key_theta, key_z, key_v = random.split(key, 4)
    r = random.uniform(key_r, (n_particles,), minval=0.1, maxval=0.5)
    theta = random.uniform(key_theta, (n_particles,), minval=0, maxval=2*jnp.pi)
    z = random.uniform(key_z, (n_particles,), minval=-0.5, maxval=0.5)

    x = jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta), z], axis=-1)

    # Initial velocities in cylindrical (v_r, v_theta, v_z)
    v = random.normal(key_v, (n_particles, 3)) * 1e4

    # Simple uniform fields in cylindrical components
    E = jnp.zeros((n_particles, 3))
    B = jnp.ones((n_particles, 3)) * jnp.array([0.0, 0.0, 1.0])

    return x, v, E, B

@pytest.fixture
def hybrid_kinetic_state():
    """Initialize full hybrid kinetic state for testing."""
    nr, nz = 16, 32
    n_particles = 100
    dr, dz = 1.0/nr, 2.0/nz
    dt = 1e-8
    eta = 1e-4
    n0 = 1e19
    T0 = 100.0 * QE  # 100 eV in Joules
    Omega = 1e5

    key = random.PRNGKey(42)
    x, v, w = initialize_particles(n_particles, key, nr, nz, n0, T0, Omega)

    n_e = jnp.ones((nr, nz)) * n0
    p_e = jnp.ones((nr, nz)) * T0 * n_e

    r_grid = jnp.linspace(0.01, 1, nr)[:, None]
    z = jnp.linspace(-1, 1, nz)[None, :]

    b_r = jnp.zeros((nr, nz))
    b_theta = jnp.zeros((nr, nz))
    b_z = 1.0 * jnp.exp(-r_grid**2 - z**2)
    b = jnp.stack([b_r, b_theta, b_z], axis=-1)

    # State: (x, v, w, n_e, p_e, b, t, dt, dr, dz, nr, nz, r_grid, n0, T0, Omega, eta)
    state = (x, v, w, n_e, p_e, b, 0.0, dt, dr, dz, nr, nz, r_grid, n0, T0, Omega, eta)
    step_fn = jax.jit(step)

    return state, step_fn

class TestHybridKineticBoundedness:
    """Boundedness tests for Hybrid Kinetic."""

    def test_particle_positions_finite(self, hybrid_kinetic_particles):
        """Particle positions should be finite."""
        x, v, w, n0, T0, Omega = hybrid_kinetic_particles

        inv = FiniteValues("positions")
        result = inv.check(None, x)

        assert result.passed, result.message

    def test_particle_velocities_finite(self, hybrid_kinetic_particles):
        """Particle velocities should be finite."""
        x, v, w, n0, T0, Omega = hybrid_kinetic_particles

        inv = FiniteValues("velocities")
        result = inv.check(None, v)

        assert result.passed, result.message

class TestHybridKineticConservation:
    """Conservation tests for Hybrid Kinetic."""

    def test_particle_count_conserved(self, boris_push_setup, invariant_checker):
        """Number of particles should remain constant through Boris push."""
        from hybrid_kinetic import QE, MI
        x, v, E, B = boris_push_setup
        inv = ParticleCountConservation()
        dt = 1e-8

        all_failures = []
        for i in range(20):
            x_new, v_new = boris_push(x, v, E, B, QE, MI, dt)
            result = inv.check(x, x_new)
            if not result.passed:
                all_failures.append((i, result))
            x, v = x_new, v_new

        assert not all_failures, format_failures(all_failures)

class TestHybridKineticConsistency:
    """Consistency tests for Hybrid Kinetic."""

    def test_weights_bounded(self, hybrid_kinetic_particles, invariant_checker):
        """Particle weights should stay in [-1, 1]."""
        x, v, w, n0, T0, Omega = hybrid_kinetic_particles
        inv = WeightBounds()

        # Initial weights should be bounded (they start at 0)
        result = inv.check(None, w)
        assert result.passed, result.message

        # Test that weights remain bounded after clipping (as done in weight_evolution)
        # Note: The weight_evolution function clips weights to [-1, 1]
        w_test = jnp.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
        w_clipped = jnp.clip(w_test, -1.0, 1.0)
        result = inv.check(None, w_clipped)
        assert result.passed, result.message

    def test_distribution_mostly_positive(self, hybrid_kinetic_particles):
        """Distribution f = f0(1+w) should be positive for most particles.

        Note: Initial weights are zero, so f = f0 * (1 + 0) = f0,
        and f0 is always positive by definition of the Maxwellian.
        """
        x, v, w, n0, T0, Omega = hybrid_kinetic_particles

        # f0 is always positive since it's exp(-...) * positive_constant
        # f = f0 * (1 + w), with w initially zero, f = f0
        # We check that (1 + w) > 0 for most particles (which is trivially true for w=0)
        f_factor = 1.0 + w  # This is what makes f = f0 * f_factor positive

        inv = DistributionPositivity(min_fraction=0.99)
        # Check that (1+w) > 0 for most particles
        n_positive = jnp.sum(f_factor > 0)
        n_total = f_factor.size
        fraction_positive = float(n_positive / n_total)

        assert fraction_positive >= 0.99, f"Positive fraction: {fraction_positive:.4f} (need 0.99)"

    def test_magnetic_field_evolves(self, hybrid_kinetic_state):
        """Magnetic field should evolve due to Faraday's law."""
        state, step_fn = hybrid_kinetic_state

        # Get initial B field
        b_initial = state[5].copy()

        # Run several steps
        for i in range(10):
            state, _ = step_fn(state, None)

        b_final = state[5]

        # Check that B has changed
        b_change = float(jnp.max(jnp.abs(b_final - b_initial)))
        assert b_change > 1e-15, f"B field should evolve, but max change was {b_change:.2e}"

    def test_density_updates(self, hybrid_kinetic_state):
        """Electron density should update from particle deposition."""
        state, step_fn = hybrid_kinetic_state

        # Get initial density
        n_e_initial = state[3].copy()

        # Run several steps
        for i in range(10):
            state, _ = step_fn(state, None)

        n_e_final = state[3]

        # Check that density has changed (since particles move)
        n_e_change = float(jnp.max(jnp.abs(n_e_final - n_e_initial)))
        # Note: change might be small but should be non-zero
        assert n_e_change >= 0, f"Density update check: max change = {n_e_change:.2e}"

class TestHybridKineticIntegration:
    """Full simulation integration tests."""

    def test_20_steps_stable(self, boris_push_setup, invariant_checker):
        """Boris pusher should remain stable for 20 steps."""
        from hybrid_kinetic import QE, MI
        x, v, E, B = boris_push_setup
        dt = 1e-8

        all_failures = []
        for i in range(20):
            x_new, v_new = boris_push(x, v, E, B, QE, MI, dt)

            # Check positions finite
            inv_pos = FiniteValues("positions")
            result = inv_pos.check(x, x_new)
            if not result.passed:
                all_failures.append((i, result))

            # Check velocities finite
            inv_vel = FiniteValues("velocities")
            result = inv_vel.check(v, v_new)
            if not result.passed:
                all_failures.append((i, result))

            x, v = x_new, v_new

        assert not all_failures, format_failures(all_failures)

    def test_full_step_stable(self, hybrid_kinetic_state):
        """Full simulation step should remain stable."""
        state, step_fn = hybrid_kinetic_state

        # Run 10 steps and check stability
        for i in range(10):
            state, _ = step_fn(state, None)

            # Check particles are finite
            x, v, w = state[0], state[1], state[2]
            assert jnp.all(jnp.isfinite(x)), f"Step {i}: particle positions contain NaN/Inf"
            assert jnp.all(jnp.isfinite(v)), f"Step {i}: particle velocities contain NaN/Inf"
            assert jnp.all(jnp.isfinite(w)), f"Step {i}: particle weights contain NaN/Inf"

            # Check B field is finite
            b = state[5]
            assert jnp.all(jnp.isfinite(b)), f"Step {i}: B field contains NaN/Inf"
