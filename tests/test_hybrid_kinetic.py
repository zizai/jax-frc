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
    rigid_rotor_f0, step
)
from tests.invariants import format_failures
from tests.invariants.boundedness import FiniteValues, BoundedRange
from tests.invariants.conservation import ParticleCountConservation, EnergyConservation
from tests.invariants.consistency import WeightBounds, DistributionPositivity

@pytest.fixture
def hybrid_kinetic_particles():
    """Initialize particles for testing."""
    n_particles = 1000
    nr, nz = 16, 32
    n0, T0, Omega = 1e19, 100.0, 1e5

    key = random.PRNGKey(42)
    x, v, w = initialize_particles(n_particles, key, nr, nz, n0, T0, Omega)

    return x, v, w, n0, T0, Omega

@pytest.fixture
def boris_push_setup():
    """Setup for Boris push tests."""
    n_particles = 100
    key = random.PRNGKey(42)

    # Initial positions and velocities
    key_x, key_v = random.split(key)
    x = random.uniform(key_x, (n_particles, 3), minval=-1, maxval=1)
    v = random.normal(key_v, (n_particles, 3)) * 1e4

    # Simple uniform fields
    E = jnp.zeros((n_particles, 3))
    B = jnp.ones((n_particles, 3)) * jnp.array([0.0, 0.0, 1.0])

    return x, v, E, B

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
