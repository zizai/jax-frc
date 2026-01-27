"""Tests for State dataclass with 3D Cartesian coordinates."""
import pytest
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.core.state import State, ParticleState


class TestStateTemperatureField:
    """Tests for Temperature field (Te) in 3D State dataclass."""

    def test_state_has_temperature_attribute(self):
        """State dataclass should have a Te attribute for electron temperature."""
        nx, ny, nz = 8, 8, 16
        state = State.zeros(nx, ny, nz)

        # Te attribute should exist
        assert hasattr(state, 'Te'), "State should have Te attribute"

    def test_state_zeros_initializes_core_fields(self):
        """State.zeros() should initialize core fields to zeros."""
        nx, ny, nz = 8, 8, 16
        state = State.zeros(nx, ny, nz)

        # Core fields should be zeros
        assert jnp.allclose(state.B, jnp.zeros((nx, ny, nz, 3))), "B should be zeros"
        assert jnp.allclose(state.E, jnp.zeros((nx, ny, nz, 3))), "E should be zeros"
        assert jnp.allclose(state.n, jnp.zeros((nx, ny, nz))), "n should be zeros"
        assert jnp.allclose(state.p, jnp.zeros((nx, ny, nz))), "p should be zeros"

    def test_state_replace_with_temperature(self):
        """State.replace() should work with Te field."""
        nx, ny, nz = 8, 8, 8
        state = State.zeros(nx, ny, nz)

        # Create a new temperature field
        new_Te = jnp.ones((nx, ny, nz)) * 100.0  # 100 eV

        # Replace Te
        new_state = state.replace(Te=new_Te)

        # Verify replacement worked
        assert jnp.allclose(new_state.Te, new_Te), "replace() should update Te field"
        # Original state Te should still be None (immutable)
        assert state.Te is None, "Original state Te should be unchanged (None)"

    def test_state_is_jax_pytree(self):
        """State should be registered as a JAX pytree for JIT compatibility."""
        nx, ny, nz = 8, 8, 8
        state = State.zeros(nx, ny, nz)

        # Set a non-zero temperature
        state = state.replace(Te=jnp.ones((nx, ny, nz)) * 50.0)

        # Flatten and unflatten via pytree
        leaves, treedef = jax.tree_util.tree_flatten(state)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Verify Te is preserved through pytree operations
        assert jnp.allclose(reconstructed.Te, state.Te), "Te should survive pytree flatten/unflatten"

    def test_state_jit_compatible_with_temperature(self):
        """JIT-compiled functions should work with State containing Te."""
        nx, ny, nz = 8, 8, 8
        state = State.zeros(nx, ny, nz)
        state = state.replace(Te=jnp.ones((nx, ny, nz)) * 25.0)

        @jax.jit
        def update_temperature(s):
            new_Te = s.Te * 2.0  # Double the temperature
            return s.replace(Te=new_Te)

        new_state = update_temperature(state)

        assert jnp.allclose(new_state.Te, jnp.ones((nx, ny, nz)) * 50.0), "JIT should work with Te field"

    def test_state_explicit_temperature_initialization(self):
        """State should accept explicit Te value in constructor."""
        nx, ny, nz = 8, 8, 8
        Te_init = jnp.ones((nx, ny, nz)) * 200.0  # 200 eV

        state = State(
            B=jnp.zeros((nx, ny, nz, 3)),
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.zeros((nx, ny, nz)),
            p=jnp.zeros((nx, ny, nz)),
            Te=Te_init,
        )

        assert jnp.allclose(state.Te, Te_init), "Explicit Te initialization should work"


class TestState3DFields:
    """Tests to ensure all 3D State fields work correctly."""

    def test_state_zeros_all_fields(self):
        """State.zeros() should initialize all required fields correctly."""
        nx, ny, nz = 8, 8, 16
        state = State.zeros(nx, ny, nz)

        # Check all 3D fields
        assert state.B.shape == (nx, ny, nz, 3)
        assert state.E.shape == (nx, ny, nz, 3)
        assert state.n.shape == (nx, ny, nz)
        assert state.p.shape == (nx, ny, nz)
        # Optional fields are None by default from zeros()
        assert state.v is None
        assert state.Te is None
        assert state.Ti is None
        assert state.particles is None

    def test_state_with_velocity(self):
        """State should accept velocity field."""
        nx, ny, nz = 8, 8, 8

        state = State.zeros(nx, ny, nz)
        v = jnp.ones((nx, ny, nz, 3)) * 1000.0  # 1 km/s
        state = state.replace(v=v)

        assert state.v is not None
        assert state.v.shape == (nx, ny, nz, 3)
        assert jnp.allclose(state.v, v)

    def test_state_with_particles(self):
        """State should accept ParticleState."""
        nx, ny, nz = 8, 8, 8
        n_particles = 1000

        particles = ParticleState(
            x=jnp.zeros((n_particles, 3)),
            v=jnp.zeros((n_particles, 3)),
            w=jnp.ones(n_particles),
            species="ions"
        )

        state = State.zeros(nx, ny, nz)
        state = state.replace(particles=particles)

        assert state.particles is not None
        assert state.particles.x.shape == (n_particles, 3)
        assert state.particles.v.shape == (n_particles, 3)
        assert state.particles.w.shape == (n_particles,)
        assert state.particles.species == "ions"

    def test_state_replace_existing_fields(self):
        """State.replace() should work for core fields."""
        nx, ny, nz = 8, 8, 8
        state = State.zeros(nx, ny, nz)

        new_n = jnp.ones((nx, ny, nz)) * 1e19
        new_state = state.replace(n=new_n)

        assert jnp.allclose(new_state.n, new_n)

    def test_state_pytree_all_fields(self):
        """Pytree registration should include all fields including optional ones."""
        nx, ny, nz = 8, 8, 8
        state = State.zeros(nx, ny, nz)
        state = state.replace(
            n=jnp.ones((nx, ny, nz)) * 2.0,
            Te=jnp.ones((nx, ny, nz)) * 100.0,
            v=jnp.ones((nx, ny, nz, 3)) * 500.0,
        )

        leaves, treedef = jax.tree_util.tree_flatten(state)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        assert jnp.allclose(reconstructed.n, state.n)
        assert jnp.allclose(reconstructed.Te, state.Te)
        assert jnp.allclose(reconstructed.v, state.v)


class TestParticleState:
    """Tests for ParticleState dataclass."""

    def test_particle_state_creation(self):
        """ParticleState should be created with correct fields."""
        n_particles = 100
        particles = ParticleState(
            x=jnp.zeros((n_particles, 3)),
            v=jnp.zeros((n_particles, 3)),
            w=jnp.ones(n_particles),
            species="electrons"
        )

        assert particles.x.shape == (n_particles, 3)
        assert particles.v.shape == (n_particles, 3)
        assert particles.w.shape == (n_particles,)
        assert particles.species == "electrons"

    def test_particle_state_is_pytree(self):
        """ParticleState should be registered as JAX pytree."""
        n_particles = 100
        particles = ParticleState(
            x=jnp.ones((n_particles, 3)),
            v=jnp.ones((n_particles, 3)) * 2.0,
            w=jnp.ones(n_particles) * 0.5,
            species="ions"
        )

        leaves, treedef = jax.tree_util.tree_flatten(particles)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        assert jnp.allclose(reconstructed.x, particles.x)
        assert jnp.allclose(reconstructed.v, particles.v)
        assert jnp.allclose(reconstructed.w, particles.w)
        assert reconstructed.species == particles.species
