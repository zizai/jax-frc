"""Tests for State dataclass with temperature field."""
import pytest
import jax
import jax.numpy as jnp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.core.state import State, ParticleState


class TestStateTemperatureField:
    """Tests for Temperature field in State dataclass."""

    def test_state_has_temperature_field(self):
        """State dataclass should have a T field for temperature."""
        nr, nz = 16, 32
        state = State.zeros(nr, nz)

        # T field should exist
        assert hasattr(state, 'T'), "State should have T field"

        # T should have correct shape (nr, nz)
        assert state.T.shape == (nr, nz), f"T shape should be ({nr}, {nz}), got {state.T.shape}"

    def test_state_zeros_initializes_temperature_to_zeros(self):
        """State.zeros() should initialize T to all zeros."""
        nr, nz = 16, 32
        state = State.zeros(nr, nz)

        # T should be all zeros
        assert jnp.allclose(state.T, jnp.zeros((nr, nz))), "T should be initialized to zeros"

    def test_state_replace_with_temperature(self):
        """State.replace() should work with T field."""
        nr, nz = 8, 8
        state = State.zeros(nr, nz)

        # Create a new temperature field
        new_T = jnp.ones((nr, nz)) * 100.0  # 100 eV

        # Replace T
        new_state = state.replace(T=new_T)

        # Verify replacement worked
        assert jnp.allclose(new_state.T, new_T), "replace() should update T field"
        # Original state should be unchanged (immutable)
        assert jnp.allclose(state.T, jnp.zeros((nr, nz))), "Original state T should be unchanged"

    def test_state_is_jax_pytree(self):
        """State should be registered as a JAX pytree for JIT compatibility."""
        nr, nz = 8, 8
        state = State.zeros(nr, nz)

        # Set a non-zero temperature
        state = state.replace(T=jnp.ones((nr, nz)) * 50.0)

        # Flatten and unflatten via pytree
        leaves, treedef = jax.tree_util.tree_flatten(state)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        # Verify T is preserved through pytree operations
        assert jnp.allclose(reconstructed.T, state.T), "T should survive pytree flatten/unflatten"

    def test_state_jit_compatible_with_temperature(self):
        """JIT-compiled functions should work with State containing T."""
        nr, nz = 8, 8
        state = State.zeros(nr, nz)
        state = state.replace(T=jnp.ones((nr, nz)) * 25.0)

        @jax.jit
        def update_temperature(s):
            new_T = s.T * 2.0  # Double the temperature
            return s.replace(T=new_T)

        new_state = update_temperature(state)

        assert jnp.allclose(new_state.T, jnp.ones((nr, nz)) * 50.0), "JIT should work with T field"

    def test_state_explicit_temperature_initialization(self):
        """State should accept explicit T value in constructor."""
        nr, nz = 8, 8
        T_init = jnp.ones((nr, nz)) * 200.0  # 200 eV

        state = State(
            psi=jnp.zeros((nr, nz)),
            n=jnp.zeros((nr, nz)),
            p=jnp.zeros((nr, nz)),
            T=T_init,
            B=jnp.zeros((nr, nz, 3)),
            E=jnp.zeros((nr, nz, 3)),
            v=jnp.zeros((nr, nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

        assert jnp.allclose(state.T, T_init), "Explicit T initialization should work"


class TestStateBackwardCompatibility:
    """Tests to ensure existing functionality still works after adding T."""

    def test_state_zeros_all_fields(self):
        """State.zeros() should initialize all fields correctly."""
        nr, nz = 16, 32
        state = State.zeros(nr, nz)

        # Check existing fields still work
        assert state.psi.shape == (nr, nz)
        assert state.n.shape == (nr, nz)
        assert state.p.shape == (nr, nz)
        assert state.B.shape == (nr, nz, 3)
        assert state.E.shape == (nr, nz, 3)
        assert state.v.shape == (nr, nz, 3)
        assert state.particles is None
        assert state.time == 0.0
        assert state.step == 0

    def test_state_zeros_with_particles(self):
        """State.zeros() with particles should still work."""
        nr, nz = 8, 8
        n_particles = 1000

        state = State.zeros(nr, nz, with_particles=True, n_particles=n_particles)

        assert state.particles is not None
        assert state.particles.n_particles == n_particles
        assert state.particles.x.shape == (n_particles, 3)
        assert state.particles.v.shape == (n_particles, 3)
        assert state.particles.w.shape == (n_particles,)

    def test_state_replace_existing_fields(self):
        """State.replace() should still work for existing fields."""
        nr, nz = 8, 8
        state = State.zeros(nr, nz)

        new_psi = jnp.ones((nr, nz)) * 0.5
        new_state = state.replace(psi=new_psi, time=1.0, step=10)

        assert jnp.allclose(new_state.psi, new_psi)
        assert new_state.time == 1.0
        assert new_state.step == 10

    def test_state_pytree_all_fields(self):
        """Pytree registration should include all fields including T."""
        nr, nz = 8, 8
        state = State.zeros(nr, nz)
        state = state.replace(
            psi=jnp.ones((nr, nz)),
            n=jnp.ones((nr, nz)) * 2.0,
            T=jnp.ones((nr, nz)) * 100.0,
            time=5.0,
            step=50
        )

        leaves, treedef = jax.tree_util.tree_flatten(state)
        reconstructed = jax.tree_util.tree_unflatten(treedef, leaves)

        assert jnp.allclose(reconstructed.psi, state.psi)
        assert jnp.allclose(reconstructed.n, state.n)
        assert jnp.allclose(reconstructed.T, state.T)
        assert reconstructed.time == state.time
        assert reconstructed.step == state.step
