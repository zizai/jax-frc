"""Tests for 3D state container."""

import jax.numpy as jnp
import pytest
import jax
from jax_frc.core.state import State


class TestState3D:
    """Test 3D state container."""

    def test_state_zeros(self):
        """Test creating zero-initialized state."""
        state = State.zeros(nx=4, ny=6, nz=8)
        assert state.B.shape == (4, 6, 8, 3)
        assert state.E.shape == (4, 6, 8, 3)
        assert state.n.shape == (4, 6, 8)
        assert state.p.shape == (4, 6, 8)

    def test_state_is_pytree(self):
        """State should be registered as JAX pytree."""
        state = State.zeros(nx=4, ny=4, nz=4)
        leaves = jax.tree_util.tree_leaves(state)
        assert len(leaves) > 0

    def test_state_replace(self):
        """Test replacing state fields."""
        state = State.zeros(nx=4, ny=4, nz=4)
        new_n = jnp.ones((4, 4, 4)) * 1e19
        state2 = state.replace(n=new_n)
        assert jnp.allclose(state2.n, 1e19)
        assert jnp.allclose(state.n, 0.0)  # Original unchanged
