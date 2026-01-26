"""Tests for circuit state dataclasses."""

import jax
import jax.numpy as jnp
import pytest


class TestCircuitParams:
    """Tests for CircuitParams dataclass."""

    def test_creation(self):
        """Can create CircuitParams with arrays."""
        from jax_frc.circuits import CircuitParams

        params = CircuitParams(
            L=jnp.array([1e-3, 1e-3]),
            R=jnp.array([0.1, 0.1]),
            C=jnp.array([jnp.inf, jnp.inf]),
        )
        assert params.L.shape == (2,)
        assert params.R.shape == (2,)

    def test_tau_property(self):
        """tau = L/R gives circuit timescale."""
        from jax_frc.circuits import CircuitParams

        params = CircuitParams(
            L=jnp.array([1e-3, 2e-3]),
            R=jnp.array([0.1, 0.1]),
            C=jnp.array([jnp.inf, jnp.inf]),
        )
        tau = params.L / params.R
        assert jnp.allclose(tau, jnp.array([0.01, 0.02]))


class TestCircuitState:
    """Tests for CircuitState dataclass."""

    def test_creation(self):
        """Can create CircuitState."""
        from jax_frc.circuits import CircuitState

        state = CircuitState(
            I_pickup=jnp.zeros(3),
            Q_pickup=jnp.zeros(3),
            I_external=jnp.zeros(2),
            Q_external=jnp.zeros(2),
            Psi_pickup=jnp.zeros(3),
            Psi_external=jnp.zeros(2),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        assert state.I_pickup.shape == (3,)
        assert state.I_external.shape == (2,)

    def test_is_pytree(self):
        """CircuitState works with JAX transformations."""
        from jax_frc.circuits import CircuitState

        state = CircuitState(
            I_pickup=jnp.array([1.0, 2.0]),
            Q_pickup=jnp.zeros(2),
            I_external=jnp.array([0.5]),
            Q_external=jnp.zeros(1),
            Psi_pickup=jnp.zeros(2),
            Psi_external=jnp.zeros(1),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        @jax.jit
        def double_currents(s):
            return s.replace(I_pickup=s.I_pickup * 2)

        new_state = double_currents(state)
        assert jnp.allclose(new_state.I_pickup, jnp.array([2.0, 4.0]))

    def test_zeros_factory(self):
        """Can create zero-initialized state."""
        from jax_frc.circuits import CircuitState

        state = CircuitState.zeros(n_pickup=3, n_external=2)
        assert state.I_pickup.shape == (3,)
        assert state.I_external.shape == (2,)
        assert jnp.all(state.I_pickup == 0)
