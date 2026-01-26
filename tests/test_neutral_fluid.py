"""Tests for neutral fluid model."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.constants import QE, MI


class TestNeutralState:
    """Tests for NeutralState dataclass."""

    def test_neutral_state_importable(self):
        """NeutralState is importable."""
        from jax_frc.models.neutral_fluid import NeutralState
        assert NeutralState is not None

    def test_neutral_state_creation(self):
        """Can create NeutralState with required fields."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * 1e-6
        mom_n = jnp.zeros((16, 32, 3))
        E_n = jnp.ones((16, 32)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)
        assert state.rho_n.shape == (16, 32)
        assert state.mom_n.shape == (16, 32, 3)
        assert state.E_n.shape == (16, 32)

    def test_neutral_state_velocity_property(self):
        """v_n = mom_n / rho_n."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * 1e-6
        mom_n = jnp.ones((16, 32, 3)) * 1e-6 * 1000  # 1000 m/s
        E_n = jnp.ones((16, 32)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)
        assert jnp.allclose(state.v_n, 1000.0)

    def test_neutral_state_pressure_property(self):
        """p_n from ideal gas EOS."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * MI * 1e19  # n = 1e19 m^-3
        mom_n = jnp.zeros((16, 32, 3))  # Stationary
        # E_n = p / (gamma - 1) for stationary gas
        gamma = 5/3
        p_target = 1e19 * 10 * QE  # n * T where T = 10 eV
        E_n = p_target / (gamma - 1)
        E_n = jnp.ones((16, 32)) * E_n
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)
        assert jnp.allclose(state.p_n, p_target, rtol=1e-5)

    def test_neutral_state_is_pytree(self):
        """NeutralState works with JAX transformations."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * 1e-6
        mom_n = jnp.zeros((16, 32, 3))
        E_n = jnp.ones((16, 32)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        # Should be able to use in JIT
        @jax.jit
        def get_density(s):
            return s.rho_n

        result = get_density(state)
        assert result.shape == (16, 32)
