"""Tests for coupled plasma-neutral state."""

import jax
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.models.neutral_fluid import NeutralState
from jax_frc.models.coupled import CoupledState, SourceRates


def test_coupled_state_creation():
    """CoupledState can be created from plasma and neutral states."""
    plasma = State.zeros(8, 1, 8)
    neutral = NeutralState(
        rho_n=jnp.ones((8, 1, 8)) * 1e-6,
        mom_n=jnp.zeros((8, 1, 8, 3)),
        E_n=jnp.ones((8, 1, 8)) * 100.0
    )
    coupled = CoupledState(plasma=plasma, neutral=neutral)

    assert coupled.plasma is plasma
    assert coupled.neutral is neutral


def test_coupled_state_is_pytree():
    """CoupledState works with JAX transformations."""
    plasma = State.zeros(8, 1, 8)
    neutral = NeutralState(
        rho_n=jnp.ones((8, 1, 8)),
        mom_n=jnp.zeros((8, 1, 8, 3)),
        E_n=jnp.ones((8, 1, 8))
    )
    coupled = CoupledState(plasma=plasma, neutral=neutral)

    # Should be able to flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(coupled)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert jnp.allclose(restored.plasma.n, coupled.plasma.n)
    assert jnp.allclose(restored.neutral.rho_n, coupled.neutral.rho_n)


def test_source_rates_creation():
    """SourceRates can be created with mass, momentum, energy."""
    rates = SourceRates(
        mass=jnp.ones((8, 1, 8)),
        momentum=jnp.zeros((8, 1, 8, 3)),
        energy=jnp.ones((8, 1, 8)) * 1e3
    )

    assert rates.mass.shape == (8, 1, 8)
    assert rates.momentum.shape == (8, 1, 8, 3)
    assert rates.energy.shape == (8, 1, 8)


def test_source_rates_is_pytree():
    """SourceRates works with JAX transformations."""
    rates = SourceRates(
        mass=jnp.ones((8, 1, 8)) * 2.0,
        momentum=jnp.ones((8, 1, 8, 3)) * 3.0,
        energy=jnp.ones((8, 1, 8)) * 4.0
    )

    # Should be able to flatten/unflatten
    leaves, treedef = jax.tree_util.tree_flatten(rates)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)

    assert jnp.allclose(restored.mass, rates.mass)
    assert jnp.allclose(restored.momentum, rates.momentum)
    assert jnp.allclose(restored.energy, rates.energy)


def test_coupled_state_jit_compatible():
    """CoupledState can be passed through JIT functions."""
    plasma = State.zeros(8, 1, 8)
    neutral = NeutralState(
        rho_n=jnp.ones((8, 1, 8)),
        mom_n=jnp.zeros((8, 1, 8, 3)),
        E_n=jnp.ones((8, 1, 8))
    )
    coupled = CoupledState(plasma=plasma, neutral=neutral)

    @jax.jit
    def extract_density(state: CoupledState):
        return state.neutral.rho_n + state.plasma.n

    result = extract_density(coupled)
    assert result.shape == (8, 1, 8)
    assert jnp.allclose(result, jnp.ones((8, 1, 8)))  # 1 + 0


def test_source_rates_jit_compatible():
    """SourceRates can be passed through JIT functions."""
    rates = SourceRates(
        mass=jnp.ones((8, 1, 8)),
        momentum=jnp.zeros((8, 1, 8, 3)),
        energy=jnp.ones((8, 1, 8)) * 100.0
    )

    @jax.jit
    def scale_rates(r: SourceRates, factor: float):
        return SourceRates(
            mass=r.mass * factor,
            momentum=r.momentum * factor,
            energy=r.energy * factor
        )

    scaled = scale_rates(rates, 2.0)
    assert jnp.allclose(scaled.mass, jnp.ones((8, 1, 8)) * 2.0)
    assert jnp.allclose(scaled.energy, jnp.ones((8, 1, 8)) * 200.0)
