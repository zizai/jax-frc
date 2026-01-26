"""Coupled plasma-neutral state and model for IMEX integration."""

from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import Array

from jax_frc.core.state import State
from jax_frc.models.neutral_fluid import NeutralState


@dataclass(frozen=True)
class SourceRates:
    """Source term rates for one fluid species.

    All rates use SI units.
    """
    mass: Array      # kg/m^3/s
    momentum: Array  # N/m^3 (vector, shape nr,nz,3)
    energy: Array    # W/m^3


@dataclass(frozen=True)
class CoupledState:
    """Combined plasma + neutral state for coupled simulations."""
    plasma: State
    neutral: NeutralState


# Register CoupledState as JAX pytree
def _coupled_state_flatten(state):
    children = (state.plasma, state.neutral)
    aux_data = None
    return children, aux_data


def _coupled_state_unflatten(aux_data, children):
    plasma, neutral = children
    return CoupledState(plasma=plasma, neutral=neutral)


jax.tree_util.register_pytree_node(
    CoupledState, _coupled_state_flatten, _coupled_state_unflatten
)


# Register SourceRates as JAX pytree
def _source_rates_flatten(rates):
    children = (rates.mass, rates.momentum, rates.energy)
    aux_data = None
    return children, aux_data


def _source_rates_unflatten(aux_data, children):
    mass, momentum, energy = children
    return SourceRates(mass=mass, momentum=momentum, energy=energy)


jax.tree_util.register_pytree_node(
    SourceRates, _source_rates_flatten, _source_rates_unflatten
)
