"""Burning plasma model with fusion, transport, and energy recovery.

Combines MHD core with nuclear burn physics, species tracking,
anomalous transport, and direct induction energy conversion.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.burn.physics import BurnPhysics, ReactionRates, PowerSources
from jax_frc.burn.species import SpeciesState, SpeciesTracker
from jax_frc.burn.conversion import DirectConversion, ConversionState


@dataclass(frozen=True)
class BurningPlasmaState:
    """Complete state for burning plasma simulation.

    Attributes:
        mhd: MHD state (B, v, p, psi, etc.)
        species: Fuel and ash densities
        rates: Current reaction rates
        power: Current power sources
        conversion: Direct conversion state
    """
    mhd: State
    species: SpeciesState
    rates: ReactionRates
    power: PowerSources
    conversion: ConversionState

    def replace(self, **kwargs) -> "BurningPlasmaState":
        """Return new state with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)


# Register BurningPlasmaState as JAX pytree
def _burning_plasma_state_flatten(state):
    children = (state.mhd, state.species, state.rates, state.power, state.conversion)
    aux_data = None
    return children, aux_data


def _burning_plasma_state_unflatten(aux_data, children):
    mhd, species, rates, power, conversion = children
    return BurningPlasmaState(
        mhd=mhd, species=species, rates=rates, power=power, conversion=conversion
    )


jax.tree_util.register_pytree_node(
    BurningPlasmaState,
    _burning_plasma_state_flatten,
    _burning_plasma_state_unflatten,
)
