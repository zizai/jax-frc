"""Species tracking for fusion fuel and ash.

Tracks D, T, 3He, 4He (ash), and protons through burn and transport.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class SpeciesState:
    """Fuel and ash species densities.

    All densities in [m^-3].
    """
    n_D: Array    # Deuterium
    n_T: Array    # Tritium
    n_He3: Array  # Helium-3
    n_He4: Array  # Helium-4 (ash)
    n_p: Array    # Protons

    @property
    def n_e(self) -> Array:
        """Electron density from quasi-neutrality [m^-3].

        n_e = n_D + n_T + n_He3 + 2*n_He4 + n_p
        (He4 is doubly charged)
        """
        return self.n_D + self.n_T + self.n_He3 + 2 * self.n_He4 + self.n_p

    def replace(self, **kwargs) -> "SpeciesState":
        """Return new SpeciesState with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)


# Register SpeciesState as JAX pytree
def _species_state_flatten(state):
    children = (state.n_D, state.n_T, state.n_He3, state.n_He4, state.n_p)
    aux_data = None
    return children, aux_data


def _species_state_unflatten(aux_data, children):
    n_D, n_T, n_He3, n_He4, n_p = children
    return SpeciesState(n_D=n_D, n_T=n_T, n_He3=n_He3, n_He4=n_He4, n_p=n_p)


jax.tree_util.register_pytree_node(
    SpeciesState, _species_state_flatten, _species_state_unflatten
)


@dataclass(frozen=True)
class SpeciesTracker:
    """Tracks fuel consumption and ash accumulation."""

    def burn_sources(self, rates) -> dict[str, Array]:
        """Compute density source terms from fusion reactions.

        Args:
            rates: ReactionRates from BurnPhysics

        Returns:
            Dictionary of dn/dt for each species [m^-3/s]
        """
        return {
            # Deuterium: consumed in DT, DD (x2), DHe3
            "D": -rates.DT - 2 * rates.DD_T - 2 * rates.DD_HE3 - rates.DHE3,

            # Tritium: consumed in DT, produced in DD->T+p
            "T": -rates.DT + rates.DD_T,

            # Helium-3: consumed in DHe3, produced in DD->He3+n
            "He3": -rates.DHE3 + rates.DD_HE3,

            # Helium-4 (ash): produced in DT and DHe3
            "He4": rates.DT + rates.DHE3,

            # Protons: produced in DD->T+p and DHe3
            "p": rates.DD_T + rates.DHE3,
        }

    def advance(
        self,
        state: SpeciesState,
        burn_sources: dict[str, Array],
        transport_divergence: dict[str, Array],
        dt: float,
    ) -> SpeciesState:
        """Advance species densities by one timestep.

        Args:
            state: Current species state
            burn_sources: dn/dt from burn reactions
            transport_divergence: -div(Gamma) for each species
            dt: Timestep [s]

        Returns:
            Updated SpeciesState
        """
        # dn/dt = burn_source - div(flux)
        n_D = state.n_D + dt * (burn_sources["D"] + transport_divergence.get("D", 0))
        n_T = state.n_T + dt * (burn_sources["T"] + transport_divergence.get("T", 0))
        n_He3 = state.n_He3 + dt * (burn_sources["He3"] + transport_divergence.get("He3", 0))
        n_He4 = state.n_He4 + dt * (burn_sources["He4"] + transport_divergence.get("He4", 0))
        n_p = state.n_p + dt * (burn_sources["p"] + transport_divergence.get("p", 0))

        # Ensure non-negative densities
        n_D = jnp.maximum(n_D, 0.0)
        n_T = jnp.maximum(n_T, 0.0)
        n_He3 = jnp.maximum(n_He3, 0.0)
        n_He4 = jnp.maximum(n_He4, 0.0)
        n_p = jnp.maximum(n_p, 0.0)

        return SpeciesState(n_D=n_D, n_T=n_T, n_He3=n_He3, n_He4=n_He4, n_p=n_p)
