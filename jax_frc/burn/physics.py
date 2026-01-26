"""Fusion reaction rates and power calculations.

Implements Bosch-Hale parameterization for D-T, D-D, and D-3He reactions.
"""

from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
from jax import Array, jit

from jax_frc.constants import (
    BOSCH_HALE_DT, BOSCH_HALE_DD_T, BOSCH_HALE_DD_HE3, BOSCH_HALE_DHE3,
    E_DT, E_DD_T, E_DD_HE3, E_DHE3,
    F_CHARGED_DT, F_CHARGED_DD_T, F_CHARGED_DD_HE3, F_CHARGED_DHE3,
)

ReactionType = Literal["DT", "DD_T", "DD_HE3", "DHE3"]


@jit(static_argnums=(1,))
def reactivity(T_keV: Array, reaction: str) -> Array:
    """Compute fusion reactivity <sigma*v> using Bosch-Hale parameterization.

    Args:
        T_keV: Temperature in keV (scalar or array)
        reaction: Reaction type ("DT", "DD_T", "DD_HE3", "DHE3")

    Returns:
        Reactivity in cm^3/s
    """
    # Get coefficients based on reaction type
    if reaction == "DT":
        coef = BOSCH_HALE_DT
    elif reaction == "DD_T":
        coef = BOSCH_HALE_DD_T
    elif reaction == "DD_HE3":
        coef = BOSCH_HALE_DD_HE3
    elif reaction == "DHE3":
        coef = BOSCH_HALE_DHE3
    else:
        raise ValueError(f"Unknown reaction: {reaction}")

    B_G = coef["B_G"]
    m_rc2 = coef["m_rc2"]
    C1, C2, C3 = coef["C1"], coef["C2"], coef["C3"]
    C4, C5, C6, C7 = coef["C4"], coef["C5"], coef["C6"], coef["C7"]

    # Ensure T is positive to avoid numerical issues
    T = jnp.maximum(T_keV, 0.1)

    # Compute theta
    numerator = T * (C2 + T * (C4 + T * C6))
    denominator = 1.0 + T * (C3 + T * (C5 + T * C7))
    theta = T / (1.0 - numerator / denominator)

    # Compute xi
    xi = (B_G**2 / (4.0 * theta)) ** (1.0 / 3.0)

    # Compute <sigma*v>
    sigma_v = C1 * theta * jnp.sqrt(xi / (m_rc2 * T**3)) * jnp.exp(-3.0 * xi)

    return sigma_v


@dataclass(frozen=True)
class ReactionRates:
    """Volumetric reaction rates for each reaction channel.

    All rates in [reactions/m³/s].
    """
    DT: Array
    DD_T: Array
    DD_HE3: Array
    DHE3: Array


@dataclass(frozen=True)
class PowerSources:
    """Fusion power density by category.

    All powers in [W/m³].
    """
    P_fusion: Array   # Total fusion power
    P_alpha: Array    # Power deposited to plasma (charged products)
    P_neutron: Array  # Power carried by neutrons
    P_charged: Array  # Power available for direct conversion


@dataclass
class BurnPhysics:
    """Fusion reaction rate and power calculations.

    Supports D-T, D-D, and D-³He reactions.
    """
    fuels: tuple[str, ...]  # ("DT", "DD", "DHE3")

    def reaction_rate(
        self, n1: Array, n2: Array, T_keV: Array, reaction: str
    ) -> Array:
        """Compute volumetric reaction rate.

        Args:
            n1: First reactant density [m⁻³]
            n2: Second reactant density [m⁻³]
            T_keV: Temperature [keV]
            reaction: Reaction type

        Returns:
            Reaction rate [reactions/m³/s]
        """
        # Convert reactivity from cm³/s to m³/s
        sigma_v = reactivity(T_keV, reaction) * 1e-6

        # Kronecker delta for identical particles (DD reactions)
        kronecker = 1.0 if reaction in ("DD_T", "DD_HE3") else 0.0

        return n1 * n2 * sigma_v / (1.0 + kronecker)

    def compute_rates(
        self, n_D: Array, n_T: Array, n_He3: Array, T_keV: Array
    ) -> ReactionRates:
        """Compute all reaction rates.

        Args:
            n_D: Deuterium density [m⁻³]
            n_T: Tritium density [m⁻³]
            n_He3: Helium-3 density [m⁻³]
            T_keV: Temperature [keV]

        Returns:
            ReactionRates for all channels
        """
        zeros = jnp.zeros_like(n_D)

        DT = self.reaction_rate(n_D, n_T, T_keV, "DT") if "DT" in self.fuels else zeros
        DD_T = self.reaction_rate(n_D, n_D, T_keV, "DD_T") if "DD" in self.fuels else zeros
        DD_HE3 = self.reaction_rate(n_D, n_D, T_keV, "DD_HE3") if "DD" in self.fuels else zeros
        DHE3 = self.reaction_rate(n_D, n_He3, T_keV, "DHE3") if "DHE3" in self.fuels else zeros

        return ReactionRates(DT=DT, DD_T=DD_T, DD_HE3=DD_HE3, DHE3=DHE3)

    def power_sources(self, rates: ReactionRates) -> PowerSources:
        """Compute fusion power densities.

        Args:
            rates: Reaction rates for all channels

        Returns:
            Power sources broken down by category
        """
        # Total fusion power per reaction
        P_DT = rates.DT * E_DT
        P_DD_T = rates.DD_T * E_DD_T
        P_DD_HE3 = rates.DD_HE3 * E_DD_HE3
        P_DHE3 = rates.DHE3 * E_DHE3

        P_fusion = P_DT + P_DD_T + P_DD_HE3 + P_DHE3

        # Charged particle power (deposited + available for direct conversion)
        P_charged = (
            P_DT * F_CHARGED_DT +
            P_DD_T * F_CHARGED_DD_T +
            P_DD_HE3 * F_CHARGED_DD_HE3 +
            P_DHE3 * F_CHARGED_DHE3
        )

        # Neutron power
        P_neutron = P_fusion - P_charged

        # Alpha heating (instant thermalization assumption)
        P_alpha = P_charged

        return PowerSources(
            P_fusion=P_fusion,
            P_alpha=P_alpha,
            P_neutron=P_neutron,
            P_charged=P_charged,
        )


# Register ReactionRates as JAX pytree
def _reaction_rates_flatten(state):
    children = (state.DT, state.DD_T, state.DD_HE3, state.DHE3)
    aux_data = None
    return children, aux_data


def _reaction_rates_unflatten(aux_data, children):
    DT, DD_T, DD_HE3, DHE3 = children
    return ReactionRates(DT=DT, DD_T=DD_T, DD_HE3=DD_HE3, DHE3=DHE3)


jax.tree_util.register_pytree_node(
    ReactionRates, _reaction_rates_flatten, _reaction_rates_unflatten
)


# Register PowerSources as JAX pytree
def _power_sources_flatten(state):
    children = (state.P_fusion, state.P_alpha, state.P_neutron, state.P_charged)
    aux_data = None
    return children, aux_data


def _power_sources_unflatten(aux_data, children):
    P_fusion, P_alpha, P_neutron, P_charged = children
    return PowerSources(
        P_fusion=P_fusion, P_alpha=P_alpha, P_neutron=P_neutron, P_charged=P_charged
    )


jax.tree_util.register_pytree_node(
    PowerSources, _power_sources_flatten, _power_sources_unflatten
)
