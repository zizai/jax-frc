"""Atomic physics coupling between plasma and neutral fluids."""

from dataclasses import dataclass
import jax.numpy as jnp

from jax_frc.constants import MI
from jax_frc.models.atomic_rates import (
    ionization_rate, recombination_rate,
    charge_exchange_rates, bremsstrahlung_loss, line_radiation_loss,
    ionization_energy_loss
)
from jax_frc.models.coupled import SourceRates


@dataclass
class AtomicCouplingConfig:
    """Configuration for atomic physics coupling."""
    include_radiation: bool = True
    impurity_fraction: float = 0.0
    Z_eff: float = 1.0


class AtomicCoupling:
    """Wraps atomic_rates module into SourceTerms protocol.

    Computes source terms for plasma-neutral coupling including:
    - Ionization (H + e -> H+ + 2e)
    - Recombination (H+ + e -> H + hv)
    - Charge exchange (H+ + H -> H + H+)
    - Radiation losses (optional)

    Conservation laws are enforced:
    - Mass: plasma_src.mass + neutral_src.mass = 0
    - Momentum: plasma_src.momentum + neutral_src.momentum = 0
    """

    def __init__(self, config: AtomicCouplingConfig):
        self.config = config

    def compute_sources(self, plasma, neutral, geometry):
        """Compute source terms for both fluids.

        Args:
            plasma: Plasma state (State object)
            neutral: Neutral fluid state (NeutralState object)
            geometry: Grid geometry

        Returns:
            (plasma_sources, neutral_sources): Tuple of SourceRates
        """
        # Extract from plasma
        n_e = plasma.n
        n_i = n_e  # Quasi-neutrality
        T_e = plasma.Te
        v_i = plasma.v

        # Extract from neutrals
        n_n = neutral.rho_n / MI
        v_n = neutral.v_n

        # Ionization and recombination
        S_ion = ionization_rate(T_e, n_e, neutral.rho_n)
        S_rec = recombination_rate(T_e, n_e, n_i)

        # Net mass source: ionization adds to plasma, recombination removes
        net_mass = S_ion - S_rec

        # Charge exchange
        R_cx, Q_cx = charge_exchange_rates(T_e, n_i, n_n, v_i, v_n)

        # Radiation (optional)
        if self.config.include_radiation:
            n_imp = self.config.impurity_fraction * n_e
            P_brem = bremsstrahlung_loss(T_e, n_e, n_i, self.config.Z_eff)
            P_line = line_radiation_loss(T_e, n_e, n_imp)
            P_ion = ionization_energy_loss(S_ion)
            P_rad = P_brem + P_line + P_ion
        else:
            P_rad = jnp.zeros_like(T_e)

        # Plasma sources
        # - Mass: gains from ionization, loses from recombination
        # - Momentum: charge exchange momentum transfer (R_cx from neutrals)
        # - Energy: loses to radiation and charge exchange thermal transfer
        plasma_sources = SourceRates(
            mass=net_mass,
            momentum=R_cx,
            energy=-P_rad - Q_cx
        )

        # Neutral sources (opposite signs for conservation)
        # - Mass: loses to ionization, gains from recombination
        # - Momentum: opposite of plasma momentum transfer
        # - Energy: gains from charge exchange thermal transfer
        neutral_sources = SourceRates(
            mass=-net_mass,
            momentum=-R_cx,
            energy=Q_cx
        )

        return plasma_sources, neutral_sources
