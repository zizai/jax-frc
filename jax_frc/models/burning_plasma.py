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


from jax_frc.transport import TransportModel
from jax_frc.models.resistive_mhd import ResistiveMHD


@dataclass
class BurningPlasmaModel:
    """Burning plasma model with fusion, transport, and energy recovery.

    Orchestrates MHD core, burn physics, species tracking,
    transport, and direct conversion modules.
    """
    mhd_core: ResistiveMHD
    burn: BurnPhysics
    species_tracker: SpeciesTracker
    transport: TransportModel
    conversion: DirectConversion

    def step(
        self,
        state: BurningPlasmaState,
        dt: float,
        geometry: Geometry,
    ) -> BurningPlasmaState:
        """Advance burning plasma state by one timestep.

        Args:
            state: Current burning plasma state
            dt: Timestep [s]
            geometry: Computational geometry

        Returns:
            Updated BurningPlasmaState
        """
        # 1. Get temperature in keV from MHD state
        T_keV = state.mhd.T  # Assuming T is already in keV

        # 2. Compute fusion reaction rates
        rates = self.burn.compute_rates(
            n_D=state.species.n_D,
            n_T=state.species.n_T,
            n_He3=state.species.n_He3,
            T_keV=T_keV,
        )

        # 3. Compute power sources
        power = self.burn.power_sources(rates)

        # 4. Compute burn source terms for species
        burn_sources = self.species_tracker.burn_sources(rates)

        # 5. Compute transport fluxes and divergences
        transport_div = {}
        for species_name, n in [
            ("D", state.species.n_D),
            ("T", state.species.n_T),
            ("He3", state.species.n_He3),
            ("He4", state.species.n_He4),
            ("p", state.species.n_p),
        ]:
            Gamma_r, Gamma_z = self.transport.particle_flux(n, geometry)
            div_Gamma = self.transport.flux_divergence(Gamma_r, Gamma_z, geometry)
            transport_div[species_name] = -div_Gamma  # -div(Gamma) is source

        # 6. Update species densities
        new_species = self.species_tracker.advance(
            state=state.species,
            burn_sources=burn_sources,
            transport_divergence=transport_div,
            dt=dt,
        )

        # 7. Compute direct conversion power from B-field change
        # (For now, use current B since MHD step not fully integrated)
        new_conversion = self.conversion.compute_power(
            B_old=state.mhd.B,
            B_new=state.mhd.B,  # TODO: integrate with MHD step
            dt=dt,
            geometry=geometry,
        )

        # 8. Update MHD state (simplified - just pass through for now)
        # Full integration would add alpha heating to energy equation
        new_mhd = state.mhd

        return BurningPlasmaState(
            mhd=new_mhd,
            species=new_species,
            rates=rates,
            power=power,
            conversion=new_conversion,
        )

    @classmethod
    def from_config(cls, config: dict) -> "BurningPlasmaModel":
        """Create BurningPlasmaModel from configuration dictionary.

        Args:
            config: Configuration dictionary with keys:
                - mhd: MHD configuration (optional, defaults to Spitzer resistivity)
                - fuels: List of fuel types, e.g. ["DT", "DD"] (optional)
                - transport: Transport coefficients (optional)
                - direct_conversion: Direct conversion parameters (optional)

        Returns:
            Configured BurningPlasmaModel instance
        """
        from jax_frc.models.resistive_mhd import ResistiveMHD

        # MHD core
        mhd_config = config.get("mhd", {"resistivity": {"type": "spitzer"}})
        mhd_core = ResistiveMHD.from_config(mhd_config)

        # Burn physics
        fuels = tuple(config.get("fuels", ["DT"]))
        burn = BurnPhysics(fuels=fuels)

        # Species tracker
        species_tracker = SpeciesTracker()

        # Transport
        transport_config = config.get("transport", {})
        transport = TransportModel(
            D_particle=transport_config.get("D_particle", 1.0),
            chi_e=transport_config.get("chi_e", 5.0),
            chi_i=transport_config.get("chi_i", 2.0),
            v_pinch=transport_config.get("v_pinch", 0.0),
        )

        # Direct conversion
        dc_config = config.get("direct_conversion", {})
        conversion = DirectConversion(
            coil_turns=dc_config.get("coil_turns", 100),
            coil_radius=dc_config.get("coil_radius", 0.6),
            circuit_resistance=dc_config.get("circuit_resistance", 0.1),
            coupling_efficiency=dc_config.get("coupling_efficiency", 0.9),
        )

        return cls(
            mhd_core=mhd_core,
            burn=burn,
            species_tracker=species_tracker,
            transport=transport,
            conversion=conversion,
        )
