"""Burning plasma model with fusion, transport, and energy recovery.

Combines MHD core with nuclear burn physics, species tracking,
anomalous transport, and circuit-based energy extraction.

The CircuitSystem replaces the legacy DirectConversion module, providing:
- Multi-coil pickup arrays for spatially resolved energy extraction
- External driven circuits with feedback control
- Proper RLC circuit dynamics with subcycling
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.burn.physics import BurnPhysics, ReactionRates, PowerSources
from jax_frc.burn.species import SpeciesState, SpeciesTracker
from jax_frc.circuits import CircuitSystem, CircuitState


@dataclass(frozen=True)
class BurningPlasmaState:
    """Complete state for burning plasma simulation.

    Attributes:
        mhd: MHD state (B, v, p, psi, etc.)
        species: Fuel and ash densities
        rates: Current reaction rates
        power: Current power sources
        circuits: Circuit system state (replaces legacy ConversionState)
    """
    mhd: State
    species: SpeciesState
    rates: ReactionRates
    power: PowerSources
    circuits: CircuitState

    def replace(self, **kwargs) -> "BurningPlasmaState":
        """Return new state with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)


# Register BurningPlasmaState as JAX pytree
def _burning_plasma_state_flatten(state):
    children = (state.mhd, state.species, state.rates, state.power, state.circuits)
    aux_data = None
    return children, aux_data


def _burning_plasma_state_unflatten(aux_data, children):
    mhd, species, rates, power, circuits = children
    return BurningPlasmaState(
        mhd=mhd, species=species, rates=rates, power=power, circuits=circuits
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
    transport, and circuit-based energy extraction.

    Attributes:
        mhd_core: Resistive MHD solver
        burn: Fusion burn physics
        species_tracker: Species density evolution
        transport: Particle and heat transport
        circuits: Circuit system for energy extraction (replaces DirectConversion)
    """
    mhd_core: ResistiveMHD
    burn: BurnPhysics
    species_tracker: SpeciesTracker
    transport: TransportModel
    circuits: CircuitSystem

    def step(
        self,
        state: BurningPlasmaState,
        dt: float,
        geometry: Geometry,
        t: float = 0.0,
    ) -> BurningPlasmaState:
        """Advance burning plasma state by one timestep.

        Args:
            state: Current burning plasma state
            dt: Timestep [s]
            geometry: Computational geometry
            t: Current simulation time [s] (for circuit waveforms)

        Returns:
            Updated BurningPlasmaState
        """
        # 1. Get temperature in keV from MHD state
        T_keV = state.mhd.Te  # Assuming Te is already in keV

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

        # 7. Update MHD state (simplified - just pass through for now)
        # Full integration would add alpha heating to energy equation
        new_mhd = state.mhd

        # 8. Step the circuit system using the plasma B-field
        new_circuits = self.circuits.step(
            circuit_state=state.circuits,
            B_plasma=new_mhd.B,
            geometry=geometry,
            t=t,
            dt=dt,
        )

        return BurningPlasmaState(
            mhd=new_mhd,
            species=new_species,
            rates=rates,
            power=power,
            circuits=new_circuits,
        )

    @classmethod
    def from_config(cls, config: dict) -> "BurningPlasmaModel":
        """Create BurningPlasmaModel from configuration dictionary.

        Supports both new circuit config format and legacy direct_conversion format.

        Args:
            config: Configuration dictionary with keys:
                - mhd: MHD configuration (optional, defaults to Spitzer resistivity)
                - fuels: List of fuel types, e.g. ["DT", "DD"] (optional)
                - transport: Transport coefficients (optional)
                - circuits: Circuit system configuration (new format)
                - direct_conversion: Legacy direct conversion parameters (deprecated)

        Returns:
            Configured BurningPlasmaModel instance

        Notes:
            If both 'circuits' and 'direct_conversion' are present, 'circuits' takes
            precedence. The 'direct_conversion' format creates an equivalent single
            pickup coil CircuitSystem for backward compatibility.
        """
        from jax_frc.models.resistive_mhd import ResistiveMHD
        from jax_frc.circuits import (
            CircuitSystem,
            CircuitParams,
            CircuitState,
            PickupCoilArray,
            ExternalCircuits,
            ExternalCircuit,
            CoilGeometry,
            CircuitDriver,
            FluxCoupling,
            waveform_from_config,
        )

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

        # Circuit system - support both new and legacy formats
        if "circuits" in config:
            # New circuit config format
            circuits = cls._create_circuit_system_from_config(config["circuits"])
        elif "direct_conversion" in config:
            # Legacy format - create equivalent single-coil CircuitSystem
            circuits = cls._create_circuit_system_from_legacy(config["direct_conversion"])
        else:
            # Default: minimal circuit system with no coils
            circuits = cls._create_default_circuit_system()

        return cls(
            mhd_core=mhd_core,
            burn=burn,
            species_tracker=species_tracker,
            transport=transport,
            circuits=circuits,
        )

    @staticmethod
    def _create_circuit_system_from_config(circuits_config: dict) -> "CircuitSystem":
        """Create CircuitSystem from new config format.

        Args:
            circuits_config: Dict with 'pickup_array' and/or 'external' keys

        Returns:
            Configured CircuitSystem
        """
        from jax_frc.circuits import (
            CircuitSystem,
            CircuitParams,
            PickupCoilArray,
            ExternalCircuits,
            ExternalCircuit,
            CoilGeometry,
            CircuitDriver,
            FluxCoupling,
            waveform_from_config,
        )

        # Pickup coil array
        pickup_config = circuits_config.get("pickup_array", {})
        if pickup_config:
            z_positions = jnp.array(pickup_config.get("z_positions", [0.0]))
            radii = jnp.array(pickup_config.get("radii", [0.6]))
            n_turns = jnp.array(pickup_config.get("n_turns", [100]))

            # Circuit parameters
            L = jnp.array(pickup_config.get("L", [1e-3] * len(z_positions)))
            R = jnp.array(pickup_config.get("R", [0.1] * len(z_positions)))
            C_raw = pickup_config.get("C", None)
            if C_raw is None:
                C = jnp.full_like(L, jnp.inf)
            else:
                C = jnp.array([jnp.inf if c is None else c for c in C_raw])

            load_resistance = jnp.array(
                pickup_config.get("load_resistance", [1.0] * len(z_positions))
            )

            pickup = PickupCoilArray(
                z_positions=z_positions,
                radii=radii,
                n_turns=n_turns,
                params=CircuitParams(L=L, R=R, C=C),
                load_resistance=load_resistance,
            )
        else:
            # Empty pickup array
            pickup = PickupCoilArray(
                z_positions=jnp.array([]),
                radii=jnp.array([]),
                n_turns=jnp.array([]),
                params=CircuitParams(
                    L=jnp.array([]), R=jnp.array([]), C=jnp.array([])
                ),
                load_resistance=jnp.array([]),
            )

        # External circuits
        external_config = circuits_config.get("external", [])
        external_circuits = []
        for ext in external_config:
            coil = CoilGeometry(
                z_center=ext.get("z_center", 0.0),
                radius=ext.get("radius", 0.8),
                length=ext.get("length", 1.0),
                n_turns=ext.get("n_turns", 50),
            )

            # Circuit parameters
            L = jnp.array([ext.get("L", 5e-3)])
            R = jnp.array([ext.get("R", 0.05)])
            C_raw = ext.get("C", None)
            C = jnp.array([jnp.inf if C_raw is None else C_raw])

            params = CircuitParams(L=L, R=R, C=C)

            # Driver
            driver_config = ext.get("driver", {"mode": "voltage"})
            mode = driver_config.get("mode", "voltage")

            if mode == "voltage":
                waveform_config = driver_config.get("waveform", {"type": "constant", "value": 0.0})
                waveform = waveform_from_config(waveform_config)
                driver = CircuitDriver(mode="voltage", waveform=waveform)
            elif mode == "feedback":
                gains = driver_config.get("feedback_gains", (1e3, 1e2, 0.0))
                driver = CircuitDriver(
                    mode="feedback",
                    feedback_gains=tuple(gains),
                    feedback_target=lambda s: 0.0,  # placeholder
                    feedback_measure=lambda s: 0.0,
                )
            else:
                driver = CircuitDriver(mode=mode)

            circuit = ExternalCircuit(
                name=ext.get("name", f"circuit_{len(external_circuits)}"),
                coil=coil,
                params=params,
                driver=driver,
            )
            external_circuits.append(circuit)

        external = ExternalCircuits(circuits=tuple(external_circuits))
        flux_coupling = FluxCoupling()

        return CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

    @staticmethod
    def _create_circuit_system_from_legacy(dc_config: dict) -> "CircuitSystem":
        """Create CircuitSystem from legacy direct_conversion config.

        Maps the old DirectConversion parameters to a single pickup coil
        CircuitSystem for backward compatibility.

        Args:
            dc_config: Legacy direct_conversion config dict

        Returns:
            Equivalent CircuitSystem with single pickup coil
        """
        from jax_frc.circuits import (
            CircuitSystem,
            CircuitParams,
            PickupCoilArray,
            ExternalCircuits,
            FluxCoupling,
        )

        coil_turns = dc_config.get("coil_turns", 100)
        coil_radius = dc_config.get("coil_radius", 0.6)
        circuit_resistance = dc_config.get("circuit_resistance", 0.1)
        coupling_efficiency = dc_config.get("coupling_efficiency", 0.9)

        # Map legacy params to circuit params
        # The old DirectConversion assumed matched load: P = V^2 / (4R)
        # This corresponds to load_resistance = circuit_resistance
        # Coupling efficiency effectively reduces the number of effective turns

        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),  # Midplane
            radii=jnp.array([coil_radius]),
            n_turns=jnp.array([coil_turns * coupling_efficiency]),  # effective turns
            params=CircuitParams(
                L=jnp.array([1e-3]),  # Default inductance (not used in old model)
                R=jnp.array([circuit_resistance]),
                C=jnp.array([jnp.inf]),  # No capacitor
            ),
            load_resistance=jnp.array([circuit_resistance]),  # Matched load
        )

        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        return CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

    @staticmethod
    def _create_default_circuit_system() -> "CircuitSystem":
        """Create default (empty) CircuitSystem."""
        from jax_frc.circuits import (
            CircuitSystem,
            CircuitParams,
            PickupCoilArray,
            ExternalCircuits,
            FluxCoupling,
        )

        pickup = PickupCoilArray(
            z_positions=jnp.array([]),
            radii=jnp.array([]),
            n_turns=jnp.array([]),
            params=CircuitParams(
                L=jnp.array([]), R=jnp.array([]), C=jnp.array([])
            ),
            load_resistance=jnp.array([]),
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        return CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )
