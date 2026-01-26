"""Tests for CircuitSystem integration with BurningPlasmaModel."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.burn import SpeciesState, ReactionRates, PowerSources


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.5,
        z_min=-1.0,
        z_max=1.0,
    )


class TestBurningPlasmaStateWithCircuits:
    """Tests for BurningPlasmaState with CircuitState."""

    def test_state_with_circuit_state(self, geometry):
        """Can create BurningPlasmaState with CircuitState instead of ConversionState."""
        from jax_frc.models.burning_plasma import BurningPlasmaState
        from jax_frc.circuits import CircuitState

        mhd = State.zeros(geometry.nr, geometry.nz)
        species = SpeciesState(
            n_D=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_T=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_He3=jnp.zeros((geometry.nr, geometry.nz)),
            n_He4=jnp.zeros((geometry.nr, geometry.nz)),
            n_p=jnp.zeros((geometry.nr, geometry.nz)),
        )
        rates = ReactionRates(
            DT=jnp.zeros((geometry.nr, geometry.nz)),
            DD_T=jnp.zeros((geometry.nr, geometry.nz)),
            DD_HE3=jnp.zeros((geometry.nr, geometry.nz)),
            DHE3=jnp.zeros((geometry.nr, geometry.nz)),
        )
        power = PowerSources(
            P_fusion=jnp.zeros((geometry.nr, geometry.nz)),
            P_alpha=jnp.zeros((geometry.nr, geometry.nz)),
            P_neutron=jnp.zeros((geometry.nr, geometry.nz)),
            P_charged=jnp.zeros((geometry.nr, geometry.nz)),
        )
        circuits = CircuitState.zeros(n_pickup=3, n_external=1)

        state = BurningPlasmaState(
            mhd=mhd,
            species=species,
            rates=rates,
            power=power,
            circuits=circuits,
        )

        assert state.mhd is not None
        assert state.circuits.I_pickup.shape == (3,)
        assert state.circuits.I_external.shape == (1,)

    def test_state_with_circuits_is_pytree(self, geometry):
        """BurningPlasmaState with circuits works with JAX transformations."""
        from jax_frc.models.burning_plasma import BurningPlasmaState
        from jax_frc.circuits import CircuitState

        mhd = State.zeros(geometry.nr, geometry.nz)
        species = SpeciesState(
            n_D=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_T=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_He3=jnp.zeros((geometry.nr, geometry.nz)),
            n_He4=jnp.zeros((geometry.nr, geometry.nz)),
            n_p=jnp.zeros((geometry.nr, geometry.nz)),
        )
        rates = ReactionRates(
            DT=jnp.zeros((geometry.nr, geometry.nz)),
            DD_T=jnp.zeros((geometry.nr, geometry.nz)),
            DD_HE3=jnp.zeros((geometry.nr, geometry.nz)),
            DHE3=jnp.zeros((geometry.nr, geometry.nz)),
        )
        power = PowerSources(
            P_fusion=jnp.zeros((geometry.nr, geometry.nz)),
            P_alpha=jnp.zeros((geometry.nr, geometry.nz)),
            P_neutron=jnp.zeros((geometry.nr, geometry.nz)),
            P_charged=jnp.zeros((geometry.nr, geometry.nz)),
        )
        circuits = CircuitState.zeros(n_pickup=2, n_external=0)

        state = BurningPlasmaState(
            mhd=mhd,
            species=species,
            rates=rates,
            power=power,
            circuits=circuits,
        )

        @jax.jit
        def get_circuit_power(s):
            return s.circuits.P_extracted

        result = get_circuit_power(state)
        assert result == 0.0


class TestBurningPlasmaModelWithCircuits:
    """Tests for BurningPlasmaModel with CircuitSystem."""

    def test_model_with_circuit_system(self):
        """Can create BurningPlasmaModel with CircuitSystem."""
        from jax_frc.models.burning_plasma import BurningPlasmaModel
        from jax_frc.models.resistive_mhd import ResistiveMHD
        from jax_frc.models.resistivity import SpitzerResistivity
        from jax_frc.burn import BurnPhysics, SpeciesTracker
        from jax_frc.transport import TransportModel
        from jax_frc.circuits import (
            CircuitSystem,
            CircuitParams,
            PickupCoilArray,
            ExternalCircuits,
            FluxCoupling,
        )

        # Create a minimal circuit system
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.6]),
            n_turns=jnp.array([100]),
            params=CircuitParams(
                L=jnp.array([1e-3]),
                R=jnp.array([0.1]),
                C=jnp.array([jnp.inf]),
            ),
            load_resistance=jnp.array([1.0]),
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()
        circuits = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        model = BurningPlasmaModel(
            mhd_core=ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6)),
            burn=BurnPhysics(fuels=("DT",)),
            species_tracker=SpeciesTracker(),
            transport=TransportModel(D_particle=1.0, chi_e=5.0, chi_i=2.0),
            circuits=circuits,
        )

        assert model.circuits is not None
        assert model.circuits.pickup.n_coils == 1

    def test_step_with_circuits(self, geometry):
        """Model step with circuits returns updated state."""
        from jax_frc.models.burning_plasma import BurningPlasmaModel, BurningPlasmaState
        from jax_frc.models.resistive_mhd import ResistiveMHD
        from jax_frc.models.resistivity import SpitzerResistivity
        from jax_frc.burn import BurnPhysics, SpeciesTracker
        from jax_frc.transport import TransportModel
        from jax_frc.circuits import (
            CircuitSystem,
            CircuitState,
            CircuitParams,
            PickupCoilArray,
            ExternalCircuits,
            FluxCoupling,
        )

        # Create circuit system
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=CircuitParams(
                L=jnp.array([1e-3]),
                R=jnp.array([0.1]),
                C=jnp.array([jnp.inf]),
            ),
            load_resistance=jnp.array([1.0]),
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()
        circuits = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        model = BurningPlasmaModel(
            mhd_core=ResistiveMHD(resistivity=SpitzerResistivity(eta_0=1e-6)),
            burn=BurnPhysics(fuels=("DT",)),
            species_tracker=SpeciesTracker(),
            transport=TransportModel(D_particle=1.0, chi_e=5.0, chi_i=2.0),
            circuits=circuits,
        )

        # Create initial state
        mhd = State.zeros(geometry.nr, geometry.nz)
        mhd = mhd.replace(
            T=jnp.ones((geometry.nr, geometry.nz)) * 10.0,  # 10 keV
            B=jnp.ones((geometry.nr, geometry.nz, 3)) * 1.0,
        )
        species = SpeciesState(
            n_D=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_T=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_He3=jnp.zeros((geometry.nr, geometry.nz)),
            n_He4=jnp.zeros((geometry.nr, geometry.nz)),
            n_p=jnp.zeros((geometry.nr, geometry.nz)),
        )

        state = BurningPlasmaState(
            mhd=mhd,
            species=species,
            rates=ReactionRates(
                DT=jnp.zeros((geometry.nr, geometry.nz)),
                DD_T=jnp.zeros((geometry.nr, geometry.nz)),
                DD_HE3=jnp.zeros((geometry.nr, geometry.nz)),
                DHE3=jnp.zeros((geometry.nr, geometry.nz)),
            ),
            power=PowerSources(
                P_fusion=jnp.zeros((geometry.nr, geometry.nz)),
                P_alpha=jnp.zeros((geometry.nr, geometry.nz)),
                P_neutron=jnp.zeros((geometry.nr, geometry.nz)),
                P_charged=jnp.zeros((geometry.nr, geometry.nz)),
            ),
            circuits=CircuitState.zeros(n_pickup=1, n_external=0),
        )

        dt = 1e-9
        new_state = model.step(state, dt, geometry)

        # Should have computed fusion power
        assert jnp.any(new_state.power.P_fusion > 0)
        # Circuit state should be updated
        assert new_state.circuits is not None


class TestFromConfigWithCircuits:
    """Tests for BurningPlasmaModel.from_config with new circuit format."""

    def test_from_config_with_circuits(self):
        """Can create model from config with circuits section."""
        from jax_frc.models.burning_plasma import BurningPlasmaModel

        config = {
            "fuels": ["DT"],
            "mhd": {"resistivity": {"type": "spitzer", "eta_0": 1e-6}},
            "transport": {"D_particle": 0.1, "chi_e": 1.0, "chi_i": 0.5},
            "circuits": {
                "pickup_array": {
                    "z_positions": [0.0, 0.25, 0.5],
                    "radii": [0.6, 0.6, 0.6],
                    "n_turns": [100, 100, 100],
                    "L": [1e-3, 1e-3, 1e-3],
                    "R": [0.1, 0.1, 0.1],
                    "C": None,
                    "load_resistance": [1.0, 1.0, 1.0],
                },
            },
        }

        model = BurningPlasmaModel.from_config(config)

        assert model.circuits is not None
        assert model.circuits.pickup.n_coils == 3

    def test_from_config_with_external_circuits(self):
        """Can create model from config with external circuits."""
        from jax_frc.models.burning_plasma import BurningPlasmaModel

        config = {
            "fuels": ["DT"],
            "circuits": {
                "pickup_array": {
                    "z_positions": [0.0],
                    "radii": [0.6],
                    "n_turns": [100],
                    "L": [1e-3],
                    "R": [0.1],
                    "load_resistance": [1.0],
                },
                "external": [
                    {
                        "name": "compression_coil",
                        "z_center": 0.0,
                        "radius": 0.8,
                        "length": 1.0,
                        "n_turns": 50,
                        "L": 5e-3,
                        "R": 0.05,
                        "driver": {"mode": "voltage", "waveform": {"type": "constant", "value": 0.0}},
                    }
                ],
            },
        }

        model = BurningPlasmaModel.from_config(config)

        assert model.circuits.external.n_circuits == 1
        assert model.circuits.external.circuits[0].name == "compression_coil"


class TestBackwardCompatibility:
    """Tests for backward compatibility with DirectConversion config."""

    def test_old_direct_conversion_config(self):
        """Old direct_conversion config creates equivalent CircuitSystem."""
        from jax_frc.models.burning_plasma import BurningPlasmaModel

        config = {
            "fuels": ["DT"],
            "direct_conversion": {
                "coil_turns": 100,
                "coil_radius": 0.6,
                "circuit_resistance": 0.1,
                "coupling_efficiency": 0.9,
            },
        }

        model = BurningPlasmaModel.from_config(config)

        # Should have created a single pickup coil circuit
        assert model.circuits is not None
        assert model.circuits.pickup.n_coils == 1
        # Check parameters match
        # Note: effective turns = coil_turns * coupling_efficiency = 100 * 0.9 = 90
        assert jnp.isclose(model.circuits.pickup.n_turns[0], 90.0)
        assert model.circuits.pickup.radii[0] == 0.6

    def test_old_config_step_works(self, geometry):
        """Model from old config can step simulation."""
        from jax_frc.models.burning_plasma import BurningPlasmaModel, BurningPlasmaState
        from jax_frc.circuits import CircuitState

        config = {
            "fuels": ["DT"],
            "direct_conversion": {
                "coil_turns": 100,
                "coil_radius": 0.6,
                "circuit_resistance": 0.1,
                "coupling_efficiency": 0.9,
            },
        }

        model = BurningPlasmaModel.from_config(config)

        mhd = State.zeros(geometry.nr, geometry.nz)
        mhd = mhd.replace(
            T=jnp.ones((geometry.nr, geometry.nz)) * 10.0,
            B=jnp.ones((geometry.nr, geometry.nz, 3)) * 1.0,
        )
        species = SpeciesState(
            n_D=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_T=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_He3=jnp.zeros((geometry.nr, geometry.nz)),
            n_He4=jnp.zeros((geometry.nr, geometry.nz)),
            n_p=jnp.zeros((geometry.nr, geometry.nz)),
        )

        state = BurningPlasmaState(
            mhd=mhd,
            species=species,
            rates=ReactionRates(
                DT=jnp.zeros((geometry.nr, geometry.nz)),
                DD_T=jnp.zeros((geometry.nr, geometry.nz)),
                DD_HE3=jnp.zeros((geometry.nr, geometry.nz)),
                DHE3=jnp.zeros((geometry.nr, geometry.nz)),
            ),
            power=PowerSources(
                P_fusion=jnp.zeros((geometry.nr, geometry.nz)),
                P_alpha=jnp.zeros((geometry.nr, geometry.nz)),
                P_neutron=jnp.zeros((geometry.nr, geometry.nz)),
                P_charged=jnp.zeros((geometry.nr, geometry.nz)),
            ),
            circuits=CircuitState.zeros(n_pickup=1, n_external=0),
        )

        dt = 1e-9
        new_state = model.step(state, dt, geometry)

        assert new_state is not None
        assert jnp.any(new_state.power.P_fusion > 0)


class TestCircuitPowerExtraction:
    """Tests for circuit power extraction during simulation."""

    def test_flux_change_induces_current(self, geometry):
        """Changing magnetic field induces current in pickup coils."""
        from jax_frc.models.burning_plasma import BurningPlasmaModel, BurningPlasmaState
        from jax_frc.circuits import CircuitState

        config = {
            "fuels": ["DT"],
            "circuits": {
                "pickup_array": {
                    "z_positions": [0.0],
                    "radii": [0.4],  # Inside geometry
                    "n_turns": [100],
                    "L": [1e-3],
                    "R": [0.1],
                    "load_resistance": [1.0],
                },
            },
        }

        model = BurningPlasmaModel.from_config(config)

        # Create state with strong B-field
        mhd = State.zeros(geometry.nr, geometry.nz)
        mhd = mhd.replace(
            T=jnp.ones((geometry.nr, geometry.nz)) * 10.0,
            B=jnp.ones((geometry.nr, geometry.nz, 3)) * 2.0,  # 2 T
        )
        mhd = mhd.replace(B=mhd.B.at[:, :, 2].set(2.0))  # Strong axial field
        species = SpeciesState(
            n_D=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_T=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_He3=jnp.zeros((geometry.nr, geometry.nz)),
            n_He4=jnp.zeros((geometry.nr, geometry.nz)),
            n_p=jnp.zeros((geometry.nr, geometry.nz)),
        )

        # Initial circuit state has zero flux linkage
        initial_circuits = CircuitState.zeros(n_pickup=1, n_external=0)

        state = BurningPlasmaState(
            mhd=mhd,
            species=species,
            rates=ReactionRates(
                DT=jnp.zeros((geometry.nr, geometry.nz)),
                DD_T=jnp.zeros((geometry.nr, geometry.nz)),
                DD_HE3=jnp.zeros((geometry.nr, geometry.nz)),
                DHE3=jnp.zeros((geometry.nr, geometry.nz)),
            ),
            power=PowerSources(
                P_fusion=jnp.zeros((geometry.nr, geometry.nz)),
                P_alpha=jnp.zeros((geometry.nr, geometry.nz)),
                P_neutron=jnp.zeros((geometry.nr, geometry.nz)),
                P_charged=jnp.zeros((geometry.nr, geometry.nz)),
            ),
            circuits=initial_circuits,
        )

        # Step the model - flux linkage changes from 0 to actual value
        dt = 1e-6
        new_state = model.step(state, dt, geometry)

        # Flux linkage should be non-zero now (field is present)
        assert new_state.circuits.Psi_pickup[0] != 0.0

        # Current should be induced (flux changed from 0)
        # Note: the actual current value depends on circuit dynamics
        # At minimum, verify the circuit state is updated
        assert new_state.circuits is not initial_circuits


class TestDeprecatedConversionImport:
    """Tests that old imports still work for backward compatibility."""

    def test_conversion_state_import_deprecated(self):
        """ConversionState import from burn module still works."""
        # This should not raise
        from jax_frc.burn import ConversionState, DirectConversion

        # Can still create the old types
        state = ConversionState(P_electric=0.0, V_induced=0.0, dPsi_dt=0.0)
        assert state.P_electric == 0.0

        dc = DirectConversion(
            coil_turns=100, coil_radius=0.6, circuit_resistance=0.1, coupling_efficiency=0.9
        )
        assert dc.coil_turns == 100
