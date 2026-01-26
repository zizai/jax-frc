"""Tests for CircuitSystem with RLC ODE integration."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.circuits import CircuitParams, CircuitState
from jax_frc.circuits.pickup import PickupCoilArray
from jax_frc.circuits.external import (
    CoilGeometry,
    CircuitDriver,
    ExternalCircuit,
    ExternalCircuits,
)
from jax_frc.circuits.coupling import FluxCoupling
from jax_frc.circuits.waveforms import make_constant, make_ramp
from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    """Standard geometry for tests."""
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.5,
        z_min=-1.0,
        z_max=1.0,
    )


@pytest.fixture
def simple_pickup():
    """Single pickup coil for simple tests."""
    params = CircuitParams(
        L=jnp.array([1e-3]),
        R=jnp.array([1.0]),
        C=jnp.array([jnp.inf]),
    )
    return PickupCoilArray(
        z_positions=jnp.array([0.0]),
        radii=jnp.array([0.3]),
        n_turns=jnp.array([100]),
        params=params,
        load_resistance=jnp.array([1.0]),
    )


@pytest.fixture
def simple_external():
    """Empty external circuits for simple tests."""
    return ExternalCircuits(circuits=())


@pytest.fixture
def flux_coupling():
    """FluxCoupling instance."""
    return FluxCoupling()


class TestCircuitSystemCreation:
    """Tests for CircuitSystem creation and basic properties."""

    def test_creation(self, simple_pickup, simple_external, flux_coupling):
        """Can create CircuitSystem."""
        from jax_frc.circuits.system import CircuitSystem

        system = CircuitSystem(
            pickup=simple_pickup,
            external=simple_external,
            flux_coupling=flux_coupling,
        )
        assert system.pickup.n_coils == 1
        assert system.external.n_circuits == 0

    def test_is_frozen_dataclass(self, simple_pickup, simple_external, flux_coupling):
        """CircuitSystem is immutable."""
        from jax_frc.circuits.system import CircuitSystem

        system = CircuitSystem(
            pickup=simple_pickup,
            external=simple_external,
            flux_coupling=flux_coupling,
        )
        with pytest.raises(AttributeError):
            system.pickup = None


class TestCircuitSystemStep:
    """Tests for CircuitSystem.step() method."""

    def test_step_returns_circuit_state(
        self, simple_pickup, simple_external, flux_coupling, geometry
    ):
        """step() returns CircuitState."""
        from jax_frc.circuits.system import CircuitSystem

        system = CircuitSystem(
            pickup=simple_pickup,
            external=simple_external,
            flux_coupling=flux_coupling,
        )
        state = CircuitState.zeros(n_pickup=1, n_external=0)
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        new_state = system.step(state, B_plasma, geometry, t=0.0, dt=1e-6)

        assert isinstance(new_state, CircuitState)
        assert new_state.I_pickup.shape == (1,)

    def test_step_preserves_shape(
        self, simple_pickup, simple_external, flux_coupling, geometry
    ):
        """step() preserves array shapes."""
        from jax_frc.circuits.system import CircuitSystem

        system = CircuitSystem(
            pickup=simple_pickup,
            external=simple_external,
            flux_coupling=flux_coupling,
        )
        state = CircuitState.zeros(n_pickup=1, n_external=0)
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        new_state = system.step(state, B_plasma, geometry, t=0.0, dt=1e-6)

        assert new_state.I_pickup.shape == state.I_pickup.shape
        assert new_state.Q_pickup.shape == state.Q_pickup.shape
        assert new_state.Psi_pickup.shape == state.Psi_pickup.shape


class TestRLCircuitODE:
    """Tests for RL circuit ODE integration (no capacitor)."""

    def test_rl_decay_no_source(self, geometry):
        """Current decays exponentially with tau = L/R when no source."""
        from jax_frc.circuits.system import CircuitSystem

        # RL circuit: L = 1e-3 H, R = 1.0 Ohm => tau = 1e-3 s
        params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([1.0]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.0]),  # No load for pure RL
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        system = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        # Start with I = 1.0 A
        state = CircuitState(
            I_pickup=jnp.array([1.0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        # Evolve for one time constant (tau = 1e-3 s)
        tau = 1e-3
        dt = tau / 100  # Small steps for accuracy
        for _ in range(100):
            state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)

        # After one tau, current should be ~ exp(-1) * I0 = 0.368
        expected = jnp.exp(-1.0)
        assert jnp.isclose(state.I_pickup[0], expected, rtol=0.05)

    def test_rl_decay_with_load_resistance(self, geometry):
        """Effective resistance includes load: R_eff = R + R_load."""
        from jax_frc.circuits.system import CircuitSystem

        # RL circuit with R = 0.5, R_load = 0.5 => R_eff = 1.0, tau = L/R_eff = 1e-3
        params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([0.5]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.5]),  # R_eff = 0.5 + 0.5 = 1.0
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        system = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        state = CircuitState(
            I_pickup=jnp.array([1.0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        # Evolve for one time constant (tau = 1e-3 s)
        tau = 1e-3
        dt = tau / 100
        for _ in range(100):
            state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)

        expected = jnp.exp(-1.0)
        assert jnp.isclose(state.I_pickup[0], expected, rtol=0.05)


class TestInducedEMF:
    """Tests for EMF induced by changing plasma flux."""

    def test_induced_current_from_flux_change(self, geometry):
        """Changing flux induces current via V_ind = -N * dPsi/dt."""
        from jax_frc.circuits.system import CircuitSystem

        params = CircuitParams(
            L=jnp.array([1e-4]),  # Small L for fast response
            R=jnp.array([1.0]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([1]),  # N=1 for simpler calculation
            params=params,
            load_resistance=jnp.array([0.0]),
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        system = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        # Initial state with known Psi
        B0 = jnp.zeros((geometry.nr, geometry.nz, 3))
        B0 = B0.at[:, :, 2].set(0.1)  # Bz = 0.1 T

        Psi0 = pickup.compute_flux_linkages(B0, geometry)
        state = CircuitState(
            I_pickup=jnp.array([0.0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=Psi0,
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        # Now increase Bz => flux increases => negative dPsi/dt => positive induced current
        B1 = jnp.zeros((geometry.nr, geometry.nz, 3))
        B1 = B1.at[:, :, 2].set(0.2)  # Bz doubled

        dt = 1e-6
        new_state = system.step(state, B1, geometry, t=0.0, dt=dt)

        # Flux increased => Lenz's law => current should oppose change
        # V_ind = -dPsi/dt (positive flux increase => negative V_ind => current opposes)
        # But sign depends on convention - just verify current changed
        assert new_state.I_pickup[0] != 0.0


class TestExternalCircuitDriven:
    """Tests for externally driven circuits."""

    def test_voltage_driven_circuit(self, geometry):
        """Voltage source drives current in external circuit."""
        from jax_frc.circuits.system import CircuitSystem

        # External circuit with constant voltage source
        coil = CoilGeometry(z_center=0.0, radius=0.4, length=0.1, n_turns=50)
        circuit_params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([1.0]),
            C=jnp.array([jnp.inf]),
        )
        driver = CircuitDriver(mode="voltage", waveform=make_constant(10.0))
        ext_circuit = ExternalCircuit(
            name="test_coil",
            coil=coil,
            params=circuit_params,
            driver=driver,
        )
        external = ExternalCircuits(circuits=(ext_circuit,))

        # Empty pickup array
        pickup_params = CircuitParams(
            L=jnp.array([]),
            R=jnp.array([]),
            C=jnp.array([]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([]),
            radii=jnp.array([]),
            n_turns=jnp.array([]),
            params=pickup_params,
            load_resistance=jnp.array([]),
        )
        flux_coupling = FluxCoupling()

        system = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        state = CircuitState.zeros(n_pickup=0, n_external=1)
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        # Evolve to steady state: I = V/R = 10/1 = 10 A
        # Time constant tau = L/R = 1e-3 s
        tau = 1e-3
        dt = tau / 100
        for _ in range(500):  # 5 tau
            state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)

        expected_I = 10.0  # V/R
        assert jnp.isclose(state.I_external[0], expected_I, rtol=0.02)


class TestRLCCircuitODE:
    """Tests for RLC circuit ODE integration with capacitor."""

    def test_lc_oscillation(self, geometry):
        """LC circuit oscillates at f = 1/(2*pi*sqrt(LC))."""
        from jax_frc.circuits.system import CircuitSystem

        # LC circuit: L = 1e-3 H, C = 1e-6 F => f = 5033 Hz, T = 0.2 ms
        L = 1e-3
        C = 1e-6
        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([0.0]),  # No damping
            C=jnp.array([C]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.0]),
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        system = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        # Start with charge on capacitor => oscillates
        Q0 = 1e-3  # 1 mC
        state = CircuitState(
            I_pickup=jnp.array([0.0]),
            Q_pickup=jnp.array([Q0]),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        # Period T = 2*pi*sqrt(LC) = 2*pi*sqrt(1e-3 * 1e-6) = 2*pi * 1e-4.5 ~ 0.2 ms
        T = 2 * jnp.pi * jnp.sqrt(L * C)
        dt = T / 100  # 100 steps per period

        # After quarter period, current should be maximum
        for _ in range(25):
            state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)

        I_max = Q0 / jnp.sqrt(L * C)  # Maximum current in LC oscillation
        assert jnp.isclose(jnp.abs(state.I_pickup[0]), I_max, rtol=0.15)


class TestSubcycling:
    """Tests for subcycling with lax.scan."""

    def test_subcycling_activates_for_large_dt(self, geometry):
        """System subcycles when dt >> L/R timescale."""
        from jax_frc.circuits.system import CircuitSystem

        # Fast circuit: L/R = 1e-6 s
        params = CircuitParams(
            L=jnp.array([1e-6]),
            R=jnp.array([1.0]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.0]),
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        system = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        state = CircuitState(
            I_pickup=jnp.array([1.0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        # Large dt = 1e-4 s >> tau = 1e-6 s
        # Without subcycling this would be unstable/inaccurate
        dt = 1e-4
        new_state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)

        # After 100*tau, current should be essentially zero
        # exp(-100) ~ 0
        assert jnp.isclose(new_state.I_pickup[0], 0.0, atol=1e-10)


class TestPowerTracking:
    """Tests for power extraction and dissipation tracking."""

    def test_power_extracted_tracked(self, geometry):
        """P_extracted is computed from load resistance."""
        from jax_frc.circuits.system import CircuitSystem

        params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([0.5]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([1.0]),
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        system = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        # Start with I = 10 A
        state = CircuitState(
            I_pickup=jnp.array([10.0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        new_state = system.step(state, B_plasma, geometry, t=0.0, dt=1e-6)

        # P_extracted ~ I^2 * R_load (approximately, current decays during step)
        # For small dt, current is still ~10 A
        # P_extracted should be positive
        assert new_state.P_extracted > 0

    def test_power_dissipated_tracked(self, geometry):
        """P_dissipated is computed from circuit resistance."""
        from jax_frc.circuits.system import CircuitSystem

        params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([1.0]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.0]),
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        system = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        state = CircuitState(
            I_pickup=jnp.array([10.0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        new_state = system.step(state, B_plasma, geometry, t=0.0, dt=1e-6)

        # P_dissipated should be positive for non-zero current
        assert new_state.P_dissipated > 0


class TestJITCompatibility:
    """Tests for JAX JIT compatibility."""

    def test_step_is_jittable(self, simple_pickup, simple_external, flux_coupling, geometry):
        """step() can be JIT-compiled."""
        from jax_frc.circuits.system import CircuitSystem

        system = CircuitSystem(
            pickup=simple_pickup,
            external=simple_external,
            flux_coupling=flux_coupling,
        )

        @jax.jit
        def jit_step(state, B, t, dt):
            return system.step(state, B, geometry, t, dt)

        state = CircuitState.zeros(n_pickup=1, n_external=0)
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        # Should not raise
        new_state = jit_step(state, B_plasma, 0.0, 1e-6)
        assert isinstance(new_state, CircuitState)

    def test_vmap_over_states(self, simple_pickup, simple_external, flux_coupling, geometry):
        """step() can be vmapped over multiple states."""
        from jax_frc.circuits.system import CircuitSystem

        system = CircuitSystem(
            pickup=simple_pickup,
            external=simple_external,
            flux_coupling=flux_coupling,
        )

        def step_fn(state):
            B = jnp.zeros((geometry.nr, geometry.nz, 3))
            return system.step(state, B, geometry, t=0.0, dt=1e-6)

        # Create batch of states
        batch_states = CircuitState(
            I_pickup=jnp.array([[1.0], [2.0], [3.0]]),
            Q_pickup=jnp.zeros((3, 1)),
            I_external=jnp.zeros((3, 0)),
            Q_external=jnp.zeros((3, 0)),
            Psi_pickup=jnp.zeros((3, 1)),
            Psi_external=jnp.zeros((3, 0)),
            P_extracted=jnp.zeros(3),
            P_dissipated=jnp.zeros(3),
        )

        # vmap over batch dimension
        vmapped_step = jax.vmap(step_fn)
        new_states = vmapped_step(batch_states)

        assert new_states.I_pickup.shape == (3, 1)

    def test_step_in_lax_scan(self, simple_pickup, simple_external, flux_coupling, geometry):
        """step() works inside lax.scan for time evolution."""
        from jax import lax
        from jax_frc.circuits.system import CircuitSystem

        system = CircuitSystem(
            pickup=simple_pickup,
            external=simple_external,
            flux_coupling=flux_coupling,
        )
        initial_state = CircuitState.zeros(n_pickup=1, n_external=0)
        initial_state = initial_state.replace(I_pickup=jnp.array([1.0]))
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))
        dt = 1e-6

        def body(state, _):
            return system.step(state, B_plasma, geometry, t=0.0, dt=dt), None

        final_state, _ = lax.scan(body, initial_state, jnp.arange(100))

        # Current should have decayed
        assert final_state.I_pickup[0] < initial_state.I_pickup[0]


class TestFluxUpdate:
    """Tests for flux linkage state updates."""

    def test_psi_updated_after_step(
        self, simple_pickup, simple_external, flux_coupling, geometry
    ):
        """Psi_pickup is updated to current flux after step."""
        from jax_frc.circuits.system import CircuitSystem

        system = CircuitSystem(
            pickup=simple_pickup,
            external=simple_external,
            flux_coupling=flux_coupling,
        )

        # Initial state with Psi = 0
        state = CircuitState.zeros(n_pickup=1, n_external=0)

        # Apply non-zero B field
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))
        B_plasma = B_plasma.at[:, :, 2].set(0.5)

        new_state = system.step(state, B_plasma, geometry, t=0.0, dt=1e-6)

        # Psi should be updated to current flux linkage
        expected_psi = simple_pickup.compute_flux_linkages(B_plasma, geometry)
        assert jnp.allclose(new_state.Psi_pickup, expected_psi, rtol=0.01)
