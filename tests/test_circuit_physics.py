"""Physics invariant tests for circuit coupling.

Tests fundamental physics conservation laws:
1. Energy conservation: P_extracted + P_dissipated = -d(E_magnetic)/dt
2. Flux conservation: In superconducting limit (R→0), total flux is constant
3. Backward compatibility: Single pickup coil with R-only should approximate DirectConversion
"""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.circuits import CircuitParams, CircuitState
from jax_frc.circuits.pickup import PickupCoilArray
from jax_frc.circuits.external import ExternalCircuits
from jax_frc.circuits.coupling import FluxCoupling
from jax_frc.circuits.system import CircuitSystem
from jax_frc.core.geometry import Geometry

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def geometry():
    """Standard geometry for physics tests."""
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.5,
        z_min=-1.0,
        z_max=1.0,
    )


class TestEnergyConservation:
    """Tests verifying energy conservation in circuit dynamics.

    Physics: P_extracted + P_dissipated = -d(E_magnetic)/dt

    The power delivered to loads plus power dissipated in resistance should
    equal the rate of change of magnetic energy in the circuits.
    For an inductor: E_magnetic = 0.5 * L * I^2
    """

    def test_energy_conservation_rl_decay(self, geometry):
        """Energy conservation during RL decay: power dissipated equals energy lost.

        For pure RL decay (no external source, no load):
        - E_initial = 0.5 * L * I0^2
        - All energy dissipated as heat: integral(P_dissipated * dt) = E_initial

        We verify instantaneous energy balance: P_dissipated ≈ -dE/dt
        """
        # Setup RL circuit: L = 1e-3 H, R = 1.0 Ohm
        L = 1e-3
        R = 1.0
        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([R]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.0]),  # No load, pure dissipation
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        system = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        # Initial current I0 = 10 A
        I0 = 10.0
        state = CircuitState(
            I_pickup=jnp.array([I0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        # Small timestep for accuracy
        dt = 1e-5

        # Compute initial magnetic energy
        E_before = 0.5 * L * state.I_pickup[0]**2

        # Step forward
        new_state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)

        # Compute final magnetic energy
        E_after = 0.5 * L * new_state.I_pickup[0]**2

        # Rate of energy change
        dE_dt = (E_after - E_before) / dt

        # For RL decay, P_dissipated should equal -dE/dt
        # (energy leaves the inductor and goes to heat)
        # Note: P_dissipated is instantaneous power at end of step
        # For better accuracy, use average current during step
        I_avg = (state.I_pickup[0] + new_state.I_pickup[0]) / 2
        P_dissipated_expected = I_avg**2 * R

        # Energy conservation: P_dissipated ≈ -dE/dt
        assert jnp.isclose(-dE_dt, P_dissipated_expected, rtol=0.05), (
            f"Energy not conserved: -dE/dt={-dE_dt:.4f}, P_dissipated={P_dissipated_expected:.4f}"
        )

    def test_energy_conservation_with_load(self, geometry):
        """Energy conservation with load resistance: P_load + P_diss = -dE/dt.

        With both coil resistance and load resistance, total power extracted
        from the inductor goes to both dissipation and useful load.
        """
        L = 1e-3
        R_coil = 0.5
        R_load = 0.5
        R_total = R_coil + R_load

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([R_coil]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([R_load]),
        )
        external = ExternalCircuits(circuits=())
        flux_coupling = FluxCoupling()

        system = CircuitSystem(
            pickup=pickup,
            external=external,
            flux_coupling=flux_coupling,
        )

        I0 = 10.0
        state = CircuitState(
            I_pickup=jnp.array([I0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        dt = 1e-5
        E_before = 0.5 * L * state.I_pickup[0]**2

        new_state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)

        E_after = 0.5 * L * new_state.I_pickup[0]**2
        dE_dt = (E_after - E_before) / dt

        # Total power = P_extracted (to load) + P_dissipated (to heat)
        I_avg = (state.I_pickup[0] + new_state.I_pickup[0]) / 2
        P_total_expected = I_avg**2 * R_total

        assert jnp.isclose(-dE_dt, P_total_expected, rtol=0.05), (
            f"Energy not conserved: -dE/dt={-dE_dt:.4f}, P_total={P_total_expected:.4f}"
        )

    def test_total_energy_dissipated_matches_initial(self, geometry):
        """Integrate over full decay: total energy dissipated equals initial energy.

        For RL decay to steady state, all initial magnetic energy should be
        dissipated as heat in the resistor.
        """
        L = 1e-3
        R = 1.0
        tau = L / R  # Time constant

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([R]),
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

        I0 = 10.0
        E_initial = 0.5 * L * I0**2

        state = CircuitState(
            I_pickup=jnp.array([I0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))

        # Integrate over 5 time constants (current decays to ~0.7% of initial)
        dt = tau / 50
        n_steps = int(5 * tau / dt)

        total_energy_dissipated = 0.0
        for _ in range(n_steps):
            I_before = state.I_pickup[0]
            new_state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)
            I_after = new_state.I_pickup[0]

            # Trapezoidal integration: P = I^2 * R
            P_avg = 0.5 * (I_before**2 + I_after**2) * R
            total_energy_dissipated += P_avg * dt

            state = new_state

        # Final magnetic energy should be nearly zero
        E_final = 0.5 * L * state.I_pickup[0]**2

        # Energy conservation: E_initial = E_final + E_dissipated
        E_conserved = E_final + total_energy_dissipated
        assert jnp.isclose(E_conserved, E_initial, rtol=0.02), (
            f"Energy not conserved: E_initial={E_initial:.6f}, "
            f"E_final+E_dissipated={E_conserved:.6f}"
        )


class TestFluxConservation:
    """Tests verifying flux conservation in superconducting limit.

    Physics: When R→0, total flux through circuit is conserved.
    Ψ_total = L*I + Ψ_external remains constant.

    This is the fundamental principle behind flux trapping in
    superconducting circuits.
    """

    def test_flux_conservation_zero_resistance(self, geometry):
        """With R=0, total flux Ψ_total = L*I + Ψ_plasma is conserved.

        When external flux changes, the circuit current adjusts to maintain
        constant total flux through the circuit.
        """
        L = 1e-3
        R = 1e-15  # Effectively zero resistance

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([R]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([1]),  # N=1 for simpler flux calculation
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

        # Initial state: some current, zero external flux
        I0 = 5.0
        state = CircuitState(
            I_pickup=jnp.array([I0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),  # Will be updated on first step
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        # Initial total flux (self-inductance only)
        Psi_total_initial = L * I0

        # Create changing plasma B-field (flux increases)
        B_plasma_0 = jnp.zeros((geometry.nr, geometry.nz, 3))

        # First step to establish initial Psi_pickup
        dt = 1e-5
        state = system.step(state, B_plasma_0, geometry, t=0.0, dt=dt)

        # Record initial state after flux is set
        I_initial = float(state.I_pickup[0])
        Psi_plasma_initial = float(state.Psi_pickup[0])
        Psi_total_initial = L * I_initial + Psi_plasma_initial

        # Now apply a changing B-field (increasing flux)
        B_plasma_1 = jnp.zeros((geometry.nr, geometry.nz, 3))
        B_plasma_1 = B_plasma_1.at[:, :, 2].set(0.1)  # Bz = 0.1 T

        # Step forward with new field
        new_state = system.step(state, B_plasma_1, geometry, t=dt, dt=dt)

        # Compute new total flux
        I_final = float(new_state.I_pickup[0])
        Psi_plasma_final = float(new_state.Psi_pickup[0])
        Psi_total_final = L * I_final + Psi_plasma_final

        # For superconducting circuit, total flux should be conserved
        # Note: Current should decrease to oppose the flux increase
        assert jnp.isclose(Psi_total_final, Psi_total_initial, rtol=0.05), (
            f"Flux not conserved: Psi_initial={Psi_total_initial:.6f}, "
            f"Psi_final={Psi_total_final:.6f}"
        )

    def test_flux_trapping_multiple_steps(self, geometry):
        """Flux conservation holds over multiple time steps.

        Verify that even with many steps and changing external flux,
        the superconducting circuit maintains constant total flux.
        """
        L = 1e-3
        R = 1e-15  # Effectively superconducting

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([R]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([1]),
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

        I0 = 5.0
        state = CircuitState(
            I_pickup=jnp.array([I0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        # Initialize with zero field
        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))
        dt = 1e-5

        # First step to establish Psi
        state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)

        # Record initial total flux
        Psi_total_initial = L * float(state.I_pickup[0]) + float(state.Psi_pickup[0])

        # Apply gradually increasing B-field over multiple steps
        n_steps = 50
        for i in range(n_steps):
            # Ramp up Bz
            Bz = 0.1 * (i + 1) / n_steps
            B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))
            B_plasma = B_plasma.at[:, :, 2].set(Bz)

            state = system.step(state, B_plasma, geometry, t=i*dt, dt=dt)

        # Final total flux should still match initial
        Psi_total_final = L * float(state.I_pickup[0]) + float(state.Psi_pickup[0])

        assert jnp.isclose(Psi_total_final, Psi_total_initial, rtol=0.05), (
            f"Flux not conserved over {n_steps} steps: "
            f"Psi_initial={Psi_total_initial:.6f}, Psi_final={Psi_total_final:.6f}"
        )

    def test_flux_decays_with_finite_resistance(self, geometry):
        """With finite R, flux decays exponentially (contrast to R=0 case).

        This confirms the flux conservation test is meaningful by showing
        that with normal resistance, total flux does decay.
        """
        L = 1e-3
        R = 1.0  # Finite resistance
        tau = L / R

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([R]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([1]),
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

        I0 = 5.0
        state = CircuitState(
            I_pickup=jnp.array([I0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        B_plasma = jnp.zeros((geometry.nr, geometry.nz, 3))
        dt = tau / 50

        # Initialize Psi
        state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)
        Psi_total_initial = L * float(state.I_pickup[0]) + float(state.Psi_pickup[0])

        # Evolve for one time constant
        for _ in range(50):
            state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)

        Psi_total_final = L * float(state.I_pickup[0]) + float(state.Psi_pickup[0])

        # With finite R, flux (current) should have decayed significantly
        # After 1 tau, current ~ 37% of initial
        assert Psi_total_final < 0.5 * Psi_total_initial, (
            f"Flux should decay with finite R: "
            f"Psi_final={Psi_total_final:.6f} should be < 0.5*{Psi_total_initial:.6f}"
        )


class TestBackwardCompatibility:
    """Tests for backward compatibility with DirectConversion behavior.

    A single pickup coil with R-only (no capacitor, no load) should
    approximately reproduce the power extraction behavior of the old
    DirectConversion module.
    """

    def test_power_scales_with_flux_change_rate(self, geometry):
        """Power extracted should scale with (dPsi/dt)^2 like DirectConversion.

        P = V^2/R where V = N * dPsi/dt, so P ~ (dPsi/dt)^2
        """
        L = 1e-4  # Small L for fast response
        R = 1.0
        N = 100  # Turns

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([R]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([N]),
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

        # Initial field
        B0 = jnp.zeros((geometry.nr, geometry.nz, 3))
        B0 = B0.at[:, :, 2].set(1.0)

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

        # Test 1: 10% flux decrease
        B1 = jnp.zeros((geometry.nr, geometry.nz, 3))
        B1 = B1.at[:, :, 2].set(0.9)  # 10% decrease

        dt = 1e-6
        state1 = system.step(state, B1, geometry, t=0.0, dt=dt)

        # Run a few more steps to let current build up
        for _ in range(10):
            state1 = system.step(state1, B1, geometry, t=0.0, dt=dt)
        P1 = state1.P_dissipated

        # Test 2: 20% flux decrease (double the rate)
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

        B2 = jnp.zeros((geometry.nr, geometry.nz, 3))
        B2 = B2.at[:, :, 2].set(0.8)  # 20% decrease

        state2 = system.step(state, B2, geometry, t=0.0, dt=dt)
        for _ in range(10):
            state2 = system.step(state2, B2, geometry, t=0.0, dt=dt)
        P2 = state2.P_dissipated

        # Power should scale roughly as (dPsi/dt)^2, so P2 ~ 4*P1
        # (allowing some tolerance due to circuit dynamics)
        ratio = P2 / P1
        assert 2.5 < ratio < 6.0, (
            f"Power scaling incorrect: P2/P1={ratio:.2f}, expected ~4.0"
        )

    def test_power_scales_with_turns_squared(self, geometry):
        """Power should scale with N^2 (V ~ N, P ~ V^2 ~ N^2).

        This matches DirectConversion behavior where P = V^2/(4R) and V ~ N.
        """
        L = 1e-4
        R = 1.0

        def measure_power(n_turns):
            params = CircuitParams(
                L=jnp.array([L]),
                R=jnp.array([R]),
                C=jnp.array([jnp.inf]),
            )
            pickup = PickupCoilArray(
                z_positions=jnp.array([0.0]),
                radii=jnp.array([0.3]),
                n_turns=jnp.array([n_turns]),
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

            B0 = jnp.zeros((geometry.nr, geometry.nz, 3))
            B0 = B0.at[:, :, 2].set(1.0)

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

            B1 = jnp.zeros((geometry.nr, geometry.nz, 3))
            B1 = B1.at[:, :, 2].set(0.9)

            dt = 1e-6
            for _ in range(10):
                state = system.step(state, B1, geometry, t=0.0, dt=dt)

            return state.P_dissipated

        P_100 = measure_power(100)
        P_200 = measure_power(200)

        # P should scale as N^2, so P_200/P_100 ~ 4
        ratio = P_200 / P_100
        assert jnp.isclose(ratio, 4.0, rtol=0.2), (
            f"Power should scale as N^2: P_200/P_100={ratio:.2f}, expected 4.0"
        )

    def test_steady_state_current_matches_ohms_law(self, geometry):
        """In steady state with constant dPsi/dt, I = V_ind/R per Ohm's law.

        When flux changes at a constant rate, the induced EMF is constant,
        and the steady-state current is V_ind/R (for RL circuit with L/R << t).
        """
        L = 1e-6  # Very small L for fast settling
        R = 1.0
        N = 100

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([R]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.3]),
            n_turns=jnp.array([N]),
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

        # Create linearly decreasing B field (constant dB/dt)
        dt = 1e-6
        n_steps = 100

        state = CircuitState(
            I_pickup=jnp.array([0.0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.zeros(0),
            Q_external=jnp.zeros(0),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.zeros(0),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        # Initialize with B0 = 1.0 T
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)
        state = system.step(state, B, geometry, t=0.0, dt=dt)

        # Apply constant dB/dt for many steps
        # Use a moderate rate that gives reasonable EMF
        dB_dt = -1e4  # -10,000 T/s
        for i in range(n_steps):
            # Track Psi from previous step for accurate dPsi/dt
            Psi_prev = float(state.Psi_pickup[0])

            Bz = 1.0 + dB_dt * (i + 1) * dt
            B = jnp.zeros((geometry.nr, geometry.nz, 3))
            B = B.at[:, :, 2].set(Bz)

            state = system.step(state, B, geometry, t=i*dt, dt=dt)

        # Get the instantaneous dPsi/dt from the last step
        Psi_now = float(state.Psi_pickup[0])
        dPsi_dt_measured = (Psi_now - Psi_prev) / dt

        # Steady state current should be V_ind/R = -dPsi/dt / R
        # V_ind = -dPsi/dt (n_turns already in Psi from compute_flux_linkages)
        V_ind_expected = -dPsi_dt_measured

        # For small L, current should have settled to V/R
        I_expected = V_ind_expected / R
        I_actual = float(state.I_pickup[0])

        # With constant dPsi/dt and L << R*dt, current should approach V/R
        # Allow 30% tolerance due to L/R dynamics
        assert jnp.isclose(I_actual, I_expected, rtol=0.3), (
            f"Current not matching Ohm's law: I_actual={I_actual:.3f}, "
            f"I_expected (V/R)={I_expected:.3f}"
        )


class TestLCOscillationEnergy:
    """Tests for energy conservation in LC oscillations.

    For undamped LC circuit, total energy E = 0.5*L*I^2 + 0.5*Q^2/C
    should remain constant.
    """

    def test_lc_energy_conservation(self, geometry):
        """Total energy in LC circuit is conserved (no damping)."""
        L = 1e-3
        C = 1e-6
        R = 0.0  # No damping

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([R]),
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

        # Initial energy in capacitor
        Q0 = 1e-3  # 1 mC
        E_initial = 0.5 * Q0**2 / C

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

        # Oscillation period
        T = 2 * jnp.pi * jnp.sqrt(L * C)
        dt = T / 100

        # Evolve for one full period
        energies = []
        for _ in range(100):
            state = system.step(state, B_plasma, geometry, t=0.0, dt=dt)
            E_L = 0.5 * L * state.I_pickup[0]**2
            E_C = 0.5 * state.Q_pickup[0]**2 / C
            E_total = E_L + E_C
            energies.append(float(E_total))

        # Energy should be conserved within tolerance
        energies = jnp.array(energies)
        E_mean = jnp.mean(energies)
        E_std = jnp.std(energies)

        # Standard deviation should be small relative to mean
        assert E_std / E_mean < 0.05, (
            f"Energy not conserved in LC oscillation: "
            f"std/mean = {E_std/E_mean:.3f}"
        )

        # Mean should be close to initial
        assert jnp.isclose(E_mean, E_initial, rtol=0.05), (
            f"Mean energy differs from initial: "
            f"E_mean={E_mean:.6f}, E_initial={E_initial:.6f}"
        )
