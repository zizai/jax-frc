"""CircuitSystem - orchestrates circuit coupling with ODE integration.

This module provides the main CircuitSystem class that integrates RLC circuit
ODEs using subcycling for numerical stability when circuit timescales (L/R)
are faster than the MHD timestep.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array, lax

from jax_frc.circuits.state import CircuitState
from jax_frc.circuits.pickup import PickupCoilArray
from jax_frc.circuits.external import ExternalCircuits
from jax_frc.circuits.coupling import FluxCoupling
from jax_frc.core.geometry import Geometry


# Maximum number of substeps for subcycling (static for JIT stability)
MAX_SUBSTEPS = 100


@dataclass(frozen=True)
class CircuitSystem:
    """Complete circuit system integrated with MHD solver.

    Orchestrates:
    - Pickup coil energy extraction
    - External driven circuits
    - Flux coupling between plasma and circuits
    - RLC circuit ODE integration with subcycling

    Attributes:
        pickup: Array of pickup coils for energy extraction
        external: Collection of external driven circuits
        flux_coupling: Computes flux linkages between plasma and circuits
    """

    pickup: PickupCoilArray
    external: ExternalCircuits
    flux_coupling: FluxCoupling

    def step(
        self,
        circuit_state: CircuitState,
        B_plasma: Array,
        geometry: Geometry,
        t: float,
        dt: float,
    ) -> CircuitState:
        """Advance all circuits by dt.

        Uses same dt as MHD for synchronization.
        Circuit timescales (L/R) are typically faster than MHD,
        so we subcycle if needed using lax.scan.

        Args:
            circuit_state: Current circuit state
            B_plasma: Plasma magnetic field (nr, nz, 3)
            geometry: Computational geometry
            t: Current simulation time [s]
            dt: Time step [s]

        Returns:
            Updated circuit state
        """
        # Compute current flux linkages
        Psi_pickup_new, Psi_external_new = self.flux_coupling.plasma_to_coils(
            B_plasma, geometry, self.pickup, self.external
        )

        # Compute induced EMF from flux change: V_ind = -dPsi/dt
        # (The n_turns factor is already included in flux linkage computation)
        dPsi_pickup_dt = (Psi_pickup_new - circuit_state.Psi_pickup) / dt
        dPsi_external_dt = (Psi_external_new - circuit_state.Psi_external) / dt

        # Get voltage sources for external circuits
        V_source_external = self._get_external_voltages(t, circuit_state)

        # Subcycle the circuit ODE integration
        new_state = self._subcycle_step(
            circuit_state,
            dPsi_pickup_dt,
            dPsi_external_dt,
            V_source_external,
            dt,
        )

        # Update flux linkages to current values
        new_state = new_state.replace(
            Psi_pickup=Psi_pickup_new,
            Psi_external=Psi_external_new,
        )

        # Compute instantaneous power
        P_extracted, P_dissipated = self._compute_power(new_state)
        new_state = new_state.replace(
            P_extracted=P_extracted,
            P_dissipated=P_dissipated,
        )

        return new_state

    def _get_external_voltages(
        self, t: float, circuit_state: CircuitState
    ) -> Array:
        """Get voltage source values for external circuits.

        Args:
            t: Current time [s]
            circuit_state: Current state (for feedback modes)

        Returns:
            V_source: Voltage for each external circuit [V]
        """
        if self.external.n_circuits == 0:
            return jnp.array([])

        # For now, we don't track error integral for feedback
        # This could be added to CircuitState if needed
        error_integral = 0.0

        voltages = []
        for circuit in self.external.circuits:
            V = circuit.driver.get_voltage(t, circuit_state, error_integral)
            voltages.append(V)

        return jnp.array(voltages)

    def _subcycle_step(
        self,
        state: CircuitState,
        dPsi_pickup_dt: Array,
        dPsi_external_dt: Array,
        V_source_external: Array,
        dt: float,
    ) -> CircuitState:
        """Subcycle circuit ODE integration for numerical stability.

        Estimates circuit timescale tau = L/R and subdivides dt if needed
        to resolve the fastest timescale.

        Uses MAX_SUBSTEPS iterations for JIT compatibility (lax.scan requires
        static length), but computes n_sub dynamically based on L/R timescale.
        Only the first n_sub iterations perform physics updates (with dt_sub = dt / n_sub),
        while remaining iterations are effectively skipped.

        Args:
            state: Current circuit state
            dPsi_pickup_dt: Rate of flux change through pickup coils [Wb/s]
            dPsi_external_dt: Rate of flux change through external coils [Wb/s]
            V_source_external: External voltage sources [V]
            dt: Total time step [s]

        Returns:
            State after dt
        """
        # Compute number of substeps based on L/R timescale
        # We want dt_sub <= 0.1 * tau_min for stability
        tau_min = self._estimate_min_timescale()
        n_sub = jnp.clip(jnp.ceil(dt / (0.1 * tau_min)), 1, MAX_SUBSTEPS)
        dt_sub = dt / n_sub

        # Pack all inputs needed for ODE step
        carry = (
            state.I_pickup,
            state.Q_pickup,
            state.I_external,
            state.Q_external,
        )

        # Closure over constant values
        def body(carry, i):
            I_pickup, Q_pickup, I_external, Q_external = carry

            # Only apply dt_sub for iterations < n_sub, else dt_effective = 0
            dt_effective = jnp.where(i < n_sub, dt_sub, 0.0)

            # Compute ODE step
            new_I_pickup, new_Q_pickup = self._ode_step_pickup(
                I_pickup, Q_pickup, dPsi_pickup_dt, dt_effective
            )
            new_I_external, new_Q_external = self._ode_step_external(
                I_external, Q_external, dPsi_external_dt, V_source_external, dt_effective
            )

            return (new_I_pickup, new_Q_pickup, new_I_external, new_Q_external), None

        # Run scan over MAX_SUBSTEPS with iteration indices
        final_carry, _ = lax.scan(body, carry, jnp.arange(MAX_SUBSTEPS))

        I_pickup, Q_pickup, I_external, Q_external = final_carry

        return state.replace(
            I_pickup=I_pickup,
            Q_pickup=Q_pickup,
            I_external=I_external,
            Q_external=Q_external,
        )

    def _estimate_min_timescale(self) -> float:
        """Estimate minimum L/R timescale across all circuits.

        Returns:
            Minimum tau = L/R [s], or inf if no circuits
        """
        tau_values = []

        # Pickup coils: effective R = R + load_resistance
        if self.pickup.n_coils > 0:
            R_eff = self.pickup.params.R + self.pickup.load_resistance
            # Avoid division by zero
            R_eff_safe = jnp.maximum(R_eff, 1e-10)
            tau_pickup = self.pickup.params.L / R_eff_safe
            tau_values.append(jnp.min(tau_pickup))

        # External circuits
        if self.external.n_circuits > 0:
            params = self.external.get_combined_params()
            R_safe = jnp.maximum(params.R, 1e-10)
            tau_external = params.L / R_safe
            tau_values.append(jnp.min(tau_external))

        if tau_values:
            return jnp.min(jnp.array(tau_values))
        else:
            return jnp.inf

    def _ode_step_pickup(
        self,
        I: Array,
        Q: Array,
        dPsi_dt: Array,
        dt: float,
    ) -> tuple[Array, Array]:
        """Single ODE step for pickup coil circuits.

        RLC circuit equation:
            L dI/dt + R_eff*I + Q/C = -dPsi/dt
            dQ/dt = I

        For pickup coils: V_source = 0, R_eff = R + load_resistance

        Uses implicit midpoint method for stability.

        Args:
            I: Current currents [A]
            Q: Current charges [C]
            dPsi_dt: Rate of flux change [Wb/s]
            dt: Time step [s]

        Returns:
            (new_I, new_Q): Updated currents and charges
        """
        if self.pickup.n_coils == 0:
            return I, Q

        L = self.pickup.params.L
        R_eff = self.pickup.params.R + self.pickup.load_resistance
        C = self.pickup.params.C

        # Source term: V = -dPsi/dt (induced EMF)
        V_ind = -dPsi_dt

        # Implicit midpoint: solve for I_new, Q_new
        # I_mid = (I + I_new) / 2
        # Q_mid = (Q + Q_new) / 2
        #
        # L * (I_new - I) / dt + R_eff * I_mid + Q_mid / C = V_ind
        # (Q_new - Q) / dt = I_mid
        #
        # From second equation: Q_new = Q + dt * I_mid = Q + dt * (I + I_new) / 2
        # Q_mid = (Q + Q_new) / 2 = Q + dt * (I + I_new) / 4
        #
        # Substituting into first equation:
        # L * (I_new - I) / dt + R_eff * (I + I_new) / 2 + (Q + dt * (I + I_new) / 4) / C = V_ind
        #
        # Rearranging for I_new:
        # I_new * (L/dt + R_eff/2 + dt/(4*C)) = V_ind - R_eff*I/2 - Q/C + L*I/dt - dt*I/(4*C)

        # Handle C = inf (no capacitor) case
        inv_C = jnp.where(jnp.isinf(C), 0.0, 1.0 / C)

        # Use safe dt to avoid division by zero; result is masked below anyway
        dt_safe = jnp.maximum(dt, 1e-30)

        # Coefficients
        a = L / dt_safe + R_eff / 2 + dt_safe * inv_C / 4
        b = V_ind - R_eff * I / 2 - Q * inv_C + L * I / dt_safe - dt_safe * I * inv_C / 4

        # Solve for I_new
        I_new_computed = b / a

        # Update Q
        I_mid = (I + I_new_computed) / 2
        Q_new_computed = Q + dt_safe * I_mid

        # If dt == 0, return unchanged values (skip this iteration)
        I_new = jnp.where(dt > 0, I_new_computed, I)
        Q_new = jnp.where(dt > 0, Q_new_computed, Q)

        return I_new, Q_new

    def _ode_step_external(
        self,
        I: Array,
        Q: Array,
        dPsi_dt: Array,
        V_source: Array,
        dt: float,
    ) -> tuple[Array, Array]:
        """Single ODE step for external circuits.

        RLC circuit equation:
            L dI/dt + R*I + Q/C = V_source - dPsi/dt
            dQ/dt = I

        Uses implicit midpoint method for stability.

        Args:
            I: Current currents [A]
            Q: Current charges [C]
            dPsi_dt: Rate of flux change [Wb/s]
            V_source: Applied voltage [V]
            dt: Time step [s]

        Returns:
            (new_I, new_Q): Updated currents and charges
        """
        if self.external.n_circuits == 0:
            return I, Q

        params = self.external.get_combined_params()
        L = params.L
        R = params.R
        C = params.C

        # Total voltage: V_source - induced EMF
        V_total = V_source - dPsi_dt

        # Handle C = inf (no capacitor) case
        inv_C = jnp.where(jnp.isinf(C), 0.0, 1.0 / C)

        # Use safe dt to avoid division by zero; result is masked below anyway
        dt_safe = jnp.maximum(dt, 1e-30)

        # Implicit midpoint coefficients (same derivation as pickup)
        a = L / dt_safe + R / 2 + dt_safe * inv_C / 4
        b = V_total - R * I / 2 - Q * inv_C + L * I / dt_safe - dt_safe * I * inv_C / 4

        # Solve for I_new
        I_new_computed = b / a

        # Update Q
        I_mid = (I + I_new_computed) / 2
        Q_new_computed = Q + dt_safe * I_mid

        # If dt == 0, return unchanged values (skip this iteration)
        I_new = jnp.where(dt > 0, I_new_computed, I)
        Q_new = jnp.where(dt > 0, Q_new_computed, Q)

        return I_new, Q_new

    def _compute_power(self, state: CircuitState) -> tuple[float, float]:
        """Compute instantaneous power extracted and dissipated.

        Args:
            state: Current circuit state

        Returns:
            (P_extracted, P_dissipated): Powers in Watts
        """
        P_extracted = 0.0
        P_dissipated = 0.0

        # Pickup coils
        if self.pickup.n_coils > 0:
            P_load, P_diss = self.pickup.compute_power(state.I_pickup)
            P_extracted = P_extracted + jnp.sum(P_load)
            P_dissipated = P_dissipated + jnp.sum(P_diss)

        # External circuits
        if self.external.n_circuits > 0:
            params = self.external.get_combined_params()
            P_diss_ext = state.I_external**2 * params.R
            P_dissipated = P_dissipated + jnp.sum(P_diss_ext)

        return P_extracted, P_dissipated


# Register CircuitSystem as JAX pytree
def _circuit_system_flatten(system):
    # All three attributes are pytrees
    children = (system.pickup, system.external, system.flux_coupling)
    aux_data = None
    return children, aux_data


def _circuit_system_unflatten(aux_data, children):
    pickup, external, flux_coupling = children
    return CircuitSystem(
        pickup=pickup,
        external=external,
        flux_coupling=flux_coupling,
    )


jax.tree_util.register_pytree_node(
    CircuitSystem,
    _circuit_system_flatten,
    _circuit_system_unflatten,
)
