"""Circuit state and parameter dataclasses."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class CircuitParams:
    """Parameters for a set of RLC circuits.

    Attributes:
        L: Inductance [H], shape (n_coils,)
        R: Resistance [Ω], shape (n_coils,)
        C: Capacitance [F], shape (n_coils,) - use jnp.inf for no capacitor
    """

    L: Array
    R: Array
    C: Array


@dataclass(frozen=True)
class CircuitState:
    """State of all circuits in the system.

    Attributes:
        I_pickup: Pickup coil currents [A], shape (n_pickup,)
        Q_pickup: Pickup capacitor charges [C], shape (n_pickup,)
        I_external: External coil currents [A], shape (n_external,)
        Q_external: External capacitor charges [C], shape (n_external,)
        Psi_pickup: Flux linkage through pickup coils [Wb], shape (n_pickup,)
        Psi_external: Flux linkage through external coils [Wb], shape (n_external,)
        P_extracted: Total power to loads [W]
        P_dissipated: Total power dissipated in resistance [W]
    """

    I_pickup: Array
    Q_pickup: Array
    I_external: Array
    Q_external: Array
    Psi_pickup: Array
    Psi_external: Array
    P_extracted: float
    P_dissipated: float

    def replace(self, **kwargs) -> "CircuitState":
        """Return new state with specified fields replaced."""
        from dataclasses import replace as dc_replace

        return dc_replace(self, **kwargs)

    @classmethod
    def zeros(cls, n_pickup: int, n_external: int) -> "CircuitState":
        """Create zero-initialized circuit state."""
        return cls(
            I_pickup=jnp.zeros(n_pickup),
            Q_pickup=jnp.zeros(n_pickup),
            I_external=jnp.zeros(n_external),
            Q_external=jnp.zeros(n_external),
            Psi_pickup=jnp.zeros(n_pickup),
            Psi_external=jnp.zeros(n_external),
            P_extracted=0.0,
            P_dissipated=0.0,
        )


# Register as JAX pytree
def _circuit_state_flatten(state):
    children = (
        state.I_pickup,
        state.Q_pickup,
        state.I_external,
        state.Q_external,
        state.Psi_pickup,
        state.Psi_external,
        state.P_extracted,
        state.P_dissipated,
    )
    aux_data = None
    return children, aux_data


def _circuit_state_unflatten(aux_data, children):
    return CircuitState(*children)


jax.tree_util.register_pytree_node(
    CircuitState,
    _circuit_state_flatten,
    _circuit_state_unflatten,
)
