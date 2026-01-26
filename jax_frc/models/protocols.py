"""Protocols for physics models supporting IMEX and source term coupling."""

from typing import Protocol, Tuple, Any
from jax import Array


class SplitRHS(Protocol):
    """Protocol for models supporting IMEX time integration.

    IMEX (Implicit-Explicit) methods split the right-hand side into:
    - Explicit terms: Non-stiff, can use explicit time stepping
    - Implicit terms: Stiff, require implicit treatment for stability

    Models implementing this protocol can be used with IMEX integrators
    like ARK methods or IMEX-RK schemes.
    """

    def explicit_rhs(self, state: Any, geometry: Any, t: float) -> Any:
        """Compute terms safe for explicit integration.

        Args:
            state: Current state of the system
            geometry: Grid/mesh geometry information
            t: Current simulation time

        Returns:
            Right-hand side contribution for explicit treatment
        """
        ...

    def implicit_rhs(self, state: Any, geometry: Any, t: float) -> Any:
        """Compute stiff terms needing implicit treatment.

        Args:
            state: Current state of the system
            geometry: Grid/mesh geometry information
            t: Current simulation time

        Returns:
            Right-hand side contribution for implicit treatment
        """
        ...

    def apply_implicit_operator(
        self, state: Any, geometry: Any, dt: float, theta: float
    ) -> Any:
        """Apply (I - theta*dt*L) for implicit solve.

        This implements the implicit operator for theta-method or
        similar implicit time integration schemes.

        Args:
            state: Current state of the system
            geometry: Grid/mesh geometry information
            dt: Time step size
            theta: Implicitness parameter (0=explicit, 1=fully implicit)

        Returns:
            Result of applying the implicit operator to state
        """
        ...


class SourceTerms(Protocol):
    """Protocol for atomic/collision source terms coupling two fluids.

    This protocol defines the interface for computing source terms
    that couple plasma and neutral fluids, such as:
    - Ionization
    - Recombination
    - Charge exchange
    - Momentum/energy transfer

    Implementations should return conservative source terms that
    preserve total particle count, momentum, and energy.
    """

    def compute_sources(
        self, plasma_state: Any, neutral_state: Any, geometry: Any
    ) -> Tuple[Any, Any]:
        """Compute source terms for both plasma and neutral fluids.

        Args:
            plasma_state: Current state of the plasma fluid
            neutral_state: Current state of the neutral fluid
            geometry: Grid/mesh geometry information

        Returns:
            Tuple of (plasma_sources, neutral_sources) where each
            contains the source terms to add to the respective fluid's
            time derivative.
        """
        ...
