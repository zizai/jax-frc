"""Base class for boundary conditions."""

from abc import ABC, abstractmethod
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

class BoundaryCondition(ABC):
    """Base class for boundary conditions."""

    @abstractmethod
    def apply(self, state: State, geometry: Geometry) -> State:
        """Apply boundary condition to state."""
        pass
