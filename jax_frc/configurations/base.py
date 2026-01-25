"""Base class for all reactor/benchmark configurations."""
from abc import ABC, abstractmethod

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel


class AbstractConfiguration(ABC):
    """Base class for all reactor/benchmark configurations.

    Each configuration encapsulates a complete simulation setup including
    geometry, initial conditions, physics model, and boundary conditions.

    Subclasses must implement the four abstract methods to define:
    - Computational geometry (domain size, resolution)
    - Initial plasma state (fields, density, temperature)
    - Physics model (resistive MHD, extended MHD, hybrid kinetic)
    - Boundary conditions (conducting walls, symmetry, etc.)

    Example usage::

        class MyConfiguration(AbstractConfiguration):
            def build_geometry(self) -> Geometry:
                return Geometry(...)

            def build_initial_state(self, geometry: Geometry) -> State:
                return State.zeros(geometry.nr, geometry.nz)

            def build_model(self) -> PhysicsModel:
                return ResistiveMHD(...)

            def build_boundary_conditions(self) -> list:
                return [ConductingWall(), ...]
    """

    name: str = "abstract"
    description: str = "Base configuration class"

    @abstractmethod
    def build_geometry(self) -> Geometry:
        """Create computational geometry for this configuration.

        Returns:
            Geometry object defining the computational domain.
        """
        ...

    @abstractmethod
    def build_initial_state(self, geometry: Geometry) -> State:
        """Create initial plasma state.

        Args:
            geometry: The computational geometry.

        Returns:
            Initial State with all fields initialized.
        """
        ...

    @abstractmethod
    def build_model(self) -> PhysicsModel:
        """Create physics model for this configuration.

        Returns:
            PhysicsModel instance (e.g., ResistiveMHD, ExtendedMHD).
        """
        ...

    @abstractmethod
    def build_boundary_conditions(self) -> list:
        """Create boundary conditions for this configuration.

        Returns:
            List of BoundaryCondition objects to apply.
        """
        ...

    def available_phases(self) -> list[str]:
        """List valid phases for this configuration.

        Override in subclasses to define multi-phase simulations
        (e.g., formation, merging, sustainment).

        Returns:
            List of phase names. Default is ["default"].
        """
        return ["default"]

    def default_runtime(self) -> dict:
        """Return suggested runtime parameters.

        Override in subclasses to provide configuration-specific
        default values for simulation duration and timestep.

        Returns:
            Dictionary with runtime parameters (t_end, dt, etc.).
        """
        return {"t_end": 1e-3, "dt": 1e-6}
