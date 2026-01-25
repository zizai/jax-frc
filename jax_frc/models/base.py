"""Abstract base class for physics models."""

from abc import ABC, abstractmethod
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

class PhysicsModel(ABC):
    """Base class for all physics models."""

    @abstractmethod
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for all evolved quantities."""
        pass

    @abstractmethod
    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Return CFL-stable timestep for this model."""
        pass

    @abstractmethod
    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Enforce physical constraints (e.g., div(B)=0)."""
        pass

    @classmethod
    def create(cls, config: dict) -> "PhysicsModel":
        """Factory method to create model from config."""
        model_type = config.get("type", "resistive_mhd")
        if model_type == "resistive_mhd":
            from jax_frc.models.resistive_mhd import ResistiveMHD
            return ResistiveMHD.from_config(config)
        elif model_type == "extended_mhd":
            from jax_frc.models.extended_mhd import ExtendedMHD
            return ExtendedMHD.from_config(config)
        elif model_type == "hybrid_kinetic":
            from jax_frc.models.hybrid_kinetic import HybridKinetic
            return HybridKinetic.from_config(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
