"""Abstract base class for physics models."""
import warnings
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

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Enforce physical constraints (e.g., div(B)=0).
        
        DEPRECATED: Constraint enforcement has moved to Solver._apply_constraints().
        This method is kept for backward compatibility but will be removed in a future version.
        """
        warnings.warn(
            "Model.apply_constraints() is deprecated. "
            "Use Solver._apply_constraints() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return state

    @classmethod
    def create(cls, config: dict) -> "PhysicsModel":
        """Factory method to create model from config."""
        model_type = config.get("type", "resistive_mhd")
        if model_type == "resistive_mhd":
            from jax_frc.models.resistive_mhd import ResistiveMHD
            eta = float(config.get("eta", config.get("resistivity", {}).get("eta_0", 1e-4)))
            return ResistiveMHD(eta=eta)
        elif model_type == "extended_mhd":
            from jax_frc.models.extended_mhd import ExtendedMHD
            from jax_frc.models.extended_mhd import TemperatureBoundaryCondition
            eta = float(config.get("eta", config.get("resistivity", {}).get("eta_0", 1e-4)))
            include_hall = bool(config.get("include_hall", True))
            include_electron_pressure = bool(config.get("include_electron_pressure", True))
            kappa_perp = float(config.get("kappa_perp", config.get("thermal", {}).get("kappa_perp", 1e18)))
            temperature_bc = None
            if "temperature_bc" in config:
                temperature_bc = TemperatureBoundaryCondition(**config["temperature_bc"])
            return ExtendedMHD(
                eta=eta,
                include_hall=include_hall,
                include_electron_pressure=include_electron_pressure,
                kappa_perp=kappa_perp,
                temperature_bc=temperature_bc,
            )
        elif model_type == "hybrid_kinetic":
            from jax_frc.models.hybrid_kinetic import HybridKinetic
            return HybridKinetic.from_config(config)
        elif model_type == "burning_plasma":
            from jax_frc.models.burning_plasma import BurningPlasmaModel
            return BurningPlasmaModel.from_config(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
