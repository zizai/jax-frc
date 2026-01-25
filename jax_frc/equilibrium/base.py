"""Base class for equilibrium solvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
import jax.numpy as jnp
from jax import Array

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State


@dataclass
class EquilibriumConstraints:
    """Constraints for equilibrium solutions.

    Attributes:
        psi_axis: Flux at magnetic axis (typically maximum)
        psi_boundary: Flux at plasma boundary (typically 0)
        total_current: Total toroidal plasma current
        beta: Plasma beta (thermal/magnetic pressure ratio)
        pressure_profile: Function p(psi) or None for default
        current_profile: Function F*F'(psi) or None for default
    """
    psi_axis: Optional[float] = None
    psi_boundary: float = 0.0
    total_current: Optional[float] = None
    beta: Optional[float] = None
    pressure_profile: Optional[Callable[[Array], Array]] = None
    current_profile: Optional[Callable[[Array], Array]] = None


class EquilibriumSolver(ABC):
    """Base class for equilibrium solvers."""

    @abstractmethod
    def solve(self, geometry: Geometry, constraints: EquilibriumConstraints,
              initial_guess: Optional[Array] = None) -> State:
        """Solve for equilibrium state.

        Args:
            geometry: Computational geometry
            constraints: Equilibrium constraints
            initial_guess: Initial guess for psi (optional)

        Returns:
            State containing equilibrium fields
        """
        pass

    @abstractmethod
    def compute_profiles(self, psi: Array, geometry: Geometry,
                        constraints: EquilibriumConstraints) -> dict:
        """Compute derived equilibrium profiles.

        Args:
            psi: Poloidal flux function
            geometry: Computational geometry
            constraints: Equilibrium constraints

        Returns:
            Dictionary containing profiles (p, j_phi, B_r, B_z, etc.)
        """
        pass
