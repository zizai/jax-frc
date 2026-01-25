"""JAX-FRC: GPU-accelerated FRC plasma simulation framework."""

__version__ = "0.1.0"

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State, ParticleState
from jax_frc.core.simulation import Simulation

__all__ = ["Geometry", "State", "ParticleState", "Simulation"]
