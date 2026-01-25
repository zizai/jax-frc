"""Core components for JAX-FRC simulation."""

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State, ParticleState
from jax_frc.core.simulation import Simulation

__all__ = ["Geometry", "State", "ParticleState", "Simulation"]
