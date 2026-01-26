"""JAX-FRC: GPU-accelerated FRC plasma simulation framework."""

__version__ = "0.1.0"

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State, ParticleState
from jax_frc.core.simulation import Simulation
from jax_frc.constants import MU0, QE, ME, MI, KB, EPSILON0, C
from jax_frc.results import SimulationResult
from jax_frc import operators
from jax_frc import physics

__all__ = [
    "Geometry",
    "State",
    "ParticleState",
    "Simulation",
    "SimulationResult",
    "MU0",
    "QE",
    "ME",
    "MI",
    "KB",
    "EPSILON0",
    "C",
    "operators",
    "physics",
]
