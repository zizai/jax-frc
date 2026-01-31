"""JAX-FRC: GPU-accelerated FRC plasma simulation framework."""

__version__ = "0.1.0"

# Core classes
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State, ParticleState
from jax_frc.core.simulation import Simulation
from jax_frc.solvers.recipe import NumericalRecipe

# Constants
from jax_frc.constants import MU0, QE, ME, MI, KB, EPSILON0, C

# Results
from jax_frc.results import SimulationResult

# 3D differential operators
from jax_frc.operators import (
    gradient_3d,
    divergence_3d,
    curl_3d,
    laplacian_3d,
)

# Submodules for qualified imports
from jax_frc import operators
from jax_frc import physics
from jax_frc import models
from jax_frc import solvers
from jax_frc import equilibrium

__all__ = [
    # Core classes
    "Geometry",
    "State",
    "ParticleState",
    "Simulation",
    "NumericalRecipe",
    "SimulationResult",
    # Constants
    "MU0",
    "QE",
    "ME",
    "MI",
    "KB",
    "EPSILON0",
    "C",
    # 3D operators
    "gradient_3d",
    "divergence_3d",
    "curl_3d",
    "laplacian_3d",
    # Submodules
    "operators",
    "physics",
    "models",
    "solvers",
    "equilibrium",
]
