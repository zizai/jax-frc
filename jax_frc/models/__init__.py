# jax_frc/models/__init__.py
"""Physics models for plasma simulation.

3D models support Cartesian coordinates with shape (nx, ny, nz, 3) for vector fields.
"""

from jax_frc.models.base import PhysicsModel
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.extended_mhd import ExtendedMHD, HaloDensityModel, TemperatureBoundaryCondition
from jax_frc.models.hybrid_kinetic import HybridKinetic
from jax_frc.models.neutral_fluid import NeutralState, NeutralFluid
from jax_frc.models.resistivity import SpitzerResistivity, ChoduraResistivity
from jax_frc.models.protocols import SplitRHS, SourceTerms
from jax_frc.models.coupled import CoupledState, CoupledModel, CoupledModelConfig, SourceRates
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig
from jax_frc.models.burning_plasma import BurningPlasmaModel, BurningPlasmaState

# Aliases for 3D models (same classes, explicit naming for clarity)
ResistiveMHD3D = ResistiveMHD
ExtendedMHD3D = ExtendedMHD
NeutralFluid3D = NeutralFluid

__all__ = [
    # Base
    "PhysicsModel",
    # MHD models (3D-compatible)
    "ResistiveMHD",
    "ExtendedMHD",
    "ResistiveMHD3D",  # Alias
    "ExtendedMHD3D",   # Alias
    # Extended MHD components
    "HaloDensityModel",
    "TemperatureBoundaryCondition",
    # Hybrid kinetic
    "HybridKinetic",
    # Neutral fluid (3D-compatible)
    "NeutralState",
    "NeutralFluid",
    "NeutralFluid3D",  # Alias
    # Resistivity models
    "SpitzerResistivity",
    "ChoduraResistivity",
    # Protocols
    "SplitRHS",
    "SourceTerms",
    # Coupled models
    "CoupledState",
    "CoupledModel",
    "CoupledModelConfig",
    "SourceRates",
    # Atomic coupling
    "AtomicCoupling",
    "AtomicCouplingConfig",
    # Burning plasma
    "BurningPlasmaModel",
    "BurningPlasmaState",
]
