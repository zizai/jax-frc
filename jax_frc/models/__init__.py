# jax_frc/models/__init__.py
"""Physics models for plasma simulation."""

from jax_frc.models.base import PhysicsModel
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.models.hybrid_kinetic import HybridKinetic
from jax_frc.models.neutral_fluid import NeutralState, NeutralFluid
from jax_frc.models.resistivity import SpitzerResistivity, ChoduraResistivity
from jax_frc.models.protocols import SplitRHS, SourceTerms
from jax_frc.models.coupled import CoupledState, CoupledModel, CoupledModelConfig, SourceRates
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig

__all__ = [
    "PhysicsModel",
    "ResistiveMHD",
    "ExtendedMHD",
    "HybridKinetic",
    "NeutralState",
    "NeutralFluid",
    "SpitzerResistivity",
    "ChoduraResistivity",
    "SplitRHS",
    "SourceTerms",
    "CoupledState",
    "CoupledModel",
    "CoupledModelConfig",
    "SourceRates",
    "AtomicCoupling",
    "AtomicCouplingConfig",
]
