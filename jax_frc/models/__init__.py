"""Physics models for JAX-FRC simulation."""

from jax_frc.models.base import PhysicsModel
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.extended_mhd import ExtendedMHD

__all__ = ["PhysicsModel", "ResistiveMHD", "ExtendedMHD"]
