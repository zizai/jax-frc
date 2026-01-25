"""Phase implementations for FRC experiments."""

from jax_frc.scenarios.phases.merging import MergingPhase
from jax_frc.scenarios.phase import PHASE_REGISTRY

# Register phases in the global registry
PHASE_REGISTRY["MergingPhase"] = MergingPhase

__all__ = [
    "MergingPhase",
]
