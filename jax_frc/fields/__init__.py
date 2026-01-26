"""External field generators for FRC simulations."""

from jax_frc.fields.coils import (
    CoilField,
    Solenoid,
    MirrorCoil,
    ThetaPinchArray,
)

__all__ = [
    "CoilField",
    "Solenoid",
    "MirrorCoil",
    "ThetaPinchArray",
]
