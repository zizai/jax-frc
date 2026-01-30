"""Normalization helpers for dimensionless units."""

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class NormScales:
    """Base scales for Alfvénic normalization."""

    L0: float
    rho0: float
    B0: float

    @property
    def v0(self) -> float:
        return self.B0 / jnp.sqrt(self.rho0)

    @property
    def T0(self) -> float:
        return self.L0 / self.v0

    @property
    def p0(self) -> float:
        return self.rho0 * self.v0**2


def to_dimless_state(state, scales: NormScales):
    """Convert physical state to dimensionless state."""
    return state.replace(
        n=state.n / scales.rho0,
        p=state.p / scales.p0,
        v=state.v / scales.v0 if state.v is not None else state.v,
        B=state.B / scales.B0,
        time=state.time / scales.T0,
    )


def to_physical_state(state, scales: NormScales):
    """Convert dimensionless state to physical units."""
    return state.replace(
        n=state.n * scales.rho0,
        p=state.p * scales.p0,
        v=state.v * scales.v0 if state.v is not None else state.v,
        B=state.B * scales.B0,
        time=state.time * scales.T0,
    )


def scale_eta_nu(eta: float, nu: float, scales: NormScales) -> tuple[float, float]:
    """Scale resistivity and viscosity to dimensionless values."""
    factor = scales.L0 * scales.v0
    return eta / factor, nu / factor
