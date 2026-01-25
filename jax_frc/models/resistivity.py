"""Resistivity models for MHD simulations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax.numpy as jnp
from jax import Array

class ResistivityModel(ABC):
    """Base class for resistivity models."""

    @abstractmethod
    def compute(self, j_phi: Array, **kwargs) -> Array:
        """Compute resistivity field."""
        pass

@dataclass
class SpitzerResistivity(ResistivityModel):
    """Classical Spitzer resistivity."""
    eta_0: float = 1e-6

    def compute(self, j_phi: Array, **kwargs) -> Array:
        return jnp.full_like(j_phi, self.eta_0)

@dataclass
class ChoduraResistivity(ResistivityModel):
    """Anomalous resistivity for reconnection."""
    eta_0: float = 1e-6
    eta_anom: float = 1e-3
    threshold: float = 1e4

    def compute(self, j_phi: Array, **kwargs) -> Array:
        j_mag = jnp.abs(j_phi)
        anomalous_factor = 0.5 * (1 + jnp.tanh((j_mag - self.threshold) / (self.threshold * 0.1)))
        return self.eta_0 + self.eta_anom * anomalous_factor
