"""Diagnostic probes for measuring plasma quantities."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import jax.numpy as jnp
from jax import Array

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry

MU0 = 1.2566e-6
QE = 1.602e-19


class Probe(ABC):
    """Base class for diagnostic probes."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the diagnostic quantity."""
        pass

    @abstractmethod
    def measure(self, state: State, geometry: Geometry) -> float:
        """Measure the diagnostic quantity from current state."""
        pass


@dataclass
class FluxProbe(Probe):
    """Measures poloidal flux at a specific location or global properties."""

    r_probe: Optional[float] = None  # If None, measure max flux
    z_probe: Optional[float] = None

    @property
    def name(self) -> str:
        if self.r_probe is None:
            return "psi_max"
        return f"psi_r{self.r_probe:.2f}_z{self.z_probe:.2f}"

    def measure(self, state: State, geometry: Geometry) -> float:
        if self.r_probe is None:
            return float(jnp.max(state.psi))

        # Interpolate to probe location
        r_idx = int((self.r_probe - geometry.r_min) / geometry.dr)
        z_idx = int((self.z_probe - geometry.z_min) / geometry.dz)
        r_idx = jnp.clip(r_idx, 0, geometry.nr - 1)
        z_idx = jnp.clip(z_idx, 0, geometry.nz - 1)
        return float(state.psi[r_idx, z_idx])


@dataclass
class EnergyProbe(Probe):
    """Measures magnetic and thermal energy."""

    energy_type: str = "magnetic"  # "magnetic", "thermal", or "total"

    @property
    def name(self) -> str:
        return f"E_{self.energy_type}"

    def measure(self, state: State, geometry: Geometry) -> float:
        if self.energy_type == "magnetic":
            # E_mag = integral(B^2 / (2*mu_0)) dV
            B_sq = jnp.sum(state.B**2, axis=-1)
            E_mag = jnp.sum(B_sq / (2 * MU0) * geometry.cell_volumes)
            return float(E_mag)

        elif self.energy_type == "thermal":
            # E_th = integral(p / (gamma-1)) dV, gamma = 5/3
            gamma = 5.0 / 3.0
            E_th = jnp.sum(state.p / (gamma - 1) * geometry.cell_volumes)
            return float(E_th)

        elif self.energy_type == "total":
            B_sq = jnp.sum(state.B**2, axis=-1)
            E_mag = jnp.sum(B_sq / (2 * MU0) * geometry.cell_volumes)
            gamma = 5.0 / 3.0
            E_th = jnp.sum(state.p / (gamma - 1) * geometry.cell_volumes)
            return float(E_mag + E_th)

        else:
            raise ValueError(f"Unknown energy type: {self.energy_type}")


@dataclass
class BetaProbe(Probe):
    """Measures plasma beta (thermal/magnetic pressure ratio)."""

    beta_type: str = "average"  # "average", "max", or "axis"

    @property
    def name(self) -> str:
        return f"beta_{self.beta_type}"

    def measure(self, state: State, geometry: Geometry) -> float:
        B_sq = jnp.sum(state.B**2, axis=-1)
        B_sq = jnp.maximum(B_sq, 1e-20)  # Avoid division by zero
        p_mag = B_sq / (2 * MU0)
        beta = state.p / p_mag

        if self.beta_type == "average":
            # Volume-averaged beta
            total_vol = jnp.sum(geometry.cell_volumes)
            return float(jnp.sum(beta * geometry.cell_volumes) / total_vol)

        elif self.beta_type == "max":
            return float(jnp.max(beta))

        elif self.beta_type == "axis":
            # Beta at magnetic axis (location of max psi)
            idx = jnp.unravel_index(jnp.argmax(state.psi), state.psi.shape)
            return float(beta[idx])

        else:
            raise ValueError(f"Unknown beta type: {self.beta_type}")


@dataclass
class CurrentProbe(Probe):
    """Measures plasma current."""

    current_type: str = "total"  # "total", "max_density"

    @property
    def name(self) -> str:
        return f"I_{self.current_type}"

    def measure(self, state: State, geometry: Geometry) -> float:
        # Compute j_phi from Delta*psi
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid
        psi = state.psi

        # Delta*psi
        psi_rr = (jnp.roll(psi, -1, axis=0) - 2*psi + jnp.roll(psi, 1, axis=0)) / dr**2
        psi_zz = (jnp.roll(psi, -1, axis=1) - 2*psi + jnp.roll(psi, 1, axis=1)) / dz**2
        psi_r = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2*dr)
        delta_star_psi = psi_rr - (1.0/r) * psi_r + psi_zz

        j_phi = -delta_star_psi / (MU0 * r)

        if self.current_type == "total":
            # Total current: I = integral(j_phi) dA
            # In 2D axisymmetric: I = integral(j_phi * r) dr dz (without 2*pi)
            I_total = jnp.sum(j_phi * geometry.dr * geometry.dz)
            return float(I_total)

        elif self.current_type == "max_density":
            return float(jnp.max(jnp.abs(j_phi)))

        else:
            raise ValueError(f"Unknown current type: {self.current_type}")


@dataclass
class SeparatrixProbe(Probe):
    """Measures separatrix properties (radius, length)."""

    property_type: str = "radius"  # "radius", "length", "xs_ratio"

    @property
    def name(self) -> str:
        return f"sep_{self.property_type}"

    def measure(self, state: State, geometry: Geometry) -> float:
        # Find separatrix as psi = 0 contour (or small value)
        psi = state.psi
        psi_threshold = jnp.max(psi) * 0.01  # 1% of max

        # Find radial extent at midplane (z=0)
        z_mid_idx = geometry.nz // 2
        psi_midplane = psi[:, z_mid_idx]

        if self.property_type == "radius":
            # Find outermost r where psi > threshold
            inside = psi_midplane > psi_threshold
            r_indices = jnp.where(inside, jnp.arange(geometry.nr), 0)
            r_sep_idx = jnp.max(r_indices)
            r_sep = geometry.r_min + r_sep_idx * geometry.dr
            return float(r_sep)

        elif self.property_type == "length":
            # Find axial extent at r of max psi
            r_max_idx = jnp.argmax(jnp.max(psi, axis=1))
            psi_axial = psi[r_max_idx, :]
            inside = psi_axial > psi_threshold
            z_indices = jnp.where(inside, jnp.arange(geometry.nz), 0)
            z_extent = (jnp.max(z_indices) - jnp.min(jnp.where(inside, z_indices, geometry.nz))) * geometry.dz
            return float(z_extent)

        elif self.property_type == "xs_ratio":
            # Elongation: length / (2 * radius)
            r_sep = self.measure(state.replace(), geometry)  # Get radius
            # Temporarily change property to get length
            old_type = self.property_type
            self.property_type = "length"
            length = self.measure(state, geometry)
            self.property_type = old_type
            return float(length / (2 * r_sep + 1e-10))

        else:
            raise ValueError(f"Unknown property type: {self.property_type}")


class DiagnosticSet:
    """Collection of probes with time history tracking."""

    def __init__(self, probes: Optional[List[Probe]] = None):
        self.probes = probes or []
        self.history: Dict[str, List[float]] = {p.name: [] for p in self.probes}
        self.times: List[float] = []

    def add_probe(self, probe: Probe) -> None:
        """Add a probe to the diagnostic set."""
        self.probes.append(probe)
        self.history[probe.name] = []

    def measure_all(self, state: State, geometry: Geometry) -> Dict[str, float]:
        """Measure all probes and record in history."""
        self.times.append(float(state.time))
        results = {}
        for probe in self.probes:
            value = probe.measure(state, geometry)
            results[probe.name] = value
            self.history[probe.name].append(value)
        return results

    def get_history(self) -> Dict[str, Any]:
        """Get full time history as dictionary."""
        return {
            "time": self.times,
            **self.history
        }

    @classmethod
    def default_set(cls) -> "DiagnosticSet":
        """Create a default set of common diagnostics."""
        return cls(probes=[
            FluxProbe(),  # psi_max
            EnergyProbe(energy_type="magnetic"),
            EnergyProbe(energy_type="thermal"),
            BetaProbe(beta_type="average"),
            CurrentProbe(current_type="total"),
        ])
