"""Merging-specific diagnostic probes."""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import jax.numpy as jnp
from jax import Array

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.diagnostics.probes import Probe

MU0 = 1.2566e-6


@dataclass
class MergingDiagnostics(Probe):
    """Computes merging-specific metrics.

    Metrics include:
    - separation_dz: Distance between magnetic nulls
    - separatrix_radius: Radius of separatrix (Rs)
    - elongation: Separatrix half-length / radius (E = Zs/Rs)
    - separatrix_beta: Plasma beta at separatrix
    - peak_pressure: Maximum pressure
    - null_positions: List of (r, z) coordinates of nulls
    - axial_velocity_at_null: Vz at null positions
    - reconnection_rate: dpsi/dt proxy at midplane
    """

    _prev_psi_midplane: Array = None
    _prev_time: float = None

    @property
    def name(self) -> str:
        return "merging"

    def measure(self, state: State, geometry: Geometry) -> float:
        """Return separation as primary scalar metric."""
        result = self.compute(state, geometry)
        return result["separation_dz"]

    def compute(self, state: State, geometry: Geometry) -> Dict[str, Any]:
        """Compute all merging diagnostics.

        Args:
            state: Current simulation state
            geometry: Computational geometry

        Returns:
            Dictionary of diagnostic values
        """
        psi = state.psi

        # Find null positions
        nulls = self._find_null_positions(psi, geometry)

        # Compute separation
        if len(nulls) >= 2:
            separation = abs(nulls[0][1] - nulls[1][1])
        else:
            separation = 0.0

        # Separatrix radius (max r where psi > threshold)
        rs = self._find_separatrix_radius(psi, geometry)

        # Elongation
        zs = self._find_separatrix_half_length(psi, geometry)
        elongation = zs / (rs + 1e-10)

        # Separatrix beta
        beta_s = self._compute_separatrix_beta(state, geometry, rs)

        # Axial velocity at nulls
        vz_at_null = self._velocity_at_nulls(state, nulls, geometry)

        # Reconnection rate (change in psi at midplane)
        recon_rate = self._compute_reconnection_rate(state, geometry)

        return {
            "separation_dz": float(separation),
            "separatrix_radius": float(rs),
            "elongation": float(elongation),
            "separatrix_beta": float(beta_s),
            "peak_pressure": float(jnp.max(state.p)),
            "null_positions": nulls,
            "axial_velocity_at_null": vz_at_null,
            "reconnection_rate": float(recon_rate),
        }

    def _find_null_positions(self, psi: Array, geometry: Geometry) -> List[Tuple[float, float]]:
        """Find magnetic null positions (local maxima of psi)."""
        nulls = []

        # Find global maximum
        idx = jnp.unravel_index(jnp.argmax(psi), psi.shape)
        r1 = geometry.r_min + idx[0] * geometry.dr
        z1 = geometry.z_min + idx[1] * geometry.dz
        nulls.append((float(r1), float(z1)))

        # For two-FRC case, find max in opposite half
        nz = psi.shape[1]
        mid = nz // 2

        if idx[1] < mid:
            # First null in left half, find max in right half
            right_half = psi[:, mid:]
            idx2 = jnp.unravel_index(jnp.argmax(right_half), right_half.shape)
            r2 = geometry.r_min + idx2[0] * geometry.dr
            z2 = geometry.z_min + (mid + idx2[1]) * geometry.dz
        else:
            # First null in right half, find max in left half
            left_half = psi[:, :mid]
            idx2 = jnp.unravel_index(jnp.argmax(left_half), left_half.shape)
            r2 = geometry.r_min + idx2[0] * geometry.dr
            z2 = geometry.z_min + idx2[1] * geometry.dz

        nulls.append((float(r2), float(z2)))
        return nulls

    def _find_separatrix_radius(self, psi: Array, geometry: Geometry) -> float:
        """Find separatrix radius at midplane."""
        threshold = jnp.max(psi) * 0.01

        z_mid_idx = psi.shape[1] // 2
        psi_midplane = psi[:, z_mid_idx]

        inside = psi_midplane > threshold
        r_indices = jnp.where(inside, jnp.arange(psi.shape[0]), 0)
        r_sep_idx = jnp.max(r_indices)

        return geometry.r_min + float(r_sep_idx) * geometry.dr

    def _find_separatrix_half_length(self, psi: Array, geometry: Geometry) -> float:
        """Find separatrix half-length."""
        threshold = jnp.max(psi) * 0.01

        r_max_idx = jnp.argmax(jnp.max(psi, axis=1))
        psi_axial = psi[r_max_idx, :]

        inside = psi_axial > threshold
        z_indices = jnp.arange(psi.shape[1])
        z_inside = jnp.where(inside, z_indices, -1)

        z_max_idx = jnp.max(z_inside)
        z_min_idx = jnp.min(jnp.where(inside, z_indices, psi.shape[1]))

        length = (z_max_idx - z_min_idx) * geometry.dz
        return float(length) / 2.0

    def _compute_separatrix_beta(self, state: State, geometry: Geometry, rs: float) -> float:
        """Compute plasma beta at separatrix."""
        r_idx = int((rs - geometry.r_min) / geometry.dr)
        r_idx = min(r_idx, state.p.shape[0] - 1)

        z_mid_idx = state.p.shape[1] // 2

        p_sep = state.p[r_idx, z_mid_idx]
        B_sq = jnp.sum(state.B[r_idx, z_mid_idx]**2)
        B_sq = jnp.maximum(B_sq, 1e-20)

        beta = 2 * MU0 * p_sep / B_sq
        return float(beta)

    def _velocity_at_nulls(self, state: State, nulls: List[Tuple[float, float]],
                          geometry: Geometry) -> List[float]:
        """Get axial velocity at null positions."""
        velocities = []
        for r, z in nulls:
            r_idx = int((r - geometry.r_min) / geometry.dr)
            z_idx = int((z - geometry.z_min) / geometry.dz)
            r_idx = min(max(r_idx, 0), state.v.shape[0] - 1)
            z_idx = min(max(z_idx, 0), state.v.shape[1] - 1)
            vz = state.v[r_idx, z_idx, 2]  # z component
            velocities.append(float(vz))
        return velocities

    def _compute_reconnection_rate(self, state: State, geometry: Geometry) -> float:
        """Compute reconnection rate as dpsi/dt at midplane X-point."""
        z_mid_idx = state.E.shape[1] // 2
        E_phi = state.E[:, z_mid_idx, 1]  # theta component
        return float(jnp.max(jnp.abs(E_phi)))
