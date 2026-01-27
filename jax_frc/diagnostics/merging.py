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
    - null_positions: List of (x, z) coordinates of nulls (midplane y-slice)
    - axial_velocity_at_null: Vz at null positions
    - reconnection_rate: max |E_y| at midplane (proxy)
    """

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
        # Use Bz at midplane as a flux proxy (3D Cartesian)
        y_mid = state.B.shape[1] // 2
        b_proxy = state.B[:, y_mid, :, 2]

        # Find null positions
        nulls = self._find_null_positions(b_proxy, geometry)

        # Compute separation
        if len(nulls) >= 2:
            separation = abs(nulls[0][1] - nulls[1][1])
        else:
            separation = 0.0

        # Separatrix radius (max r where psi > threshold)
        rs = self._find_separatrix_radius(b_proxy, geometry)

        # Elongation
        zs = self._find_separatrix_half_length(b_proxy, geometry)
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

    def _find_null_positions(self, b_proxy: Array, geometry: Geometry) -> List[Tuple[float, float]]:
        """Find magnetic null positions (local maxima of Bz proxy)."""
        nulls = []

        # Find global maximum
        idx = jnp.unravel_index(jnp.argmax(b_proxy), b_proxy.shape)
        x1 = geometry.x_min + idx[0] * geometry.dx
        z1 = geometry.z_min + idx[1] * geometry.dz
        nulls.append((float(x1), float(z1)))

        # For two-FRC case, find max in opposite half
        nz = b_proxy.shape[1]
        mid = nz // 2

        if idx[1] < mid:
            # First null in left half, find max in right half
            right_half = b_proxy[:, mid:]
            idx2 = jnp.unravel_index(jnp.argmax(right_half), right_half.shape)
            x2 = geometry.x_min + idx2[0] * geometry.dx
            z2 = geometry.z_min + (mid + idx2[1]) * geometry.dz
        else:
            # First null in right half, find max in left half
            left_half = b_proxy[:, :mid]
            idx2 = jnp.unravel_index(jnp.argmax(left_half), left_half.shape)
            x2 = geometry.x_min + idx2[0] * geometry.dx
            z2 = geometry.z_min + idx2[1] * geometry.dz

        nulls.append((float(x2), float(z2)))
        return nulls

    def _find_separatrix_radius(self, b_proxy: Array, geometry: Geometry) -> float:
        """Find separatrix radius at midplane using Bz proxy."""
        threshold = jnp.max(b_proxy) * 0.01

        z_mid_idx = b_proxy.shape[1] // 2
        proxy_midplane = b_proxy[:, z_mid_idx]

        inside = proxy_midplane > threshold
        x_indices = jnp.where(inside, jnp.arange(b_proxy.shape[0]), 0)
        x_sep_idx = jnp.max(x_indices)

        return geometry.x_min + float(x_sep_idx) * geometry.dx

    def _find_separatrix_half_length(self, b_proxy: Array, geometry: Geometry) -> float:
        """Find separatrix half-length using Bz proxy."""
        threshold = jnp.max(b_proxy) * 0.01

        x_max_idx = jnp.argmax(jnp.max(b_proxy, axis=1))
        proxy_axial = b_proxy[x_max_idx, :]

        inside = proxy_axial > threshold
        z_indices = jnp.arange(b_proxy.shape[1])
        z_inside = jnp.where(inside, z_indices, -1)

        z_max_idx = jnp.max(z_inside)
        z_min_idx = jnp.min(jnp.where(inside, z_indices, b_proxy.shape[1]))

        length = (z_max_idx - z_min_idx) * geometry.dz
        return float(length) / 2.0

    def _compute_separatrix_beta(self, state: State, geometry: Geometry, rs: float) -> float:
        """Compute plasma beta at separatrix."""
        x_idx = int((rs - geometry.x_min) / geometry.dx)
        x_idx = min(x_idx, state.p.shape[0] - 1)

        y_mid_idx = state.p.shape[1] // 2
        z_mid_idx = state.p.shape[2] // 2

        p_sep = state.p[x_idx, y_mid_idx, z_mid_idx]
        B_sq = jnp.sum(state.B[x_idx, y_mid_idx, z_mid_idx]**2)
        B_sq = jnp.maximum(B_sq, 1e-20)

        beta = 2 * MU0 * p_sep / B_sq
        return float(beta)

    def _velocity_at_nulls(
        self, state: State, nulls: List[Tuple[float, float]], geometry: Geometry
    ) -> List[float]:
        """Get axial velocity at null positions."""
        if state.v is None:
            return [0.0 for _ in nulls]
        velocities = []
        y_mid_idx = state.v.shape[1] // 2
        for x, z in nulls:
            x_idx = int((x - geometry.x_min) / geometry.dx)
            z_idx = int((z - geometry.z_min) / geometry.dz)
            x_idx = min(max(x_idx, 0), state.v.shape[0] - 1)
            z_idx = min(max(z_idx, 0), state.v.shape[2] - 1)
            vz = state.v[x_idx, y_mid_idx, z_idx, 2]  # z component
            velocities.append(float(vz))
        return velocities

    def _compute_reconnection_rate(self, state: State, geometry: Geometry) -> float:
        """Compute reconnection rate proxy as max E_y at midplane."""
        y_mid_idx = state.E.shape[1] // 2
        z_mid_idx = state.E.shape[2] // 2
        E_y = state.E[:, y_mid_idx, z_mid_idx, 1]
        return float(jnp.max(jnp.abs(E_y)))
