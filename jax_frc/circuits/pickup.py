"""Pickup coil array for energy extraction."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, vmap

from jax_frc.circuits.state import CircuitParams
from jax_frc.core.geometry import Geometry


@dataclass(frozen=True)
class PickupCoilArray:
    """Array of pickup coils at different axial positions.

    Each coil has its own RLC circuit for energy extraction.

    Attributes:
        z_positions: Axial positions of coil centers [m], shape (n_coils,)
        radii: Coil radii [m], shape (n_coils,)
        n_turns: Turns per coil, shape (n_coils,)
        params: Circuit parameters (L, R, C) for each coil
        load_resistance: External load resistance [Ω], shape (n_coils,)
    """

    z_positions: Array
    radii: Array
    n_turns: Array
    params: CircuitParams
    load_resistance: Array

    @property
    def n_coils(self) -> int:
        """Number of pickup coils."""
        return self.z_positions.shape[0]

    def compute_flux_linkages(self, B: Array, geometry: Geometry) -> Array:
        """Compute flux linkage Ψ = N * ∫B·dA for each coil.

        Integrates Bz over the area within coil radius at each coil's
        z-position. Uses linear interpolation for z-positions between
        grid points.

        Args:
            B: Magnetic field (nr, nz, 3) with components (Br, Bphi, Bz)
            geometry: Computational geometry

        Returns:
            Psi: Flux linkage for each coil [Wb], shape (n_coils,)
        """
        Bz = B[:, :, 2]  # Axial component
        r = geometry.r  # 1D radial coordinates
        z = geometry.z  # 1D axial coordinates
        dr = geometry.dr

        def flux_for_coil(z_coil, radius, n_turn):
            # Find z index (interpolate between grid points)
            z_idx_float = (z_coil - geometry.z_min) / geometry.dz
            z_idx = jnp.clip(z_idx_float.astype(int), 0, geometry.nz - 2)
            z_frac = z_idx_float - z_idx

            # Interpolate Bz at coil z-position
            Bz_at_z = (1 - z_frac) * Bz[:, z_idx] + z_frac * Bz[:, z_idx + 1]

            # Mask for r < coil radius
            mask = r < radius

            # Integrate: Psi = integral(Bz * 2*pi*r * dr) for r < radius
            flux = jnp.sum(Bz_at_z * 2 * jnp.pi * r * dr * mask)

            return n_turn * flux

        # Vectorize over all coils using vmap for JIT compatibility
        Psi = vmap(flux_for_coil)(self.z_positions, self.radii, self.n_turns)

        return Psi

    def compute_power(self, I: Array) -> tuple[Array, Array]:
        """Compute power extracted and dissipated.

        Args:
            I: Current in each coil [A], shape (n_coils,)

        Returns:
            P_load: Power to external loads [W], shape (n_coils,)
            P_dissipated: Power dissipated in coil resistance [W], shape (n_coils,)
        """
        P_load = I**2 * self.load_resistance
        P_dissipated = I**2 * self.params.R
        return P_load, P_dissipated
