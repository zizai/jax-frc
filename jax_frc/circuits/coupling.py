"""Flux coupling between plasma and circuits.

This module provides bidirectional coupling:
- plasma_to_coils: Compute plasma flux threading each circuit
- coils_to_plasma: Compute B-field contribution from external coil currents
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array, vmap

from jax_frc.circuits.pickup import PickupCoilArray
from jax_frc.circuits.external import ExternalCircuits
from jax_frc.core.geometry import Geometry


@dataclass(frozen=True)
class FluxCoupling:
    """Computes flux linkages between plasma and all circuits.

    Provides bidirectional coupling:
    - plasma_to_coils: Compute plasma flux threading each circuit
    - coils_to_plasma: Compute B-field contribution from external coil currents

    This class is stateless and JIT-compatible.
    """

    def plasma_to_coils(
        self,
        B_plasma: Array,
        geometry: Geometry,
        pickup: PickupCoilArray,
        external: ExternalCircuits,
    ) -> tuple[Array, Array]:
        """Compute plasma flux threading each circuit.

        Integrates the plasma B-field over each coil's area to compute
        flux linkages. Uses the existing compute_flux_linkages methods
        on PickupCoilArray and computes flux for external circuits.

        Args:
            B_plasma: Plasma magnetic field (nx, ny, nz, 3) with (Bx, By, Bz)
            geometry: Computational geometry
            pickup: Pickup coil array
            external: External circuits

        Returns:
            Psi_pickup: Flux linkage for each pickup coil [Wb], shape (n_pickup,)
            Psi_external: Flux linkage for each external coil [Wb], shape (n_external,)
        """
        # Compute flux through pickup coils using existing method
        Psi_pickup = pickup.compute_flux_linkages(B_plasma, geometry)

        # Compute flux through external coils
        Psi_external = self._compute_external_flux(B_plasma, geometry, external)

        return Psi_pickup, Psi_external

    def _compute_external_flux(
        self,
        B: Array,
        geometry: Geometry,
        external: ExternalCircuits,
    ) -> Array:
        """Compute flux linkage through external coils.

        Args:
            B: Magnetic field (nr, nz, 3)
            geometry: Computational geometry
            external: External circuits

        Returns:
            Psi: Flux linkage for each external coil [Wb], shape (n_external,)
        """
        if external.n_circuits == 0:
            return jnp.array([])

        y_mid = B.shape[1] // 2
        Bz = B[:, y_mid, :, 2]  # Axial component at midplane
        r = geometry.x
        dr = geometry.dx

        # Extract coil parameters
        z_centers, radii, lengths, n_turns = external._get_coil_params()

        def flux_for_coil(z_center: float, radius: float, n_turn: int) -> float:
            """Compute flux for a single external coil."""
            # Find z index (interpolate between grid points)
            z_idx_float = (z_center - geometry.z_min) / geometry.dz
            z_idx = jnp.clip(z_idx_float.astype(int), 0, geometry.nz - 2)
            z_frac = z_idx_float - z_idx

            # Interpolate Bz at coil z-position
            Bz_at_z = (1 - z_frac) * Bz[:, z_idx] + z_frac * Bz[:, z_idx + 1]

            # Mask for r < coil radius
            mask = r < radius

            # Integrate: Psi = integral(Bz * 2*pi*r * dr) for r < radius
            flux = jnp.sum(Bz_at_z * 2 * jnp.pi * r * dr * mask)

            return n_turn * flux

        # Vectorize over all external coils
        Psi = vmap(flux_for_coil)(z_centers, radii, n_turns)

        return Psi

    def coils_to_plasma(
        self,
        I_external: Array,
        external: ExternalCircuits,
        geometry: Geometry,
    ) -> Array:
        """Compute B-field from external coil currents.

        Uses the existing compute_b_field method on ExternalCircuits
        to compute the magnetic field contribution from all external
        coil currents.

        Args:
            I_external: Current in each external coil [A], shape (n_external,)
            external: External circuits
            geometry: Computational geometry

        Returns:
            B_coils: Magnetic field from coils (nr, nz, 3)
        """
        return external.compute_b_field(I_external, geometry)
