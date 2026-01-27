"""Anomalous transport model with configurable diffusivities.

Computes particle and energy fluxes for fusion plasma transport.
"""

from dataclasses import dataclass
from typing import Union

import jax.numpy as jnp
from jax import Array

from jax_frc.core.geometry import Geometry


@dataclass(frozen=True)
class TransportModel:
    """Anomalous transport with configurable diffusivities.

    Attributes:
        D_particle: Particle diffusivity [m²/s]
        chi_e: Electron thermal diffusivity [m²/s]
        chi_i: Ion thermal diffusivity [m²/s]
        v_pinch: Inward pinch velocity [m/s], default 0
    """
    D_particle: Union[float, Array]
    chi_e: Union[float, Array]
    chi_i: Union[float, Array]
    v_pinch: Union[float, Array] = 0.0

    def particle_flux(
        self, n: Array, geometry: Geometry
    ) -> tuple[Array, Array]:
        """Compute particle flux Gamma = -D*grad(n) + n*v_pinch.

        Args:
            n: Number density [m⁻³]
            geometry: Computational geometry

        Returns:
            (Gamma_r, Gamma_z): Flux components [m⁻²s⁻¹]
        """
        # Compute gradients using central differences
        dn_dr = self._gradient_r(n, geometry)
        dn_dz = self._gradient_z(n, geometry)

        # Diffusive flux
        Gamma_r = -self.D_particle * dn_dr + n * self.v_pinch
        Gamma_z = -self.D_particle * dn_dz

        return Gamma_r, Gamma_z

    def energy_flux(
        self, n: Array, T: Array, geometry: Geometry
    ) -> tuple[Array, Array]:
        """Compute energy flux q = -n*chi*grad(T).

        Uses combined chi = (chi_e + chi_i) / 2 for single-T model.

        Args:
            n: Number density [m⁻³]
            T: Temperature [keV or eV, consistent units]
            geometry: Computational geometry

        Returns:
            (q_r, q_z): Heat flux components [keV*m⁻²s⁻¹]
        """
        chi = (self.chi_e + self.chi_i) / 2

        dT_dr = self._gradient_r(T, geometry)
        dT_dz = self._gradient_z(T, geometry)

        q_r = -n * chi * dT_dr
        q_z = -n * chi * dT_dz

        return q_r, q_z

    def flux_divergence(
        self, flux_r: Array, flux_z: Array, geometry: Geometry
    ) -> Array:
        """Compute divergence of flux in cylindrical coordinates.

        div(F) = (1/r) * d(r*F_r)/dr + dF_z/dz

        Args:
            flux_r: Radial flux component
            flux_z: Axial flux component
            geometry: Computational geometry

        Returns:
            Divergence field
        """
        y_mid = geometry.ny // 2
        r = geometry.x_grid[:, y_mid, :]
        dr, dz = geometry.dx, geometry.dz

        # d(r*F_r)/dr
        rFr = r * flux_r
        drFr_dr = (jnp.roll(rFr, -1, axis=0) - jnp.roll(rFr, 1, axis=0)) / (2 * dr)

        # dF_z/dz
        dFz_dz = (jnp.roll(flux_z, -1, axis=1) - jnp.roll(flux_z, 1, axis=1)) / (2 * dz)

        return (1 / r) * drFr_dr + dFz_dz

    def _gradient_r(self, f: Array, geometry: Geometry) -> Array:
        """Central difference gradient in r."""
        return (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * geometry.dx)

    def _gradient_z(self, f: Array, geometry: Geometry) -> Array:
        """Central difference gradient in z."""
        return (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * geometry.dz)
