"""Energy partition diagnostics for plasma simulations.

This module provides diagnostics for tracking how energy is distributed
among magnetic, kinetic, and thermal components in FRC simulations.
"""

from dataclasses import dataclass
from typing import Dict

import jax.numpy as jnp

from jax_frc.diagnostics.probes import Probe
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.constants import MU0, MI


@dataclass
class EnergyDiagnostics(Probe):
    """Compute energy partition: magnetic, kinetic, thermal.

    This diagnostic tracks the distribution of energy across the three
    main energy channels in an MHD plasma:

    - Magnetic energy: E_mag = (1/2mu0) * integral(B^2) dV
    - Kinetic energy: E_kin = (1/2) * integral(rho * v^2) dV
    - Thermal energy: E_th = integral(p / (gamma-1)) dV

    All integrals are performed in cylindrical coordinates with the
    volume element dV = 2*pi*r*dr*dz.

    Attributes:
        gamma: Adiabatic index (ratio of specific heats). Default 5/3 for ideal gas.
        ion_mass: Ion mass in kg. Default is proton mass.
    """

    gamma: float = 5.0 / 3.0  # adiabatic index
    ion_mass: float = MI  # ion mass for kinetic energy calculation

    @property
    def name(self) -> str:
        """Name of the diagnostic quantity."""
        return "energy_partition"

    def measure(self, state: State, geometry: Geometry) -> float:
        """Return total energy as the primary metric.

        Args:
            state: Current simulation state containing fields
            geometry: Computational geometry defining the domain

        Returns:
            Total energy (magnetic + kinetic + thermal) in Joules
        """
        result = self.compute(state, geometry)
        return result["E_total"]

    def compute(self, state: State, geometry: Geometry) -> Dict[str, float]:
        """Compute all energy components.

        Args:
            state: Current simulation state containing fields
            geometry: Computational geometry defining the domain

        Returns:
            Dictionary with keys:
            - E_magnetic: Magnetic field energy [J]
            - E_kinetic: Bulk flow kinetic energy [J]
            - E_thermal: Thermal/internal energy [J]
            - E_total: Sum of all components [J]
        """
        E_mag = self._compute_magnetic_energy(state, geometry)
        E_kin = self._compute_kinetic_energy(state, geometry)
        E_therm = self._compute_thermal_energy(state, geometry)
        E_total = E_mag + E_kin + E_therm

        return {
            "E_magnetic": float(E_mag),
            "E_kinetic": float(E_kin),
            "E_thermal": float(E_therm),
            "E_total": float(E_total),
        }

    def _compute_magnetic_energy(self, state: State, geometry: Geometry) -> float:
        """Compute magnetic energy.

        E_magnetic = (1/2mu0) * integral(B^2) dV
                   = (1/2mu0) * integral((B_r^2 + B_phi^2 + B_z^2) * 2*pi*r) dr dz

        Args:
            state: Current simulation state
            geometry: Computational geometry

        Returns:
            Magnetic energy in Joules
        """
        # B field has shape (nr, nz, 3) with components [B_r, B_phi, B_z]
        B_squared = jnp.sum(state.B**2, axis=-1)  # B_r^2 + B_phi^2 + B_z^2

        # Integrate over volume: (1/2mu0) * B^2 * dV
        # cell_volumes already includes 2*pi*r factor for cylindrical
        E_mag = jnp.sum(B_squared / (2.0 * MU0) * geometry.cell_volumes)

        return float(E_mag)

    def _compute_kinetic_energy(self, state: State, geometry: Geometry) -> float:
        """Compute bulk flow kinetic energy.

        E_kinetic = (1/2) * integral(rho * v^2) dV
                  = (1/2) * integral(rho * (v_r^2 + v_phi^2 + v_z^2) * 2*pi*r) dr dz

        where rho = n * m_ion (mass density from number density)

        Args:
            state: Current simulation state
            geometry: Computational geometry

        Returns:
            Kinetic energy in Joules
        """
        if state.v is None:
            return 0.0
        # v field has shape (nr, nz, 3) with components [v_r, v_phi, v_z]
        v_squared = jnp.sum(state.v**2, axis=-1)  # v_r^2 + v_phi^2 + v_z^2

        # Mass density: rho = n * m_ion
        rho = state.n * self.ion_mass

        # Integrate over volume: (1/2) * rho * v^2 * dV
        E_kin = jnp.sum(0.5 * rho * v_squared * geometry.cell_volumes)

        return float(E_kin)

    def _compute_thermal_energy(self, state: State, geometry: Geometry) -> float:
        """Compute thermal/internal energy.

        E_thermal = integral(p / (gamma - 1)) dV
                  = integral(p / (gamma - 1) * 2*pi*r) dr dz

        For an ideal gas with gamma = 5/3, this represents the internal
        energy associated with particle thermal motion.

        Args:
            state: Current simulation state
            geometry: Computational geometry

        Returns:
            Thermal energy in Joules
        """
        # Integrate over volume: p / (gamma - 1) * dV
        E_therm = jnp.sum(state.p / (self.gamma - 1.0) * geometry.cell_volumes)

        return float(E_therm)
