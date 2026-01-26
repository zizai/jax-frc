"""Direct induction energy conversion.

Computes electrical power recovery from time-varying magnetic flux.
"""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from jax_frc.core.geometry import Geometry


@dataclass(frozen=True)
class ConversionState:
    """State of direct conversion system.

    Attributes:
        P_electric: Recovered electrical power [W]
        V_induced: Induced voltage [V]
        dPsi_dt: Rate of flux change [Wb/s]
    """
    P_electric: float
    V_induced: float
    dPsi_dt: float


@dataclass(frozen=True)
class DirectConversion:
    """Direct induction energy recovery from magnetic flux change.

    Models power extraction via induction coils as the FRC plasma
    expands/compresses against the magnetic field.

    Attributes:
        coil_turns: Number of turns in pickup coil
        coil_radius: Coil radius [m]
        circuit_resistance: Total circuit resistance [Ohm]
        coupling_efficiency: Flux linkage efficiency (0-1)
    """
    coil_turns: int
    coil_radius: float
    circuit_resistance: float
    coupling_efficiency: float

    def compute_power(
        self,
        B_old: Array,
        B_new: Array,
        dt: float,
        geometry: Geometry,
    ) -> ConversionState:
        """Compute induced power from magnetic field change.

        Args:
            B_old: Magnetic field at previous timestep (nr, nz, 3)
            B_new: Magnetic field at current timestep (nr, nz, 3)
            dt: Timestep [s]
            geometry: Computational geometry

        Returns:
            ConversionState with power and voltage
        """
        # Compute magnetic flux through coil
        Psi_old = self._flux_integral(B_old, geometry)
        Psi_new = self._flux_integral(B_new, geometry)

        # Rate of flux change
        dPsi_dt = (Psi_new - Psi_old) / dt

        # Induced voltage: V = -N * dPsi/dt * eta_coupling
        V_induced = -self.coil_turns * dPsi_dt * self.coupling_efficiency

        # Power to matched load: P = V^2 / (4R)
        P_electric = V_induced**2 / (4 * self.circuit_resistance)

        return ConversionState(
            P_electric=float(P_electric),
            V_induced=float(V_induced),
            dPsi_dt=float(dPsi_dt),
        )

    def _flux_integral(self, B: Array, geometry: Geometry) -> float:
        """Integrate Bz over area within coil radius.

        Psi = integral(Bz * 2*pi*r * dr) for r < coil_radius
        """
        r = geometry.r_grid
        Bz = B[:, :, 2]  # Axial component

        # Sum over midplane (z=0, or just average over z)
        # For simplicity, take flux at z midpoint
        nz_mid = geometry.nz // 2
        Bz_mid = Bz[:, nz_mid]
        r_mid = r[:, nz_mid]
        mask_mid = r_mid < self.coil_radius

        # Flux = integral(Bz * 2*pi*r * dr)
        dr = geometry.dr
        flux = jnp.sum(Bz_mid * 2 * jnp.pi * r_mid * dr * mask_mid)

        return float(flux)

    def back_reaction_power(self, state: ConversionState) -> float:
        """Power extracted from plasma (for energy conservation).

        Returns:
            Power that should be subtracted from plasma energy [W]
        """
        return state.P_electric
