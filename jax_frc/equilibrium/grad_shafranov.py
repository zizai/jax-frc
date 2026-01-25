"""Grad-Shafranov equilibrium solver."""

from dataclasses import dataclass
from typing import Optional, Tuple
import jax.numpy as jnp
from jax import jit, lax
from jax import Array

from jax_frc.equilibrium.base import EquilibriumSolver, EquilibriumConstraints
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State

MU0 = 1.2566e-6


@dataclass
class GradShafranovSolver(EquilibriumSolver):
    """Iterative Grad-Shafranov equilibrium solver.

    Solves: Delta* psi = -mu_0 * r^2 * dp/dpsi - F * dF/dpsi

    Uses Picard iteration with successive over-relaxation (SOR).
    """

    max_iterations: int = 1000
    tolerance: float = 1e-8
    omega: float = 1.5  # SOR relaxation parameter

    def solve(self, geometry: Geometry, constraints: EquilibriumConstraints,
              initial_guess: Optional[Array] = None) -> State:
        """Solve Grad-Shafranov equation iteratively."""
        nr, nz = geometry.nr, geometry.nz
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Initial guess: simple FRC-like profile
        if initial_guess is None:
            psi = self._initial_frc_guess(geometry)
        else:
            psi = initial_guess

        # Get profile functions
        if constraints.pressure_profile is not None:
            p_func = constraints.pressure_profile
        else:
            p_func = lambda psi_norm: 1.0 - psi_norm**2  # Parabolic default

        if constraints.current_profile is not None:
            ff_func = constraints.current_profile
        else:
            ff_func = lambda psi_norm: jnp.zeros_like(psi_norm)  # No toroidal field

        # Iterative solution
        psi, converged, n_iter = self._solve_iteration(
            psi, geometry, constraints, p_func, ff_func
        )

        # Build complete state
        profiles = self.compute_profiles(psi, geometry, constraints)
        state = State.zeros(nr, nz)
        state = state.replace(
            psi=psi,
            p=profiles["pressure"],
            B=profiles["B"],
            n=profiles["density"]
        )

        return state

    def _solve_iteration(self, psi_init, geometry, constraints, p_func, ff_func):
        """Run Picard iteration to solve GS equation."""
        nr, nz = geometry.nr, geometry.nz
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        def iteration_step(carry, _):
            psi, error = carry

            # Normalize psi for profile evaluation
            psi_max = jnp.max(psi)
            psi_min = jnp.min(psi)
            psi_range = jnp.maximum(psi_max - psi_min, 1e-10)
            psi_norm = (psi - psi_min) / psi_range

            # Compute RHS: -mu_0 * r^2 * dp/dpsi - F*dF/dpsi
            # For FRC, typically F=0 (no toroidal field), so just pressure term
            p = p_func(psi_norm)
            dp_dpsi = jnp.gradient(p, axis=0) / (psi_range / nr)  # Approximate

            rhs = -MU0 * r**2 * dp_dpsi

            # SOR update for Laplace* operator
            psi_new = self._sor_step(psi, rhs, r, dr, dz, self.omega)

            # Apply boundary conditions
            psi_new = psi_new.at[0, :].set(psi_new[1, :])  # Neumann at r=0
            psi_new = psi_new.at[-1, :].set(constraints.psi_boundary)
            psi_new = psi_new.at[:, 0].set(constraints.psi_boundary)
            psi_new = psi_new.at[:, -1].set(constraints.psi_boundary)

            # Compute error
            error_new = jnp.max(jnp.abs(psi_new - psi))

            return (psi_new, error_new), error_new

        # Run iterations
        init_carry = (psi_init, jnp.inf)
        (psi_final, final_error), errors = lax.scan(
            iteration_step, init_carry, jnp.arange(self.max_iterations)
        )

        converged = final_error < self.tolerance
        n_iter = jnp.sum(errors > self.tolerance)

        return psi_final, converged, n_iter

    @staticmethod
    @jit
    def _sor_step(psi, rhs, r, dr, dz, omega):
        """Single SOR iteration step for Delta* operator."""
        nr, nz = psi.shape

        # Coefficients for Delta* = d^2/dr^2 - (1/r)*d/dr + d^2/dz^2
        # Discretized: a_E*psi_E + a_W*psi_W + a_N*psi_N + a_S*psi_S + a_P*psi_P = rhs
        a_E = 1/dr**2 + 1/(2*r*dr)  # East (r+dr)
        a_W = 1/dr**2 - 1/(2*r*dr)  # West (r-dr)
        a_N = 1/dz**2               # North (z+dz)
        a_S = 1/dz**2               # South (z-dz)
        a_P = -2/dr**2 - 2/dz**2    # Center

        # Jacobi-like update (vectorized)
        psi_E = jnp.roll(psi, -1, axis=0)
        psi_W = jnp.roll(psi, 1, axis=0)
        psi_N = jnp.roll(psi, -1, axis=1)
        psi_S = jnp.roll(psi, 1, axis=1)

        psi_gs = (rhs - a_E*psi_E - a_W*psi_W - a_N*psi_N - a_S*psi_S) / a_P

        # SOR relaxation
        psi_new = psi + omega * (psi_gs - psi)

        return psi_new

    def _initial_frc_guess(self, geometry: Geometry) -> Array:
        """Generate initial guess for FRC equilibrium."""
        r = geometry.r_grid
        z = geometry.z_grid
        r_s = (geometry.r_max - geometry.r_min) / 2  # Separatrix radius
        z_s = (geometry.z_max - geometry.z_min) / 4  # Separatrix half-length

        # Simple FRC profile: psi ~ (1 - (r/r_s)^2) * exp(-(z/z_s)^2)
        r_norm = (r - geometry.r_min) / (geometry.r_max - geometry.r_min)
        psi = (1 - (2*r_norm - 1)**2) * jnp.exp(-(z / z_s)**2)
        psi = jnp.maximum(psi, 0)

        return psi

    def compute_profiles(self, psi: Array, geometry: Geometry,
                        constraints: EquilibriumConstraints) -> dict:
        """Compute derived equilibrium profiles from psi."""
        dr, dz = geometry.dr, geometry.dz
        r = geometry.r_grid

        # Magnetic field: B_r = -(1/r)*dpsi/dz, B_z = (1/r)*dpsi/dr
        dpsi_dr = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2*dr)
        dpsi_dz = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) / (2*dz)

        B_r = -dpsi_dz / r
        B_z = dpsi_dr / r
        B_phi = jnp.zeros_like(psi)  # No toroidal field for FRC

        B = jnp.stack([B_r, B_phi, B_z], axis=-1)

        # Current density: j_phi = -Delta*psi / (mu_0 * r)
        delta_star_psi = self._laplace_star(psi, dr, dz, r)
        j_phi = -delta_star_psi / (MU0 * r)

        # Pressure from profile function
        psi_max = jnp.max(psi)
        psi_norm = psi / jnp.maximum(psi_max, 1e-10)
        if constraints.pressure_profile is not None:
            pressure = constraints.pressure_profile(psi_norm)
        else:
            pressure = 1e4 * (1 - psi_norm**2)  # Default parabolic

        # Density (assume isothermal for simplicity)
        T0 = 1000.0  # eV
        density = pressure / (T0 * 1.602e-19)  # n = p / (kT)

        return {
            "B": B,
            "B_r": B_r,
            "B_z": B_z,
            "B_phi": B_phi,
            "j_phi": j_phi,
            "pressure": pressure,
            "density": density,
            "psi_norm": psi_norm,
        }

    @staticmethod
    def _laplace_star(psi, dr, dz, r):
        """Compute Delta* operator."""
        psi_rr = (jnp.roll(psi, -1, axis=0) - 2*psi + jnp.roll(psi, 1, axis=0)) / dr**2
        psi_zz = (jnp.roll(psi, -1, axis=1) - 2*psi + jnp.roll(psi, 1, axis=1)) / dz**2
        psi_r = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2*dr)
        return psi_rr - (1.0/r) * psi_r + psi_zz
