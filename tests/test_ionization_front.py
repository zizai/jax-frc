# tests/test_ionization_front.py
"""Validation test: Ionization front propagation."""

import jax.numpy as jnp
import jax.lax as lax
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.neutral_fluid import NeutralState
from jax_frc.models.coupled import CoupledState
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig
from jax_frc.constants import MI, QE


def test_ionization_front_mass_conservation():
    """Total mass (plasma + neutral) conserved during ionization.

    This tests the fundamental conservation property of the coupling.
    """
    n_e = 1e19  # Initial plasma density
    T_e = 100 * QE  # 100 eV - hot enough for ionization
    rho_n = 1e-5  # Initial neutral density [kg/m^3]

    geometry = Geometry(
        coord_system="cylindrical",
        nr=8, nz=8,
        r_min=0.01, r_max=0.1,
        z_min=-0.1, z_max=0.1
    )

    plasma = State.zeros(8, 8)
    plasma = plasma.replace(
        n=jnp.ones((8, 8)) * n_e,
        T=jnp.ones((8, 8)) * T_e,
        p=jnp.ones((8, 8)) * n_e * T_e * 2,
        v=jnp.zeros((8, 8, 3))
    )

    neutral = NeutralState(
        rho_n=jnp.ones((8, 8)) * rho_n,
        mom_n=jnp.zeros((8, 8, 3)),
        E_n=jnp.ones((8, 8)) * rho_n * 300 * QE / MI  # Room temp neutrals
    )

    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))

    # Initial total mass
    initial_plasma_mass = jnp.sum(plasma.n * MI)
    initial_neutral_mass = jnp.sum(neutral.rho_n)
    initial_total = initial_plasma_mass + initial_neutral_mass

    # Get source rates
    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # Source rates should conserve mass: plasma gains = neutral loses
    plasma_mass_rate = jnp.sum(plasma_src.mass)
    neutral_mass_rate = jnp.sum(neutral_src.mass)

    assert jnp.allclose(plasma_mass_rate + neutral_mass_rate, 0.0, atol=1e-30), \
        "Mass source terms don't sum to zero"

    # After a small time step, total mass should be conserved
    dt = 1e-8
    new_plasma_n = plasma.n + plasma_src.mass / MI * dt
    new_neutral_rho = neutral.rho_n + neutral_src.mass * dt

    final_plasma_mass = jnp.sum(new_plasma_n * MI)
    final_neutral_mass = jnp.sum(new_neutral_rho)
    final_total = final_plasma_mass + final_neutral_mass

    relative_error = jnp.abs(final_total - initial_total) / initial_total
    # Use 1e-6 tolerance to account for float32 precision
    assert relative_error < 1e-6, f"Mass conservation violated: {relative_error:.2e}"


def test_ionization_creates_plasma():
    """Hot plasma ionizes cold neutrals, increasing plasma density."""
    n_e = 1e18  # Lower initial plasma density
    T_e = 200 * QE  # 200 eV - very hot
    rho_n = 1e-4  # Significant neutral density

    geometry = Geometry(
        coord_system="cylindrical",
        nr=4, nz=4,
        r_min=0.01, r_max=0.1,
        z_min=-0.1, z_max=0.1
    )

    plasma = State.zeros(4, 4)
    plasma = plasma.replace(
        n=jnp.ones((4, 4)) * n_e,
        T=jnp.ones((4, 4)) * T_e,
        p=jnp.ones((4, 4)) * n_e * T_e * 2
    )

    neutral = NeutralState(
        rho_n=jnp.ones((4, 4)) * rho_n,
        mom_n=jnp.zeros((4, 4, 3)),
        E_n=jnp.ones((4, 4)) * 100.0
    )

    coupling = AtomicCoupling(AtomicCouplingConfig(include_radiation=False))
    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # At high temperature, ionization >> recombination
    # So plasma should gain mass (plasma_src.mass > 0)
    # and neutrals should lose mass (neutral_src.mass < 0)
    assert jnp.all(plasma_src.mass > 0), "Plasma should gain mass from ionization"
    assert jnp.all(neutral_src.mass < 0), "Neutrals should lose mass to ionization"
