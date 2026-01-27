"""Tests for AtomicCoupling source term computation."""

import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.models.neutral_fluid import NeutralState
from jax_frc.models.atomic_coupling import AtomicCoupling, AtomicCouplingConfig
from jax_frc.constants import MI, QE


def test_compute_sources_returns_opposite_mass_rates():
    """Mass source terms should have opposite signs (conservation)."""
    config = AtomicCouplingConfig(include_radiation=False)
    coupling = AtomicCoupling(config)

    geometry = Geometry(
        nx=8, ny=8, nz=8,
        x_min=-0.5, x_max=0.5,
        y_min=-0.5, y_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    plasma = State.zeros(8, 8, 8)
    n_e = 1e19
    T_e = 100 * QE
    plasma = plasma.replace(
        n=jnp.ones((8, 8, 8)) * n_e,
        Te=jnp.ones((8, 8, 8)) * T_e,
        p=jnp.ones((8, 8, 8)) * n_e * T_e * 2,
        v=jnp.zeros((8, 8, 8, 3))
    )

    neutral = NeutralState(
        rho_n=jnp.ones((8, 8, 8)) * 1e-6,
        mom_n=jnp.zeros((8, 8, 8, 3)),
        E_n=jnp.ones((8, 8, 8)) * 100.0
    )

    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # Mass conservation
    total_mass_rate = plasma_src.mass + neutral_src.mass
    assert jnp.allclose(total_mass_rate, 0.0, atol=1e-30)


def test_compute_sources_momentum_conservation():
    """Momentum source terms should have opposite signs."""
    config = AtomicCouplingConfig(include_radiation=False)
    coupling = AtomicCoupling(config)

    geometry = Geometry(
        nx=8, ny=8, nz=8,
        x_min=-0.5, x_max=0.5,
        y_min=-0.5, y_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    plasma = State.zeros(8, 8, 8)
    plasma = plasma.replace(
        n=jnp.ones((8, 8, 8)) * 1e19,
        Te=jnp.ones((8, 8, 8)) * 100 * QE,
        p=jnp.ones((8, 8, 8)) * 1e19 * 100 * QE * 2,
        v=jnp.zeros((8, 8, 8, 3))
    )

    neutral = NeutralState(
        rho_n=jnp.ones((8, 8, 8)) * 1e-6,
        mom_n=jnp.zeros((8, 8, 8, 3)).at[:, :, :, 2].set(1e-6 * 1000),
        E_n=jnp.ones((8, 8, 8)) * 100.0
    )

    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # Momentum conservation
    total_mom = plasma_src.momentum + neutral_src.momentum
    assert jnp.allclose(total_mom, 0.0, atol=1e-25)


def test_compute_sources_with_radiation():
    """Test that radiation configuration is handled."""
    config_with_rad = AtomicCouplingConfig(
        include_radiation=True,
        impurity_fraction=0.01,
        Z_eff=1.5
    )
    coupling = AtomicCoupling(config_with_rad)

    geometry = Geometry(
        nx=8, ny=8, nz=8,
        x_min=-0.5, x_max=0.5,
        y_min=-0.5, y_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    plasma = State.zeros(8, 8, 8)
    plasma = plasma.replace(
        n=jnp.ones((8, 8, 8)) * 1e19,
        Te=jnp.ones((8, 8, 8)) * 100 * QE,
        p=jnp.ones((8, 8, 8)) * 1e19 * 100 * QE * 2,
        v=jnp.zeros((8, 8, 8, 3))
    )

    neutral = NeutralState(
        rho_n=jnp.ones((8, 8, 8)) * 1e-6,
        mom_n=jnp.zeros((8, 8, 8, 3)),
        E_n=jnp.ones((8, 8, 8)) * 100.0
    )

    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # Mass conservation still holds with radiation
    total_mass_rate = plasma_src.mass + neutral_src.mass
    assert jnp.allclose(total_mass_rate, 0.0, atol=1e-30)

    # Momentum conservation still holds
    total_mom = plasma_src.momentum + neutral_src.momentum
    assert jnp.allclose(total_mom, 0.0, atol=1e-25)

    # Radiation should cause plasma energy loss (negative energy source)
    # At 100 eV and 1e19 m^-3, expect some radiation loss
    assert jnp.all(plasma_src.energy < 0), "Plasma should lose energy to radiation"


def test_ionization_dominates_at_high_temperature():
    """At high Te, ionization should dominate over recombination."""
    config = AtomicCouplingConfig(include_radiation=False)
    coupling = AtomicCoupling(config)

    geometry = Geometry(
        nx=8, ny=8, nz=8,
        x_min=-0.5, x_max=0.5,
        y_min=-0.5, y_max=0.5,
        z_min=-0.5, z_max=0.5
    )

    # High temperature plasma (1 keV)
    plasma = State.zeros(8, 8, 8)
    plasma = plasma.replace(
        n=jnp.ones((8, 8, 8)) * 1e19,
        Te=jnp.ones((8, 8, 8)) * 1000 * QE,  # 1 keV
        p=jnp.ones((8, 8, 8)) * 1e19 * 1000 * QE * 2,
        v=jnp.zeros((8, 8, 8, 3))
    )

    # Significant neutral density
    neutral = NeutralState(
        rho_n=jnp.ones((8, 8, 8)) * 1e-5,
        mom_n=jnp.zeros((8, 8, 8, 3)),
        E_n=jnp.ones((8, 8, 8)) * 100.0
    )

    plasma_src, neutral_src = coupling.compute_sources(plasma, neutral, geometry)

    # At high Te with neutrals present, net ionization should dominate
    # Plasma gains mass (positive), neutrals lose mass (negative)
    assert jnp.all(plasma_src.mass > 0), "Plasma should gain mass from ionization"
    assert jnp.all(neutral_src.mass < 0), "Neutrals should lose mass to ionization"
