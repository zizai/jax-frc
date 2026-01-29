"""Tests for OrszagTangConfiguration."""
import pytest
import jax.numpy as jnp


def test_orszag_tang_builds_geometry():
    """Configuration creates Cartesian geometry."""
    from jax_frc.configurations import OrszagTangConfiguration

    config = OrszagTangConfiguration()
    geometry = config.build_geometry()

    assert geometry.x_min == 0.0
    assert geometry.nx == 256
    assert geometry.ny == 1
    assert geometry.nz == 256


def test_orszag_tang_initial_velocity():
    """Initial velocity has vortex pattern."""
    from jax_frc.configurations import OrszagTangConfiguration

    config = OrszagTangConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # vx = -v0 * sin(z), so vx should be nonzero
    assert jnp.max(jnp.abs(state.v[:, :, :, 0])) > 0.5


def test_orszag_tang_initial_magnetic():
    """Initial B field has vortex pattern."""
    from jax_frc.configurations import OrszagTangConfiguration

    config = OrszagTangConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Bx = -B0 * sin(z), so Bx should be nonzero
    assert jnp.max(jnp.abs(state.B[:, :, :, 0])) > 0.5


def test_orszag_tang_uniform_density():
    """Density should be uniform."""
    from jax_frc.configurations import OrszagTangConfiguration

    config = OrszagTangConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    assert jnp.allclose(state.n, state.n[0, 0, 0], rtol=0.01)


def test_orszag_tang_in_registry():
    """Configuration is in the registry."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'OrszagTangConfiguration' in CONFIGURATION_REGISTRY
