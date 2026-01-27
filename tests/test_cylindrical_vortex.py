"""Tests for CylindricalVortexConfiguration."""
import pytest
import jax.numpy as jnp


def test_cylindrical_vortex_builds_geometry():
    """Configuration creates Cartesian geometry."""
    from jax_frc.configurations import CylindricalVortexConfiguration

    config = CylindricalVortexConfiguration()
    geometry = config.build_geometry()

    assert geometry.x_min == 0.2
    assert geometry.x_max == 1.2
    assert geometry.nx == 256
    assert geometry.ny == 4
    assert geometry.nz == 256


def test_cylindrical_vortex_initial_velocity():
    """Initial velocity has vortex pattern."""
    from jax_frc.configurations import CylindricalVortexConfiguration

    config = CylindricalVortexConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # vr = -v0 * sin(z), so vr should be nonzero
    assert jnp.max(jnp.abs(state.v[:, :, :, 0])) > 0.5


def test_cylindrical_vortex_initial_magnetic():
    """Initial B field has vortex pattern."""
    from jax_frc.configurations import CylindricalVortexConfiguration

    config = CylindricalVortexConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Br = -B0 * sin(z), so Br should be nonzero
    assert jnp.max(jnp.abs(state.B[:, :, :, 0])) > 0.5


def test_cylindrical_vortex_uniform_density():
    """Density should be uniform."""
    from jax_frc.configurations import CylindricalVortexConfiguration

    config = CylindricalVortexConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    assert jnp.allclose(state.n, state.n[0, 0, 0], rtol=0.01)


def test_cylindrical_vortex_in_registry():
    """Configuration is in the registry."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'CylindricalVortexConfiguration' in CONFIGURATION_REGISTRY
