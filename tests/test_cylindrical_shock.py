"""Tests for CylindricalShockConfiguration."""
import pytest
import jax.numpy as jnp


def test_cylindrical_shock_builds_geometry():
    """Configuration creates cylindrical geometry."""
    from jax_frc.configurations import CylindricalShockConfiguration

    config = CylindricalShockConfiguration()
    geometry = config.build_geometry()

    assert geometry.coord_system == "cylindrical"
    assert geometry.nr == 16  # Minimal r resolution
    assert geometry.nz == 512
    assert geometry.z_min == -1.0
    assert geometry.z_max == 1.0


def test_cylindrical_shock_initial_conditions():
    """Initial state has Brio-Wu left/right states."""
    from jax_frc.configurations import CylindricalShockConfiguration

    config = CylindricalShockConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Left state (z < 0): rho=1.0, p=1.0
    left_idx = geometry.nz // 4
    assert jnp.allclose(state.n[0, left_idx], 1.0, rtol=0.01)
    assert jnp.allclose(state.p[0, left_idx], 1.0, rtol=0.01)

    # Right state (z > 0): rho=0.125, p=0.1
    right_idx = 3 * geometry.nz // 4
    assert jnp.allclose(state.n[0, right_idx], 0.125, rtol=0.01)
    assert jnp.allclose(state.p[0, right_idx], 0.1, rtol=0.01)


def test_cylindrical_shock_magnetic_field():
    """B field has Bz=0.75 constant, Br reverses."""
    from jax_frc.configurations import CylindricalShockConfiguration

    config = CylindricalShockConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Bz should be constant 0.75
    assert jnp.allclose(state.B[:, :, 2], 0.75, rtol=0.01)

    # Br should be +1 on left, -1 on right
    left_idx = geometry.nz // 4
    right_idx = 3 * geometry.nz // 4
    assert state.B[0, left_idx, 0] > 0.5  # Br > 0 on left
    assert state.B[0, right_idx, 0] < -0.5  # Br < 0 on right


def test_cylindrical_shock_builds_model():
    """Configuration creates ResistiveMHD model."""
    from jax_frc.configurations import CylindricalShockConfiguration
    from jax_frc.models.resistive_mhd import ResistiveMHD

    config = CylindricalShockConfiguration()
    model = config.build_model()

    # Should use resistive MHD (not extended)
    assert isinstance(model, ResistiveMHD)


def test_cylindrical_shock_in_registry():
    """Configuration is in the registry."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'CylindricalShockConfiguration' in CONFIGURATION_REGISTRY
