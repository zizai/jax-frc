"""Tests for CylindricalGEMConfiguration."""
import pytest
import jax.numpy as jnp


def test_cylindrical_gem_builds_geometry():
    """Configuration creates correct domain."""
    from jax_frc.configurations import CylindricalGEMConfiguration

    config = CylindricalGEMConfiguration()
    geometry = config.build_geometry()

    assert geometry.nx == 256
    assert geometry.ny == 4
    assert geometry.nz == 512


def test_cylindrical_gem_harris_sheet():
    """Initial Br follows tanh profile."""
    from jax_frc.configurations import CylindricalGEMConfiguration

    config = CylindricalGEMConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Br should be ~+B0 for z >> lambda, ~-B0 for z << -lambda
    center_x = geometry.nx // 2
    center_y = geometry.ny // 2
    z_idx_pos = 3 * geometry.nz // 4  # z > 0
    z_idx_neg = geometry.nz // 4  # z < 0

    assert state.B[center_x, center_y, z_idx_pos, 0] > 0.5 * config.B0
    assert state.B[center_x, center_y, z_idx_neg, 0] < -0.5 * config.B0


def test_cylindrical_gem_density_profile():
    """Density has sech^2 + background profile."""
    from jax_frc.configurations import CylindricalGEMConfiguration

    config = CylindricalGEMConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Density should peak at z=0
    center_z = geometry.nz // 2
    edge_z = 0
    center_x = geometry.nx // 2
    center_y = geometry.ny // 2

    assert state.n[center_x, center_y, center_z] > state.n[center_x, center_y, edge_z]


def test_cylindrical_gem_uses_extended_mhd():
    """Configuration uses ExtendedMHD with Hall."""
    from jax_frc.configurations import CylindricalGEMConfiguration
    from jax_frc.models.extended_mhd import ExtendedMHD

    config = CylindricalGEMConfiguration()
    model = config.build_model()

    assert isinstance(model, ExtendedMHD)


def test_cylindrical_gem_in_registry():
    """Configuration is in the registry."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'CylindricalGEMConfiguration' in CONFIGURATION_REGISTRY
