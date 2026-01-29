"""Tests for BrioWuShockConfiguration."""
import pytest
import jax.numpy as jnp


def test_brio_wu_shock_builds_geometry():
    """Configuration creates Cartesian geometry."""
    from jax_frc.configurations import BrioWuShockConfiguration

    config = BrioWuShockConfiguration()
    geometry = config.build_geometry()

    assert geometry.nx == 1  # Pseudo-1D
    assert geometry.ny == 1  # Pseudo-1D
    assert geometry.nz == 512
    assert geometry.z_min == -1.0
    assert geometry.z_max == 1.0


def test_brio_wu_shock_initial_conditions():
    """Initial state has Brio-Wu left/right states."""
    from jax_frc.configurations import BrioWuShockConfiguration

    config = BrioWuShockConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Left state (z < 0): rho=1.0, p=1.0
    left_idx = geometry.nz // 4
    x_idx = geometry.nx // 2
    y_idx = geometry.ny // 2
    assert jnp.allclose(state.n[x_idx, y_idx, left_idx], 1.0, rtol=0.01)
    assert jnp.allclose(state.p[x_idx, y_idx, left_idx], 1.0, rtol=0.01)

    # Right state (z > 0): rho=0.125, p=0.1
    right_idx = 3 * geometry.nz // 4
    assert jnp.allclose(state.n[x_idx, y_idx, right_idx], 0.125, rtol=0.01)
    assert jnp.allclose(state.p[x_idx, y_idx, right_idx], 0.1, rtol=0.01)


def test_brio_wu_shock_magnetic_field():
    """B field has Bz=0.75 constant, Bx reverses."""
    from jax_frc.configurations import BrioWuShockConfiguration

    config = BrioWuShockConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Bz should be constant 0.75
    assert jnp.allclose(state.B[:, :, :, 2], 0.75, rtol=0.01)

    # Bx should be +1 on left, -1 on right
    left_idx = geometry.nz // 4
    right_idx = 3 * geometry.nz // 4
    x_idx = geometry.nx // 2
    y_idx = geometry.ny // 2
    assert state.B[x_idx, y_idx, left_idx, 0] > 0.5  # Bx > 0 on left
    assert state.B[x_idx, y_idx, right_idx, 0] < -0.5  # Bx < 0 on right


def test_brio_wu_shock_builds_model():
    """Configuration creates ResistiveMHD model."""
    from jax_frc.configurations import BrioWuShockConfiguration
    from jax_frc.models.resistive_mhd import ResistiveMHD

    config = BrioWuShockConfiguration()
    model = config.build_model()

    # Should use resistive MHD (not extended)
    assert isinstance(model, ResistiveMHD)


def test_brio_wu_shock_in_registry():
    """Configuration is in the registry."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'BrioWuShockConfiguration' in CONFIGURATION_REGISTRY
