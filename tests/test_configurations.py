"""Tests for configuration base class."""
import pytest
from abc import ABC


def test_abstract_configuration_cannot_instantiate():
    """AbstractConfiguration should not be instantiatable."""
    from jax_frc.configurations.base import AbstractConfiguration

    with pytest.raises(TypeError):
        AbstractConfiguration()


def test_abstract_configuration_has_required_methods():
    """AbstractConfiguration defines required abstract methods."""
    from jax_frc.configurations.base import AbstractConfiguration

    assert hasattr(AbstractConfiguration, 'build_geometry')
    assert hasattr(AbstractConfiguration, 'build_initial_state')
    assert hasattr(AbstractConfiguration, 'build_model')
    assert hasattr(AbstractConfiguration, 'build_boundary_conditions')


def test_abstract_configuration_is_abc():
    """AbstractConfiguration should be an ABC subclass."""
    from jax_frc.configurations.base import AbstractConfiguration

    assert issubclass(AbstractConfiguration, ABC)


def test_abstract_configuration_has_helper_methods():
    """AbstractConfiguration has non-abstract helper methods."""
    from jax_frc.configurations.base import AbstractConfiguration

    assert hasattr(AbstractConfiguration, 'available_phases')
    assert hasattr(AbstractConfiguration, 'default_runtime')


def test_abstract_configuration_importable_from_package():
    """AbstractConfiguration should be importable from configurations package."""
    from jax_frc.configurations import AbstractConfiguration

    assert AbstractConfiguration is not None


def test_cartesian_helper_geometry():
    from tests.utils.cartesian import make_geometry

    geom = make_geometry(nx=4, ny=1, nz=8)

    assert geom.nx == 4 and geom.ny == 1 and geom.nz == 8


# MagneticDiffusionConfiguration tests
def test_magnetic_diffusion_builds_geometry():
    """MagneticDiffusionConfiguration creates valid 3D Cartesian geometry."""
    from jax_frc.configurations import MagneticDiffusionConfiguration

    config = MagneticDiffusionConfiguration()
    geometry = config.build_geometry()

    assert geometry.nx > 0
    assert geometry.ny > 0
    assert geometry.nz > 0


def test_magnetic_diffusion_builds_initial_state():
    """MagneticDiffusionConfiguration creates state with Gaussian B_z in x-y plane."""
    from jax_frc.configurations import MagneticDiffusionConfiguration
    import jax.numpy as jnp
    from tests.utils.cartesian import make_geometry

    config = MagneticDiffusionConfiguration()
    geometry = make_geometry(nx=8, ny=8, nz=1)
    state = config.build_initial_state(geometry)

    # B_z should have Gaussian profile (peak in center)
    x_center = geometry.nx // 2
    y_center = geometry.ny // 2
    z_center = geometry.nz // 2
    # Center should have higher B_z than edge (in x-y plane)
    assert state.B[x_center, y_center, z_center, 2] > state.B[0, y_center, z_center, 2]


def test_magnetic_diffusion_builds_model():
    """MagneticDiffusionConfiguration creates ExtendedMHD model by default."""
    from jax_frc.configurations import MagneticDiffusionConfiguration
    from jax_frc.models.extended_mhd import ExtendedMHD

    config = MagneticDiffusionConfiguration()
    model = config.build_model()

    assert isinstance(model, ExtendedMHD)
    # Verify Hall and electron pressure are disabled for diffusion test
    assert model.include_hall is False
    assert model.include_electron_pressure is False


# FrozenFluxConfiguration tests
def test_frozen_flux_default_grid_dims_cartesian():
    """FrozenFluxConfiguration defaults to pseudo-2D Cartesian grid (thin z)."""
    from jax_frc.configurations import FrozenFluxConfiguration

    config = FrozenFluxConfiguration()

    assert config.nx == 64
    assert config.ny == 64
    assert config.nz == 1  # Thin z for pseudo-2D in x-y plane


def test_frozen_flux_builds_geometry():
    """FrozenFluxConfiguration creates valid geometry."""
    from jax_frc.configurations import FrozenFluxConfiguration

    config = FrozenFluxConfiguration()
    geometry = config.build_geometry()

    assert geometry.nx > 0
    assert geometry.ny > 0
    assert geometry.nz > 0


def test_frozen_flux_builds_initial_state():
    """FrozenFluxConfiguration creates state with magnetic loop and rotation velocity."""
    from jax_frc.configurations import FrozenFluxConfiguration
    import jax.numpy as jnp

    config = FrozenFluxConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # B should have non-zero values (magnetic loop)
    assert jnp.max(jnp.abs(state.B)) > 0
    # v should have rotation pattern (v_x = -omega*y, v_y = omega*x)
    assert jnp.max(jnp.abs(state.v)) > 0


def test_frozen_flux_builds_model():
    """FrozenFluxConfiguration creates ResistiveMHD model."""
    from jax_frc.configurations import FrozenFluxConfiguration
    from jax_frc.models.resistive_mhd import ResistiveMHD

    config = FrozenFluxConfiguration()
    model = config.build_model()

    assert isinstance(model, ResistiveMHD)


# Configuration Registry tests
def test_configuration_registry_has_magnetic_diffusion():
    """Registry contains MagneticDiffusionConfiguration."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'MagneticDiffusionConfiguration' in CONFIGURATION_REGISTRY
    assert CONFIGURATION_REGISTRY['MagneticDiffusionConfiguration'] is not None


def test_configuration_registry_has_frozen_flux():
    """Registry contains FrozenFluxConfiguration."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'FrozenFluxConfiguration' in CONFIGURATION_REGISTRY
    assert CONFIGURATION_REGISTRY['FrozenFluxConfiguration'] is not None


def test_configuration_registry_creates_instance():
    """Registry can create configuration instances."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    ConfigClass = CONFIGURATION_REGISTRY['MagneticDiffusionConfiguration']
    config = ConfigClass()

    assert config.name == "magnetic_diffusion"
