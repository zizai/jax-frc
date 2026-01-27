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


# MagneticDiffusionConfiguration tests
def test_magnetic_diffusion_builds_geometry():
    """MagneticDiffusionConfiguration creates valid geometry."""
    from jax_frc.configurations import MagneticDiffusionConfiguration

    config = MagneticDiffusionConfiguration()
    geometry = config.build_geometry()

    assert geometry.nr > 0
    assert geometry.nz > 0
    assert geometry.coord_system == "cylindrical"


def test_magnetic_diffusion_builds_initial_state():
    """MagneticDiffusionConfiguration creates state with Gaussian B_z."""
    from jax_frc.configurations import MagneticDiffusionConfiguration
    import jax.numpy as jnp

    config = MagneticDiffusionConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # B_z should have Gaussian profile (peak in center)
    center_idx = geometry.nz // 2
    assert state.B[geometry.nr // 2, center_idx, 2] > state.B[geometry.nr // 2, 0, 2]


def test_magnetic_diffusion_builds_model():
    """MagneticDiffusionConfiguration creates ResistiveMHD model."""
    from jax_frc.configurations import MagneticDiffusionConfiguration
    from jax_frc.models.resistive_mhd import ResistiveMHD

    config = MagneticDiffusionConfiguration()
    model = config.build_model()

    assert isinstance(model, ResistiveMHD)


# FrozenFluxConfiguration tests
def test_frozen_flux_builds_geometry():
    """FrozenFluxConfiguration creates valid geometry."""
    from jax_frc.configurations import FrozenFluxConfiguration

    config = FrozenFluxConfiguration()
    geometry = config.build_geometry()

    assert geometry.nr > 0
    assert geometry.nz > 0
    assert geometry.coord_system == "cylindrical"


def test_frozen_flux_builds_initial_state():
    """FrozenFluxConfiguration creates state with uniform B_phi and v_r."""
    from jax_frc.configurations import FrozenFluxConfiguration
    import jax.numpy as jnp

    config = FrozenFluxConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # B_phi should be uniform
    assert jnp.allclose(state.B[:, :, 1], config.B_phi_0)
    # v_r should be uniform
    assert jnp.allclose(state.v[:, :, 0], config.v_r)


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
