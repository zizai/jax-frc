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


# SlabDiffusionConfiguration tests
def test_slab_diffusion_builds_geometry():
    """SlabDiffusionConfiguration creates valid geometry."""
    from jax_frc.configurations.analytic import SlabDiffusionConfiguration

    config = SlabDiffusionConfiguration()
    geometry = config.build_geometry()

    assert geometry.nr > 0
    assert geometry.nz > 0
    assert geometry.coord_system == "cylindrical"


def test_slab_diffusion_builds_initial_state():
    """SlabDiffusionConfiguration creates state with Gaussian temperature."""
    from jax_frc.configurations.analytic import SlabDiffusionConfiguration
    import jax.numpy as jnp

    config = SlabDiffusionConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Temperature should have Gaussian profile (peak in center)
    center_idx = geometry.nz // 2
    assert state.T[geometry.nr // 2, center_idx] > state.T[geometry.nr // 2, 0]


def test_slab_diffusion_builds_model():
    """SlabDiffusionConfiguration creates ExtendedMHD with thermal transport."""
    from jax_frc.configurations.analytic import SlabDiffusionConfiguration
    from jax_frc.models.extended_mhd import ExtendedMHD

    config = SlabDiffusionConfiguration()
    model = config.build_model()

    assert isinstance(model, ExtendedMHD)
    assert model.thermal is not None


# Configuration Registry tests
def test_configuration_registry_has_slab_diffusion():
    """Registry contains SlabDiffusionConfiguration."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'SlabDiffusionConfiguration' in CONFIGURATION_REGISTRY
    assert CONFIGURATION_REGISTRY['SlabDiffusionConfiguration'] is not None


def test_configuration_registry_creates_instance():
    """Registry can create configuration instances."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    ConfigClass = CONFIGURATION_REGISTRY['SlabDiffusionConfiguration']
    config = ConfigClass()

    assert config.name == "slab_diffusion"
