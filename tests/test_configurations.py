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
