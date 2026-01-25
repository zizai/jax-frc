"""Configuration classes for reactor and benchmark setups."""
from .base import AbstractConfiguration
from .analytic import SlabDiffusionConfiguration

CONFIGURATION_REGISTRY = {
    'SlabDiffusionConfiguration': SlabDiffusionConfiguration,
}

__all__ = ['AbstractConfiguration', 'SlabDiffusionConfiguration', 'CONFIGURATION_REGISTRY']
