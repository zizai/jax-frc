"""Configuration classes for reactor and benchmark setups."""
from .base import AbstractConfiguration
from .analytic import SlabDiffusionConfiguration
from .linear_configuration import (
    LinearConfiguration,
    ConfigurationResult,
    TransitionSpec,
    PhaseSpec,
)
from .frc_merging import (
    BelovaMergingConfiguration,
    BelovaCase1Configuration,
    BelovaCase2Configuration,
    BelovaCase4Configuration,
)

CONFIGURATION_REGISTRY = {
    'SlabDiffusionConfiguration': SlabDiffusionConfiguration,
    'LinearConfiguration': LinearConfiguration,
    'BelovaMergingConfiguration': BelovaMergingConfiguration,
    'BelovaCase1Configuration': BelovaCase1Configuration,
    'BelovaCase2Configuration': BelovaCase2Configuration,
    'BelovaCase4Configuration': BelovaCase4Configuration,
}

__all__ = [
    'AbstractConfiguration',
    'SlabDiffusionConfiguration',
    'LinearConfiguration',
    'ConfigurationResult',
    'TransitionSpec',
    'PhaseSpec',
    'BelovaMergingConfiguration',
    'BelovaCase1Configuration',
    'BelovaCase2Configuration',
    'BelovaCase4Configuration',
    'CONFIGURATION_REGISTRY',
]
