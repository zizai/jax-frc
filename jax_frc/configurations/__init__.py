"""Configuration classes for reactor and benchmark setups."""

# Core abstractions
from jax_frc.configurations.base import AbstractConfiguration
from jax_frc.configurations.phase import (
    Phase,
    PhaseResult,
    PHASE_REGISTRY,
    register_phase,
)

# Transitions
from jax_frc.configurations.transitions import (
    Transition,
    timeout,
    condition,
    any_of,
    all_of,
    separation_below,
    temperature_above,
    flux_below,
    velocity_below,
    transition_from_spec,
)

# Configuration implementations
from jax_frc.configurations.linear_configuration import (
    LinearConfiguration,
    TransitionSpec,
    PhaseSpec,
    ConfigurationResult,
)
from jax_frc.configurations.frc_merging import (
    BelovaMergingConfiguration,
    BelovaCase1Configuration,
    BelovaCase2Configuration,
    BelovaCase4Configuration,
)
from jax_frc.configurations.magnetic_diffusion import MagneticDiffusionConfiguration
from jax_frc.configurations.frozen_flux import FrozenFluxConfiguration
from jax_frc.configurations.orszag_tang import OrszagTangConfiguration
from jax_frc.configurations.gem_reconnection import GEMReconnectionConfiguration
from jax_frc.configurations.brio_wu_shock import BrioWuShockConfiguration

# Import phases submodule to trigger registration
from jax_frc.configurations import phases

CONFIGURATION_REGISTRY = {
    'MagneticDiffusionConfiguration': MagneticDiffusionConfiguration,
    'FrozenFluxConfiguration': FrozenFluxConfiguration,
    'OrszagTangConfiguration': OrszagTangConfiguration,
    'GEMReconnectionConfiguration': GEMReconnectionConfiguration,
    'BrioWuShockConfiguration': BrioWuShockConfiguration,
    'LinearConfiguration': LinearConfiguration,
    'BelovaMergingConfiguration': BelovaMergingConfiguration,
    'BelovaCase1Configuration': BelovaCase1Configuration,
    'BelovaCase2Configuration': BelovaCase2Configuration,
    'BelovaCase4Configuration': BelovaCase4Configuration,
}

__all__ = [
    # Core abstractions
    'AbstractConfiguration',
    'Phase',
    'PhaseResult',
    'PHASE_REGISTRY',
    'register_phase',
    # Transitions
    'Transition',
    'timeout',
    'condition',
    'any_of',
    'all_of',
    'separation_below',
    'temperature_above',
    'flux_below',
    'velocity_below',
    'transition_from_spec',
    # Configuration implementations
    'MagneticDiffusionConfiguration',
    'FrozenFluxConfiguration',
    'OrszagTangConfiguration',
    'GEMReconnectionConfiguration',
    'BrioWuShockConfiguration',
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
