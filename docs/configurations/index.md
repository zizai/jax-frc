# Configurations

The configurations module provides a unified system for defining simulation setups, including geometry, initial conditions, physics models, and multi-phase execution.

## Package Structure

```
jax_frc/configurations/
├── base.py              # AbstractConfiguration base class
├── linear_configuration.py  # LinearConfiguration for YAML-based setup
├── frc_merging.py       # Belova merging configurations
├── magnetic_diffusion.py    # Magnetic diffusion test case
├── phase.py             # Phase base class and registry
├── transitions.py       # Transition conditions
└── phases/
    └── merging.py       # MergingPhase implementation
```

## Basic Usage

### Using Pre-built Configurations

```python
from jax_frc.configurations import BelovaCase2Configuration

# Create configuration
config = BelovaCase2Configuration()

# Build simulation components
geometry = config.build_geometry()
state = config.build_initial_state(geometry)
model = config.build_model()
```

### Using LinearConfiguration with YAML

```python
from jax_frc.configurations import LinearConfiguration

# Load from YAML
config = LinearConfiguration.from_yaml("cases/my_simulation.yaml")

# Or create programmatically
config = LinearConfiguration(
    name="my_sim",
    geometry={"nr": 64, "nz": 128, "r_max": 1.0, "z_max": 2.0},
    model={"type": "extended_mhd", "resistivity": {"eta_0": 1e-6}},
    phases=[{"class": "MergingPhase", "config": {"separation": 2.0}}]
)
```

## Configuration Classes

| Class | Description |
|-------|-------------|
| `AbstractConfiguration` | Base class defining the configuration interface |
| `LinearConfiguration` | YAML-driven configuration with phase support |
| `BelovaMergingConfiguration` | Base FRC merging configuration |
| `BelovaCase1Configuration` | Large FRC, no compression |
| `BelovaCase2Configuration` | Small FRC, no compression |
| `BelovaCase4Configuration` | Large FRC with compression |
| `MagneticDiffusionConfiguration` | Magnetic diffusion test case |

## Transition Conditions

Control when phases end and transition to the next:

```python
from jax_frc.configurations import timeout, separation_below, any_of, all_of

# Transition after 20 Alfven times
transition = timeout(20.0)

# Transition when FRCs merge
transition = separation_below(0.3, geometry)

# Combine conditions
transition = any_of(
    separation_below(0.3, geometry),
    timeout(20.0)
)
```

| Condition | Description |
|-----------|-------------|
| `timeout(t)` | Transition after t Alfven times |
| `separation_below(d, geom)` | Transition when FRC separation < d |
| `temperature_above(T)` | Transition when peak temperature > T |
| `flux_below(psi)` | Transition when flux drops below threshold |
| `velocity_below(v)` | Transition when max velocity < v |
| `any_of(*conditions)` | Transition when any condition is met |
| `all_of(*conditions)` | Transition when all conditions are met |

## Available Phases

- [FRC Merging](merging.md) - Two-FRC collision simulation

## Configuration Registry

Access configurations by name:

```python
from jax_frc.configurations import CONFIGURATION_REGISTRY

# List available configurations
print(CONFIGURATION_REGISTRY.keys())

# Create by name
ConfigClass = CONFIGURATION_REGISTRY["BelovaCase2Configuration"]
config = ConfigClass()
```
