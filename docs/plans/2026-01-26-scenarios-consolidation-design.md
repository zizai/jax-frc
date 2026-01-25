# Scenarios Module Consolidation into Configurations

**Date:** 2026-01-26
**Status:** Approved

## Goal

Consolidate the `jax_frc/scenarios/` module into `jax_frc/configurations/`. The scenarios module now only contains phases and transitions, which are building blocks for configurations. A single unified module reduces confusion and simplifies imports.

## File Operations

### Moves

| Source | Destination |
|--------|-------------|
| `scenarios/phase.py` | `configurations/phase.py` |
| `scenarios/transitions.py` | `configurations/transitions.py` |
| `scenarios/phases/` | `configurations/phases/` |

### Deletions

- `jax_frc/scenarios/__init__.py`
- `jax_frc/scenarios/` directory

## Import Updates

### Internal (configurations module)

`linear_configuration.py`:
```python
# Before
from jax_frc.scenarios import Phase, PhaseResult, PHASE_REGISTRY
from jax_frc.scenarios import transition_from_spec

# After
from jax_frc.configurations.phase import Phase, PhaseResult, PHASE_REGISTRY
from jax_frc.configurations.transitions import transition_from_spec
```

### Moved files

`phases/merging.py` and `phases/__init__.py`:
```python
# Before
from jax_frc.scenarios.phase import ...

# After
from jax_frc.configurations.phase import ...
```

### Tests

- Rename `tests/test_scenarios.py` → `tests/test_phases.py`
- Update all imports from `jax_frc.scenarios` → `jax_frc.configurations`

## Public API

All exports unified in `configurations/__init__.py`:

```python
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
from jax_frc.configurations.analytic import SlabDiffusionConfiguration

# Import phases submodule to trigger registration
from jax_frc.configurations import phases

# Registry
CONFIGURATION_REGISTRY = { ... }
```

## Verification

After refactor, run:
1. `py -m pytest tests/ -v` - All tests pass
2. `python examples.py` - Examples work
3. `python scripts/run_example.py belova_case1` - YAML execution works
