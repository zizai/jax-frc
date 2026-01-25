# Scenarios

The scenario framework enables multi-phase simulations with automatic transitions.

## Package Structure

```
jax_frc/scenarios/
├── scenario.py       # Multi-phase scenario orchestrator
├── phase.py          # Phase base class with transitions
├── transitions.py    # Transition conditions (timeout, separation, etc.)
└── phases/
    └── merging.py    # FRC merging phase implementation
```

## Basic Usage

```python
from jax_frc.scenarios import Scenario, timeout
from jax_frc.scenarios.phases.merging import MergingPhase
from jax_frc.scenarios.transitions import separation_below, any_of

# Create a merging phase with transition conditions
merge_phase = MergingPhase(
    name="frc_merge",
    transition=any_of(
        separation_below(0.3, geometry),  # Complete when FRCs merge
        timeout(20.0)                      # Or timeout after 20 tA
    ),
    separation=2.0,
    initial_velocity=0.1,
    compression={"mirror_ratio": 1.5, "ramp_time": 10.0}
)

# Create and run scenario
scenario = Scenario(
    name="frc_merging",
    phases=[merge_phase],
    geometry=geometry,
    initial_state=initial_state,
    dt=0.01
)
result = scenario.run()
```

## Transition Conditions

Available transition conditions:

| Condition | Description |
|-----------|-------------|
| `timeout(t)` | Transition after t Alfven times |
| `separation_below(d, geom)` | Transition when FRC separation < d |
| `any_of(*conditions)` | Transition when any condition is met |
| `all_of(*conditions)` | Transition when all conditions are met |

## Multi-Phase Simulations

Chain multiple phases together:

```python
scenario = Scenario(
    name="full_experiment",
    phases=[formation_phase, merge_phase, equilibrium_phase],
    geometry=geometry,
    initial_state=initial_state,
    dt=0.01
)
```

## Available Phases

- [FRC Merging](merging.md) - Two-FRC collision simulation
