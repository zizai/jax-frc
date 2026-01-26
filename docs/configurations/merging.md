# FRC Merging Simulation

Simulate two-FRC collision experiments based on Belova et al. validation cases.

## Overview

The merging phase implements FRC collision physics including:
- Initial FRC separation and velocity
- Optional mirror compression
- Reconnection tracking
- Merge completion detection

## Basic Usage

```python
from jax_frc.configurations.frc_merging import BelovaCase2Configuration

# Run small FRC merging (complete merge expected)
config = BelovaCase2Configuration()
geometry = config.build_geometry()
state = config.build_initial_state(geometry)

# Access merging diagnostics
from jax_frc.diagnostics.merging import MergingDiagnostics

diag = MergingDiagnostics()
metrics = diag.compute(state, geometry)
print(f"Separation: {metrics['separation_dz']}")
print(f"Reconnection rate: {metrics['reconnection_rate']}")
```

## MergingPhase Configuration

```python
from jax_frc.configurations.phases.merging import MergingPhase
from jax_frc.configurations.transitions import separation_below, timeout, any_of

merge_phase = MergingPhase(
    name="frc_merge",
    transition=any_of(
        separation_below(0.3, geometry),
        timeout(20.0)
    ),
    separation=2.0,            # Initial FRC separation (in d_i)
    initial_velocity=0.1,      # Collision velocity (in v_A)
    compression={              # Optional compression
        "mirror_ratio": 1.5,
        "ramp_time": 10.0
    }
)
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | str | Phase identifier |
| `transition` | Condition | When to end this phase |
| `separation` | float | Initial FRC separation |
| `initial_velocity` | float | Collision velocity |
| `compression` | dict | Mirror compression settings |

## Belova Validation Cases

Based on Belova et al. experimental validation:

| Case | Description | Expected Outcome |
|------|-------------|------------------|
| Case 1 | Large FRC, no compression | Partial merge, doublet |
| Case 2 | Small FRC, no compression | Complete merge ~5-7 tA |
| Case 4 | Large FRC with compression | Complete merge ~20-25 tA |

### Running Validation Cases

```python
from jax_frc.configurations.frc_merging import (
    BelovaCase1Configuration,
    BelovaCase2Configuration,
    BelovaCase4Configuration,
)

# Case 1: Large FRC, no compression
config1 = BelovaCase1Configuration()
geometry1 = config1.build_geometry()
state1 = config1.build_initial_state(geometry1)

# Case 2: Small FRC, no compression
config2 = BelovaCase2Configuration()
geometry2 = config2.build_geometry()
state2 = config2.build_initial_state(geometry2)

# Case 4: Large FRC with compression
config4 = BelovaCase4Configuration()
geometry4 = config4.build_geometry()
state4 = config4.build_initial_state(geometry4)
```

## Merging Diagnostics

Track merging progress with specialized diagnostics:

```python
from jax_frc.diagnostics.merging import MergingDiagnostics

diag = MergingDiagnostics()
metrics = diag.compute(state, geometry)
```

### Available Metrics

| Metric | Description |
|--------|-------------|
| `separation_dz` | Distance between O-points |
| `reconnection_rate` | Flux annihilation rate |
| `merged` | Boolean: FRCs fully merged |
| `o_point_positions` | (z1, z2) positions of O-points |
| `x_point_position` | Position of X-point (if exists) |
