# AGATE Validation Snapshots Design

**Date**: 2026-01-30
**Status**: Draft
**Goal**: Enhance AGATE runner and validation tests to compare JAX-FRC against AGATE at multiple time snapshots across three resolutions.

## Overview

Currently, validation tests compare JAX-FRC against AGATE only at the final simulation time. This design extends the validation to compare at 40 evenly-spaced snapshots throughout the simulation, providing better coverage of temporal evolution.

### Key Requirements

1. **Multiple snapshots**: 40 evenly-spaced snapshots per simulation
2. **Strict time matching**: JAX-FRC outputs at exactly the same times as AGATE
3. **Three resolutions**: [128, 128, 1], [256, 256, 1], [512, 512, 1]
4. **Both test cases**: Orszag-Tang (ideal MHD) and GEM reconnection (Hall MHD)
5. **Comprehensive metrics**: Per-field and aggregate physical metrics

## AGATE Runner Changes

### Configuration Updates

```python
CASE_CONFIGS = {
    "orszag_tang": {
        "physics": "ideal_mhd",
        "hall": False,
        "end_time": 0.48,
        "cfl": 0.4,
        "num_snapshots": 40,
    },
    "gem_reconnection": {
        "physics": "hall_mhd",
        "hall": True,
        "end_time": 12.0,
        "cfl": 0.4,
        "num_snapshots": 40,
    },
}

RESOLUTIONS = [
    [128, 128, 1],
    [256, 256, 1],
    [512, 512, 1],
]
```

### Snapshot Timing

For Orszag-Tang (t=0 to t=0.48, 40 snapshots):
- Δt = 0.48 / 39 = 0.0123 between snapshots
- Output times: [0.0, 0.0123, 0.0246, ..., 0.48]

For GEM (t=0 to t=12.0, 40 snapshots):
- Δt = 12.0 / 39 = 0.308 between snapshots
- Output times: [0.0, 0.308, 0.615, ..., 12.0]

### Config YAML Format

```yaml
# orszag_tang_256.config.yaml
case: orszag_tang
resolution: [256, 256, 1]
physics: ideal_mhd
hall: false
end_time: 0.48
cfl: 0.4
num_snapshots: 40
snapshot_times: [0.0, 0.0123, 0.0246, ..., 0.48]
agate_version: "1.0.0"
generated_at: "2026-01-30T00:00:00Z"
```

### Output Files

```
validation/references/agate/orszag_tang/256/
├── orszag_tang_256.grid.h5
├── orszag_tang_256.state_000.h5  # t=0.0
├── orszag_tang_256.state_001.h5  # t=0.0123
├── ...
├── orszag_tang_256.state_039.h5  # t=0.48
└── orszag_tang_256.config.yaml
```

## Validation Test Changes

### Load Snapshot Configuration

```python
def load_agate_config(case: str, resolution: list[int]) -> dict:
    """Load AGATE config including snapshot_times."""
    res_str = f"{resolution[0]}"
    config_path = f"validation/references/agate/{case}/{res_str}/{case}_{res_str}.config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
```

### JAX-FRC Outputs at Exact Times

```python
def run_simulation_with_snapshots(cfg, snapshot_times: list[float]) -> list[State]:
    """Run JAX-FRC, capturing state at each snapshot time."""
    states = []
    state = initial_state
    for target_time in snapshot_times:
        while state.time < target_time - 1e-12:
            dt = min(cfg["dt"], target_time - state.time)
            state = solver.step(state, dt, model, geometry)
        states.append(state)
    return states
```

### Compare All Snapshots

```python
def validate_all_snapshots(jax_states, case, resolution):
    """Compare JAX-FRC vs AGATE at each snapshot."""
    all_errors = []
    for i, jax_state in enumerate(jax_states):
        agate_fields = load_agate_snapshot(case, resolution, snapshot_idx=i)
        errors = compute_field_l2_errors(jax_state, agate_fields)
        all_errors.append({"time": jax_state.time, "errors": errors})
    return all_errors
```

## Metrics

### Per-Field Metrics (spatial comparison at each snapshot)

For fields: density, velocity, B, pressure

- **L2 error**: Normalized L2 norm of difference
- **Max abs error**: Maximum absolute difference
- **Relative error**: Relative to reference magnitude

### Aggregate Physical Metrics (time-series comparison)

For each aggregate quantity:
- Total energy
- Magnetic energy
- Kinetic energy
- Thermal energy
- Enstrophy (fluid: ∫|ω|² dV where ω = ∇×v)
- Reconnected flux (for GEM)

Time-series comparison metrics:
- **Mean residual**: Average of (JAX - AGATE) over time
- **Std residual**: Standard deviation of residuals
- **Relative error**: ||JAX - AGATE|| / ||AGATE||

## Reporting

### Console Output

```
Resolution [256, 256, 1]:
  Validating 40 snapshots...

  Per-Field Statistics:
    Field      L2 Error   Max Abs Error   Rel Error   Status
    ------------------------------------------------------------
    density    0.082      0.182           0.145       PASS
    velocity   0.045      0.098           0.076       PASS
    B          0.023      0.057           0.041       PASS
    pressure   0.091      0.195           0.158       PASS

  Aggregate Time-Series Statistics:
    Metric           Mean Resid   Std Resid   Rel Error   Status
    ----------------------------------------------------------------
    total_energy     0.0012       0.0008      0.015       PASS
    magnetic_energy  0.0018       0.0011      0.022       PASS
    kinetic_energy   0.0025       0.0015      0.031       PASS
    enstrophy        0.0031       0.0019      0.038       PASS

  Overall: PASS
```

### HTML Report Plots

1. **Time-series evolution**: JAX vs AGATE for each aggregate metric
2. **Residual plot**: (JAX - AGATE) vs time for each metric
3. **Per-field error evolution**: L2 error vs snapshot time
4. **Field comparison**: Side-by-side JAX vs AGATE at t=0, t_mid, t_final

## Pass/Fail Criteria

### Thresholds

```python
L2_ERROR_THRESHOLD = 0.20
MAX_ABS_ERROR_THRESHOLD = 0.20
RELATIVE_ERROR_THRESHOLD = 0.20
```

### Criteria

- Each per-field metric must be below threshold at all snapshots
- Each aggregate time-series metric must be below threshold
- Same thresholds apply to both quick and full test modes

## Testing Strategy

### Quick Test Mode

- 5 snapshots (t=0, t=0.25*end, t=0.5*end, t=0.75*end, t=end)
- Single resolution [128, 128, 1]
- **All metrics must still pass** (same thresholds)
- Purpose: Fast CI/development iteration

### Full Test Mode

- 40 snapshots (evenly spaced)
- All resolutions: [128, 128, 1], [256, 256, 1], [512, 512, 1]
- All metrics must pass

### Unit Tests

```python
def test_snapshot_times_evenly_spaced():
    """Verify snapshot times are evenly distributed."""
    config = get_expected_config("orszag_tang", [256, 256, 1])
    times = config["snapshot_times"]
    assert len(times) == 40
    assert times[0] == 0.0
    assert times[-1] == 0.48

def test_run_generates_all_snapshots():
    """Verify all snapshot files are generated."""
    run_agate_simulation("orszag_tang", [128, 128, 1], output_dir)
    for i in range(40):
        assert (output_dir / f"orszag_tang_128.state_{i:03d}.h5").exists()
```

## Implementation Tasks

1. **Update AGATE runner** - Add snapshot output at regular intervals
2. **Update config format** - Store resolution as [nx, ny, nz] and snapshot_times
3. **Update validation tests** - Iterate over all snapshots
4. **Add aggregate metrics** - Compute and compare time-series
5. **Update reporting** - Add time-series plots and per-snapshot tables
6. **Update quick test mode** - Use 5 snapshots, same pass criteria
