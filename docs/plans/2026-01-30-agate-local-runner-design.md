# AGATE Local Runner Design

**Date**: 2026-01-30
**Status**: Draft
**Goal**: Replace Zenodo downloads with local AGATE execution for reproducible, self-contained validation

## Overview

Currently, JAX-FRC validation downloads pre-computed AGATE reference data from Zenodo. This design replaces that with local AGATE execution, enabling:

1. **Reproducibility** - Regenerate reference data with specific parameters/versions
2. **CI/CD self-containment** - No external dependencies on Zenodo

### High-Level Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Validation Run                                │
├─────────────────────────────────────────────────────────────────┤
│  1. Check if reference data exists in validation/references/agate/│
│  2. If missing/invalid → Run AGATE to generate it               │
│  3. Run JAX-FRC simulation                                       │
│  4. Compare JAX-FRC vs AGATE reference data                      │
│  5. Generate HTML report                                         │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### 1. AGATE Runner Module

**New File**: `validation/utils/agate_runner.py`

```python
"""Run AGATE simulations to generate reference data."""

from pathlib import Path
from typing import Literal

def run_agate_simulation(
    case: Literal["orszag_tang", "gem_reconnection"],
    resolution: int,
    output_dir: Path,
    overwrite: bool = False
) -> Path:
    """Run AGATE simulation and save results.

    Args:
        case: Which test case to run
        resolution: Grid resolution (e.g., 256, 512)
        output_dir: Where to save HDF5 output files
        overwrite: If True, regenerate even if files exist

    Returns:
        Path to the output directory containing grid.h5 and state files
    """
```

### 2. Case Configurations

| Case | AGATE Scenario | Physics | End Time | Notes |
|------|---------------|---------|----------|-------|
| Orszag-Tang | `OTVortex` | Ideal MHD | t=0.48 | `hall=False` |
| GEM Reconnection | `ReconnectionGEM` | Hall MHD | t=12.0 | `hall=True` |

### 3. Directory Structure

```
validation/references/agate/
├── orszag_tang/
│   ├── 256/
│   │   ├── orszag_tang_256.grid.h5
│   │   ├── orszag_tang_256.state_000000.h5
│   │   ├── orszag_tang_256.state_final.h5
│   │   └── orszag_tang_256.config.yaml
│   └── 512/
│       └── ...
└── gem_reconnection/
    └── 512/
        ├── gem_reconnection_512.grid.h5
        ├── gem_reconnection_512.state_000000.h5
        ├── gem_reconnection_512.state_final.h5
        └── gem_reconnection_512.config.yaml
```

### 4. Configuration Tracking

Each generated reference includes a `{case}_{resolution}.config.yaml` file:

```yaml
# AGATE reference data configuration
case: orszag_tang
resolution: 256
physics: ideal_mhd
hall: false
end_time: 0.48
cfl: 0.4
agate_version: "1.0.0"
generated_at: "2026-01-30T20:00:00Z"
```

### 5. Cache Validation

```python
def is_cache_valid(case: str, resolution: int, output_dir: Path) -> bool:
    """Check if cached data matches current configuration."""
    config_file = output_dir / f"{case}_{resolution}.config.yaml"
    if not config_file.exists():
        return False

    with open(config_file) as f:
        cached_config = yaml.safe_load(f)
    expected_config = get_expected_config(case, resolution)

    # Compare key parameters
    keys_to_check = ["case", "resolution", "physics", "hall", "end_time"]
    return all(cached_config.get(k) == expected_config.get(k) for k in keys_to_check)
```

## Integration

### Modified AgateDataLoader

```python
class AgateDataLoader:
    def ensure_files(self, case: str, resolution: int) -> Path:
        """Ensure reference data exists, generating if needed."""
        output_dir = self.base_dir / case / str(resolution)

        if is_cache_valid(case, resolution, output_dir):
            return output_dir

        # Generate using AGATE
        from validation.utils.agate_runner import run_agate_simulation
        run_agate_simulation(case, resolution, output_dir, overwrite=True)
        return output_dir
```

## AGATE Simulation Implementation

### Orszag-Tang

```python
def run_orszag_tang(resolution: int, output_dir: Path):
    from agate.framework.scenario import OTVortex
    from agate.framework.roller import Roller
    from agate.framework.fileHandler import fileHandler

    scenario = OTVortex(divClean=True, hall=False)
    roller = Roller.autodefault(
        scenario,
        ncells=resolution,
        options={"cfl": 0.4, "slopeName": "mcbeta", "mcbeta": 1.3}
    )
    roller.orient("numpy")
    roller.roll(start_time=0.0, end_time=0.48)

    handler = fileHandler(directory=str(output_dir), prefix=f"orszag_tang_{resolution}")
    handler.outputGrid(roller.grid)
    handler.outputState(roller.grid, roller.state, roller.time)
```

### GEM Reconnection

```python
def run_gem_reconnection(resolution: int, output_dir: Path):
    from agate.framework.scenario import ReconnectionGEM
    from agate.framework.roller import Roller

    scenario = ReconnectionGEM(divClean=True, hall=True, guide_field=0.0)
    roller = Roller.autodefault(scenario, ncells=resolution, options={"cfl": 0.4})
    roller.orient("numpy")
    roller.roll(start_time=0.0, end_time=12.0)
    # ... save output ...
```

## CI/CD Integration

### GitHub Actions Workflow

```yaml
validation:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4

    - name: Checkout AGATE
      uses: actions/checkout@v4
      with:
        repository: user/agate-open-source
        path: agate-open-source

    - name: Install dependencies
      run: |
        pip install -e .
        pip install -e ./agate-open-source

    - name: Run validation
      run: python scripts/run_validation.py --category regression
```

### Cache Strategy

- First run: AGATE generates reference data (~10-30 min)
- Subsequent runs: Use cached data (fast)
- Optional: Use GitHub Actions cache to persist between runs

## Error Handling

### AGATE Import Failure

```python
try:
    from validation.utils.agate_runner import run_agate_simulation
except ImportError:
    raise RuntimeError(
        "AGATE not installed. Install with: pip install -e ../agate-open-source"
    )
```

### Simulation Failure

- Raise clear error with case/resolution info
- Clean up partial files on failure
- Log AGATE output for debugging

### Timeout Expectations

| Case | Resolution | Expected Time |
|------|------------|---------------|
| OT | 256 | ~2-5 minutes |
| OT | 1024 | ~30-60 minutes |
| GEM | 512 | ~1-2 hours |

### Version Mismatch

- Store `agate_version` in config.yaml
- Log warning (not error) if AGATE version differs from cached data

## Testing

### Unit Tests (`tests/test_agate_runner.py`)

```python
def test_config_validation_detects_mismatch():
    """Config mismatch should trigger regeneration."""

def test_config_validation_accepts_valid_cache():
    """Valid cache should be reused."""

def test_run_orszag_tang_generates_expected_files():
    """OT simulation should produce grid.h5, state files, config.yaml."""

def test_run_gem_generates_expected_files():
    """GEM simulation should produce expected output files."""
```

## Dependencies

- AGATE installed as editable: `pip install -e ../agate-open-source`
- PyYAML for config files
- No Zenodo dependency (can be removed or kept as fallback)
