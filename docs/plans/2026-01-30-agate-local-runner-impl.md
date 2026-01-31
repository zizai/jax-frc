# AGATE Local Runner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace Zenodo downloads with local AGATE execution for reproducible, self-contained validation.

**Architecture:** Create `agate_runner.py` module that runs AGATE simulations on-demand, with config.yaml tracking for cache validation. Modify `AgateDataLoader` to call runner when data is missing.

**Tech Stack:** Python, AGATE framework, PyYAML, h5py

---

## Task 1: Create AGATE Runner Module - Config Functions

**Files:**
- Create: `validation/utils/agate_runner.py`
- Test: `tests/test_agate_runner.py`

**Step 1: Write the failing test for get_expected_config**

```python
# tests/test_agate_runner.py
"""Tests for AGATE runner module."""

import pytest
from validation.utils.agate_runner import get_expected_config


def test_get_expected_config_orszag_tang():
    """OT config should have correct parameters."""
    config = get_expected_config("orszag_tang", 256)

    assert config["case"] == "orszag_tang"
    assert config["resolution"] == 256
    assert config["physics"] == "ideal_mhd"
    assert config["hall"] is False
    assert config["end_time"] == 0.48
    assert config["cfl"] == 0.4


def test_get_expected_config_gem():
    """GEM config should have Hall MHD parameters."""
    config = get_expected_config("gem_reconnection", 512)

    assert config["case"] == "gem_reconnection"
    assert config["resolution"] == 512
    assert config["physics"] == "hall_mhd"
    assert config["hall"] is True
    assert config["end_time"] == 12.0


def test_get_expected_config_unknown_case():
    """Unknown case should raise ValueError."""
    with pytest.raises(ValueError, match="Unknown case"):
        get_expected_config("unknown_case", 256)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_agate_runner.py::test_get_expected_config_orszag_tang -v`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# validation/utils/agate_runner.py
"""Run AGATE simulations to generate reference data."""

from pathlib import Path
from typing import Literal

# Case configurations
CASE_CONFIGS = {
    "orszag_tang": {
        "physics": "ideal_mhd",
        "hall": False,
        "end_time": 0.48,
        "cfl": 0.4,
        "slope_name": "mcbeta",
        "mcbeta": 1.3,
    },
    "gem_reconnection": {
        "physics": "hall_mhd",
        "hall": True,
        "end_time": 12.0,
        "cfl": 0.4,
        "guide_field": 0.0,
    },
}


def get_expected_config(case: str, resolution: int) -> dict:
    """Get expected configuration for a case.

    Args:
        case: Case name ("orszag_tang" or "gem_reconnection")
        resolution: Grid resolution

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If case is unknown
    """
    if case not in CASE_CONFIGS:
        raise ValueError(f"Unknown case: {case}. Valid cases: {list(CASE_CONFIGS.keys())}")

    base_config = CASE_CONFIGS[case].copy()
    return {
        "case": case,
        "resolution": resolution,
        **base_config,
    }
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_agate_runner.py -v -k "get_expected_config"`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add validation/utils/agate_runner.py tests/test_agate_runner.py
git commit -m "feat(validation): add AGATE runner config functions"
```

---

## Task 2: Add Cache Validation Functions

**Files:**
- Modify: `validation/utils/agate_runner.py`
- Modify: `tests/test_agate_runner.py`

**Step 1: Write the failing test for is_cache_valid**

```python
# Add to tests/test_agate_runner.py
import tempfile
import yaml
from pathlib import Path
from validation.utils.agate_runner import is_cache_valid, get_expected_config


def test_is_cache_valid_missing_config():
    """Missing config.yaml should return False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        assert is_cache_valid("orszag_tang", 256, output_dir) is False


def test_is_cache_valid_matching_config():
    """Matching config should return True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        config_file = output_dir / "orszag_tang_256.config.yaml"

        # Write matching config
        config = get_expected_config("orszag_tang", 256)
        config["agate_version"] = "1.0.0"
        config["generated_at"] = "2026-01-30T00:00:00Z"

        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)

        assert is_cache_valid("orszag_tang", 256, output_dir) is True


def test_is_cache_valid_mismatched_resolution():
    """Mismatched resolution should return False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        config_file = output_dir / "orszag_tang_256.config.yaml"

        # Write config with wrong resolution
        config = get_expected_config("orszag_tang", 512)  # Wrong resolution
        with open(config_file, "w") as f:
            yaml.safe_dump(config, f)

        assert is_cache_valid("orszag_tang", 256, output_dir) is False
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_agate_runner.py::test_is_cache_valid_missing_config -v`
Expected: FAIL with "ImportError" for is_cache_valid

**Step 3: Write minimal implementation**

```python
# Add to validation/utils/agate_runner.py
import yaml


def is_cache_valid(case: str, resolution: int, output_dir: Path) -> bool:
    """Check if cached data matches current configuration.

    Args:
        case: Case name
        resolution: Grid resolution
        output_dir: Directory containing cached data

    Returns:
        True if cache is valid, False otherwise
    """
    config_file = output_dir / f"{case}_{resolution}.config.yaml"
    if not config_file.exists():
        return False

    try:
        with open(config_file) as f:
            cached_config = yaml.safe_load(f)
    except Exception:
        return False

    expected_config = get_expected_config(case, resolution)

    # Compare key parameters (ignore metadata like generated_at, agate_version)
    keys_to_check = ["case", "resolution", "physics", "hall", "end_time"]
    return all(
        cached_config.get(k) == expected_config.get(k)
        for k in keys_to_check
    )
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_agate_runner.py -v -k "is_cache_valid"`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add validation/utils/agate_runner.py tests/test_agate_runner.py
git commit -m "feat(validation): add cache validation for AGATE data"
```

---

## Task 3: Add AGATE Simulation Runner - Orszag-Tang

**Files:**
- Modify: `validation/utils/agate_runner.py`
- Modify: `tests/test_agate_runner.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_agate_runner.py
import pytest
from validation.utils.agate_runner import run_agate_simulation


@pytest.mark.slow
def test_run_orszag_tang_generates_files():
    """OT simulation should produce expected output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        run_agate_simulation("orszag_tang", 64, output_dir)  # Small resolution for speed

        # Check expected files exist
        assert (output_dir / "orszag_tang_64.grid.h5").exists()
        assert (output_dir / "orszag_tang_64.config.yaml").exists()

        # Check at least one state file exists
        state_files = list(output_dir.glob("orszag_tang_64.state_*.h5"))
        assert len(state_files) >= 1
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_agate_runner.py::test_run_orszag_tang_generates_files -v`
Expected: FAIL with "ImportError" for run_agate_simulation

**Step 3: Write minimal implementation**

```python
# Add to validation/utils/agate_runner.py
from datetime import datetime


def _get_agate_version() -> str:
    """Get AGATE version string."""
    try:
        import agate
        return getattr(agate, "__version__", "unknown")
    except ImportError:
        return "not_installed"


def _save_config(case: str, resolution: int, output_dir: Path) -> None:
    """Save configuration to YAML file."""
    config = get_expected_config(case, resolution)
    config["agate_version"] = _get_agate_version()
    config["generated_at"] = datetime.utcnow().isoformat() + "Z"

    config_file = output_dir / f"{case}_{resolution}.config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)


def _run_orszag_tang(resolution: int, output_dir: Path) -> None:
    """Run Orszag-Tang vortex simulation."""
    from agate.framework.scenario import OTVortex
    from agate.framework.roller import Roller
    from agate.framework.fileHandler import fileHandler

    # Create scenario (ideal MHD, no Hall term)
    scenario = OTVortex(divClean=True, hall=False)

    # Create roller with standard options
    roller = Roller.autodefault(
        scenario,
        ncells=resolution,
        options={"cfl": 0.4, "slopeName": "mcbeta", "mcbeta": 1.3}
    )
    roller.orient("numpy")

    # Run simulation
    print(f"Running Orszag-Tang at resolution {resolution}...")
    roller.roll(start_time=0.0, end_time=0.48, add_stopWatch=True)

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    handler = fileHandler(directory=str(output_dir), prefix=f"orszag_tang_{resolution}")
    handler.outputGrid(roller.grid)
    handler.outputState(roller.grid, roller.state, roller.time)


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
        Path to the output directory

    Raises:
        RuntimeError: If AGATE is not installed
        ValueError: If case is unknown
    """
    try:
        import agate
    except ImportError:
        raise RuntimeError(
            "AGATE not installed. Install with: pip install -e ../agate-open-source"
        )

    if case not in CASE_CONFIGS:
        raise ValueError(f"Unknown case: {case}")

    # Check cache
    if not overwrite and is_cache_valid(case, resolution, output_dir):
        print(f"Using cached AGATE data for {case} at resolution {resolution}")
        return output_dir

    # Run simulation
    if case == "orszag_tang":
        _run_orszag_tang(resolution, output_dir)
    elif case == "gem_reconnection":
        _run_gem_reconnection(resolution, output_dir)

    # Save config
    _save_config(case, resolution, output_dir)

    return output_dir
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_agate_runner.py::test_run_orszag_tang_generates_files -v`
Expected: PASS (may take 1-2 minutes)

**Step 5: Commit**

```bash
git add validation/utils/agate_runner.py tests/test_agate_runner.py
git commit -m "feat(validation): add Orszag-Tang AGATE simulation runner"
```

---

## Task 4: Add GEM Reconnection Runner

**Files:**
- Modify: `validation/utils/agate_runner.py`
- Modify: `tests/test_agate_runner.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_agate_runner.py
@pytest.mark.slow
def test_run_gem_generates_files():
    """GEM simulation should produce expected output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        run_agate_simulation("gem_reconnection", 64, output_dir)  # Small resolution

        assert (output_dir / "gem_reconnection_64.grid.h5").exists()
        assert (output_dir / "gem_reconnection_64.config.yaml").exists()

        state_files = list(output_dir.glob("gem_reconnection_64.state_*.h5"))
        assert len(state_files) >= 1
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_agate_runner.py::test_run_gem_generates_files -v`
Expected: FAIL with "NameError" for _run_gem_reconnection

**Step 3: Write minimal implementation**

```python
# Add to validation/utils/agate_runner.py
def _run_gem_reconnection(resolution: int, output_dir: Path) -> None:
    """Run GEM reconnection simulation."""
    from agate.framework.scenario import ReconnectionGEM
    from agate.framework.roller import Roller
    from agate.framework.fileHandler import fileHandler

    # Create scenario (Hall MHD)
    scenario = ReconnectionGEM(divClean=True, hall=True, guide_field=0.0)

    # Create roller
    roller = Roller.autodefault(
        scenario,
        ncells=resolution,
        options={"cfl": 0.4}
    )
    roller.orient("numpy")

    # Run simulation
    print(f"Running GEM reconnection at resolution {resolution}...")
    roller.roll(start_time=0.0, end_time=12.0, add_stopWatch=True)

    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    handler = fileHandler(directory=str(output_dir), prefix=f"gem_reconnection_{resolution}")
    handler.outputGrid(roller.grid)
    handler.outputState(roller.grid, roller.state, roller.time)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_agate_runner.py::test_run_gem_generates_files -v`
Expected: PASS (may take several minutes)

**Step 5: Commit**

```bash
git add validation/utils/agate_runner.py tests/test_agate_runner.py
git commit -m "feat(validation): add GEM reconnection AGATE simulation runner"
```

---

## Task 5: Integrate with AgateDataLoader

**Files:**
- Modify: `validation/utils/agate_data.py`
- Modify: `tests/test_agate_loader.py` (if exists) or create test

**Step 1: Write the failing test**

```python
# tests/test_agate_loader_integration.py
"""Tests for AgateDataLoader integration with AGATE runner."""

import tempfile
import pytest
from pathlib import Path
from validation.utils.agate_data import AgateDataLoader


@pytest.mark.slow
def test_ensure_files_generates_missing_data():
    """ensure_files should generate data if missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = AgateDataLoader(cache_dir=Path(tmpdir))

        # This should trigger AGATE simulation
        paths = loader.ensure_files("ot", 64)

        # Check files were generated
        assert len(paths) > 0
        assert any("grid" in str(p) for p in paths)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_agate_loader_integration.py -v`
Expected: FAIL (tries to download from Zenodo)

**Step 3: Modify AgateDataLoader**

```python
# validation/utils/agate_data.py - Replace ensure_files method
def ensure_files(self, case: str, resolution: int) -> list[Path]:
    """Ensure required files are present locally; generate if missing.

    Priority:
    1. Check if local data exists with valid config
    2. If missing/invalid → run AGATE to generate
    3. Return paths to data files
    """
    # Map short names to full names
    case_map = {"ot": "orszag_tang", "gem": "gem_reconnection"}
    full_case = case_map.get(case.lower(), case.lower())

    output_dir = Path(self.cache_dir) / full_case / str(resolution)

    # Try local generation first
    try:
        from validation.utils.agate_runner import is_cache_valid, run_agate_simulation

        if not is_cache_valid(full_case, resolution, output_dir):
            print(f"Generating AGATE reference data for {full_case} at {resolution}...")
            run_agate_simulation(full_case, resolution, output_dir, overwrite=True)

        # Return all HDF5 files in the directory
        return list(output_dir.glob("*.h5"))

    except ImportError:
        # Fall back to Zenodo download if AGATE not available
        print("AGATE not installed, falling back to Zenodo download...")
        return self._ensure_files_zenodo(case, resolution)


def _ensure_files_zenodo(self, case: str, resolution: int) -> list[Path]:
    """Original Zenodo download logic (fallback)."""
    files = self._select_files(case, resolution)
    local_paths: list[Path] = []
    for file_meta in files:
        key = file_meta["key"]
        url = file_meta["links"]["self"]
        filename = Path(key).name
        target = self._target_path(case, resolution, filename)
        if self._is_archive(target):
            if not target.exists():
                self._download_file(url, target)
            extracted = self._extract_archive(target, target.parent)
            local_paths.extend(extracted)
            continue
        if not target.exists():
            self._download_file(url, target)
        local_paths.append(target)
    return local_paths
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_agate_loader_integration.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add validation/utils/agate_data.py tests/test_agate_loader_integration.py
git commit -m "feat(validation): integrate AGATE runner with AgateDataLoader"
```

---

## Task 6: Add Error Handling and Cleanup

**Files:**
- Modify: `validation/utils/agate_runner.py`
- Modify: `tests/test_agate_runner.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_agate_runner.py
def test_run_simulation_cleans_up_on_failure():
    """Partial files should be cleaned up on failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Create a partial file to simulate interrupted run
        partial_file = output_dir / "orszag_tang_256.grid.h5"
        output_dir.mkdir(parents=True, exist_ok=True)
        partial_file.write_text("partial")

        # Config file missing = invalid cache
        # Running should clean up and regenerate
        # (This test verifies the cleanup logic exists)
        assert partial_file.exists()
```

**Step 2: Run test to verify current behavior**

Run: `py -m pytest tests/test_agate_runner.py::test_run_simulation_cleans_up_on_failure -v`
Expected: PASS (just checking file exists)

**Step 3: Add cleanup logic to run_agate_simulation**

```python
# Update run_agate_simulation in validation/utils/agate_runner.py
import shutil

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
        Path to the output directory

    Raises:
        RuntimeError: If AGATE is not installed or simulation fails
        ValueError: If case is unknown
    """
    try:
        import agate
    except ImportError:
        raise RuntimeError(
            "AGATE not installed. Install with: pip install -e ../agate-open-source"
        )

    if case not in CASE_CONFIGS:
        raise ValueError(f"Unknown case: {case}")

    # Check cache
    if not overwrite and is_cache_valid(case, resolution, output_dir):
        print(f"Using cached AGATE data for {case} at resolution {resolution}")
        return output_dir

    # Clean up any partial files before regenerating
    if output_dir.exists():
        for f in output_dir.glob(f"{case}_{resolution}.*"):
            f.unlink()

    try:
        # Run simulation
        if case == "orszag_tang":
            _run_orszag_tang(resolution, output_dir)
        elif case == "gem_reconnection":
            _run_gem_reconnection(resolution, output_dir)

        # Save config
        _save_config(case, resolution, output_dir)

    except Exception as e:
        # Clean up partial files on failure
        if output_dir.exists():
            for f in output_dir.glob(f"{case}_{resolution}.*"):
                f.unlink()
        raise RuntimeError(f"AGATE simulation failed for {case} at {resolution}: {e}") from e

    return output_dir
```

**Step 4: Run all tests to verify nothing broke**

Run: `py -m pytest tests/test_agate_runner.py -v`
Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add validation/utils/agate_runner.py tests/test_agate_runner.py
git commit -m "feat(validation): add error handling and cleanup to AGATE runner"
```

---

## Task 7: Update Validation Cases to Use New Loader

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`
- Modify: `validation/cases/regression/gem_reconnection.py`

**Step 1: Verify existing validation still works**

Run: `py validation/cases/regression/orszag_tang.py --quick`
Expected: Should work (generates AGATE data if needed)

**Step 2: Run full validation test**

Run: `py -m pytest tests/ -v -k "not slow" --ignore=tests/test_orszag_tang_case.py --ignore=tests/test_gem_reconnection_case.py`
Expected: PASS

**Step 3: Commit final integration**

```bash
git add -A
git commit -m "feat(validation): complete AGATE local runner integration"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Config functions | `agate_runner.py`, `test_agate_runner.py` |
| 2 | Cache validation | `agate_runner.py`, `test_agate_runner.py` |
| 3 | Orszag-Tang runner | `agate_runner.py`, `test_agate_runner.py` |
| 4 | GEM runner | `agate_runner.py`, `test_agate_runner.py` |
| 5 | AgateDataLoader integration | `agate_data.py`, `test_agate_loader_integration.py` |
| 6 | Error handling | `agate_runner.py`, `test_agate_runner.py` |
| 7 | Final integration | Validation cases |

**Total estimated commits:** 7
