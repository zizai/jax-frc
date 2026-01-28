# AGATE Validation Benchmarks Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add AGATE-aligned OT and GEM validation cases with automatic Zenodo downloads, scalar time-series metrics, and block-bootstrap regression checks.

**Architecture:** Introduce a small AGATE loader utility to download/cache Zenodo record 15084058 into `validation/references/agate/`, a regression utility that aligns time-series and computes block-bootstrap CI on mean relative error, and two new validation case scripts that run JAX-FRC at 256/512/1024 and compare to AGATE metrics.

**Tech Stack:** Python, JAX, h5py, numpy/jax.numpy, matplotlib, pytest.

## Task 1: Add AGATE data loader utility

**Files:**
- Create: `validation/utils/agate_data.py`
- Test: `tests/test_agate_loader.py`

**Step 1: Write the failing test**

```python
from pathlib import Path
import json
import pytest

from validation.utils.agate_data import AgateDataLoader


def test_agate_loader_cache_layout(tmp_path, monkeypatch):
    loader = AgateDataLoader(cache_dir=tmp_path)
    # stub record metadata, ensure it chooses the right directory structure
    record = {
        "files": [
            {"key": "GEM_data/hallGEM512.grid.h5", "links": {"self": "https://example/grid.h5"}},
            {"key": "GEM_data/hallGEM512.state_0000.h5", "links": {"self": "https://example/state0.h5"}},
        ]
    }
    monkeypatch.setattr(loader, "_fetch_record", lambda: record)
    needed = loader._select_files("gem", 512)
    assert all("hallGEM512" in f["key"] for f in needed)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_agate_loader.py::test_agate_loader_cache_layout -v`
Expected: FAIL (module or class missing).

**Step 3: Write minimal implementation**

```python
# validation/utils/agate_data.py
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ZENODO_RECORD = "https://zenodo.org/api/records/15084058"


@dataclass
class AgateDataLoader:
    cache_dir: Path = Path("validation/references/agate")

    def _fetch_record(self) -> dict:
        with urllib.request.urlopen(ZENODO_RECORD) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _select_files(self, case: str, resolution: int) -> list[dict]:
        record = self._fetch_record()
        files = record.get("files", [])
        if case == "gem":
            return [f for f in files if f["key"].lower().startswith("gem_data/") and f"{resolution}" in f["key"]]
        if case == "ot":
            return [f for f in files if f["key"].lower().startswith("hall_ot_data/") and f"{resolution}" in f["key"]]
        raise ValueError("Unknown case")
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_agate_loader.py::test_agate_loader_cache_layout -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add validation/utils/agate_data.py tests/test_agate_loader.py
git commit -m "feat(validation): add agate data loader"
```

## Task 2: Add regression utility (block bootstrap CI)

**Files:**
- Create: `validation/utils/regression.py`
- Test: `tests/test_validation_regression.py`

**Step 1: Write the failing test**

```python
import numpy as np
from validation.utils.regression import block_bootstrap_ci


def test_block_bootstrap_ci_bounds():
    rng = np.random.default_rng(0)
    errors = rng.normal(0.05, 0.01, size=200)
    mean, lo, hi = block_bootstrap_ci(errors, block_size=10, n_boot=200)
    assert lo <= mean <= hi
    assert hi - lo > 0
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_regression.py::test_block_bootstrap_ci_bounds -v`
Expected: FAIL (module missing).

**Step 3: Write minimal implementation**

```python
# validation/utils/regression.py
import numpy as np


def block_bootstrap_ci(errors: np.ndarray, block_size: int, n_boot: int = 500, alpha: float = 0.05):
    errors = np.asarray(errors)
    n = len(errors)
    n_blocks = max(1, n // block_size)
    means = []
    for _ in range(n_boot):
        idx = np.random.randint(0, n_blocks, size=n_blocks)
        sample = np.concatenate([errors[i * block_size:(i + 1) * block_size] for i in idx])
        means.append(np.mean(sample))
    means = np.asarray(means)
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return float(np.mean(errors)), float(lo), float(hi)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_regression.py::test_block_bootstrap_ci_bounds -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add validation/utils/regression.py tests/test_validation_regression.py
git commit -m "feat(validation): add block bootstrap regression utility"
```

## Task 3: Add Orszag-Tang validation case

**Files:**
- Create: `validation/cases/mhd_regression/orszag_tang.py`
- Modify: `validation/cases/mhd_regression/__init__.py`

**Step 1: Write the failing test**

```python
from validation.cases.mhd_regression.orszag_tang import main


def test_orszag_tang_quick_smoke():
    assert main(quick_test=True) in (True, False)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_orszag_tang_case.py::test_orszag_tang_quick_smoke -v`
Expected: FAIL (module missing).

**Step 3: Write minimal implementation**

- Mirror pattern from `validation/cases/mhd_regression/cylindrical_vortex.py`.
- Use `CylindricalVortexConfiguration` (Orszag-Tang) from `jax_frc.configurations.validation_benchmarks`.
- Run 256/512/1024 (or smaller in quick mode).
- Compute metrics for JAX-FRC and AGATE and run regression with `block_bootstrap_ci`.

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_orszag_tang_case.py::test_orszag_tang_quick_smoke -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add validation/cases/mhd_regression/orszag_tang.py validation/cases/mhd_regression/__init__.py tests/test_orszag_tang_case.py
git commit -m "feat(validation): add orszag-tang regression case"
```

## Task 4: Add GEM reconnection validation case

**Files:**
- Create: `validation/cases/hall_reconnection/reconnection_gem.py`
- Modify: `validation/cases/hall_reconnection/__init__.py`

**Step 1: Write the failing test**

```python
from validation.cases.hall_reconnection.reconnection_gem import main


def test_reconnection_gem_quick_smoke():
    assert main(quick_test=True) in (True, False)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_reconnection_gem_case.py::test_reconnection_gem_quick_smoke -v`
Expected: FAIL (module missing).

**Step 3: Write minimal implementation**

- Mirror pattern from `validation/cases/hall_reconnection/cylindrical_gem.py`.
- Use `CylindricalGEMConfiguration` from `jax_frc.configurations.validation_benchmarks`.
- Run 256/512/1024 (or smaller in quick mode).
- Compute metrics for JAX-FRC and AGATE and run regression with `block_bootstrap_ci`.

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_reconnection_gem_case.py::test_reconnection_gem_quick_smoke -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add validation/cases/hall_reconnection/reconnection_gem.py validation/cases/hall_reconnection/__init__.py tests/test_reconnection_gem_case.py
git commit -m "feat(validation): add GEM reconnection regression case"
```

## Task 5: Documentation updates

**Files:**
- Modify: `validation/README.md`
- Modify: `validation_guide.md`

**Step 1: Update docs**
- Add the two new cases and mention AGATE auto-download/cache location.

**Step 2: Commit**

```bash
git add validation/README.md validation_guide.md
git commit -m "docs(validation): add AGATE regression cases"
```

## Task 6: Run focused tests

**Step 1: Run unit tests for new utilities**
Run: `py -m pytest tests/test_agate_loader.py tests/test_validation_regression.py -v`
Expected: PASS.

**Step 2: Run quick validation cases**
Run: `py validation/cases/mhd_regression/orszag_tang.py --quick`
Run: `py validation/cases/hall_reconnection/reconnection_gem.py --quick`
Expected: Both run to completion (physics checks may be relaxed in quick mode).

**Step 3: Commit (if changes made)**

```bash
git add -A
git commit -m "test(validation): run quick AGATE cases"
```
