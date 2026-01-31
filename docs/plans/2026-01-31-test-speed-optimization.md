# Test Speed Optimization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce `pytest tests/ -k "not slow" -v` runtime toward ~5 minutes without removing coverage or adding dependencies.

**Architecture:** Target the slowest tests first (IMEX validation, IMEX diffusion, Orszag–Tang quick smoke, linear solver IMEX, MMS convergence, energy integration). Use smaller grids/step counts, cached fixtures, and quick-mode parameters while preserving the same invariants and assertions.

**Tech Stack:** Python 3.12, JAX, pytest.

---

### Task 1: Record baseline slow list (chunked) and confirm top offenders

**Files:**
- None

**Step 1: Run a small chunk to verify the durations workflow**

Run: `.venv/bin/python -m pytest tests/test_imex_diffusion.py -k "not slow" -q --durations=10`  
Expected: PASS, durations show the two slow IMEX diffusion tests.

**Step 2: Capture IMEX validation single-test durations (4 commands)**

Run (one at a time):
```
.venv/bin/python -m pytest tests/test_imex_validation.py::TestResistiveDiffusionAnalytic::test_1d_diffusion_decay_rate -q --durations=3
.venv/bin/python -m pytest tests/test_imex_validation.py::TestResistiveDiffusionAnalytic::test_decay_rate_scales_with_resistivity -q --durations=3
.venv/bin/python -m pytest tests/test_imex_validation.py::TestResistiveDiffusionAnalytic::test_uniform_field_is_stationary -q --durations=3
.venv/bin/python -m pytest tests/test_imex_validation.py::TestResistiveDiffusionAnalytic::test_diffusion_preserves_pattern_shape -q --durations=3
```
Expected: PASS, durations recorded for each test.

**Step 3: Confirm Orszag–Tang quick smoke duration**

Run: `.venv/bin/python -m pytest tests/test_orszag_tang_case.py::test_orszag_tang_quick_smoke -q --durations=3`  
Expected: PASS, duration printed.

**Step 4: Commit nothing (baseline only)**

No changes yet.

---

### Task 2: Speed up IMEX analytic validation tests

**Files:**
- Modify: `tests/test_imex_validation.py`

**Step 1: Write the failing test**

Add a tiny grid/step variant for one test (same assertions, smaller parameters) so it fails with the current large defaults (timeout risk). Example intended change shown below.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_imex_validation.py::TestResistiveDiffusionAnalytic::test_1d_diffusion_decay_rate -v`  
Expected: FAIL (timeout or long runtime).

**Step 3: Write minimal implementation (reduce grids/steps)**

Update parameters:
```python
# test_1d_diffusion_decay_rate
nx, ny, nz = 8, 1, 16
dt = 5e-5
n_steps = 10

# test_diffusion_preserves_pattern_shape
nx, ny, nz = 6, 1, 12
dt = 1e-5
n_steps = 3

# test_uniform_field_is_stationary
nx, ny, nz = 6, 1, 12
dt = 1e-4
n_steps = 5

# test_decay_rate_scales_with_resistivity
nx, ny, nz = 6, 1, 12
dt = 2e-5
n_steps = 6
```

**Step 4: Run tests to verify they pass**

Run:
```
.venv/bin/python -m pytest tests/test_imex_validation.py -k "ResistiveDiffusionAnalytic" -v
```
Expected: PASS, runtime reduced.

**Step 5: Commit**

```
git add tests/test_imex_validation.py
git commit -m "test: reduce IMEX validation grid and step counts"
```

---

### Task 3: Speed up IMEX diffusion analytic tests

**Files:**
- Modify: `tests/test_imex_diffusion.py`

**Step 1: Write the failing test**

Change grid/step counts to smaller values (see below) so current runtime no longer matches expectations if you run the old code.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_imex_diffusion.py -v`  
Expected: FAIL (if parameters not updated).

**Step 3: Write minimal implementation**

```python
# test_gaussian_diffusion_converges
geometry = make_geometry(nx=16, ny=1, nz=16, extent=0.5)
dt = 2e-5
n_steps = int(t_final / dt)

# test_imex_large_timestep_stable
geometry = make_geometry(nx=8, ny=1, nz=8, extent=0.5)
for _ in range(5):
    state = solver.step(...)
```
If needed, relax error tolerance slightly (e.g., 0.07 -> 0.10) but only if required to keep the same qualitative assertion.

**Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_imex_diffusion.py -v`  
Expected: PASS.

**Step 5: Commit**

```
git add tests/test_imex_diffusion.py
git commit -m "test: reduce IMEX diffusion grid size and steps"
```

---

### Task 4: Speed up Orszag–Tang quick smoke

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`

**Step 1: Write the failing test**

Update quick-mode parameters and run the quick smoke (should fail until code updated).

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_orszag_tang_case.py::test_orszag_tang_quick_smoke -v`  
Expected: FAIL (if quick mode not updated).

**Step 3: Write minimal implementation**

```python
QUICK_RESOLUTIONS = ([64, 64, 1],)
QUICK_NUM_SNAPSHOTS = 3

def get_quick_snapshot_times(end_time: float) -> list[float]:
    return [0.0, end_time / 2.0, end_time]

def setup_configuration(quick_test: bool, resolution: list[int]) -> dict:
    t_end = 0.12 if quick_test else 0.48
    return {
        "nx": resolution[0],
        "nz": resolution[2] if resolution[2] > 1 else resolution[0],
        "t_end": t_end,
        "dt": 1e-4,
        "use_cfl": True,
    }
```
Also update `main()` quick snapshot selection to use the new 3-snapshot list if needed.

**Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_orszag_tang_case.py::test_orszag_tang_quick_smoke -v`  
Expected: PASS, runtime reduced.

**Step 5: Commit**

```
git add validation/cases/regression/orszag_tang.py
git commit -m "test: speed up Orszag–Tang quick smoke mode"
```

---

### Task 5: Speed up linear solver IMEX tests by shrinking fixtures

**Files:**
- Modify: `tests/test_linear_solvers.py`

**Step 1: Write the failing test**

Reduce geometry fixture sizes to smaller grids; existing behavior should still hold but will fail until fixture updated.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_linear_solvers.py::TestImexFullStep::test_imex_step_stable_with_moderate_dt -v`  
Expected: FAIL if fixture not updated.

**Step 3: Write minimal implementation**

Update fixtures:
```python
class TestImexDiffusion:
    @pytest.fixture
    def geometry(self):
        return make_geometry(nx=8, ny=2, nz=16)

class TestImexExplicit:
    @pytest.fixture
    def geometry(self):
        return make_geometry(nx=8, ny=2, nz=16)

class TestImexFullStep:
    @pytest.fixture
    def geometry(self):
        return make_geometry(nx=6, ny=2, nz=12)
```
Optionally reduce loop counts in `test_imex_step_stable_with_moderate_dt` from 10 to 5 steps.

**Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_linear_solvers.py -k "Imex" -v`  
Expected: PASS, runtime reduced.

**Step 5: Commit**

```
git add tests/test_linear_solvers.py
git commit -m "test: shrink IMEX linear solver fixtures"
```

---

### Task 6: Speed up MMS convergence tests

**Files:**
- Modify: `tests/test_mhd_mms_convergence.py`

**Step 1: Write the failing test**

Lower coarse/fine sizes to 6/12 and run convergence tests.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_mhd_mms_convergence.py -v`  
Expected: FAIL before updates.

**Step 3: Write minimal implementation**

```python
geom_coarse = make_mms_geometry(6)
geom_fine = make_mms_geometry(12)

# Hall + EP test
geom_coarse = make_mms_geometry_2d(6)
geom_fine = make_mms_geometry_2d(12)
```

**Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_mhd_mms_convergence.py -v`  
Expected: PASS, runtime reduced.

**Step 5: Commit**

```
git add tests/test_mhd_mms_convergence.py
git commit -m "test: reduce MMS convergence grid sizes"
```

---

### Task 7: Speed up energy integration tests

**Files:**
- Modify: `tests/test_energy_integration.py`

**Step 1: Write the failing test**

Reduce geometry sizes and step counts; run one test to confirm failure until code is updated.

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_energy_integration.py::TestEndToEndTemperatureEvolution::test_simulation_runs_without_error -v`  
Expected: FAIL before updates.

**Step 3: Write minimal implementation**

```python
# geometry fixtures
return make_geometry(nx=12, ny=2, nz=16)

# run steps
n_steps = 5  # was 10
n_steps = 8  # was 20

# diffusion conservation
n_steps = 3  # was 5

# 1D heat conduction test
geometry = Geometry(nx=8, ny=2, nz=32, ...)
n_steps = 12  # was 30
dt = 2e-5
```

**Step 4: Run tests to verify pass**

Run: `.venv/bin/python -m pytest tests/test_energy_integration.py -v`  
Expected: PASS, runtime reduced.

**Step 5: Commit**

```
git add tests/test_energy_integration.py
git commit -m "test: reduce energy integration grid sizes and steps"
```

---

### Task 8: Re-run chunked durations and compare totals

**Files:**
- None

**Step 1: Re-run slow chunks**

Run:
```
.venv/bin/python -m pytest tests/test_imex_validation.py -k "ResistiveDiffusionAnalytic" -q --durations=10
.venv/bin/python -m pytest tests/test_imex_diffusion.py -q --durations=10
.venv/bin/python -m pytest tests/test_orszag_tang_case.py::test_orszag_tang_quick_smoke -q --durations=3
.venv/bin/python -m pytest tests/test_linear_solvers.py -k "Imex" -q --durations=10
.venv/bin/python -m pytest tests/test_mhd_mms_convergence.py -q --durations=10
.venv/bin/python -m pytest tests/test_energy_integration.py -q --durations=10
```
Expected: PASS, durations reduced.

**Step 2: Evaluate whether we meet ~5 minutes**

If still above target, repeat with next tranche (JIT tests and hybrid kinetic) or adjust steps further.

**Step 3: Commit any remaining adjustments**

```
git status
git add -A
git commit -m "test: reduce slow test runtimes"
```

---

**Plan complete and saved to `docs/plans/2026-01-31-test-speed-optimization.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

Which approach?
