# Codebase Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove legacy root-level files and consolidate all tests into `tests/` directory.

**Architecture:** Delete 5 legacy implementation files, move `physics_utils.py` into the package, delete 3 test files that import legacy code, move 1 test file from `jax_frc/tests/` to `tests/`.

**Tech Stack:** Python, pytest, git

---

### Task 1: Verify Baseline Tests Pass

**Files:**
- None modified

**Step 1: Run all tests to establish baseline**

Run: `py -m pytest tests/ -k "not slow" -v --tb=short 2>&1 | tail -30`
Expected: All tests PASS (may have some skips)

**Step 2: Record test count**

Run: `py -m pytest tests/ -k "not slow" --collect-only -q 2>&1 | tail -5`
Expected: Note the number of tests collected

---

### Task 2: Move physics_utils.py to jax_frc/physics.py

**Files:**
- Move: `physics_utils.py` → `jax_frc/physics.py`
- Modify: `jax_frc/__init__.py`

**Step 1: Copy physics_utils.py to jax_frc/physics.py**

Run: `cp physics_utils.py jax_frc/physics.py`

**Step 2: Update imports to use jax_frc.constants**

Edit `jax_frc/physics.py` to replace local constant definitions with imports:

Replace:
```python
MU0 = 1.2566e-6
QE = 1.602e-19
ME = 9.109e-31
MI = 1.673e-27
KB = 1.381e-23
```

With:
```python
from jax_frc.constants import MU0, QE, ME, MI, KB, EPSILON0
```

And update `compute_debye_length` and `compute_plasma_frequency` to use `EPSILON0` instead of hardcoded `8.854e-12`.

**Step 3: Add module docstring**

Add at top of `jax_frc/physics.py`:
```python
"""Plasma physics utility functions.

Provides common plasma parameter calculations including:
- Characteristic speeds (Alfven, sound)
- Length scales (Larmor radius, skin depth, Debye length)
- Dimensionless numbers (beta, Reynolds, Lundquist, Mach)
- FRC-specific quantities (separatrix radius, volume)
- Energy calculations (magnetic, kinetic, thermal)
"""
```

**Step 4: Update jax_frc/__init__.py to export physics module**

Add to imports:
```python
from jax_frc import physics
```

Add `"physics"` to `__all__` list.

**Step 5: Verify import works**

Run: `py -c "from jax_frc import physics; print(physics.compute_alfven_speed(0.1, 1e19))"`
Expected: Prints a numeric value (Alfven speed)

**Step 6: Commit**

```bash
git add jax_frc/physics.py jax_frc/__init__.py
git commit -m "feat: add physics module with plasma utility functions"
```

---

### Task 3: Move test_simulation.py to tests/

**Files:**
- Move: `jax_frc/tests/test_simulation.py` → `tests/test_simulation_integration.py`
- Delete: `jax_frc/tests/__init__.py`
- Delete: `jax_frc/tests/` directory

**Step 1: Copy test file to new location**

Run: `cp jax_frc/tests/test_simulation.py tests/test_simulation_integration.py`

**Step 2: Verify the moved test runs**

Run: `py -m pytest tests/test_simulation_integration.py -v --tb=short 2>&1 | tail -20`
Expected: Tests run (may pass or fail depending on fixtures)

**Step 3: Delete jax_frc/tests directory**

Run: `rm -rf jax_frc/tests/`

**Step 4: Commit**

```bash
git add tests/test_simulation_integration.py
git rm -rf jax_frc/tests/
git commit -m "refactor: move simulation integration tests to tests/"
```

---

### Task 4: Delete Legacy Test Files

**Files:**
- Delete: `tests/test_resistive_mhd.py`
- Delete: `tests/test_extended_mhd.py`
- Delete: `tests/test_hybrid_kinetic.py`

**Step 1: Delete the three test files**

Run: `rm tests/test_resistive_mhd.py tests/test_extended_mhd.py tests/test_hybrid_kinetic.py`

**Step 2: Verify remaining tests still pass**

Run: `py -m pytest tests/ -k "not slow" -v --tb=short 2>&1 | tail -30`
Expected: Tests pass (count will be lower)

**Step 3: Commit**

```bash
git rm tests/test_resistive_mhd.py tests/test_extended_mhd.py tests/test_hybrid_kinetic.py
git commit -m "refactor: remove tests that depend on legacy root files"
```

---

### Task 5: Delete Legacy Root Files

**Files:**
- Delete: `resistive_mhd.py`
- Delete: `extended_mhd.py`
- Delete: `hybrid_kinetic.py`
- Delete: `examples.py`
- Delete: `test_simulations.py`
- Delete: `physics_utils.py`

**Step 1: Delete all legacy root files**

Run: `rm resistive_mhd.py extended_mhd.py hybrid_kinetic.py examples.py test_simulations.py physics_utils.py`

**Step 2: Verify no remaining imports of deleted files**

Run: `grep -r "from resistive_mhd\|from extended_mhd\|from hybrid_kinetic\|from physics_utils\|import resistive_mhd\|import extended_mhd\|import hybrid_kinetic\|import physics_utils" --include="*.py" . 2>/dev/null || echo "No remaining imports"`
Expected: "No remaining imports" (or only matches in .worktrees which are separate)

**Step 3: Run final test suite**

Run: `py -m pytest tests/ -k "not slow" -v --tb=short 2>&1 | tail -30`
Expected: All tests pass

**Step 4: Commit**

```bash
git rm resistive_mhd.py extended_mhd.py hybrid_kinetic.py examples.py test_simulations.py physics_utils.py
git commit -m "refactor: remove legacy root-level implementation files

These standalone functional implementations are replaced by the
class-based versions in jax_frc/models/. The physics utilities
have been moved to jax_frc/physics.py."
```

---

### Task 6: Final Verification

**Files:**
- None modified

**Step 1: Run full test suite**

Run: `py -m pytest tests/ -v --tb=short 2>&1 | tail -50`
Expected: All tests pass

**Step 2: Verify package imports work**

Run: `py -c "import jax_frc; from jax_frc.models import ResistiveMHD, ExtendedMHD, HybridKinetic; from jax_frc import physics; print('All imports OK')"`
Expected: "All imports OK"

**Step 3: List remaining root files**

Run: `ls -la *.py 2>/dev/null || echo "No .py files in root"`
Expected: "No .py files in root"

**Step 4: Verify test structure**

Run: `find tests -name "*.py" -type f | head -20`
Expected: List of test files all under `tests/`

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Verify baseline | None |
| 2 | Move physics_utils.py | +jax_frc/physics.py, ~jax_frc/__init__.py |
| 3 | Move test_simulation.py | +tests/test_simulation_integration.py, -jax_frc/tests/ |
| 4 | Delete legacy tests | -3 test files |
| 5 | Delete legacy root files | -6 root files |
| 6 | Final verification | None |

**Total commits:** 5
**Net file change:** -8 files (delete 11, add 2, modify 1)
