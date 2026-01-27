# Remove ValidationRunner Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove the deprecated ValidationRunner API and its tests/docs, leaving validation driven by standalone scripts.

**Architecture:** Delete the ValidationRunner module, remove its public exports, and replace references in tests and docs. Add a small regression test to assert ValidationRunner is no longer importable.

**Tech Stack:** Python, pytest, JAX validation utilities.

### Task 1: Add regression test for removed ValidationRunner

**Files:**
- Create: `tests/test_validation_api.py`

**Step 1: Write the failing test**

```python
import pytest


def test_validation_runner_removed():
    with pytest.raises(ImportError):
        from jax_frc.validation import ValidationRunner
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_api.py::test_validation_runner_removed -v`
Expected: FAIL (ValidationRunner still importable).

### Task 2: Remove ValidationRunner code and references

**Files:**
- Delete: `jax_frc/validation/runner.py`
- Modify: `jax_frc/validation/__init__.py:1-22`
- Delete: `tests/test_validation_runner.py`
- Delete: `tests/test_validation_integration.py`
- Modify: `docs/api/validation.md:1-60`
- Modify: `docs/api/index.md:50-70`

**Step 1: Remove ValidationRunner implementation**

- Delete `jax_frc/validation/runner.py`.
- Remove `ValidationRunner` import/export from `jax_frc/validation/__init__.py`.

**Step 2: Remove tests that depend on ValidationRunner**

- Delete `tests/test_validation_runner.py`.
- Delete `tests/test_validation_integration.py`.

**Step 3: Update docs to reference script-based validation**

- `docs/api/validation.md`: replace ValidationRunner examples with script usage (e.g., `python validation/cases/analytic/magnetic_diffusion.py`), and remove YAML Runner section.
- `docs/api/index.md`: remove `runner.py` line from module listing.

**Step 4: Run tests to verify it passes**

Run: `py -m pytest tests/test_validation_api.py::test_validation_runner_removed -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/test_validation_api.py jax_frc/validation/__init__.py docs/api/validation.md docs/api/index.md
git rm jax_frc/validation/runner.py tests/test_validation_runner.py tests/test_validation_integration.py
git commit -m "remove(validation): drop ValidationRunner API"
```
