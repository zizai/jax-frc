# Remove Validation Tests Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove unit tests in `tests/` that import `validation.*` while keeping `jax_frc.validation.*` tests intact.

**Architecture:** Identify tests that import `validation.*`, then delete those test files. Verify remaining tests run.

**Tech Stack:** Python, pytest, JAX (project codebase)

### Task 1: Identify validation-importing tests

**Files:**
- Inspect: `tests/`

**Step 1: Run a targeted search for validation imports**

Use Serena MCP search for `validation` imports in `tests/`:
- `from validation.` patterns
- `import validation` patterns

Expected: A list of test files containing validation imports.

**Step 2: Record the exact file list to remove**

Expected list (verify with search output):
- `tests/test_validation_regression.py`
- `tests/test_reconnection_gem_case.py`
- `tests/test_agate_loader_integration.py`
- `tests/test_agate_loader.py`
- `tests/test_agate_runner.py`
- `tests/test_validation_reporting.py`
- `tests/test_validation_cases.py`
- `tests/test_orszag_tang_case.py`
- `tests/test_agate_snapshots.py`

### Task 2: Remove validation tests

**Files:**
- Delete: `tests/test_validation_regression.py`
- Delete: `tests/test_reconnection_gem_case.py`
- Delete: `tests/test_agate_loader_integration.py`
- Delete: `tests/test_agate_loader.py`
- Delete: `tests/test_agate_runner.py`
- Delete: `tests/test_validation_reporting.py`
- Delete: `tests/test_validation_cases.py`
- Delete: `tests/test_orszag_tang_case.py`
- Delete: `tests/test_agate_snapshots.py`

**Step 1: Delete the files listed above**

Run: `git rm <each file>`

Expected: Files removed from the working tree.

**Step 2: Confirm no remaining validation imports in tests**

Use Serena MCP search to confirm no `validation` imports remain in `tests/` (both `from validation.` and `import validation`).

Expected: No matches.

### Task 3: Verify remaining tests

**Files:**
- Test: `tests/`

**Step 1: Run the fast test subset**

Run: `py -m pytest tests/ -k "not slow"`

Expected: PASS (no validation tests collected).

### Task 4: Commit changes

**Files:**
- Modify: `tests/` (deletions)

**Step 1: Stage deletions**

Run: `git add tests/`

**Step 2: Commit**

Run: `git commit -m "test: remove validation suite tests"`

Expected: Commit created with deletions.
