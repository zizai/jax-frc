# Scenarios Module Consolidation Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Consolidate `jax_frc/scenarios/` into `jax_frc/configurations/` for a unified module structure.

**Architecture:** Move phase.py, transitions.py, and phases/ into configurations/. Update all imports across the codebase. Delete the scenarios/ module.

**Tech Stack:** Python, JAX

---

## Task 1: Move phase.py and Update Its Internal Import

**Files:**
- Move: `jax_frc/scenarios/phase.py` → `jax_frc/configurations/phase.py`
- Modify: The moved file's internal import

**Step 1: Copy phase.py to configurations/**

```bash
cp jax_frc/scenarios/phase.py jax_frc/configurations/phase.py
```

**Step 2: Update internal import in the copied file**

Change line 7 in `jax_frc/configurations/phase.py`:
```python
# Before
from jax_frc.scenarios.transitions import Transition

# After
from jax_frc.configurations.transitions import Transition
```

**Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('jax_frc/configurations/phase.py').read())"`
Expected: No output (successful parse)

**Step 4: Commit**

```bash
git add jax_frc/configurations/phase.py
git commit -m "refactor: copy phase.py to configurations module"
```

---

## Task 2: Move transitions.py

**Files:**
- Move: `jax_frc/scenarios/transitions.py` → `jax_frc/configurations/transitions.py`

**Step 1: Copy transitions.py to configurations/**

```bash
cp jax_frc/scenarios/transitions.py jax_frc/configurations/transitions.py
```

**Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('jax_frc/configurations/transitions.py').read())"`
Expected: No output (successful parse)

**Step 3: Commit**

```bash
git add jax_frc/configurations/transitions.py
git commit -m "refactor: copy transitions.py to configurations module"
```

---

## Task 3: Move phases/ Directory and Update Its Imports

**Files:**
- Move: `jax_frc/scenarios/phases/` → `jax_frc/configurations/phases/`
- Modify: `jax_frc/configurations/phases/__init__.py`
- Modify: `jax_frc/configurations/phases/merging.py`

**Step 1: Copy phases/ directory**

```bash
cp -r jax_frc/scenarios/phases jax_frc/configurations/phases
```

**Step 2: Update imports in phases/__init__.py**

Replace entire file `jax_frc/configurations/phases/__init__.py`:
```python
"""Phase implementations for FRC experiments."""

from jax_frc.configurations.phases.merging import MergingPhase
from jax_frc.configurations.phase import PHASE_REGISTRY

# Register phases in the global registry
PHASE_REGISTRY["MergingPhase"] = MergingPhase

__all__ = [
    "MergingPhase",
]
```

**Step 3: Update imports in phases/merging.py**

In `jax_frc/configurations/phases/merging.py`, change lines 7-8:
```python
# Before
from jax_frc.scenarios.phase import Phase
from jax_frc.scenarios.transitions import Transition

# After
from jax_frc.configurations.phase import Phase
from jax_frc.configurations.transitions import Transition
```

**Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('jax_frc/configurations/phases/__init__.py').read())"`
Run: `python -c "import ast; ast.parse(open('jax_frc/configurations/phases/merging.py').read())"`
Expected: No output for both

**Step 5: Commit**

```bash
git add jax_frc/configurations/phases/
git commit -m "refactor: copy phases/ to configurations module"
```

---

## Task 4: Update linear_configuration.py Imports

**Files:**
- Modify: `jax_frc/configurations/linear_configuration.py` (lines 17-18)

**Step 1: Update imports**

Change lines 17-18 in `jax_frc/configurations/linear_configuration.py`:
```python
# Before
from jax_frc.scenarios.phase import Phase, PhaseResult, PHASE_REGISTRY
from jax_frc.scenarios.transitions import transition_from_spec

# After
from jax_frc.configurations.phase import Phase, PhaseResult, PHASE_REGISTRY
from jax_frc.configurations.transitions import transition_from_spec
```

**Step 2: Verify import works**

Run: `python -c "from jax_frc.configurations.linear_configuration import LinearConfiguration; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add jax_frc/configurations/linear_configuration.py
git commit -m "refactor: update linear_configuration imports to configurations module"
```

---

## Task 5: Update configurations/__init__.py with Unified Exports

**Files:**
- Modify: `jax_frc/configurations/__init__.py`

**Step 1: Replace entire __init__.py**

Replace `jax_frc/configurations/__init__.py` with:
```python
"""Configuration classes for reactor and benchmark setups."""

# Core abstractions
from jax_frc.configurations.base import AbstractConfiguration
from jax_frc.configurations.phase import (
    Phase,
    PhaseResult,
    PHASE_REGISTRY,
    register_phase,
)

# Transitions
from jax_frc.configurations.transitions import (
    Transition,
    timeout,
    condition,
    any_of,
    all_of,
    separation_below,
    temperature_above,
    flux_below,
    velocity_below,
    transition_from_spec,
)

# Configuration implementations
from jax_frc.configurations.linear_configuration import (
    LinearConfiguration,
    TransitionSpec,
    PhaseSpec,
    ConfigurationResult,
)
from jax_frc.configurations.frc_merging import (
    BelovaMergingConfiguration,
    BelovaCase1Configuration,
    BelovaCase2Configuration,
    BelovaCase4Configuration,
)
from jax_frc.configurations.analytic import SlabDiffusionConfiguration

# Import phases submodule to trigger registration
from jax_frc.configurations import phases

CONFIGURATION_REGISTRY = {
    'SlabDiffusionConfiguration': SlabDiffusionConfiguration,
    'LinearConfiguration': LinearConfiguration,
    'BelovaMergingConfiguration': BelovaMergingConfiguration,
    'BelovaCase1Configuration': BelovaCase1Configuration,
    'BelovaCase2Configuration': BelovaCase2Configuration,
    'BelovaCase4Configuration': BelovaCase4Configuration,
}

__all__ = [
    # Core abstractions
    'AbstractConfiguration',
    'Phase',
    'PhaseResult',
    'PHASE_REGISTRY',
    'register_phase',
    # Transitions
    'Transition',
    'timeout',
    'condition',
    'any_of',
    'all_of',
    'separation_below',
    'temperature_above',
    'flux_below',
    'velocity_below',
    'transition_from_spec',
    # Configuration implementations
    'SlabDiffusionConfiguration',
    'LinearConfiguration',
    'ConfigurationResult',
    'TransitionSpec',
    'PhaseSpec',
    'BelovaMergingConfiguration',
    'BelovaCase1Configuration',
    'BelovaCase2Configuration',
    'BelovaCase4Configuration',
    'CONFIGURATION_REGISTRY',
]
```

**Step 2: Verify module imports**

Run: `python -c "from jax_frc.configurations import Phase, timeout, LinearConfiguration, PHASE_REGISTRY; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add jax_frc/configurations/__init__.py
git commit -m "refactor: add unified exports to configurations module"
```

---

## Task 6: Update Test Files

**Files:**
- Rename: `tests/test_scenarios.py` → `tests/test_phases.py`
- Modify: `tests/test_phases.py` (imports)
- Modify: `tests/test_linear_configuration.py` (imports)
- Modify: `tests/test_merging_phase.py` (imports)

**Step 1: Rename test_scenarios.py**

```bash
git mv tests/test_scenarios.py tests/test_phases.py
```

**Step 2: Update imports in test_phases.py**

In `tests/test_phases.py`, update the imports at the top:
```python
# Before
from jax_frc.scenarios.transitions import Transition, timeout, condition, any_of, all_of
from jax_frc.scenarios.phase import Phase, PhaseResult

# After
from jax_frc.configurations import Transition, timeout, condition, any_of, all_of
from jax_frc.configurations import Phase, PhaseResult
```

Also update the inline imports in test methods:
```python
# Before (lines 98, 125, 145, 165, 185)
from jax_frc.scenarios.transitions import separation_below
from jax_frc.scenarios.transitions import temperature_above
from jax_frc.scenarios.transitions import flux_below
from jax_frc.scenarios.transitions import velocity_below
from jax_frc.scenarios.phase import PhaseResult

# After
from jax_frc.configurations import separation_below
from jax_frc.configurations import temperature_above
from jax_frc.configurations import flux_below
from jax_frc.configurations import velocity_below
from jax_frc.configurations import PhaseResult
```

**Step 3: Update imports in test_linear_configuration.py**

In `tests/test_linear_configuration.py`, change the import:
```python
# Before
from jax_frc.scenarios import (

# After
from jax_frc.configurations import (
```

**Step 4: Update imports in test_merging_phase.py**

In `tests/test_merging_phase.py`, change:
```python
# Before
from jax_frc.scenarios.phases.merging import MergingPhase
from jax_frc.scenarios.transitions import timeout

# After
from jax_frc.configurations.phases.merging import MergingPhase
from jax_frc.configurations import timeout
```

**Step 5: Run tests to verify**

Run: `py -m pytest tests/test_phases.py tests/test_linear_configuration.py tests/test_merging_phase.py -v`
Expected: All tests pass (except pre-existing failures in backward compatibility tests)

**Step 6: Commit**

```bash
git add tests/test_phases.py tests/test_linear_configuration.py tests/test_merging_phase.py
git commit -m "refactor: update test imports to configurations module"
```

---

## Task 7: Delete scenarios/ Module

**Files:**
- Delete: `jax_frc/scenarios/` (entire directory)

**Step 1: Remove scenarios/ directory**

```bash
git rm -r jax_frc/scenarios/
```

**Step 2: Run full test suite**

Run: `py -m pytest tests/ -v --ignore=tests/test_frc_merging_configuration.py::TestBackwardCompatibility`
Expected: All tests pass (198 passing)

**Step 3: Commit**

```bash
git commit -m "refactor: remove scenarios module (consolidated into configurations)"
```

---

## Task 8: Final Verification

**Step 1: Run full test suite**

Run: `py -m pytest tests/ -q`
Expected: 198 passed, 6 failed (pre-existing backward compatibility failures)

**Step 2: Verify example execution**

Run: `python scripts/run_example.py belova_case1 --dry-run` (if --dry-run exists, otherwise just verify import)
Run: `python -c "from jax_frc.configurations import BelovaCase1Configuration; c = BelovaCase1Configuration(); print(c.name)"`
Expected: `belova_case1`

**Step 3: Final commit (if any remaining changes)**

```bash
git status
# If clean, no action needed
```

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Move phase.py | +1 |
| 2 | Move transitions.py | +1 |
| 3 | Move phases/ | +2 |
| 4 | Update linear_configuration.py | ~1 |
| 5 | Update configurations/__init__.py | ~1 |
| 6 | Update test files | ~3 |
| 7 | Delete scenarios/ | -5 |
| 8 | Final verification | 0 |

Total: 8 tasks, ~8 commits
