# Physics Model Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Connect physics models to scenario runner and add Belova paper validation tests.

**Architecture:** Scenario owns PhysicsModel and Solver. _run_phase calls solver.step() in time loop. Diagnostics recorded at intervals into PhaseResult.history.

**Tech Stack:** JAX, pytest, dataclasses

---

## Task 1: Add physics_model and solver to Scenario

**Files:**
- Modify: `jax_frc/scenarios/scenario.py:1-51`
- Test: `tests/test_scenarios.py`

**Step 1: Write the failing test**

Add to `tests/test_scenarios.py`:

```python
def test_scenario_requires_physics_model_and_solver():
    """Scenario should require physics_model and solver fields."""
    from jax_frc.models.base import PhysicsModel
    from jax_frc.solvers.base import Solver

    # Check that Scenario has these attributes in its signature
    import inspect
    sig = inspect.signature(Scenario)
    params = list(sig.parameters.keys())

    assert "physics_model" in params
    assert "solver" in params
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_scenarios.py::test_scenario_requires_physics_model_and_solver -v`
Expected: FAIL with AssertionError

**Step 3: Update imports and add fields to Scenario**

In `jax_frc/scenarios/scenario.py`, update imports at top:

```python
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.scenarios.phase import Phase, PhaseResult
from jax_frc.models.base import PhysicsModel
from jax_frc.solvers.base import Solver
```

Update Scenario dataclass (add after `initial_state` field, before `dt`):

```python
@dataclass
class Scenario:
    """Orchestrates multi-phase FRC simulations.

    Runs phases in sequence, passing state between them.
    Each phase runs until its transition triggers.

    Attributes:
        name: Scenario identifier
        phases: Ordered list of phases to run
        geometry: Computational geometry
        initial_state: Starting state (or None if first phase creates it)
        physics_model: Physics model for evolution equations
        solver: Time integrator
        dt: Timestep for simulation
        config: Optional per-phase configuration overrides
    """

    name: str
    phases: List[Phase]
    geometry: Geometry
    initial_state: Optional[State]
    physics_model: PhysicsModel
    solver: Solver
    dt: float
    config: Dict[str, dict] = field(default_factory=dict)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_scenarios.py::test_scenario_requires_physics_model_and_solver -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/scenarios/scenario.py tests/test_scenarios.py
git commit -m "feat(scenarios): add physics_model and solver to Scenario"
```

---

## Task 2: Update _run_phase to call solver

**Files:**
- Modify: `jax_frc/scenarios/scenario.py:91-135`
- Test: `tests/test_scenarios.py`

**Step 1: Write the failing test**

Add to `tests/test_scenarios.py`:

```python
def test_scenario_calls_solver_step():
    """Scenario._run_phase should call solver.step() for physics evolution."""
    from unittest.mock import MagicMock, patch
    from jax_frc.core.geometry import Geometry
    from jax_frc.core.state import State
    from jax_frc.scenarios import Scenario
    from jax_frc.scenarios.phase import Phase
    from jax_frc.scenarios.transitions import timeout

    geometry = Geometry(
        coord_system="cylindrical", nr=8, nz=16,
        r_min=0.01, r_max=1.0, z_min=-1.0, z_max=1.0
    )
    state = State.zeros(nr=8, nz=16)

    # Mock physics model and solver
    mock_model = MagicMock()
    mock_solver = MagicMock()

    # Solver.step returns updated state with incremented time
    def fake_step(s, dt, model, geom):
        return s.replace(time=s.time + dt, step=s.step + 1)
    mock_solver.step.side_effect = fake_step

    phase = Phase(name="test", transition=timeout(0.05))

    scenario = Scenario(
        name="test",
        phases=[phase],
        geometry=geometry,
        initial_state=state,
        physics_model=mock_model,
        solver=mock_solver,
        dt=0.01,
    )

    result = scenario.run()

    # Solver should have been called multiple times
    assert mock_solver.step.call_count >= 5
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_scenarios.py::test_scenario_calls_solver_step -v`
Expected: FAIL (solver.step not called)

**Step 3: Update _run_phase to call solver**

In `jax_frc/scenarios/scenario.py`, replace the time loop in `_run_phase`:

```python
    def _run_phase(self, phase: Phase, state: State, config: dict) -> PhaseResult:
        """Run a single phase to completion.

        Args:
            phase: Phase to run
            state: Initial state for this phase
            config: Phase configuration

        Returns:
            PhaseResult with final state and diagnostics
        """
        # Setup phase
        start_time = float(state.time) if state else 0.0
        state = phase.setup(state, self.geometry, config)
        initial_state = state

        t = start_time
        termination = "condition_met"

        # Run until transition triggers
        while True:
            # Check for completion
            complete, reason = phase.is_complete(state, t)
            if complete:
                termination = reason
                break

            # Apply step hook (time-varying BCs, etc.)
            state = phase.step_hook(state, self.geometry, t)

            # Advance physics via solver
            state = self.solver.step(state, self.dt, self.physics_model, self.geometry)
            t = float(state.time)

        # Cleanup
        state = phase.on_complete(state, self.geometry)

        return PhaseResult(
            name=phase.name,
            initial_state=initial_state,
            final_state=state,
            start_time=start_time,
            end_time=t,
            termination=termination,
        )
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_scenarios.py::test_scenario_calls_solver_step -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/scenarios/scenario.py tests/test_scenarios.py
git commit -m "feat(scenarios): integrate solver.step() into phase time loop"
```

---

## Task 3: Fix existing tests that don't provide physics_model/solver

**Files:**
- Modify: `tests/test_scenarios.py`
- Modify: `tests/test_merging_integration.py`
- Modify: `tests/test_merging_phase.py`
- Modify: `examples/merging_examples.py`

**Step 1: Run all tests to see failures**

Run: `py -m pytest tests/ -v --tb=short 2>&1 | head -80`
Expected: Multiple failures due to missing physics_model/solver args

**Step 2: Create test fixtures for mock model/solver**

Add to `tests/conftest.py`:

```python
import pytest
from unittest.mock import MagicMock

@pytest.fixture
def mock_physics_model():
    """Mock physics model that returns unchanged state."""
    model = MagicMock()
    model.compute_rhs.return_value = None
    model.apply_constraints.side_effect = lambda s, g: s
    return model

@pytest.fixture
def mock_solver():
    """Mock solver that just increments time."""
    solver = MagicMock()
    def fake_step(state, dt, model, geometry):
        return state.replace(time=state.time + dt, step=state.step + 1)
    solver.step.side_effect = fake_step
    return solver
```

**Step 3: Update test_scenarios.py to use fixtures**

Update `TestScenario` class tests to accept and use the fixtures:

```python
class TestScenario:
    """Tests for Scenario runner."""

    def test_scenario_runs_single_phase(self, mock_physics_model, mock_solver):
        """Single phase scenario runs and returns result."""
        geometry = Geometry(
            coord_system="cylindrical", nr=8, nz=16,
            r_min=0.01, r_max=1.0, z_min=-1.0, z_max=1.0
        )
        state = State.zeros(nr=8, nz=16)
        phase = Phase(name="test", transition=timeout(0.05))

        scenario = Scenario(
            name="test_scenario",
            phases=[phase],
            geometry=geometry,
            initial_state=state,
            physics_model=mock_physics_model,
            solver=mock_solver,
            dt=0.01,
        )

        result = scenario.run()

        assert result.success
        assert len(result.phase_results) == 1
        assert result.phase_results[0].name == "test"

    def test_scenario_chains_phases(self, mock_physics_model, mock_solver):
        """Multi-phase scenario passes state between phases."""
        geometry = Geometry(
            coord_system="cylindrical", nr=8, nz=16,
            r_min=0.01, r_max=1.0, z_min=-1.0, z_max=1.0
        )
        state = State.zeros(nr=8, nz=16)

        phase1 = Phase(name="phase1", transition=timeout(0.02))
        phase2 = Phase(name="phase2", transition=timeout(0.02))

        scenario = Scenario(
            name="multi_phase",
            phases=[phase1, phase2],
            geometry=geometry,
            initial_state=state,
            physics_model=mock_physics_model,
            solver=mock_solver,
            dt=0.01,
        )

        result = scenario.run()

        assert result.success
        assert len(result.phase_results) == 2
```

**Step 4: Update test_merging_integration.py**

Update to use real or mock physics:

```python
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.solvers.explicit import EulerSolver

class TestMergingIntegration:
    """Integration tests for full merging workflow."""

    def test_scenario_runs_to_completion(self):
        """Full scenario runs without errors."""
        scenario = belova_case2()
        result = scenario.run()

        assert result.success
        assert len(result.phase_results) == 1
        assert result.total_time > 0
```

**Step 5: Update merging_examples.py to create physics_model/solver**

Update `belova_case1`, `belova_case2`, `belova_case4` functions (see Task 5 for full implementation).

**Step 6: Run all tests**

Run: `py -m pytest tests/ -v`
Expected: All PASS

**Step 7: Commit**

```bash
git add tests/conftest.py tests/test_scenarios.py tests/test_merging_integration.py tests/test_merging_phase.py
git commit -m "test: update tests to provide physics_model and solver"
```

---

## Task 4: Add history tracking to PhaseResult

**Files:**
- Modify: `jax_frc/scenarios/phase.py:10-21`
- Test: `tests/test_scenarios.py`

**Step 1: Write the failing test**

Add to `tests/test_scenarios.py`:

```python
def test_phase_result_has_history():
    """PhaseResult should have a history dict."""
    from jax_frc.scenarios.phase import PhaseResult
    from jax_frc.core.state import State

    state = State.zeros(nr=8, nz=16)
    result = PhaseResult(
        name="test",
        initial_state=state,
        final_state=state,
        start_time=0.0,
        end_time=1.0,
        termination="condition_met",
        history={"time": [0.0, 0.5, 1.0], "metric": [1.0, 0.5, 0.2]}
    )

    assert result.history["time"] == [0.0, 0.5, 1.0]
    assert result.history["metric"] == [1.0, 0.5, 0.2]
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_scenarios.py::test_phase_result_has_history -v`
Expected: FAIL (history not accepted or wrong default)

**Step 3: Update PhaseResult dataclass**

In `jax_frc/scenarios/phase.py`, update PhaseResult:

```python
@dataclass
class PhaseResult:
    """Result of running a phase."""

    name: str
    initial_state: State
    final_state: State
    start_time: float
    end_time: float
    termination: str  # "condition_met", "timeout", "error"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    history: Dict[str, List[float]] = field(default_factory=dict)
```

Update imports at top:

```python
from dataclasses import dataclass, field
from typing import Dict, Any, List
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_scenarios.py::test_phase_result_has_history -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/scenarios/phase.py tests/test_scenarios.py
git commit -m "feat(scenarios): add history field to PhaseResult"
```

---

## Task 5: Add diagnostics list and output_interval to Scenario

**Files:**
- Modify: `jax_frc/scenarios/scenario.py`
- Test: `tests/test_scenarios.py`

**Step 1: Write the failing test**

Add to `tests/test_scenarios.py`:

```python
def test_scenario_has_diagnostics_fields():
    """Scenario should have diagnostics list and output_interval."""
    import inspect
    sig = inspect.signature(Scenario)
    params = list(sig.parameters.keys())

    assert "diagnostics" in params
    assert "output_interval" in params
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_scenarios.py::test_scenario_has_diagnostics_fields -v`
Expected: FAIL

**Step 3: Add fields to Scenario**

In `jax_frc/scenarios/scenario.py`, add import and fields:

```python
from jax_frc.diagnostics.probes import Probe
```

Update Scenario dataclass (add after `config`):

```python
    config: Dict[str, dict] = field(default_factory=dict)
    diagnostics: List[Probe] = field(default_factory=list)
    output_interval: int = 100
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_scenarios.py::test_scenario_has_diagnostics_fields -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/scenarios/scenario.py tests/test_scenarios.py
git commit -m "feat(scenarios): add diagnostics list and output_interval to Scenario"
```

---

## Task 6: Record diagnostics in _run_phase

**Files:**
- Modify: `jax_frc/scenarios/scenario.py:91-135`
- Test: `tests/test_scenarios.py`

**Step 1: Write the failing test**

Add to `tests/test_scenarios.py`:

```python
def test_scenario_records_diagnostics_history(mock_physics_model, mock_solver):
    """Scenario should record diagnostics at output_interval."""
    from jax_frc.diagnostics.probes import Probe
    from dataclasses import dataclass

    @dataclass
    class CountingProbe(Probe):
        @property
        def name(self) -> str:
            return "counter"

        def measure(self, state, geometry) -> float:
            return float(state.step)

    geometry = Geometry(
        coord_system="cylindrical", nr=8, nz=16,
        r_min=0.01, r_max=1.0, z_min=-1.0, z_max=1.0
    )
    state = State.zeros(nr=8, nz=16)
    phase = Phase(name="test", transition=timeout(0.1))

    scenario = Scenario(
        name="test",
        phases=[phase],
        geometry=geometry,
        initial_state=state,
        physics_model=mock_physics_model,
        solver=mock_solver,
        dt=0.01,
        diagnostics=[CountingProbe()],
        output_interval=2,  # Record every 2 steps
    )

    result = scenario.run()

    # Should have recorded history
    history = result.phase_results[0].history
    assert "time" in history
    assert "counter" in history
    assert len(history["time"]) > 0
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_scenarios.py::test_scenario_records_diagnostics_history -v`
Expected: FAIL (history empty)

**Step 3: Update _run_phase to record diagnostics**

In `jax_frc/scenarios/scenario.py`, update `_run_phase`:

```python
    def _run_phase(self, phase: Phase, state: State, config: dict) -> PhaseResult:
        """Run a single phase to completion.

        Args:
            phase: Phase to run
            state: Initial state for this phase
            config: Phase configuration

        Returns:
            PhaseResult with final state and diagnostics
        """
        # Setup phase
        start_time = float(state.time) if state else 0.0
        state = phase.setup(state, self.geometry, config)
        initial_state = state

        t = start_time
        termination = "condition_met"

        # Initialize history tracking
        history: Dict[str, List[float]] = {"time": []}
        for probe in self.diagnostics:
            history[probe.name] = []

        # Run until transition triggers
        while True:
            # Check for completion
            complete, reason = phase.is_complete(state, t)
            if complete:
                termination = reason
                break

            # Record diagnostics at intervals
            if state.step % self.output_interval == 0:
                history["time"].append(t)
                for probe in self.diagnostics:
                    history[probe.name].append(probe.measure(state, self.geometry))

            # Apply step hook (time-varying BCs, etc.)
            state = phase.step_hook(state, self.geometry, t)

            # Advance physics via solver
            state = self.solver.step(state, self.dt, self.physics_model, self.geometry)
            t = float(state.time)

        # Cleanup
        state = phase.on_complete(state, self.geometry)

        return PhaseResult(
            name=phase.name,
            initial_state=initial_state,
            final_state=state,
            start_time=start_time,
            end_time=t,
            termination=termination,
            history=history,
        )
```

Add import at top:

```python
from typing import List, Optional, Dict, Any
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_scenarios.py::test_scenario_records_diagnostics_history -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/scenarios/scenario.py tests/test_scenarios.py
git commit -m "feat(scenarios): record diagnostics history in _run_phase"
```

---

## Task 7: Update merging examples with physics_model and solver

**Files:**
- Modify: `examples/merging_examples.py`
- Test: Run examples

**Step 1: Update imports**

At top of `examples/merging_examples.py`:

```python
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.scenarios import Scenario, timeout
from jax_frc.scenarios.phases.merging import MergingPhase
from jax_frc.scenarios.transitions import separation_below, any_of
from jax_frc.models.base import PhysicsModel
from jax_frc.solvers.base import Solver
import jax.numpy as jnp
```

**Step 2: Update belova_case1**

```python
def belova_case1(model_type: str = "resistive_mhd") -> Scenario:
    """Large FRC merging without compression (paper Fig. 1-2).

    Parameters:
        S* = 25.6, E = 2.9, xs = 0.69, beta_s = 0.2
        Initial separation: dZ = 180 (normalized)
        Initial velocity: Vz = 0.2 vA

    Expected outcome: Partial merge, doublet configuration

    Args:
        model_type: "resistive_mhd" or "hybrid_kinetic"
    """
    geometry = create_default_geometry(rc=1.0, zc=5.0, nr=64, nz=512)
    initial_state = create_initial_frc(
        geometry,
        s_star=25.6,
        elongation=2.9,
        xs=0.69,
        beta_s=0.2
    )

    model_config = {
        "type": model_type,
        "resistivity": {"type": "chodura", "eta_0": 1e-6, "eta_anom": 1e-3}
    }
    physics_model = PhysicsModel.create(model_config)
    solver = Solver.create({"type": "rk4"})

    merge_phase = MergingPhase(
        name="merge_no_compression",
        transition=any_of(
            separation_below(0.5, geometry),
            timeout(30.0)
        ),
        separation=3.0,
        initial_velocity=0.2,
        compression=None,
    )

    return Scenario(
        name="belova_case1_large_frc",
        phases=[merge_phase],
        geometry=geometry,
        initial_state=initial_state,
        physics_model=physics_model,
        solver=solver,
        dt=0.001,
    )
```

**Step 3: Update belova_case2**

```python
def belova_case2(model_type: str = "resistive_mhd") -> Scenario:
    """Small FRC merging without compression (paper Fig. 3-4).

    Parameters:
        S* = 20, E = 1.5, xs = 0.53, beta_s = 0.2
        Initial separation: dZ = 75 (normalized)
        Initial velocity: Vz = 0.1 vA

    Expected outcome: Complete merge by ~5-7 tA

    Args:
        model_type: "resistive_mhd" or "hybrid_kinetic"
    """
    geometry = create_default_geometry(rc=1.0, zc=3.0, nr=64, nz=256)
    initial_state = create_initial_frc(
        geometry,
        s_star=20.0,
        elongation=1.5,
        xs=0.53,
        beta_s=0.2
    )

    model_config = {
        "type": model_type,
        "resistivity": {"type": "chodura", "eta_0": 1e-6, "eta_anom": 1e-3}
    }
    physics_model = PhysicsModel.create(model_config)
    solver = Solver.create({"type": "rk4"})

    merge_phase = MergingPhase(
        name="merge_small_frc",
        transition=any_of(
            separation_below(0.3, geometry),
            timeout(15.0)
        ),
        separation=1.5,
        initial_velocity=0.1,
        compression=None,
    )

    return Scenario(
        name="belova_case2_small_frc",
        phases=[merge_phase],
        geometry=geometry,
        initial_state=initial_state,
        physics_model=physics_model,
        solver=solver,
        dt=0.001,
    )
```

**Step 4: Add belova_case3**

```python
def belova_case3(separation: float = 1.5, model_type: str = "resistive_mhd") -> Scenario:
    """Small FRC with variable separation (paper Section 2.3).

    Tests sensitivity of merging to initial separation.

    Args:
        separation: Initial separation in normalized units
                    1.5 ~ dZ=75, 2.2 ~ dZ=110, 2.5 ~ dZ=125, 3.7 ~ dZ=185
        model_type: "resistive_mhd" or "hybrid_kinetic"
    """
    zc = max(3.0, separation * 1.5)
    geometry = create_default_geometry(rc=1.0, zc=zc, nr=64, nz=256)
    initial_state = create_initial_frc(
        geometry,
        s_star=20.0,
        elongation=1.5,
        xs=0.53,
        beta_s=0.2
    )

    model_config = {
        "type": model_type,
        "resistivity": {"type": "chodura", "eta_0": 1e-6, "eta_anom": 1e-3}
    }
    physics_model = PhysicsModel.create(model_config)
    solver = Solver.create({"type": "rk4"})

    max_time = 50.0 if separation > 2.5 else 25.0

    merge_phase = MergingPhase(
        name="merge_separation_test",
        transition=any_of(
            separation_below(0.3, geometry),
            timeout(max_time)
        ),
        separation=separation,
        initial_velocity=0.1,
        compression=None,
    )

    return Scenario(
        name=f"belova_case3_sep{separation}",
        phases=[merge_phase],
        geometry=geometry,
        initial_state=initial_state,
        physics_model=physics_model,
        solver=solver,
        dt=0.001,
    )
```

**Step 5: Update belova_case4**

```python
def belova_case4(model_type: str = "resistive_mhd") -> Scenario:
    """Large FRC with compression (paper Fig. 6-7).

    Parameters: Same as case1 but with compression
        Mirror ratio: 1.5
        Ramp time: 19 tA

    Expected outcome: Complete merge by ~20-25 tA

    Args:
        model_type: "resistive_mhd" or "hybrid_kinetic"
    """
    geometry = create_default_geometry(rc=1.0, zc=5.0, nr=64, nz=512)
    initial_state = create_initial_frc(
        geometry,
        s_star=25.6,
        elongation=2.9,
        xs=0.69,
        beta_s=0.2
    )

    model_config = {
        "type": model_type,
        "resistivity": {"type": "chodura", "eta_0": 1e-6, "eta_anom": 1e-3}
    }
    physics_model = PhysicsModel.create(model_config)
    solver = Solver.create({"type": "rk4"})

    merge_phase = MergingPhase(
        name="merge_with_compression",
        transition=any_of(
            separation_below(0.3, geometry),
            timeout(40.0)
        ),
        separation=3.0,
        initial_velocity=0.0,
        compression={
            "base_field": 1.0,
            "mirror_ratio": 1.5,
            "ramp_time": 19.0,
            "profile": "cosine",
        },
    )

    return Scenario(
        name="belova_case4_compression",
        phases=[merge_phase],
        geometry=geometry,
        initial_state=initial_state,
        physics_model=physics_model,
        solver=solver,
        dt=0.001,
    )
```

**Step 6: Run existing tests**

Run: `py -m pytest tests/test_merging_integration.py -v`
Expected: PASS

**Step 7: Commit**

```bash
git add examples/merging_examples.py
git commit -m "feat(examples): add physics_model/solver to Belova cases, add case3"
```

---

## Task 8: Create Belova validation test file

**Files:**
- Create: `tests/test_belova_validation.py`

**Step 1: Create test file with imports and Case 2 tests**

```python
# tests/test_belova_validation.py
"""Quantitative validation tests against Belova et al. (arXiv:2501.03425v1)."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
from merging_examples import (
    belova_case1, belova_case2, belova_case3, belova_case4
)

from jax_frc.diagnostics.merging import MergingDiagnostics


class TestBelovaCase2:
    """Small FRC merging - expect complete merge by ~5-7 tA."""

    @pytest.mark.slow
    def test_complete_merge_mhd(self):
        """MHD: separation reaches near-zero."""
        scenario = belova_case2(model_type="resistive_mhd")
        result = scenario.run()

        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        assert final_diag["separation_dz"] < 0.5
        assert result.total_time <= 15.0

    @pytest.mark.slow
    def test_elongation_increase(self):
        """Elongation should increase after merge."""
        scenario = belova_case2()

        # Setup phase to get two-FRC initial state
        phase = scenario.phases[0]
        two_frc_state = phase.setup(
            scenario.initial_state, scenario.geometry, {}
        )
        initial_diag = MergingDiagnostics().compute(
            two_frc_state, scenario.geometry
        )

        result = scenario.run()
        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        ratio = final_diag["elongation"] / (initial_diag["elongation"] + 1e-10)
        assert ratio > 1.2  # Should increase
```

**Step 2: Run tests**

Run: `py -m pytest tests/test_belova_validation.py -v -k "Case2" --tb=short`
Expected: Tests run (may be slow)

**Step 3: Commit**

```bash
git add tests/test_belova_validation.py
git commit -m "test: add Belova Case 2 validation tests"
```

---

## Task 9: Add remaining validation test classes

**Files:**
- Modify: `tests/test_belova_validation.py`

**Step 1: Add Case 1 tests**

Append to `tests/test_belova_validation.py`:

```python
class TestBelovaCase1:
    """Large FRC - expect partial merge, doublet formation."""

    @pytest.mark.slow
    def test_doublet_formation_mhd(self):
        """MHD: should NOT fully merge."""
        scenario = belova_case1(model_type="resistive_mhd")
        result = scenario.run()

        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        # Should remain as doublet (two nulls)
        assert final_diag["separation_dz"] > 0.3
        assert len(final_diag["null_positions"]) == 2
```

**Step 2: Add Case 3 tests**

```python
class TestBelovaCase3:
    """Small FRC with varying initial separation."""

    @pytest.mark.slow
    @pytest.mark.parametrize("separation,should_merge", [
        (1.5, True),   # dZ=75 equivalent
        (2.2, True),   # dZ=110 equivalent
    ])
    def test_merge_vs_separation(self, separation, should_merge):
        """Merging depends on initial separation."""
        scenario = belova_case3(separation=separation)
        result = scenario.run()

        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        if should_merge:
            assert final_diag["separation_dz"] < 1.0

    @pytest.mark.slow
    def test_large_separation_no_merge(self):
        """Very large separation prevents merging."""
        scenario = belova_case3(separation=3.7)
        result = scenario.run()

        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        assert final_diag["separation_dz"] > 2.0
```

**Step 3: Add Case 4 tests**

```python
class TestBelovaCase4:
    """Large FRC with compression - expect complete merge."""

    @pytest.mark.slow
    def test_compression_enables_merge(self):
        """Compression should enable complete merge."""
        scenario = belova_case4(model_type="resistive_mhd")
        result = scenario.run()

        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        assert final_diag["separation_dz"] < 0.5
```

**Step 4: Run all validation tests**

Run: `py -m pytest tests/test_belova_validation.py -v --tb=short`

**Step 5: Commit**

```bash
git add tests/test_belova_validation.py
git commit -m "test: add Belova Cases 1, 3, 4 validation tests"
```

---

## Task 10: Run full test suite and verify

**Files:**
- None (verification only)

**Step 1: Run all tests**

Run: `py -m pytest tests/ -v --tb=short`
Expected: All PASS (validation tests may be slow)

**Step 2: Run quick smoke test**

Run: `py -m pytest tests/ -v --ignore=tests/test_belova_validation.py`
Expected: All PASS quickly

**Step 3: Final commit**

```bash
git add -A
git status
# If any uncommitted changes:
git commit -m "chore: final cleanup for physics integration"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Add physics_model/solver to Scenario | scenario.py |
| 2 | Update _run_phase to call solver | scenario.py |
| 3 | Fix existing tests | test_*.py, conftest.py |
| 4 | Add history to PhaseResult | phase.py |
| 5 | Add diagnostics/output_interval | scenario.py |
| 6 | Record diagnostics in _run_phase | scenario.py |
| 7 | Update merging examples | merging_examples.py |
| 8 | Create Case 2 validation tests | test_belova_validation.py |
| 9 | Add Cases 1, 3, 4 tests | test_belova_validation.py |
| 10 | Verify full test suite | - |
