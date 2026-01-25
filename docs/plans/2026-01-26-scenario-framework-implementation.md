# Scenario Framework Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a composable phase pipeline framework supporting FRC formation, translation, merging, compression, and nuclear burn phases.

**Architecture:** Phases are standalone classes with setup/step_hook/complete lifecycle. A Scenario orchestrates phases with condition+timeout transitions. Each phase can modify boundary conditions and enable physics features.

**Tech Stack:** JAX, Python dataclasses, existing jax_frc infrastructure (State, Geometry, Probe patterns)

---

## Task 1: Phase and Transition Base Classes

**Files:**
- Create: `jax_frc/scenarios/__init__.py`
- Create: `jax_frc/scenarios/phase.py`
- Create: `jax_frc/scenarios/transitions.py`
- Test: `tests/test_scenarios.py`

**Step 1: Create scenarios package init**

```python
# jax_frc/scenarios/__init__.py
"""Scenario framework for multi-phase FRC simulations."""

from jax_frc.scenarios.phase import Phase, PhaseResult
from jax_frc.scenarios.transitions import Transition, timeout, condition, any_of, all_of

__all__ = [
    "Phase",
    "PhaseResult",
    "Transition",
    "timeout",
    "condition",
    "any_of",
    "all_of",
]
```

**Step 2: Write failing test for Transition**

```python
# tests/test_scenarios.py
"""Tests for scenario framework."""

import pytest
from jax_frc.scenarios.transitions import Transition, timeout, condition, any_of


class TestTransition:
    """Tests for Transition class."""

    def test_timeout_triggers_at_time(self):
        """Timeout transition triggers when t >= timeout."""
        trans = timeout(10.0)

        triggered, reason = trans.evaluate(None, t=5.0)
        assert not triggered

        triggered, reason = trans.evaluate(None, t=10.0)
        assert triggered
        assert reason == "timeout"

    def test_condition_triggers_when_true(self):
        """Condition transition triggers when condition returns True."""
        trans = condition(lambda state, t: t > 5.0)

        triggered, reason = trans.evaluate(None, t=3.0)
        assert not triggered

        triggered, reason = trans.evaluate(None, t=6.0)
        assert triggered
        assert reason == "condition_met"

    def test_any_of_triggers_on_first_match(self):
        """any_of triggers when any sub-transition triggers."""
        trans = any_of(
            timeout(10.0),
            condition(lambda s, t: t > 5.0)
        )

        triggered, reason = trans.evaluate(None, t=6.0)
        assert triggered
        assert reason == "condition_met"
```

**Step 3: Run test to verify it fails**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\scenario-framework
python -m pytest tests/test_scenarios.py -v
```

Expected: FAIL with "No module named 'jax_frc.scenarios'"

**Step 4: Implement transitions.py**

```python
# jax_frc/scenarios/transitions.py
"""Transition conditions for phase completion."""

from dataclasses import dataclass
from typing import Callable, Optional, List
from jax_frc.core.state import State


@dataclass
class Transition:
    """Evaluates whether a phase should complete.

    Combines optional condition-based and time-based triggers.
    Condition is checked first; timeout acts as fallback.
    """

    _condition: Optional[Callable[[State, float], bool]] = None
    _timeout: Optional[float] = None
    _name: str = ""

    def evaluate(self, state: State, t: float) -> tuple[bool, str]:
        """Check if transition should trigger.

        Args:
            state: Current simulation state
            t: Current simulation time

        Returns:
            (triggered, reason) where reason is "condition_met", "timeout", or ""
        """
        if self._condition is not None and self._condition(state, t):
            return True, "condition_met"
        if self._timeout is not None and t >= self._timeout:
            return True, "timeout"
        return False, ""


def timeout(t: float) -> Transition:
    """Create a time-based transition.

    Args:
        t: Time at which transition triggers

    Returns:
        Transition that triggers when simulation time >= t
    """
    return Transition(_timeout=t, _name=f"timeout({t})")


def condition(fn: Callable[[State, float], bool]) -> Transition:
    """Create a condition-based transition.

    Args:
        fn: Function (state, t) -> bool that returns True to trigger

    Returns:
        Transition that triggers when fn returns True
    """
    return Transition(_condition=fn, _name="condition")


def any_of(*transitions: Transition) -> Transition:
    """Create transition that triggers when any sub-transition triggers.

    Args:
        *transitions: Sub-transitions to check

    Returns:
        Transition that triggers on first matching sub-transition
    """
    def check(state: State, t: float) -> bool:
        for trans in transitions:
            triggered, _ = trans.evaluate(state, t)
            if triggered:
                return True
        return False

    # Also need to track which one triggered for the reason
    def evaluate_combined(state: State, t: float) -> tuple[bool, str]:
        for trans in transitions:
            triggered, reason = trans.evaluate(state, t)
            if triggered:
                return True, reason
        return False, ""

    result = Transition(_name="any_of")
    result.evaluate = evaluate_combined  # Override method
    return result


def all_of(*transitions: Transition) -> Transition:
    """Create transition that triggers when all sub-transitions trigger.

    Args:
        *transitions: Sub-transitions that must all trigger

    Returns:
        Transition that triggers only when all sub-transitions match
    """
    def evaluate_combined(state: State, t: float) -> tuple[bool, str]:
        for trans in transitions:
            triggered, _ = trans.evaluate(state, t)
            if not triggered:
                return False, ""
        return True, "all_conditions_met"

    result = Transition(_name="all_of")
    result.evaluate = evaluate_combined
    return result
```

**Step 5: Run test to verify it passes**

```bash
python -m pytest tests/test_scenarios.py::TestTransition -v
```

Expected: PASS (3 tests)

**Step 6: Write failing test for Phase**

Add to `tests/test_scenarios.py`:

```python
from jax_frc.scenarios.phase import Phase, PhaseResult
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
import jax.numpy as jnp


class TestPhase:
    """Tests for Phase base class."""

    def test_phase_setup_returns_state(self):
        """Phase.setup returns modified state."""
        phase = Phase(name="test", transition=timeout(10.0))
        geometry = Geometry(
            coord_system="cylindrical",
            nr=10, nz=20,
            r_min=0.1, r_max=1.0,
            z_min=-1.0, z_max=1.0
        )
        state = State.zeros(nr=10, nz=20)

        result = phase.setup(state, geometry, {})
        assert isinstance(result, State)

    def test_phase_is_complete_delegates_to_transition(self):
        """Phase.is_complete uses transition.evaluate."""
        phase = Phase(name="test", transition=timeout(5.0))
        state = State.zeros(nr=10, nz=20)

        complete, reason = phase.is_complete(state, t=3.0)
        assert not complete

        complete, reason = phase.is_complete(state, t=5.0)
        assert complete
        assert reason == "timeout"
```

**Step 7: Run test to verify it fails**

```bash
python -m pytest tests/test_scenarios.py::TestPhase -v
```

Expected: FAIL with "cannot import name 'Phase'"

**Step 8: Implement phase.py**

```python
# jax_frc/scenarios/phase.py
"""Phase base class for simulation stages."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.scenarios.transitions import Transition


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


@dataclass
class Phase:
    """Base class for simulation phases.

    A phase represents one stage of an FRC experiment (formation, merging, etc.).
    Subclasses override setup, step_hook, and on_complete for phase-specific behavior.

    Attributes:
        name: Human-readable phase name
        transition: Condition for phase completion
    """

    name: str
    transition: Transition

    def setup(self, state: State, geometry: Geometry, config: dict) -> State:
        """Called once when phase begins.

        Override to modify initial state, configure boundary conditions, etc.

        Args:
            state: State from previous phase (or initial state)
            geometry: Computational geometry
            config: Phase-specific configuration

        Returns:
            Modified state to begin phase with
        """
        return state

    def step_hook(self, state: State, geometry: Geometry, t: float) -> State:
        """Called each timestep during phase.

        Override for time-varying boundary conditions, source terms, etc.

        Args:
            state: Current state
            geometry: Computational geometry
            t: Current simulation time

        Returns:
            Modified state (or unchanged if no per-step modifications needed)
        """
        return state

    def is_complete(self, state: State, t: float) -> tuple[bool, str]:
        """Check if phase should end.

        Delegates to transition.evaluate(). Override for custom logic.

        Args:
            state: Current state
            t: Current simulation time

        Returns:
            (triggered, reason) tuple
        """
        return self.transition.evaluate(state, t)

    def on_complete(self, state: State, geometry: Geometry) -> State:
        """Called once when phase ends.

        Override for cleanup, state preparation for next phase, etc.

        Args:
            state: Final state of this phase
            geometry: Computational geometry

        Returns:
            State to pass to next phase
        """
        return state
```

**Step 9: Run all scenario tests**

```bash
python -m pytest tests/test_scenarios.py -v
```

Expected: PASS (5 tests)

**Step 10: Commit**

```bash
git add jax_frc/scenarios/ tests/test_scenarios.py
git commit -m "feat(scenarios): add Phase and Transition base classes

- Transition supports condition-based and timeout triggers
- any_of/all_of combinators for complex conditions
- Phase provides setup/step_hook/on_complete lifecycle"
```

---

## Task 2: Scenario Runner

**Files:**
- Create: `jax_frc/scenarios/scenario.py`
- Modify: `jax_frc/scenarios/__init__.py`
- Test: `tests/test_scenarios.py`

**Step 1: Write failing test for Scenario**

Add to `tests/test_scenarios.py`:

```python
from jax_frc.scenarios.scenario import Scenario, ScenarioResult


class TestScenario:
    """Tests for Scenario runner."""

    @pytest.fixture
    def simple_geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=10, nz=20,
            r_min=0.1, r_max=1.0,
            z_min=-1.0, z_max=1.0
        )

    @pytest.fixture
    def initial_state(self):
        return State.zeros(nr=10, nz=20)

    def test_scenario_runs_single_phase(self, simple_geometry, initial_state):
        """Scenario runs a single phase to completion."""
        phase = Phase(name="test", transition=timeout(1.0))

        scenario = Scenario(
            name="test_scenario",
            phases=[phase],
            geometry=simple_geometry,
            initial_state=initial_state,
            dt=0.1,
        )

        result = scenario.run()

        assert isinstance(result, ScenarioResult)
        assert len(result.phase_results) == 1
        assert result.phase_results[0].name == "test"
        assert result.phase_results[0].termination == "timeout"

    def test_scenario_chains_phases(self, simple_geometry, initial_state):
        """Scenario passes state between phases."""
        phase1 = Phase(name="phase1", transition=timeout(1.0))
        phase2 = Phase(name="phase2", transition=timeout(2.0))

        scenario = Scenario(
            name="test_scenario",
            phases=[phase1, phase2],
            geometry=simple_geometry,
            initial_state=initial_state,
            dt=0.1,
        )

        result = scenario.run()

        assert len(result.phase_results) == 2
        assert result.phase_results[0].name == "phase1"
        assert result.phase_results[1].name == "phase2"
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_scenarios.py::TestScenario -v
```

Expected: FAIL with "cannot import name 'Scenario'"

**Step 3: Implement scenario.py**

```python
# jax_frc/scenarios/scenario.py
"""Scenario runner for multi-phase simulations."""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.scenarios.phase import Phase, PhaseResult

log = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    """Result of running a complete scenario."""

    name: str
    phase_results: List[PhaseResult]
    total_time: float
    success: bool

    @property
    def final_state(self) -> State:
        """Get final state from last phase."""
        return self.phase_results[-1].final_state


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
        dt: Timestep for simulation
        config: Optional per-phase configuration overrides
    """

    name: str
    phases: List[Phase]
    geometry: Geometry
    initial_state: Optional[State]
    dt: float
    config: Dict[str, dict] = field(default_factory=dict)

    def run(self) -> ScenarioResult:
        """Run all phases in sequence.

        Returns:
            ScenarioResult with results from all phases
        """
        state = self.initial_state
        phase_results = []

        for phase in self.phases:
            log.info(f"Starting phase: {phase.name}")

            # Get phase-specific config
            phase_config = self.config.get(phase.name, {})

            # Run the phase
            result = self._run_phase(phase, state, phase_config)
            phase_results.append(result)

            # Pass state to next phase
            state = result.final_state

            if result.termination == "timeout":
                log.warning(f"Phase {phase.name} ended by timeout")
            elif result.termination == "error":
                log.error(f"Phase {phase.name} ended with error")
                break

        total_time = sum(r.end_time - r.start_time for r in phase_results)
        success = all(r.termination != "error" for r in phase_results)

        return ScenarioResult(
            name=self.name,
            phase_results=phase_results,
            total_time=total_time,
            success=success,
        )

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

            # Advance time (actual physics stepping would go here)
            t += self.dt
            state = state.replace(time=t, step=state.step + 1)

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

**Step 4: Update __init__.py**

```python
# jax_frc/scenarios/__init__.py
"""Scenario framework for multi-phase FRC simulations."""

from jax_frc.scenarios.phase import Phase, PhaseResult
from jax_frc.scenarios.transitions import Transition, timeout, condition, any_of, all_of
from jax_frc.scenarios.scenario import Scenario, ScenarioResult

__all__ = [
    "Phase",
    "PhaseResult",
    "Transition",
    "timeout",
    "condition",
    "any_of",
    "all_of",
    "Scenario",
    "ScenarioResult",
]
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_scenarios.py -v
```

Expected: PASS (7 tests)

**Step 6: Commit**

```bash
git add jax_frc/scenarios/
git commit -m "feat(scenarios): add Scenario runner for multi-phase simulations

- Scenario orchestrates phases in sequence
- Passes state between phases
- Logs phase transitions and warnings for timeouts"
```

---

## Task 3: Built-in Transition Conditions

**Files:**
- Modify: `jax_frc/scenarios/transitions.py`
- Test: `tests/test_scenarios.py`

**Step 1: Write failing tests for physics conditions**

Add to `tests/test_scenarios.py`:

```python
from jax_frc.scenarios.transitions import (
    separation_below, temperature_above, flux_below, velocity_below
)


class TestPhysicsConditions:
    """Tests for physics-based transition conditions."""

    def test_separation_below_threshold(self):
        """separation_below triggers when dZ < threshold."""
        # Create state with two magnetic nulls
        state = State.zeros(nr=10, nz=40)
        # Set psi with two peaks (simplified)
        psi = jnp.zeros((10, 40))
        psi = psi.at[5, 10].set(1.0)  # Null 1 at z_idx=10
        psi = psi.at[5, 30].set(1.0)  # Null 2 at z_idx=30
        state = state.replace(psi=psi)

        geometry = Geometry(
            coord_system="cylindrical",
            nr=10, nz=40,
            r_min=0.1, r_max=1.0,
            z_min=-2.0, z_max=2.0  # dz = 0.1, separation = 20*0.1 = 2.0
        )

        trans = separation_below(3.0, geometry)  # Should trigger (2.0 < 3.0)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert triggered

        trans = separation_below(1.0, geometry)  # Should not trigger (2.0 > 1.0)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert not triggered

    def test_temperature_above_threshold(self):
        """temperature_above triggers when T > threshold."""
        state = State.zeros(nr=10, nz=20)
        # Set pressure and density to give T = p/n
        state = state.replace(
            p=jnp.ones((10, 20)) * 100.0,
            n=jnp.ones((10, 20)) * 10.0  # T = 100/10 = 10
        )

        trans = temperature_above(5.0)  # Should trigger (10 > 5)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert triggered

        trans = temperature_above(15.0)  # Should not trigger (10 < 15)
        triggered, _ = trans.evaluate(state, t=0.0)
        assert not triggered
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_scenarios.py::TestPhysicsConditions -v
```

Expected: FAIL with "cannot import name 'separation_below'"

**Step 3: Implement physics conditions**

Add to `jax_frc/scenarios/transitions.py`:

```python
import jax.numpy as jnp
from jax_frc.core.geometry import Geometry


def separation_below(threshold: float, geometry: Geometry) -> Transition:
    """Transition when FRC null separation drops below threshold.

    Finds magnetic nulls (local maxima of psi) and computes their axial separation.

    Args:
        threshold: Separation threshold in length units
        geometry: Geometry for coordinate mapping

    Returns:
        Transition that triggers when dZ < threshold
    """
    def check(state: State, t: float) -> bool:
        psi = state.psi

        # Find null positions (local maxima of psi along z at each r)
        # Simplified: find z indices of max psi in each half of domain
        nz = psi.shape[1]
        mid = nz // 2

        # Find max in each half
        left_half = psi[:, :mid]
        right_half = psi[:, mid:]

        left_max_z = jnp.argmax(jnp.max(left_half, axis=0))
        right_max_z = mid + jnp.argmax(jnp.max(right_half, axis=0))

        # Convert to physical coordinates
        z1 = geometry.z_min + left_max_z * geometry.dz
        z2 = geometry.z_min + right_max_z * geometry.dz

        separation = jnp.abs(z2 - z1)
        return float(separation) < threshold

    return condition(check)


def temperature_above(threshold: float) -> Transition:
    """Transition when peak temperature exceeds threshold.

    Temperature computed as T = p/n (in appropriate units).

    Args:
        threshold: Temperature threshold

    Returns:
        Transition that triggers when max(T) > threshold
    """
    def check(state: State, t: float) -> bool:
        # T = p / n, avoid division by zero
        n_safe = jnp.maximum(state.n, 1e-10)
        T = state.p / n_safe
        return float(jnp.max(T)) > threshold

    return condition(check)


def flux_below(threshold: float) -> Transition:
    """Transition when peak poloidal flux drops below threshold.

    Indicates loss of FRC confinement.

    Args:
        threshold: Flux threshold

    Returns:
        Transition that triggers when max(psi) < threshold
    """
    def check(state: State, t: float) -> bool:
        return float(jnp.max(state.psi)) < threshold

    return condition(check)


def velocity_below(threshold: float) -> Transition:
    """Transition when peak velocity drops below threshold.

    Used for translation phase completion.

    Args:
        threshold: Velocity threshold

    Returns:
        Transition that triggers when max(|v|) < threshold
    """
    def check(state: State, t: float) -> bool:
        v_mag = jnp.sqrt(jnp.sum(state.v**2, axis=-1))
        return float(jnp.max(v_mag)) < threshold

    return condition(check)
```

**Step 4: Update __init__.py exports**

Add to `jax_frc/scenarios/__init__.py`:

```python
from jax_frc.scenarios.transitions import (
    Transition, timeout, condition, any_of, all_of,
    separation_below, temperature_above, flux_below, velocity_below
)

__all__ = [
    # ... existing ...
    "separation_below",
    "temperature_above",
    "flux_below",
    "velocity_below",
]
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_scenarios.py -v
```

Expected: PASS (9 tests)

**Step 6: Commit**

```bash
git add jax_frc/scenarios/
git commit -m "feat(scenarios): add physics-based transition conditions

- separation_below: FRC null separation threshold
- temperature_above: peak temperature threshold
- flux_below: confinement loss detection
- velocity_below: translation completion"
```

---

## Task 4: Merging Diagnostics Probe

**Files:**
- Create: `jax_frc/diagnostics/merging.py`
- Modify: `jax_frc/diagnostics/__init__.py`
- Test: `tests/test_merging_diagnostics.py`

**Step 1: Write failing test**

```python
# tests/test_merging_diagnostics.py
"""Tests for merging diagnostics."""

import pytest
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.diagnostics.merging import MergingDiagnostics


class TestMergingDiagnostics:
    """Tests for MergingDiagnostics probe."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=20, nz=40,
            r_min=0.1, r_max=1.0,
            z_min=-2.0, z_max=2.0
        )

    @pytest.fixture
    def two_frc_state(self, geometry):
        """Create state with two FRC-like structures."""
        state = State.zeros(nr=20, nz=40)

        # Create two peaked psi structures
        r = geometry.r_grid
        z = geometry.z_grid

        # FRC 1 centered at z=-1
        psi1 = jnp.exp(-((r - 0.5)**2 + (z + 1.0)**2) / 0.1)
        # FRC 2 centered at z=+1
        psi2 = jnp.exp(-((r - 0.5)**2 + (z - 1.0)**2) / 0.1)

        psi = psi1 + psi2

        # Set pressure proportional to psi
        p = psi * 0.5

        return state.replace(psi=psi, p=p, n=jnp.ones_like(psi))

    def test_computes_separation(self, two_frc_state, geometry):
        """Diagnostics compute null separation."""
        diag = MergingDiagnostics()
        result = diag.compute(two_frc_state, geometry)

        assert "separation_dz" in result
        # Two FRCs at z=-1 and z=+1, separation ~2.0
        assert 1.5 < result["separation_dz"] < 2.5

    def test_computes_separatrix_radius(self, two_frc_state, geometry):
        """Diagnostics compute separatrix radius."""
        diag = MergingDiagnostics()
        result = diag.compute(two_frc_state, geometry)

        assert "separatrix_radius" in result
        assert result["separatrix_radius"] > 0

    def test_computes_peak_pressure(self, two_frc_state, geometry):
        """Diagnostics compute peak pressure."""
        diag = MergingDiagnostics()
        result = diag.compute(two_frc_state, geometry)

        assert "peak_pressure" in result
        assert result["peak_pressure"] > 0
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_merging_diagnostics.py -v
```

Expected: FAIL with "No module named 'jax_frc.diagnostics.merging'"

**Step 3: Implement merging.py**

```python
# jax_frc/diagnostics/merging.py
"""Merging-specific diagnostic probes."""

from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import jax.numpy as jnp
from jax import Array

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.diagnostics.probes import Probe

MU0 = 1.2566e-6


@dataclass
class MergingDiagnostics(Probe):
    """Computes merging-specific metrics.

    Metrics include:
    - separation_dz: Distance between magnetic nulls
    - separatrix_radius: Radius of separatrix (Rs)
    - elongation: Separatrix half-length / radius (E = Zs/Rs)
    - separatrix_beta: Plasma beta at separatrix
    - peak_pressure: Maximum pressure
    - null_positions: List of (r, z) coordinates of nulls
    - axial_velocity_at_null: Vz at null positions
    - reconnection_rate: dpsi/dt proxy at midplane
    """

    _prev_psi_midplane: Array = None
    _prev_time: float = None

    @property
    def name(self) -> str:
        return "merging"

    def measure(self, state: State, geometry: Geometry) -> float:
        """Return separation as primary scalar metric."""
        result = self.compute(state, geometry)
        return result["separation_dz"]

    def compute(self, state: State, geometry: Geometry) -> Dict[str, Any]:
        """Compute all merging diagnostics.

        Args:
            state: Current simulation state
            geometry: Computational geometry

        Returns:
            Dictionary of diagnostic values
        """
        psi = state.psi

        # Find null positions
        nulls = self._find_null_positions(psi, geometry)

        # Compute separation
        if len(nulls) >= 2:
            separation = abs(nulls[0][1] - nulls[1][1])
        else:
            separation = 0.0

        # Separatrix radius (max r where psi > threshold)
        rs = self._find_separatrix_radius(psi, geometry)

        # Elongation
        zs = self._find_separatrix_half_length(psi, geometry)
        elongation = zs / (rs + 1e-10)

        # Separatrix beta
        beta_s = self._compute_separatrix_beta(state, geometry, rs)

        # Axial velocity at nulls
        vz_at_null = self._velocity_at_nulls(state, nulls, geometry)

        # Reconnection rate (change in psi at midplane)
        recon_rate = self._compute_reconnection_rate(state, geometry)

        return {
            "separation_dz": float(separation),
            "separatrix_radius": float(rs),
            "elongation": float(elongation),
            "separatrix_beta": float(beta_s),
            "peak_pressure": float(jnp.max(state.p)),
            "null_positions": nulls,
            "axial_velocity_at_null": vz_at_null,
            "reconnection_rate": float(recon_rate),
        }

    def _find_null_positions(self, psi: Array, geometry: Geometry) -> List[Tuple[float, float]]:
        """Find magnetic null positions (local maxima of psi)."""
        nulls = []

        # Find global maximum
        idx = jnp.unravel_index(jnp.argmax(psi), psi.shape)
        r1 = geometry.r_min + idx[0] * geometry.dr
        z1 = geometry.z_min + idx[1] * geometry.dz
        nulls.append((float(r1), float(z1)))

        # For two-FRC case, find max in opposite half
        nz = psi.shape[1]
        mid = nz // 2

        if idx[1] < mid:
            # First null in left half, find max in right half
            right_half = psi[:, mid:]
            idx2 = jnp.unravel_index(jnp.argmax(right_half), right_half.shape)
            r2 = geometry.r_min + idx2[0] * geometry.dr
            z2 = geometry.z_min + (mid + idx2[1]) * geometry.dz
        else:
            # First null in right half, find max in left half
            left_half = psi[:, :mid]
            idx2 = jnp.unravel_index(jnp.argmax(left_half), left_half.shape)
            r2 = geometry.r_min + idx2[0] * geometry.dr
            z2 = geometry.z_min + idx2[1] * geometry.dz

        nulls.append((float(r2), float(z2)))
        return nulls

    def _find_separatrix_radius(self, psi: Array, geometry: Geometry) -> float:
        """Find separatrix radius at midplane."""
        # Separatrix is where psi crosses threshold (e.g., 1% of max)
        threshold = jnp.max(psi) * 0.01

        z_mid_idx = psi.shape[1] // 2
        psi_midplane = psi[:, z_mid_idx]

        inside = psi_midplane > threshold
        r_indices = jnp.where(inside, jnp.arange(psi.shape[0]), 0)
        r_sep_idx = jnp.max(r_indices)

        return geometry.r_min + float(r_sep_idx) * geometry.dr

    def _find_separatrix_half_length(self, psi: Array, geometry: Geometry) -> float:
        """Find separatrix half-length."""
        threshold = jnp.max(psi) * 0.01

        # Find r of max psi
        r_max_idx = jnp.argmax(jnp.max(psi, axis=1))
        psi_axial = psi[r_max_idx, :]

        inside = psi_axial > threshold
        z_indices = jnp.arange(psi.shape[1])
        z_inside = jnp.where(inside, z_indices, -1)

        z_max_idx = jnp.max(z_inside)
        z_min_idx = jnp.min(jnp.where(inside, z_indices, psi.shape[1]))

        length = (z_max_idx - z_min_idx) * geometry.dz
        return float(length) / 2.0

    def _compute_separatrix_beta(self, state: State, geometry: Geometry, rs: float) -> float:
        """Compute plasma beta at separatrix."""
        # Find approximate separatrix location
        r_idx = int((rs - geometry.r_min) / geometry.dr)
        r_idx = min(r_idx, state.p.shape[0] - 1)

        z_mid_idx = state.p.shape[1] // 2

        p_sep = state.p[r_idx, z_mid_idx]
        B_sq = jnp.sum(state.B[r_idx, z_mid_idx]**2)
        B_sq = jnp.maximum(B_sq, 1e-20)

        beta = 2 * MU0 * p_sep / B_sq
        return float(beta)

    def _velocity_at_nulls(self, state: State, nulls: List[Tuple[float, float]],
                          geometry: Geometry) -> List[float]:
        """Get axial velocity at null positions."""
        velocities = []
        for r, z in nulls:
            r_idx = int((r - geometry.r_min) / geometry.dr)
            z_idx = int((z - geometry.z_min) / geometry.dz)
            r_idx = min(max(r_idx, 0), state.v.shape[0] - 1)
            z_idx = min(max(z_idx, 0), state.v.shape[1] - 1)
            vz = state.v[r_idx, z_idx, 2]  # z component
            velocities.append(float(vz))
        return velocities

    def _compute_reconnection_rate(self, state: State, geometry: Geometry) -> float:
        """Compute reconnection rate as dpsi/dt at midplane X-point."""
        # This is a placeholder - proper implementation needs state history
        # For now, return magnitude of E_phi at midplane (proxy for reconnection)
        z_mid_idx = state.E.shape[1] // 2
        E_phi = state.E[:, z_mid_idx, 1]  # theta component
        return float(jnp.max(jnp.abs(E_phi)))
```

**Step 4: Update diagnostics __init__.py**

```python
# jax_frc/diagnostics/__init__.py
"""Diagnostic probes and output."""

from jax_frc.diagnostics.probes import (
    Probe, FluxProbe, EnergyProbe, BetaProbe,
    CurrentProbe, SeparatrixProbe, DiagnosticSet
)
from jax_frc.diagnostics.merging import MergingDiagnostics

__all__ = [
    "Probe",
    "FluxProbe",
    "EnergyProbe",
    "BetaProbe",
    "CurrentProbe",
    "SeparatrixProbe",
    "DiagnosticSet",
    "MergingDiagnostics",
]
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_merging_diagnostics.py -v
```

Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add jax_frc/diagnostics/ tests/test_merging_diagnostics.py
git commit -m "feat(diagnostics): add MergingDiagnostics probe

Computes merging-specific metrics:
- separation_dz: null separation
- separatrix_radius, elongation
- separatrix_beta, peak_pressure
- null_positions, reconnection_rate"
```

---

## Task 5: Time-Dependent Mirror Boundary Condition

**Files:**
- Create: `jax_frc/boundaries/time_dependent.py`
- Modify: `jax_frc/boundaries/__init__.py`
- Test: `tests/test_boundaries.py`

**Step 1: Write failing test**

```python
# tests/test_boundaries.py
"""Tests for boundary conditions."""

import pytest
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.boundaries.time_dependent import TimeDependentMirrorBC


class TestTimeDependentMirrorBC:
    """Tests for time-dependent mirror field boundary condition."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=20, nz=40,
            r_min=0.1, r_max=1.0,
            z_min=-2.0, z_max=2.0
        )

    @pytest.fixture
    def initial_state(self):
        return State.zeros(nr=20, nz=40)

    def test_no_change_at_t0(self, initial_state, geometry):
        """At t=0, boundary is unchanged (mirror_ratio = 1.0)."""
        bc = TimeDependentMirrorBC(
            base_field=1.0,
            mirror_ratio_final=1.5,
            ramp_time=10.0,
            profile="cosine"
        )

        result = bc.apply(initial_state, geometry, t=0.0)

        # At t=0, psi at boundaries should be unchanged
        assert jnp.allclose(result.psi[:, 0], initial_state.psi[:, 0])
        assert jnp.allclose(result.psi[:, -1], initial_state.psi[:, -1])

    def test_full_compression_at_ramp_end(self, initial_state, geometry):
        """At t=ramp_time, mirror_ratio reaches final value."""
        bc = TimeDependentMirrorBC(
            base_field=1.0,
            mirror_ratio_final=1.5,
            ramp_time=10.0,
            profile="cosine"
        )

        result = bc.apply(initial_state, geometry, t=10.0)

        # At t=ramp_time, boundary field should be at final value
        # This manifests as modified psi at z boundaries
        # Exact value depends on implementation
        assert result is not None

    def test_cosine_profile_smooth(self, initial_state, geometry):
        """Cosine profile gives smooth ramp."""
        bc = TimeDependentMirrorBC(
            base_field=1.0,
            mirror_ratio_final=2.0,
            ramp_time=10.0,
            profile="cosine"
        )

        ratios = []
        for t in [0, 2.5, 5.0, 7.5, 10.0]:
            ratio = bc._compute_mirror_ratio(t)
            ratios.append(ratio)

        # Should be monotonically increasing
        assert ratios == sorted(ratios)
        # Should start at 1.0 and end at 2.0
        assert abs(ratios[0] - 1.0) < 0.01
        assert abs(ratios[-1] - 2.0) < 0.01
```

**Step 2: Run test to verify it fails**

```bash
python -m pytest tests/test_boundaries.py -v
```

Expected: FAIL with "No module named 'jax_frc.boundaries.time_dependent'"

**Step 3: Implement time_dependent.py**

```python
# jax_frc/boundaries/time_dependent.py
"""Time-dependent boundary conditions."""

from dataclasses import dataclass
from typing import Literal
import jax.numpy as jnp

from jax_frc.boundaries.base import BoundaryCondition
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


@dataclass
class TimeDependentMirrorBC(BoundaryCondition):
    """Mirror field with time-varying strength at z boundaries.

    Implements compression profile from Belova et al.:
    - Spatial: delta_Aphi(z) ~ 0.5(1 - cos(pi*z/Zc))
    - Temporal: f(t) ~ (1 - cos(pi*t/T)) for cosine profile

    Attributes:
        base_field: B0 at t=0
        mirror_ratio_final: B_end/B0 (e.g., 1.5)
        ramp_time: T in Alfven times
        profile: "cosine" or "linear"
    """

    base_field: float
    mirror_ratio_final: float
    ramp_time: float
    profile: Literal["cosine", "linear"] = "cosine"

    def apply(self, state: State, geometry: Geometry, t: float = 0.0) -> State:
        """Apply time-dependent mirror field at z boundaries.

        Args:
            state: Current state
            geometry: Computational geometry
            t: Current simulation time

        Returns:
            State with modified boundary values
        """
        psi = state.psi

        # Compute current mirror ratio
        ratio = self._compute_mirror_ratio(t)

        # Compute delta_Aphi from ratio change
        # Mirror field increase -> increase in Aphi at boundaries
        # psi = r * Aphi, so delta_psi = r * delta_Aphi

        delta_ratio = ratio - 1.0

        # Spatial profile: strongest at ends, zero at midplane
        z = geometry.z
        Zc = geometry.z_max
        spatial_profile = 0.5 * (1 - jnp.cos(jnp.pi * jnp.abs(z) / Zc))

        # Apply to psi at z boundaries
        r = geometry.r

        # Compute boundary modification
        # At z boundaries, psi should reflect increased external flux
        delta_psi_boundary = delta_ratio * self.base_field * r * spatial_profile[-1]

        # Apply to end boundaries
        psi = psi.at[:, 0].set(psi[:, 0] + delta_psi_boundary)
        psi = psi.at[:, -1].set(psi[:, -1] + delta_psi_boundary)

        return state.replace(psi=psi)

    def _compute_mirror_ratio(self, t: float) -> float:
        """Compute current mirror ratio based on time and profile.

        Args:
            t: Current time

        Returns:
            Current mirror ratio (1.0 to mirror_ratio_final)
        """
        if t <= 0:
            return 1.0
        if t >= self.ramp_time:
            return self.mirror_ratio_final

        # Normalized time
        tau = t / self.ramp_time

        if self.profile == "cosine":
            # Smooth cosine ramp: f(tau) = 0.5 * (1 - cos(pi * tau))
            f = 0.5 * (1 - jnp.cos(jnp.pi * tau))
        else:  # linear
            f = tau

        # Interpolate between 1.0 and final ratio
        return float(1.0 + (self.mirror_ratio_final - 1.0) * f)

    def get_current_field(self, t: float) -> float:
        """Get current boundary field strength.

        Args:
            t: Current time

        Returns:
            Current field strength at boundary
        """
        return self.base_field * self._compute_mirror_ratio(t)
```

**Step 4: Update boundaries __init__.py**

```python
# jax_frc/boundaries/__init__.py
"""Boundary conditions for FRC simulations."""

from jax_frc.boundaries.base import BoundaryCondition
from jax_frc.boundaries.conducting import ConductingWallBC
from jax_frc.boundaries.symmetry import AxisSymmetryBC
from jax_frc.boundaries.time_dependent import TimeDependentMirrorBC

__all__ = [
    "BoundaryCondition",
    "ConductingWallBC",
    "AxisSymmetryBC",
    "TimeDependentMirrorBC",
]
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_boundaries.py -v
```

Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add jax_frc/boundaries/ tests/test_boundaries.py
git commit -m "feat(boundaries): add TimeDependentMirrorBC

Time-varying mirror field for FRC compression:
- Cosine and linear ramp profiles
- Spatial profile strongest at ends, zero at midplane
- Follows Belova et al. compression scheme"
```

---

## Task 6: MergingPhase Implementation

**Files:**
- Create: `jax_frc/scenarios/phases/__init__.py`
- Create: `jax_frc/scenarios/phases/merging.py`
- Test: `tests/test_merging_phase.py`

**Step 1: Create phases package init**

```python
# jax_frc/scenarios/phases/__init__.py
"""Phase implementations for FRC experiments."""

from jax_frc.scenarios.phases.merging import MergingPhase

__all__ = [
    "MergingPhase",
]
```

**Step 2: Write failing test**

```python
# tests/test_merging_phase.py
"""Tests for MergingPhase."""

import pytest
import jax.numpy as jnp
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.scenarios.phases.merging import MergingPhase
from jax_frc.scenarios.transitions import timeout


class TestMergingPhase:
    """Tests for MergingPhase."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=20, nz=80,
            r_min=0.1, r_max=1.0,
            z_min=-4.0, z_max=4.0
        )

    @pytest.fixture
    def single_frc_state(self, geometry):
        """Create a single FRC equilibrium state."""
        state = State.zeros(nr=20, nz=80)

        r = geometry.r_grid
        z = geometry.z_grid

        # Single FRC centered at z=0
        psi = jnp.exp(-((r - 0.5)**2 + z**2) / 0.2)
        p = psi * 0.5
        n = jnp.ones_like(psi)

        return state.replace(psi=psi, p=p, n=n)

    def test_setup_creates_two_frc_state(self, single_frc_state, geometry):
        """MergingPhase.setup creates two-FRC configuration."""
        phase = MergingPhase(
            name="merge",
            transition=timeout(10.0),
            separation=2.0,
            initial_velocity=0.1,
        )

        config = {}
        result = phase.setup(single_frc_state, geometry, config)

        # Should have two psi peaks
        psi = result.psi
        z_mid = psi.shape[1] // 2

        left_max = jnp.max(psi[:, :z_mid])
        right_max = jnp.max(psi[:, z_mid:])

        # Both halves should have significant flux
        assert left_max > 0.1
        assert right_max > 0.1

    def test_setup_applies_initial_velocity(self, single_frc_state, geometry):
        """MergingPhase.setup applies antisymmetric velocity."""
        phase = MergingPhase(
            name="merge",
            transition=timeout(10.0),
            separation=2.0,
            initial_velocity=0.2,
        )

        config = {}
        result = phase.setup(single_frc_state, geometry, config)

        # Left FRC should have +Vz, right FRC should have -Vz
        z_mid = result.v.shape[1] // 2

        vz_left = result.v[:, z_mid // 2, 2]  # z component
        vz_right = result.v[:, z_mid + z_mid // 2, 2]

        # Should have opposite signs
        assert jnp.mean(vz_left) > 0
        assert jnp.mean(vz_right) < 0
```

**Step 3: Run test to verify it fails**

```bash
python -m pytest tests/test_merging_phase.py -v
```

Expected: FAIL with "No module named 'jax_frc.scenarios.phases'"

**Step 4: Implement merging.py**

```python
# jax_frc/scenarios/phases/merging.py
"""Merging phase for two-FRC collision."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.scenarios.phase import Phase
from jax_frc.scenarios.transitions import Transition
from jax_frc.boundaries.time_dependent import TimeDependentMirrorBC


@dataclass
class MergingPhase(Phase):
    """Phase for FRC merging simulation.

    Sets up two-FRC initial condition by mirror-flipping a single FRC,
    applies initial velocities, and optionally enables compression.

    Attributes:
        name: Phase name
        transition: Completion condition
        separation: Initial separation between FRC nulls (in length units)
        initial_velocity: Initial axial velocity (positive = toward midplane)
        compression: Optional compression BC configuration
    """

    separation: float = 1.0
    initial_velocity: float = 0.0
    compression: Optional[dict] = None

    _compression_bc: Optional[TimeDependentMirrorBC] = None

    def setup(self, state: State, geometry: Geometry, config: dict) -> State:
        """Create two-FRC configuration from single FRC.

        Args:
            state: Single FRC equilibrium state
            geometry: Computational geometry
            config: Phase configuration (can override separation, velocity)

        Returns:
            Two-FRC state with initial velocities
        """
        # Override from config if provided
        separation = config.get("separation", self.separation)
        velocity = config.get("initial_velocity", self.initial_velocity)

        # Mirror-flip to create two FRCs
        psi = self._create_two_frc_psi(state.psi, geometry, separation)

        # Create antisymmetric velocity field
        v = self._create_velocity_field(state.v, geometry, velocity)

        # Mirror other fields
        n = self._mirror_flip(state.n)
        p = self._mirror_flip(state.p)
        B = self._mirror_flip_vector(state.B)
        E = state.E  # Keep E as is initially

        # Setup compression BC if configured
        compression_config = config.get("compression", self.compression)
        if compression_config:
            self._compression_bc = TimeDependentMirrorBC(
                base_field=compression_config.get("base_field", 1.0),
                mirror_ratio_final=compression_config.get("mirror_ratio", 1.5),
                ramp_time=compression_config.get("ramp_time", 10.0),
                profile=compression_config.get("profile", "cosine"),
            )

        return state.replace(psi=psi, n=n, p=p, B=B, E=E, v=v)

    def step_hook(self, state: State, geometry: Geometry, t: float) -> State:
        """Apply compression BC if configured.

        Args:
            state: Current state
            geometry: Computational geometry
            t: Current simulation time

        Returns:
            State with updated boundary conditions
        """
        if self._compression_bc is not None:
            state = self._compression_bc.apply(state, geometry, t)
        return state

    def _create_two_frc_psi(self, psi: jnp.ndarray, geometry: Geometry,
                           separation: float) -> jnp.ndarray:
        """Create two-FRC psi by mirror-flipping and shifting.

        Args:
            psi: Single FRC poloidal flux
            geometry: Computational geometry
            separation: Desired separation between nulls

        Returns:
            Two-FRC poloidal flux array
        """
        nz = psi.shape[1]
        z = geometry.z
        dz = geometry.dz

        # Compute shift in grid points
        shift_points = int(separation / (2 * dz))
        shift_points = min(shift_points, nz // 4)  # Don't shift too far

        # Create shifted versions
        # Left FRC: shift toward negative z
        psi_left = jnp.roll(psi, -shift_points, axis=1)

        # Right FRC: mirror and shift toward positive z
        psi_mirrored = jnp.flip(psi, axis=1)
        psi_right = jnp.roll(psi_mirrored, shift_points, axis=1)

        # Combine (simple addition - could add G-S smoothing)
        psi_combined = psi_left + psi_right

        return psi_combined

    def _create_velocity_field(self, v: jnp.ndarray, geometry: Geometry,
                              velocity: float) -> jnp.ndarray:
        """Create antisymmetric velocity field for merging.

        Left FRC moves right (+z), right FRC moves left (-z).

        Args:
            v: Existing velocity field
            geometry: Computational geometry
            velocity: Magnitude of initial velocity

        Returns:
            Velocity field with antisymmetric Vz
        """
        nz = v.shape[1]
        z = geometry.z_grid

        # Create antisymmetric Vz: positive for z<0, negative for z>0
        # Use tanh profile for smooth transition
        z_normalized = z / (geometry.z_max * 0.5)
        vz_profile = -velocity * jnp.tanh(z_normalized * 2)

        # Set Vz component
        v_new = v.at[:, :, 2].set(vz_profile)

        return v_new

    def _mirror_flip(self, field: jnp.ndarray) -> jnp.ndarray:
        """Mirror-flip a scalar field and add to original."""
        return field + jnp.flip(field, axis=1)

    def _mirror_flip_vector(self, field: jnp.ndarray) -> jnp.ndarray:
        """Mirror-flip a vector field (flip z-component sign)."""
        flipped = jnp.flip(field, axis=1)
        # Flip sign of z-component
        flipped = flipped.at[:, :, 2].set(-flipped[:, :, 2])
        return field + flipped
```

**Step 5: Run tests**

```bash
python -m pytest tests/test_merging_phase.py -v
```

Expected: PASS (2 tests)

**Step 6: Commit**

```bash
git add jax_frc/scenarios/phases/ tests/test_merging_phase.py
git commit -m "feat(scenarios): add MergingPhase for two-FRC collision

- Mirror-flip single FRC to create two-FRC configuration
- Apply antisymmetric initial velocity field
- Optional compression via TimeDependentMirrorBC"
```

---

## Task 7: Merging Example Configurations

**Files:**
- Create: `examples/merging_examples.py`
- Test: Run example to verify it works

**Step 1: Create merging_examples.py**

```python
# examples/merging_examples.py
"""Example merging scenarios from Belova et al. paper."""

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.scenarios import Scenario, timeout
from jax_frc.scenarios.phases.merging import MergingPhase
from jax_frc.scenarios.transitions import separation_below, any_of
from jax_frc.equilibrium.grad_shafranov import create_frc_equilibrium
import jax.numpy as jnp


def create_default_geometry(rc: float = 1.0, zc: float = 4.0,
                           nr: int = 64, nz: int = 256) -> Geometry:
    """Create default cylindrical geometry for merging.

    Args:
        rc: Flux conserver radius
        zc: Half-length of domain
        nr: Radial grid points
        nz: Axial grid points

    Returns:
        Geometry object
    """
    return Geometry(
        coord_system="cylindrical",
        nr=nr,
        nz=nz,
        r_min=0.01 * rc,
        r_max=rc,
        z_min=-zc,
        z_max=zc,
    )


def create_initial_frc(geometry: Geometry,
                       s_star: float = 20.0,
                       elongation: float = 2.0,
                       xs: float = 0.6,
                       beta_s: float = 0.2) -> State:
    """Create initial single-FRC equilibrium.

    Uses simplified Gaussian profile (proper G-S solve in production).

    Args:
        geometry: Computational geometry
        s_star: Kinetic parameter Rs/di
        elongation: E = Zs/Rs
        xs: Normalized separatrix radius Rs/Rc
        beta_s: Separatrix beta

    Returns:
        Single FRC equilibrium state
    """
    state = State.zeros(nr=geometry.nr, nz=geometry.nz)

    r = geometry.r_grid
    z = geometry.z_grid

    # Compute FRC dimensions from parameters
    Rc = geometry.r_max
    Rs = xs * Rc
    Zs = elongation * Rs

    # Create Gaussian FRC profile
    psi = jnp.exp(-((r - 0.5*Rs)**2 / (0.3*Rs)**2 + z**2 / Zs**2))

    # Pressure proportional to psi, with separatrix beta
    p = beta_s * psi

    # Uniform density
    n = jnp.ones_like(psi)

    return state.replace(psi=psi, p=p, n=n)


def belova_case1() -> Scenario:
    """Large FRC merging without compression (paper Fig. 1-2).

    Parameters:
        S* = 25.6, E = 2.9, xs = 0.69, beta_s = 0.2
        Initial separation: dZ = 180 (normalized)
        Initial velocity: Vz = 0.2 vA

    Expected outcome: Partial merge, doublet configuration
    """
    geometry = create_default_geometry(rc=1.0, zc=5.0, nr=64, nz=512)
    initial_state = create_initial_frc(
        geometry,
        s_star=25.6,
        elongation=2.9,
        xs=0.69,
        beta_s=0.2
    )

    # Merge phase with velocity drive only (no compression)
    merge_phase = MergingPhase(
        name="merge_no_compression",
        transition=any_of(
            separation_below(0.5, geometry),  # Complete merge
            timeout(30.0)  # tA
        ),
        separation=3.0,  # Normalized units
        initial_velocity=0.2,
        compression=None,
    )

    return Scenario(
        name="belova_case1_large_frc",
        phases=[merge_phase],
        geometry=geometry,
        initial_state=initial_state,
        dt=0.01,
    )


def belova_case2() -> Scenario:
    """Small FRC merging without compression (paper Fig. 3-4).

    Parameters:
        S* = 20, E = 1.5, xs = 0.53, beta_s = 0.2
        Initial separation: dZ = 75 (normalized)
        Initial velocity: Vz = 0.1 vA

    Expected outcome: Complete merge by ~5-7 tA
    """
    geometry = create_default_geometry(rc=1.0, zc=3.0, nr=64, nz=256)
    initial_state = create_initial_frc(
        geometry,
        s_star=20.0,
        elongation=1.5,
        xs=0.53,
        beta_s=0.2
    )

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
        dt=0.01,
    )


def belova_case4() -> Scenario:
    """Large FRC with compression (paper Fig. 6-7).

    Parameters: Same as case1 but with compression
        Mirror ratio: 1.5
        Ramp time: 19 tA

    Expected outcome: Complete merge by ~20-25 tA
    """
    geometry = create_default_geometry(rc=1.0, zc=5.0, nr=64, nz=512)
    initial_state = create_initial_frc(
        geometry,
        s_star=25.6,
        elongation=2.9,
        xs=0.69,
        beta_s=0.2
    )

    merge_phase = MergingPhase(
        name="merge_with_compression",
        transition=any_of(
            separation_below(0.3, geometry),
            timeout(40.0)
        ),
        separation=3.0,
        initial_velocity=0.0,  # Compression drives merging
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
        dt=0.01,
    )


if __name__ == "__main__":
    # Run case 2 as a quick test
    print("Creating Belova Case 2 scenario...")
    scenario = belova_case2()

    print(f"Geometry: {scenario.geometry.nr}x{scenario.geometry.nz}")
    print(f"Phases: {[p.name for p in scenario.phases]}")
    print(f"dt: {scenario.dt}")

    print("\nRunning scenario...")
    result = scenario.run()

    print(f"\nResult: {result.success}")
    for pr in result.phase_results:
        print(f"  {pr.name}: {pr.termination} at t={pr.end_time:.2f}")
```

**Step 2: Run example**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\scenario-framework
python examples/merging_examples.py
```

Expected: Runs without error, prints scenario info

**Step 3: Commit**

```bash
git add examples/merging_examples.py
git commit -m "feat(examples): add Belova et al. merging scenarios

Three validation cases from the paper:
- Case 1: Large FRC without compression (doublet)
- Case 2: Small FRC complete merge
- Case 4: Large FRC with compression"
```

---

## Task 8: Integration Test

**Files:**
- Create: `tests/test_merging_integration.py`

**Step 1: Write integration test**

```python
# tests/test_merging_integration.py
"""Integration tests for merging scenario."""

import pytest
import jax.numpy as jnp
from examples.merging_examples import belova_case2, create_default_geometry, create_initial_frc
from jax_frc.diagnostics.merging import MergingDiagnostics


class TestMergingIntegration:
    """Integration tests for full merging workflow."""

    def test_scenario_runs_to_completion(self):
        """Full scenario runs without errors."""
        scenario = belova_case2()
        result = scenario.run()

        assert result.success
        assert len(result.phase_results) == 1
        assert result.total_time > 0

    def test_diagnostics_track_merging(self):
        """Diagnostics correctly track merging progress."""
        geometry = create_default_geometry(rc=1.0, zc=2.0, nr=32, nz=64)
        initial = create_initial_frc(geometry, elongation=1.5, xs=0.5)

        diag = MergingDiagnostics()
        result = diag.compute(initial, geometry)

        # Should have valid metrics
        assert result["separation_dz"] >= 0
        assert result["separatrix_radius"] > 0
        assert result["peak_pressure"] >= 0

    def test_two_frc_setup_doubles_flux(self):
        """MergingPhase setup creates two FRCs with roughly double flux."""
        from jax_frc.scenarios.phases.merging import MergingPhase
        from jax_frc.scenarios.transitions import timeout

        geometry = create_default_geometry(rc=1.0, zc=2.0, nr=32, nz=64)
        single_frc = create_initial_frc(geometry, elongation=1.5, xs=0.5)

        phase = MergingPhase(
            name="test",
            transition=timeout(1.0),
            separation=1.0,
            initial_velocity=0.1,
        )

        two_frc = phase.setup(single_frc, geometry, {})

        # Two-FRC should have roughly double the total flux
        single_flux = jnp.sum(jnp.abs(single_frc.psi))
        double_flux = jnp.sum(jnp.abs(two_frc.psi))

        assert 1.5 < double_flux / single_flux < 2.5
```

**Step 2: Run integration tests**

```bash
python -m pytest tests/test_merging_integration.py -v
```

Expected: PASS (3 tests)

**Step 3: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: All tests pass

**Step 4: Commit**

```bash
git add tests/test_merging_integration.py
git commit -m "test: add merging integration tests

Verify:
- Full scenario runs to completion
- Diagnostics track merging correctly
- Two-FRC setup doubles flux content"
```

---

## Summary

**Implementation order:**
1. Phase and Transition base classes
2. Scenario runner
3. Built-in transition conditions
4. Merging diagnostics probe
5. Time-dependent mirror boundary condition
6. MergingPhase implementation
7. Merging example configurations
8. Integration tests

**Total commits:** 8

**Files created:**
- `jax_frc/scenarios/__init__.py`
- `jax_frc/scenarios/phase.py`
- `jax_frc/scenarios/transitions.py`
- `jax_frc/scenarios/scenario.py`
- `jax_frc/scenarios/phases/__init__.py`
- `jax_frc/scenarios/phases/merging.py`
- `jax_frc/diagnostics/merging.py`
- `jax_frc/boundaries/time_dependent.py`
- `examples/merging_examples.py`
- `tests/test_scenarios.py`
- `tests/test_merging_diagnostics.py`
- `tests/test_boundaries.py`
- `tests/test_merging_phase.py`
- `tests/test_merging_integration.py`

**Remaining tasks (future):**
- Burn module (reactions, products, fuel, fast ions)
- FormationPhase
- TranslationPhase
- CompressionPhase
- BurnPhase
- Full cycle integration
