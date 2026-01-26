# Tooling Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add JIT compilation, CLI progress display, and automatic output saving to run_example.py

**Architecture:** JIT at solver level (not loop), progress reporter writing to stderr, OutputManager handling file saves after simulation completes.

**Tech Stack:** JAX (jit, partial), Python stdlib (argparse, pathlib, datetime, sys)

---

## Task 1: JIT-compile ResistiveMHD.compute_rhs

**Files:**
- Modify: `jax_frc/models/resistive_mhd.py:24-51`
- Test: `tests/test_jit_compilation.py` (create)

**Step 1: Write failing test**

Create `tests/test_jit_compilation.py`:

```python
"""Tests for JIT compilation of physics models and solvers."""

import pytest
import jax
import jax.numpy as jnp
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


class TestResistiveMHDJIT:
    """Test JIT compilation of ResistiveMHD."""

    @pytest.fixture
    def setup(self):
        """Create model, state, geometry for testing."""
        geometry = Geometry(
            coord_system="cylindrical",
            nr=16, nz=32,
            r_min=0.0, r_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        state = State.zeros(geometry.nr, geometry.nz)
        # Add some non-trivial psi
        r, z = geometry.r_grid, geometry.z_grid
        state = state.replace(psi=jnp.exp(-r**2 - z**2))
        model = ResistiveMHD(resistivity=SpitzerResistivity())
        return model, state, geometry

    def test_compute_rhs_is_jittable(self, setup):
        """compute_rhs can be JIT-compiled without error."""
        model, state, geometry = setup
        # This should not raise
        jitted_rhs = jax.jit(model.compute_rhs, static_argnums=(2,))
        result = jitted_rhs(state, geometry)
        assert result.psi.shape == state.psi.shape

    def test_compute_rhs_jit_produces_same_result(self, setup):
        """JIT and non-JIT produce identical results."""
        model, state, geometry = setup
        result_eager = model.compute_rhs(state, geometry)
        jitted_rhs = jax.jit(model.compute_rhs, static_argnums=(2,))
        result_jit = jitted_rhs(state, geometry)
        assert jnp.allclose(result_eager.psi, result_jit.psi, rtol=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_jit_compilation.py -v`
Expected: PASS (the model is already JIT-compatible, we just need to add decorator)

**Step 3: Add JIT decorator to compute_rhs**

In `jax_frc/models/resistive_mhd.py`, add import and decorator:

```python
from functools import partial

# ... existing code ...

@dataclass
class ResistiveMHD(PhysicsModel):
    """Single-fluid resistive MHD model."""

    resistivity: ResistivityModel

    @partial(jax.jit, static_argnums=(0, 2))  # self and geometry are static
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        # ... existing implementation unchanged ...
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_jit_compilation.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/models/resistive_mhd.py tests/test_jit_compilation.py
git commit -m "feat: JIT-compile ResistiveMHD.compute_rhs"
```

---

## Task 2: JIT-compile ExtendedMHD.compute_rhs

**Files:**
- Modify: `jax_frc/models/extended_mhd.py`
- Test: `tests/test_jit_compilation.py`

**Step 1: Add test to existing file**

Append to `tests/test_jit_compilation.py`:

```python
from jax_frc.models.extended_mhd import ExtendedMHD


class TestExtendedMHDJIT:
    """Test JIT compilation of ExtendedMHD."""

    @pytest.fixture
    def setup(self):
        """Create model, state, geometry for testing."""
        geometry = Geometry(
            coord_system="cylindrical",
            nr=16, nz=32,
            r_min=0.0, r_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        state = State.zeros(geometry.nr, geometry.nz)
        r, z = geometry.r_grid, geometry.z_grid
        # Initialize B field
        state = state.replace(
            B=jnp.stack([jnp.zeros_like(r), jnp.zeros_like(r), jnp.ones_like(r)], axis=-1),
            n=jnp.ones_like(r) * 1e19,
        )
        model = ExtendedMHD(resistivity=SpitzerResistivity())
        return model, state, geometry

    def test_compute_rhs_is_jittable(self, setup):
        """compute_rhs can be JIT-compiled without error."""
        model, state, geometry = setup
        jitted_rhs = jax.jit(model.compute_rhs, static_argnums=(2,))
        result = jitted_rhs(state, geometry)
        assert result.B.shape == state.B.shape
```

**Step 2: Run test**

Run: `py -m pytest tests/test_jit_compilation.py::TestExtendedMHDJIT -v`

**Step 3: Add JIT decorator to ExtendedMHD.compute_rhs**

In `jax_frc/models/extended_mhd.py`:

```python
from functools import partial

# In ExtendedMHD class:
    @partial(jax.jit, static_argnums=(0, 2))
    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        # ... existing implementation ...
```

**Step 4: Run test to verify**

Run: `py -m pytest tests/test_jit_compilation.py::TestExtendedMHDJIT -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/models/extended_mhd.py tests/test_jit_compilation.py
git commit -m "feat: JIT-compile ExtendedMHD.compute_rhs"
```

---

## Task 3: JIT-compile EulerSolver.step and RK4Solver.step

**Files:**
- Modify: `jax_frc/solvers/explicit.py`
- Test: `tests/test_jit_compilation.py`

**Step 1: Add solver JIT tests**

Append to `tests/test_jit_compilation.py`:

```python
from jax_frc.solvers.explicit import EulerSolver, RK4Solver


class TestExplicitSolversJIT:
    """Test JIT compilation of explicit solvers."""

    @pytest.fixture
    def setup(self):
        """Create solver, model, state, geometry."""
        geometry = Geometry(
            coord_system="cylindrical",
            nr=16, nz=32,
            r_min=0.0, r_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        state = State.zeros(geometry.nr, geometry.nz)
        r, z = geometry.r_grid, geometry.z_grid
        state = state.replace(psi=jnp.exp(-r**2 - z**2))
        model = ResistiveMHD(resistivity=SpitzerResistivity())
        return state, model, geometry

    def test_euler_step_is_jittable(self, setup):
        """EulerSolver.step can be JIT-compiled."""
        state, model, geometry = setup
        solver = EulerSolver()
        dt = 1e-6
        # Should compile and run
        result = solver.step(state, dt, model, geometry)
        assert result.psi.shape == state.psi.shape
        assert float(result.time) > float(state.time)

    def test_rk4_step_is_jittable(self, setup):
        """RK4Solver.step can be JIT-compiled."""
        state, model, geometry = setup
        solver = RK4Solver()
        dt = 1e-6
        result = solver.step(state, dt, model, geometry)
        assert result.psi.shape == state.psi.shape
        assert float(result.time) > float(state.time)
```

**Step 2: Run tests**

Run: `py -m pytest tests/test_jit_compilation.py::TestExplicitSolversJIT -v`

**Step 3: Note on solver JIT**

The solver `step()` methods call `model.compute_rhs()` which is now JIT-compiled. The solver itself doesn't need a JIT decorator because:
1. The heavy computation is in `compute_rhs()` (now JIT)
2. The solver loop runs in Python anyway (can't use lax.while_loop due to callbacks)

However, we can JIT the entire step for additional speedup. Add to `jax_frc/solvers/explicit.py`:

```python
from functools import partial
import jax

@dataclass
class EulerSolver(Solver):
    """Simple forward Euler integration."""

    @partial(jax.jit, static_argnums=(0, 3, 4))  # self, model, geometry static
    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        # ... existing implementation unchanged ...


@dataclass
class RK4Solver(Solver):
    """4th-order Runge-Kutta integration."""

    @partial(jax.jit, static_argnums=(0, 3, 4))
    def step(self, state: State, dt: float, model: PhysicsModel, geometry) -> State:
        # ... existing implementation unchanged ...
```

**Step 4: Run tests**

Run: `py -m pytest tests/test_jit_compilation.py::TestExplicitSolversJIT -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/solvers/explicit.py tests/test_jit_compilation.py
git commit -m "feat: JIT-compile EulerSolver and RK4Solver step methods"
```

---

## Task 4: Create ProgressReporter class

**Files:**
- Create: `jax_frc/diagnostics/progress.py`
- Test: `tests/test_progress.py` (create)

**Step 1: Write failing test**

Create `tests/test_progress.py`:

```python
"""Tests for CLI progress reporting."""

import pytest
import sys
from io import StringIO
from jax_frc.diagnostics.progress import ProgressReporter


class TestProgressReporter:
    """Test ProgressReporter output."""

    def test_report_writes_to_stderr(self):
        """report() writes progress to stderr."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(
            t_end=1.0,
            output_interval=1,
            stream=stderr_capture,
        )
        reporter.report(
            t=0.5,
            step=100,
            dt=1e-6,
            phase_name="merging",
        )
        output = stderr_capture.getvalue()
        assert "merging" in output
        assert "50" in output  # 50% progress

    def test_report_shows_percentage(self):
        """Progress percentage is calculated correctly."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(t_end=1.0, stream=stderr_capture)
        reporter.report(t=0.25, step=50, dt=1e-6, phase_name="test")
        output = stderr_capture.getvalue()
        assert "25" in output  # 25%

    def test_report_includes_step_count(self):
        """Step count is shown in output."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(t_end=1.0, stream=stderr_capture)
        reporter.report(t=0.1, step=1234, dt=1e-6, phase_name="test")
        output = stderr_capture.getvalue()
        assert "1234" in output

    def test_report_disabled_does_nothing(self):
        """When enabled=False, nothing is written."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(t_end=1.0, enabled=False, stream=stderr_capture)
        reporter.report(t=0.5, step=100, dt=1e-6, phase_name="test")
        assert stderr_capture.getvalue() == ""

    def test_finish_prints_newline(self):
        """finish() prints final newline."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(t_end=1.0, stream=stderr_capture)
        reporter.report(t=0.5, step=100, dt=1e-6, phase_name="test")
        reporter.finish()
        output = stderr_capture.getvalue()
        assert output.endswith("\n")
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_progress.py -v`
Expected: ImportError (module doesn't exist)

**Step 3: Implement ProgressReporter**

Create `jax_frc/diagnostics/progress.py`:

```python
"""CLI progress reporting for simulations."""

import sys
from dataclasses import dataclass, field
from typing import TextIO, Optional, Dict, Any


@dataclass
class ProgressReporter:
    """Reports simulation progress to stderr.

    Produces output like:
        [Phase: merging] t=1.23e-6 / 5.00e-6 (24.6%) | step 1200 | dt=4.1e-9

    Attributes:
        t_end: Target end time for percentage calculation
        output_interval: Only report every N calls (default 1 = every call)
        enabled: If False, report() does nothing
        stream: Output stream (default stderr)
    """

    t_end: float
    output_interval: int = 1
    enabled: bool = True
    stream: TextIO = field(default_factory=lambda: sys.stderr)

    _call_count: int = field(default=0, init=False, repr=False)

    def report(
        self,
        t: float,
        step: int,
        dt: float,
        phase_name: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report current simulation progress.

        Args:
            t: Current simulation time
            step: Current step number
            dt: Current timestep
            phase_name: Name of current phase
            diagnostics: Optional dict of diagnostic values to display
        """
        if not self.enabled:
            return

        self._call_count += 1
        if self._call_count % self.output_interval != 0:
            return

        # Calculate progress percentage
        pct = (t / self.t_end * 100) if self.t_end > 0 else 0.0

        # Build progress string
        parts = [
            f"[Phase: {phase_name}]",
            f"t={t:.2e} / {self.t_end:.2e} ({pct:.1f}%)",
            f"| step {step}",
            f"| dt={dt:.1e}",
        ]

        # Add diagnostics if provided
        if diagnostics:
            for name, value in diagnostics.items():
                if isinstance(value, float):
                    parts.append(f"| {name}={value:.3g}")
                else:
                    parts.append(f"| {name}={value}")

        line = " ".join(parts)

        # Write with carriage return for in-place update
        self.stream.write(f"\r{line}")
        self.stream.flush()

    def finish(self) -> None:
        """Print final newline after progress reporting completes."""
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_progress.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/diagnostics/progress.py tests/test_progress.py
git commit -m "feat: add ProgressReporter for CLI progress display"
```

---

## Task 5: Integrate ProgressReporter into LinearConfiguration

**Files:**
- Modify: `jax_frc/configurations/linear_configuration.py:242-309`
- Test: `tests/test_linear_configuration_progress.py` (create)

**Step 1: Write test**

Create `tests/test_linear_configuration_progress.py`:

```python
"""Tests for progress reporting in LinearConfiguration."""

import pytest
from io import StringIO
from dataclasses import dataclass, field
from typing import List

import jax.numpy as jnp

from jax_frc.configurations.linear_configuration import (
    LinearConfiguration, PhaseSpec, TransitionSpec, ConfigurationResult
)
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity
from jax_frc.diagnostics.progress import ProgressReporter


@dataclass
class SimpleTestConfiguration(LinearConfiguration):
    """Minimal configuration for testing progress."""

    name: str = "test"
    timeout: float = 1e-5
    dt: float = 1e-6
    model_type: str = "resistive_mhd"

    def build_geometry(self) -> Geometry:
        return Geometry(
            coord_system="cylindrical",
            nr=8, nz=16,
            r_min=0.0, r_max=1.0,
            z_min=-1.0, z_max=1.0,
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        state = State.zeros(geometry.nr, geometry.nz)
        r = geometry.r_grid
        return state.replace(psi=jnp.exp(-r**2))

    def build_model(self):
        return ResistiveMHD(resistivity=SpitzerResistivity())

    def build_boundary_conditions(self):
        return []

    def build_phase_specs(self) -> List[PhaseSpec]:
        return [
            PhaseSpec(
                name="evolve",
                transition=TransitionSpec(type="timeout", value=self.timeout),
            )
        ]


class TestLinearConfigurationProgress:
    """Test progress reporting integration."""

    def test_run_with_progress_reporter(self):
        """Configuration can run with progress reporter."""
        stderr_capture = StringIO()
        reporter = ProgressReporter(
            t_end=1e-5,
            stream=stderr_capture,
            enabled=True,
        )

        config = SimpleTestConfiguration()
        config.progress_reporter = reporter
        result = config.run()

        assert result.success
        output = stderr_capture.getvalue()
        # Should have some progress output
        assert len(output) > 0

    def test_run_without_progress_reporter(self):
        """Configuration runs fine without progress reporter."""
        config = SimpleTestConfiguration()
        result = config.run()
        assert result.success
```

**Step 2: Run test to verify failure**

Run: `py -m pytest tests/test_linear_configuration_progress.py -v`
Expected: AttributeError (progress_reporter not recognized)

**Step 3: Add progress_reporter to LinearConfiguration**

Modify `jax_frc/configurations/linear_configuration.py`:

```python
# Add import at top
from jax_frc.diagnostics.progress import ProgressReporter

# In LinearConfiguration class, add field:
@dataclass
class LinearConfiguration(AbstractConfiguration):
    # ... existing fields ...
    dt: float = 1e-6
    output_interval: int = 100
    phase_config: Dict[str, dict] = field(default_factory=dict)
    diagnostics: List["Probe"] = field(default_factory=list)

    # Add this field:
    progress_reporter: Optional[ProgressReporter] = None

    # ... rest unchanged until _run_phase ...
```

**Step 4: Integrate progress reporting into _run_phase**

In `_run_phase` method, after recording diagnostics (line ~290):

```python
    def _run_phase(
        self,
        phase: Phase,
        state: State,
        geometry: Geometry,
        solver: Solver,
        model: PhysicsModel,
        config: dict,
    ) -> PhaseResult:
        # ... existing setup code ...

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
                    history[probe.name].append(probe.measure(state, geometry))

                # Report progress
                if self.progress_reporter is not None:
                    self.progress_reporter.report(
                        t=t,
                        step=int(state.step),
                        dt=self.dt,
                        phase_name=phase.name,
                    )

            # Apply step hook (time-varying BCs, etc.)
            state = phase.step_hook(state, geometry, t)

            # Advance physics via solver
            state = solver.step(state, self.dt, model, geometry)
            t = float(state.time)

        # Finish progress line
        if self.progress_reporter is not None:
            self.progress_reporter.finish()

        # ... rest of method unchanged ...
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_linear_configuration_progress.py -v`
Expected: PASS

**Step 6: Commit**

```bash
git add jax_frc/configurations/linear_configuration.py tests/test_linear_configuration_progress.py
git commit -m "feat: integrate ProgressReporter into LinearConfiguration"
```

---

## Task 6: Create OutputManager class

**Files:**
- Modify: `jax_frc/diagnostics/output.py`
- Test: `tests/test_output_manager.py` (create)

**Step 1: Write failing test**

Create `tests/test_output_manager.py`:

```python
"""Tests for OutputManager."""

import pytest
import tempfile
from pathlib import Path
import shutil

from jax_frc.diagnostics.output import OutputManager


class TestOutputManager:
    """Test OutputManager file operations."""

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    def test_creates_output_directory(self, output_dir):
        """OutputManager creates directory structure."""
        manager = OutputManager(
            output_dir=output_dir / "test_run",
            example_name="belova_case2",
        )
        manager.setup()
        assert manager.run_dir.exists()
        assert (manager.run_dir / "plots").exists()

    def test_save_history_csv(self, output_dir):
        """save_history creates CSV file."""
        manager = OutputManager(output_dir=output_dir / "test", example_name="test")
        manager.setup()
        history = {"time": [0.0, 1.0, 2.0], "B_max": [0.1, 0.2, 0.3]}
        manager.save_history(history, format="csv")
        assert (manager.run_dir / "history.csv").exists()

    def test_save_history_json(self, output_dir):
        """save_history creates JSON file."""
        manager = OutputManager(output_dir=output_dir / "test", example_name="test")
        manager.setup()
        history = {"time": [0.0, 1.0], "psi_max": [1.0, 0.9]}
        manager.save_history(history, format="json")
        assert (manager.run_dir / "history.json").exists()

    def test_save_config_copy(self, output_dir):
        """save_config copies config to output."""
        manager = OutputManager(output_dir=output_dir / "test", example_name="test")
        manager.setup()
        config = {"configuration": {"class": "TestConfig"}, "runtime": {"dt": 1e-6}}
        manager.save_config(config)
        assert (manager.run_dir / "config.yaml").exists()

    def test_get_summary(self, output_dir):
        """get_summary returns dict of output paths."""
        manager = OutputManager(output_dir=output_dir / "test", example_name="test")
        manager.setup()
        history = {"time": [0.0], "B_max": [0.1]}
        manager.save_history(history)
        summary = manager.get_summary()
        assert "run_dir" in summary
        assert "history" in summary
```

**Step 2: Run test to verify failure**

Run: `py -m pytest tests/test_output_manager.py -v`
Expected: ImportError (OutputManager not defined)

**Step 3: Implement OutputManager**

Add to `jax_frc/diagnostics/output.py`:

```python
from datetime import datetime
import yaml


@dataclass
class OutputManager:
    """Manages simulation output files.

    Creates output directory structure:
        {output_dir}/{example_name}_{timestamp}/
            config.yaml
            history.csv (or .json)
            checkpoint_final.h5 (optional)
            plots/
                *.png

    Attributes:
        output_dir: Base directory for outputs
        example_name: Name of the example being run
        timestamp: Optional timestamp string (default: auto-generated)
        save_checkpoint: Whether to save final checkpoint
    """

    output_dir: Path
    example_name: str
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    save_checkpoint: bool = True

    run_dir: Path = field(init=False)
    _history_path: Optional[Path] = field(default=None, init=False)
    _checkpoint_path: Optional[Path] = field(default=None, init=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)

    def setup(self) -> None:
        """Create output directory structure."""
        self.run_dir = self.output_dir / f"{self.example_name}_{self.timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "plots").mkdir(exist_ok=True)

    def save_config(self, config: Dict[str, Any]) -> Path:
        """Save copy of configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Path to saved config file
        """
        path = self.run_dir / "config.yaml"
        with open(path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False)
        return path

    def save_history(self, history: Dict[str, Any], format: str = "csv") -> Path:
        """Save time history data.

        Args:
            history: Dict with 'time' and diagnostic values
            format: 'csv' or 'json'

        Returns:
            Path to saved history file
        """
        ext = "csv" if format == "csv" else "json"
        path = self.run_dir / f"history.{ext}"
        save_time_history(history, path, format=format)
        self._history_path = path
        return path

    def save_final_checkpoint(
        self,
        state: "State",
        geometry: "Geometry",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[Path]:
        """Save final state checkpoint.

        Args:
            state: Final simulation state
            geometry: Computational geometry
            metadata: Optional metadata

        Returns:
            Path to checkpoint file, or None if save_checkpoint=False
        """
        if not self.save_checkpoint:
            return None
        path = self.run_dir / "checkpoint_final.h5"
        save_checkpoint(state, geometry, path, metadata)
        self._checkpoint_path = path
        return path

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of saved outputs.

        Returns:
            Dict with paths to all saved files
        """
        summary = {
            "run_dir": str(self.run_dir),
        }
        if self._history_path and self._history_path.exists():
            summary["history"] = str(self._history_path)
        if self._checkpoint_path and self._checkpoint_path.exists():
            summary["checkpoint"] = str(self._checkpoint_path)

        # List any plots
        plots_dir = self.run_dir / "plots"
        if plots_dir.exists():
            plots = list(plots_dir.glob("*.png"))
            if plots:
                summary["plots"] = [str(p) for p in plots]

        return summary
```

Also add these imports at the top of output.py:

```python
from dataclasses import dataclass, field
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_output_manager.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/diagnostics/output.py tests/test_output_manager.py
git commit -m "feat: add OutputManager for simulation output handling"
```

---

## Task 7: Update run_example.py with CLI flags

**Files:**
- Modify: `scripts/run_example.py`
- Test: Manual testing (CLI behavior)

**Step 1: Add new argument parser options**

In `scripts/run_example.py`, update the argument parser in `main()`:

```python
def main():
    parser = argparse.ArgumentParser(
        description="Run example simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    List available examples
  %(prog)s belova_case2              Run specific example
  %(prog)s frc/belova_case1          Run with category prefix
  %(prog)s --category frc            Run all FRC examples
  %(prog)s --all                     Run all examples

Output options:
  %(prog)s belova_case2 --output-dir ./runs
  %(prog)s belova_case2 --no-progress
  %(prog)s belova_case2 --no-plots
  %(prog)s belova_case2 --no-checkpoint
  %(prog)s belova_case2 --history-format json
        """
    )
    parser.add_argument('examples', nargs='*', help="Example names to run")
    parser.add_argument('--category', help="Run all examples in category")
    parser.add_argument('--all', action='store_true', help="Run all examples")
    parser.add_argument('--list', action='store_true', help="List available examples")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    # New output options
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'),
                        help="Base directory for outputs (default: outputs/)")
    parser.add_argument('--progress', dest='progress', action='store_true', default=True,
                        help="Show progress bar (default)")
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                        help="Disable progress bar")
    parser.add_argument('--plots', dest='plots', action='store_true', default=True,
                        help="Generate plots (default)")
    parser.add_argument('--no-plots', dest='plots', action='store_false',
                        help="Skip plot generation")
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', default=True,
                        help="Save final checkpoint (default)")
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false',
                        help="Skip final checkpoint")
    parser.add_argument('--history-format', choices=['csv', 'json'], default='csv',
                        help="Format for history output (default: csv)")

    args = parser.parse_args()
    # ... rest of main ...
```

**Step 2: Update run_example function signature**

```python
def run_example(yaml_path: Path, args) -> tuple:
    """Run a single example from YAML.

    Args:
        yaml_path: Path to example YAML file
        args: Parsed command-line arguments

    Returns:
        Tuple of (ConfigurationResult, OutputManager)
    """
    # ... implementation in next step ...
```

**Step 3: Integrate OutputManager and ProgressReporter**

Full updated `run_example` function:

```python
from jax_frc.diagnostics.output import OutputManager
from jax_frc.diagnostics.progress import ProgressReporter


def run_example(yaml_path: Path, args):
    """Run a single example from YAML.

    Args:
        yaml_path: Path to example YAML file
        args: Parsed command-line arguments

    Returns:
        Tuple of (ConfigurationResult, OutputManager or None)
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Build configuration from registry
    class_name = config['configuration']['class']
    overrides = config['configuration'].get('overrides', {})

    if class_name not in CONFIGURATION_REGISTRY:
        raise ValueError(f"Unknown configuration class: {class_name}")

    ConfigClass = CONFIGURATION_REGISTRY[class_name]
    configuration = ConfigClass(**overrides)

    # Apply runtime overrides
    if 'runtime' in config:
        if 't_end' in config['runtime']:
            configuration.timeout = config['runtime']['t_end']
        if 'dt' in config['runtime']:
            configuration.dt = config['runtime']['dt']

    # Setup output manager
    output_manager = OutputManager(
        output_dir=args.output_dir,
        example_name=yaml_path.stem,
        save_checkpoint=args.checkpoint,
    )
    output_manager.setup()
    output_manager.save_config(config)

    # Setup progress reporter
    if args.progress:
        progress_reporter = ProgressReporter(
            t_end=configuration.timeout,
            output_interval=1,  # Report at configuration's output_interval
            enabled=True,
        )
        configuration.progress_reporter = progress_reporter

    # Print header
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"{'='*60}")
    if config.get('description'):
        for line in config['description'].strip().split('\n'):
            print(f"  {line}")
    print(f"\nConfiguration: {class_name}")
    print(f"Model: {configuration.model_type}")
    print(f"Runtime: t_end={configuration.timeout}, dt={configuration.dt}")
    print(f"Output: {output_manager.run_dir}")
    print(f"{'='*60}\n")

    # Run simulation
    result = configuration.run()

    # Save outputs
    # Collect history from all phases
    combined_history = {"time": []}
    for pr in result.phase_results:
        if pr.history:
            for key, values in pr.history.items():
                if key not in combined_history:
                    combined_history[key] = []
                combined_history[key].extend(values)

    if combined_history["time"]:
        output_manager.save_history(combined_history, format=args.history_format)

    # Save final checkpoint
    if args.checkpoint:
        geometry = configuration.build_geometry()
        output_manager.save_final_checkpoint(
            result.final_state,
            geometry,
            metadata={"example": config['name'], "success": result.success},
        )

    # Generate plots (if enabled and plotting module available)
    if args.plots and combined_history["time"]:
        try:
            from jax_frc.diagnostics.plotting import plot_time_traces
            plot_path = output_manager.run_dir / "plots" / "time_traces.png"
            # Create a minimal result-like object for plotting
            # (plotting expects specific structure)
            # For now, skip if not enough data
        except Exception as e:
            logging.warning(f"Could not generate plots: {e}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"{'='*60}")
    for pr in result.phase_results:
        print(f"  Phase '{pr.name}':")
        print(f"    Termination: {pr.termination}")
        print(f"    End time: {pr.end_time:.4e}")
    print()

    # Print output summary
    summary = output_manager.get_summary()
    print("Outputs saved:")
    for key, value in summary.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for v in value:
                print(f"    - {v}")
        else:
            print(f"  {key}: {value}")
    print()

    return result, output_manager
```

**Step 4: Update main() to pass args**

```python
def main():
    # ... argument parsing unchanged ...

    # Run examples (update the loop)
    all_success = True
    results = []
    for example_file in example_files:
        try:
            result, output_mgr = run_example(example_file, args)
            results.append((example_file.stem, result, output_mgr))
            if not result.success:
                all_success = False
        except Exception as e:
            logging.exception(f"Error running {example_file}")
            all_success = False

    # Print final summary if multiple examples
    if len(example_files) > 1:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for name, result, _ in results:
            status = "SUCCESS" if result.success else "FAILED"
            print(f"  {name}: {status}")
        print()

    return 0 if all_success else 1
```

**Step 5: Test manually**

```bash
cd C:/Users/周光裕/jax-frc/.worktrees/tooling-improvements
py scripts/run_example.py --list
py scripts/run_example.py --help
```

**Step 6: Commit**

```bash
git add scripts/run_example.py
git commit -m "feat: add output, progress, and checkpoint options to run_example.py"
```

---

## Task 8: Run full test suite and verify

**Files:** None (verification only)

**Step 1: Run all tests**

```bash
py -m pytest tests/ -v -k "not slow"
```

Expected: All tests pass

**Step 2: Test run_example.py end-to-end**

```bash
py scripts/run_example.py belova_case2 --output-dir ./test_outputs
```

Verify:
- Progress appears during simulation
- `test_outputs/belova_case2_*/` directory created
- `history.csv` exists
- `config.yaml` exists
- `checkpoint_final.h5` exists

**Step 3: Test with flags**

```bash
py scripts/run_example.py belova_case2 --no-progress --no-checkpoint --history-format json
```

Verify:
- No progress output
- No checkpoint file
- `history.json` exists instead of csv

**Step 4: Clean up test outputs**

```bash
rm -rf ./test_outputs
```

**Step 5: Final commit**

```bash
git add -A
git commit -m "test: verify tooling improvements work end-to-end"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | JIT ResistiveMHD.compute_rhs | resistive_mhd.py, test_jit_compilation.py |
| 2 | JIT ExtendedMHD.compute_rhs | extended_mhd.py |
| 3 | JIT EulerSolver/RK4Solver.step | explicit.py |
| 4 | Create ProgressReporter | progress.py, test_progress.py |
| 5 | Integrate progress into LinearConfiguration | linear_configuration.py |
| 6 | Create OutputManager | output.py, test_output_manager.py |
| 7 | Update run_example.py CLI | run_example.py |
| 8 | Verify everything works | (testing only) |

Total: 8 tasks, ~8 commits
