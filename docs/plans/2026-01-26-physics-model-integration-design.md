# Physics Model Integration and Belova Validation Design

**Date:** 2026-01-26
**Status:** Draft
**Authors:** Design session with Claude

---

## Overview

This design integrates the existing physics models (Resistive MHD, Extended MHD, Hybrid Kinetic) with the scenario runner framework, then adds quantitative validation tests against Belova et al. (arXiv:2501.03425v1) merging results.

## Goals

- Connect physics models and solvers to the scenario runner
- Enable real physics evolution during scenario execution
- Validate merging simulations against published results
- Compare MHD vs Hybrid model behavior

## Non-Goals

- Multi-model scenarios (different models per phase) - future extension
- 3D simulations
- Exact numerical reproduction of HYM code results

---

## Part A: Physics Integration

### Scenario Class Changes

Add two required fields to `Scenario`:

```python
@dataclass
class Scenario:
    name: str
    phases: List[Phase]
    geometry: Geometry
    initial_state: Optional[State]
    physics_model: PhysicsModel      # NEW
    solver: Solver                   # NEW
    dt: float
    config: Dict[str, dict] = field(default_factory=dict)
```

### Physics Stepping in _run_phase

Replace the current time-increment loop:

```python
# OLD: Just increments time
t += self.dt
state = state.replace(time=t, step=state.step + 1)
```

With actual physics stepping:

```python
# NEW: Actual physics stepping
state = phase.step_hook(state, self.geometry, t)  # Apply phase BCs
state = self.solver.step(state, self.dt, self.physics_model, self.geometry)
t = float(state.time)  # Solver updates time internally
```

The solver's `step()` already handles:
- Time and step increment
- Calling `model.apply_constraints()`

### Updated Merging Examples

Add `model_type` parameter to example functions:

```python
from jax_frc.models import PhysicsModel
from jax_frc.solvers import Solver

def belova_case2(model_type: str = "resistive_mhd") -> Scenario:
    """Small FRC merging without compression (paper Fig. 3-4).

    Args:
        model_type: "resistive_mhd" or "hybrid_kinetic" for comparison
    """
    geometry = create_default_geometry(rc=1.0, zc=3.0, nr=64, nz=256)
    initial_state = create_initial_frc(
        geometry, s_star=20.0, elongation=1.5, xs=0.53, beta_s=0.2
    )

    # Create physics model from config
    model_config = {
        "type": model_type,
        "resistivity": {"type": "chodura", "eta_0": 1e-6, "eta_anom": 1e-3}
    }
    physics_model = PhysicsModel.create(model_config)

    # Solver - RK4 for accuracy
    solver = Solver.create({"type": "rk4"})

    merge_phase = MergingPhase(
        name="merge_small_frc",
        transition=any_of(separation_below(0.3, geometry), timeout(15.0)),
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

---

## Part B: Validation Infrastructure

### Time History Tracking

Add diagnostics recording to `Scenario`:

```python
@dataclass
class Scenario:
    # ... existing fields ...
    diagnostics: List[Probe] = field(default_factory=list)
    output_interval: int = 100  # Record every N steps
```

Add history to `PhaseResult`:

```python
@dataclass
class PhaseResult:
    name: str
    initial_state: State
    final_state: State
    start_time: float
    end_time: float
    termination: str
    history: Dict[str, List[float]] = field(default_factory=dict)  # NEW
```

Recording in `_run_phase`:

```python
history = {probe.name: [] for probe in self.diagnostics}
history["time"] = []

while True:
    complete, reason = phase.is_complete(state, t)
    if complete:
        termination = reason
        break

    state = phase.step_hook(state, self.geometry, t)
    state = self.solver.step(state, self.dt, self.physics_model, self.geometry)
    t = float(state.time)

    # Record diagnostics at intervals
    if state.step % self.output_interval == 0:
        history["time"].append(t)
        for probe in self.diagnostics:
            history[probe.name].append(probe.measure(state, self.geometry))
```

---

## Validation Test Cases

### Test File: tests/test_belova_validation.py

#### Case 2: Small FRC Complete Merge

```python
class TestBelovaCase2:
    """Small FRC merging - expect complete merge by ~5-7 tA."""

    def test_complete_merge_mhd(self):
        """MHD: separation reaches near-zero by 7 tA."""
        scenario = belova_case2(model_type="resistive_mhd")
        result = scenario.run()

        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        assert final_diag["separation_dz"] < 0.5  # Near-zero separation
        assert result.total_time <= 10.0          # Should complete, not timeout

    def test_elongation_increase(self):
        """Elongation should increase ~1.7x after complete merge."""
        scenario = belova_case2()
        initial_diag = MergingDiagnostics().compute(
            scenario.initial_state, scenario.geometry
        )

        result = scenario.run()
        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        ratio = final_diag["elongation"] / initial_diag["elongation"]
        assert 1.4 < ratio < 2.0  # Paper: ~1.7x for complete merge

    def test_merging_timescale(self):
        """Separation should drop to <0.5 between 5-7 tA."""
        scenario = belova_case2()
        scenario.diagnostics = [MergingDiagnostics()]
        result = scenario.run()

        times = result.phase_results[0].history["time"]
        separations = result.phase_results[0].history["merging"]

        merge_time = next(t for t, sep in zip(times, separations) if sep < 0.5)
        assert 5.0 <= merge_time <= 7.0
```

#### Case 1: Large FRC Doublet Formation

```python
class TestBelovaCase1:
    """Large FRC - expect partial merge, doublet formation."""

    def test_doublet_formation_mhd(self):
        """MHD: should NOT fully merge, final dZ ~ 40 (normalized)."""
        scenario = belova_case1(model_type="resistive_mhd")
        result = scenario.run()

        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        assert final_diag["separation_dz"] > 0.3   # NOT fully merged
        assert len(final_diag["null_positions"]) == 2  # Two nulls = doublet

    def test_elongation_doublet(self):
        """Elongation should increase ~2.3x for doublet."""
        scenario = belova_case1()
        initial_diag = MergingDiagnostics().compute(
            scenario.initial_state, scenario.geometry
        )

        result = scenario.run()
        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        ratio = final_diag["elongation"] / initial_diag["elongation"]
        assert 1.8 < ratio < 2.8  # Paper: ~2.3x for doublet
```

#### Case 4: Compression-Driven Merge

```python
class TestBelovaCase4:
    """Large FRC with compression - expect complete merge."""

    def test_compression_enables_merge(self):
        """Compression should enable complete merge by 20-25 tA."""
        scenario = belova_case4(model_type="resistive_mhd")
        result = scenario.run()

        final_diag = MergingDiagnostics().compute(
            result.final_state, scenario.geometry
        )

        # Should fully merge (unlike Case 1 without compression)
        assert final_diag["separation_dz"] < 0.5
        assert 15.0 <= result.total_time <= 30.0
```

#### MHD vs Hybrid Comparison

```python
class TestMHDvsHybridComparison:
    """Compare MHD and Hybrid results per Belova et al. Section 4."""

    @pytest.fixture
    def case2_both_models(self):
        """Run Case 2 with both MHD and Hybrid."""
        mhd_scenario = belova_case2(model_type="resistive_mhd")
        hybrid_scenario = belova_case2(model_type="hybrid_kinetic")
        return mhd_scenario.run(), hybrid_scenario.run(), mhd_scenario.geometry

    def test_global_dynamics_similar(self, case2_both_models):
        """Paper: global dynamics are similar between MHD and Hybrid."""
        mhd, hybrid, geometry = case2_both_models

        mhd_diag = MergingDiagnostics().compute(mhd.final_state, geometry)
        hybrid_diag = MergingDiagnostics().compute(hybrid.final_state, geometry)

        # Final elongation should be similar (~2.5 for both)
        assert abs(mhd_diag["elongation"] - hybrid_diag["elongation"]) < 0.5

    def test_hybrid_less_complete_merge(self, case2_both_models):
        """Paper: Hybrid shows less complete merging than MHD."""
        mhd, hybrid, geometry = case2_both_models

        mhd_sep = MergingDiagnostics().compute(mhd.final_state, geometry)["separation_dz"]
        hybrid_sep = MergingDiagnostics().compute(hybrid.final_state, geometry)["separation_dz"]

        # Hybrid should have slightly larger final separation
        assert hybrid_sep >= mhd_sep * 0.8

    def test_hybrid_merging_slower(self, case2_both_models):
        """Paper: Hybrid merges by ~6-7 tA vs MHD ~5 tA."""
        mhd, hybrid, geometry = case2_both_models

        assert hybrid.total_time >= mhd.total_time * 0.9
```

---

## Validation Criteria Summary

From Belova et al. paper:

| Case | Parameters | Expected Outcome |
|------|------------|------------------|
| 1 | S*=25.6, E=2.9, xs=0.69, no compression | Partial merge, doublet (dZ~40), E increase ~2.3x |
| 2 | S*=20, E=1.5, xs=0.53, no compression | Complete merge by ~5-7 tA, E increase ~1.7x |
| 4 | Case 1 + compression (1.5x mirror) | Complete merge by ~20-25 tA |

MHD vs Hybrid differences:
- Global dynamics similar
- Hybrid shows less complete merging
- Hybrid merging is slower
- Hybrid has reduced axial oscillations (FLR viscosity)

---

## Implementation Order

1. **Scenario physics integration**
   - Add `physics_model` and `solver` to `Scenario`
   - Update `_run_phase` to call solver
   - Update merging examples with model_type parameter

2. **History tracking**
   - Add `diagnostics` list and `output_interval` to `Scenario`
   - Add `history` dict to `PhaseResult`
   - Record diagnostics in time loop

3. **Validation tests**
   - Create `tests/test_belova_validation.py`
   - Implement Case 2 tests (fastest to run)
   - Implement Case 1 tests (doublet formation)
   - Implement Case 4 tests (compression)
   - Implement MHD vs Hybrid comparison tests

---

## Files to Modify

- `jax_frc/scenarios/scenario.py` - Add physics_model, solver, diagnostics, history recording
- `jax_frc/scenarios/phase.py` - Add history to PhaseResult
- `examples/merging_examples.py` - Add model_type parameter, create model/solver

## Files to Create

- `tests/test_belova_validation.py` - Quantitative validation tests
