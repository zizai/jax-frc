# run_example.py Tooling Improvements Design

**Date:** 2026-01-26
**Status:** Approved
**Issues addressed:** issues.md items 1-3

## Summary

Improve `scripts/run_example.py` with:
1. JIT compilation of simulation step for performance
2. CLI progress display during simulation
3. Automatic history saving, visualization, and output files

## Current State

- **History tracking**: Exists in `LinearConfiguration` (every 100 steps), but not saved
- **Output utilities**: Checkpoint and export functions exist but aren't integrated
- **Progress display**: None
- **JIT compilation**: Step functions not JIT-compiled

## Design

### 1. JIT Compilation Strategy

JIT-compile at the solver level, not the loop level.

**Why not JIT the whole loop?** JAX's `lax.while_loop` requires pure functions with static shapes. The current loop has dynamic termination conditions, history accumulation, and callbacks that don't fit cleanly.

**What to JIT:**
- `Solver.step()` - with `static_argnums` for model and geometry
- `Model.compute_rhs()` - the physics computation
- `TimeController.compute_dt()` - timestep computation

**Example:**
```python
from functools import partial

@partial(jax.jit, static_argnums=(3, 4))  # model, geometry are static
def step(self, state: State, dt: float, model: PhysicsModel, geometry: Geometry) -> State:
    ...
```

**Expected outcome:** Each step runs fully on GPU without Python round-trips. The Python loop orchestrates and collects history infrequently (every 100 steps).

### 2. CLI Progress Display

Add periodic updates to stderr showing simulation progress.

**Format:**
```
[Phase: acceleration] t=1.23e-6 / 5.00e-6 (24.6%) | step 1200 | dt=4.1e-9 | B_max=0.42
```

**Implementation:**
- New `ProgressReporter` class in `jax_frc/diagnostics/progress.py`
- Takes `output_interval`, `t_end`, optional probe names to display
- Has `report(state, phase_name, dt)` method
- Writes to stderr so stdout stays clean for piping
- Updates at same interval as history collection (default 100 steps)

**CLI control:**
```bash
python scripts/run_example.py belova_case2 --progress        # Enable (default)
python scripts/run_example.py belova_case2 --no-progress     # Quiet mode
```

### 3. Output Integration

Automatic output handling with configurable options.

**Directory structure:**
```
outputs/
  frc_belova_case2_20260126_143052/
    config.yaml          # Copy of input config for reproducibility
    history.csv          # Time series data
    checkpoint_final.h5  # Final state (optional)
    plots/
      time_traces.png
      fields_final.png
      overview.html      # Index linking all plots
```

**CLI options:**
```bash
# Basic run with default outputs
python scripts/run_example.py belova_case2

# Control output location
python scripts/run_example.py belova_case2 --output-dir ./my_runs

# Control what gets saved
python scripts/run_example.py belova_case2 --no-plots
python scripts/run_example.py belova_case2 --no-checkpoint
python scripts/run_example.py belova_case2 --history-format json

# Batch mode
python scripts/run_example.py --all --no-progress --no-plots
```

**Implementation:**
- Add `OutputManager` class in `jax_frc/diagnostics/output.py`
- Handles directory creation, file writing, plot generation
- After `configuration.run()` returns, extract history and save
- Print summary with paths to generated files

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `jax_frc/diagnostics/progress.py` | `ProgressReporter` class |

### Modified Files

| File | Changes |
|------|---------|
| `jax_frc/solvers/explicit.py` | Add `@jax.jit` to `step()` methods |
| `jax_frc/solvers/imex.py` | Add `@jax.jit` to `step()` methods |
| `jax_frc/solvers/semi_implicit.py` | Add `@jax.jit` to `step()` methods |
| `jax_frc/models/base.py` | Ensure `compute_rhs()` is JIT-friendly |
| `jax_frc/configurations/linear_configuration.py` | Integrate `ProgressReporter` |
| `jax_frc/diagnostics/output.py` | Add `OutputManager` class |
| `scripts/run_example.py` | Add CLI args, integrate output/progress |

## Implementation Order

**Phase 1: JIT compilation**
1. Audit existing solvers for JIT decorators
2. Add `@jax.jit` with proper `static_argnums`
3. Run tests to verify nothing breaks
4. Benchmark before/after

**Phase 2: Progress display**
1. Create `ProgressReporter` class
2. Integrate into `LinearConfiguration.run_phase()`
3. Add `--progress/--no-progress` flags

**Phase 3: Output integration**
1. Add `OutputManager` class
2. Modify run_example.py to save history, generate plots
3. Add CLI flags for output control

**Phase 4: Testing & polish**
1. Add unit tests for new classes
2. Update existing examples to verify they work
3. Add docstrings

## Testing Strategy

- **Unit tests:** `ProgressReporter` (mock stderr), `OutputManager` (verify file creation)
- **Integration test:** Run small example, verify outputs exist
- **Benchmark:** Compare step time before/after JIT

## Dependencies

None new. Uses only stdlib (`argparse`, `datetime`, `pathlib`).
