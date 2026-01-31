# AGATE Validation Snapshots Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance AGATE runner and validation tests to compare JAX-FRC against AGATE at 40 evenly-spaced time snapshots across three resolutions.

**Architecture:** Update AGATE runner to output multiple snapshots during simulation. Update validation tests to load all snapshots, run JAX-FRC to exact matching times, and compute per-field and aggregate time-series metrics. Add time-series plots to HTML reports.

**Tech Stack:** Python, h5py, numpy, matplotlib, yaml, jax_frc.validation.metrics

---

## Task 1: Update AGATE Runner Configuration

**Files:**
- Modify: `validation/utils/agate_runner.py`

**Step 1: Add num_snapshots to CASE_CONFIGS**

In `CASE_CONFIGS`, add `num_snapshots: 40` to each case:

```python
CASE_CONFIGS = {
    "orszag_tang": {
        "physics": "ideal_mhd",
        "hall": False,
        "end_time": 0.48,
        "cfl": 0.4,
        "slope_name": "mcbeta",
        "mcbeta": 1.3,
        "num_snapshots": 40,
    },
    "gem_reconnection": {
        "physics": "hall_mhd",
        "hall": True,
        "end_time": 12.0,
        "cfl": 0.4,
        "guide_field": 0.0,
        "num_snapshots": 40,
    },
}
```

**Step 2: Update get_expected_config to compute snapshot_times**

```python
def get_expected_config(case: str, resolution: list[int]) -> dict:
    """Get expected configuration for a case.

    Args:
        case: Case name ("orszag_tang" or "gem_reconnection")
        resolution: Grid resolution as [nx, ny, nz]

    Returns:
        Configuration dictionary with snapshot_times
    """
    if case not in CASE_CONFIGS:
        raise ValueError(f"Unknown case: {case}")

    base_config = CASE_CONFIGS[case].copy()
    end_time = base_config["end_time"]
    num_snapshots = base_config["num_snapshots"]

    # Compute evenly-spaced snapshot times
    snapshot_times = [
        i * end_time / (num_snapshots - 1)
        for i in range(num_snapshots)
    ]

    return {
        "case": case,
        "resolution": resolution,
        "snapshot_times": snapshot_times,
        **base_config,
    }
```

**Step 3: Run tests**

```bash
py -m pytest tests/test_agate_runner.py -v
```

**Step 4: Commit**

```bash
git add validation/utils/agate_runner.py
git commit -m "feat(agate): add num_snapshots config and snapshot_times computation"
```

---

## Task 2: Update AGATE Runner to Output Multiple Snapshots

**Files:**
- Modify: `validation/utils/agate_runner.py`

**Step 1: Update _run_orszag_tang to output snapshots**

Replace the single `outputState` call with a loop that outputs at each snapshot time:

```python
def _run_orszag_tang(resolution: list[int], output_dir: Path) -> None:
    """Run Orszag-Tang vortex simulation with multiple snapshots."""
    from agate.framework.scenario import OTVortex
    from agate.framework.roller import Roller
    from agate.framework.fileHandler import fileHandler

    config = get_expected_config("orszag_tang", resolution)
    snapshot_times = config["snapshot_times"]

    scenario = OTVortex(divClean=True, hall=False)
    roller = Roller.autodefault(
        scenario,
        ncells=resolution[:2],  # Use nx, ny from resolution
        options={"cfl": 0.4, "slopeName": "mcbeta", "mcbeta": 1.3}
    )
    roller.orient("numpy")

    output_dir.mkdir(parents=True, exist_ok=True)
    res_str = f"{resolution[0]}"
    handler = fileHandler(directory=str(output_dir), prefix=f"orszag_tang_{res_str}")
    handler.outputGrid(roller.grid)

    print(f"Running Orszag-Tang at resolution {resolution}...")
    for i, target_time in enumerate(snapshot_times):
        if i == 0:
            # Output initial state
            handler.outputState(roller.grid, roller.state, roller.time, suffix=f"state_{i:03d}")
            continue
        try:
            roller.roll(start_time=roller.time, end_time=target_time, add_stopWatch=False)
        except Exception as e:
            raise RuntimeError(f"Orszag-Tang simulation failed at t={target_time}: {e}") from e
        handler.outputState(roller.grid, roller.state, roller.time, suffix=f"state_{i:03d}")
```

**Step 2: Update _run_gem_reconnection similarly**

```python
def _run_gem_reconnection(resolution: list[int], output_dir: Path) -> None:
    """Run GEM reconnection simulation with multiple snapshots."""
    from agate.framework.scenario import ReconnectionGEM
    from agate.framework.roller import Roller
    from agate.framework.fileHandler import fileHandler

    config = get_expected_config("gem_reconnection", resolution)
    snapshot_times = config["snapshot_times"]

    scenario = ReconnectionGEM(divClean=True, hall=True, guide_field=0.0)
    roller = Roller.autodefault(
        scenario,
        ncells=resolution[:2],
        options={"cfl": 0.4}
    )
    roller.orient("numpy")

    output_dir.mkdir(parents=True, exist_ok=True)
    res_str = f"{resolution[0]}"
    handler = fileHandler(directory=str(output_dir), prefix=f"gem_reconnection_{res_str}")
    handler.outputGrid(roller.grid)

    print(f"Running GEM reconnection at resolution {resolution}...")
    for i, target_time in enumerate(snapshot_times):
        if i == 0:
            handler.outputState(roller.grid, roller.state, roller.time, suffix=f"state_{i:03d}")
            continue
        try:
            roller.roll(start_time=roller.time, end_time=target_time, add_stopWatch=False)
        except Exception as e:
            raise RuntimeError(f"GEM simulation failed at t={target_time}: {e}") from e
        handler.outputState(roller.grid, roller.state, roller.time, suffix=f"state_{i:03d}")
```

**Step 3: Update run_agate_simulation signature**

Change `resolution: int` to `resolution: list[int]`:

```python
def run_agate_simulation(
    case: Literal["orszag_tang", "gem_reconnection"],
    resolution: list[int],
    output_dir: Path,
    overwrite: bool = False
) -> Path:
```

**Step 4: Update is_cache_valid for new config format**

```python
def is_cache_valid(case: str, resolution: list[int], output_dir: Path) -> bool:
    res_str = f"{resolution[0]}"
    config_file = output_dir / f"{case}_{res_str}.config.yaml"
    # ... rest unchanged but check num_snapshots too
```

**Step 5: Run tests**

```bash
py -m pytest tests/test_agate_runner.py -v
```

**Step 6: Commit**

```bash
git add validation/utils/agate_runner.py
git commit -m "feat(agate): output 40 snapshots during simulation"
```

---

## Task 3: Update Config YAML Format

**Files:**
- Modify: `validation/utils/agate_runner.py`

**Step 1: Update _save_config to include snapshot_times**

```python
def _save_config(case: str, resolution: list[int], output_dir: Path) -> None:
    """Save configuration to YAML file."""
    config = get_expected_config(case, resolution)
    config["agate_version"] = _get_agate_version()
    config["generated_at"] = datetime.now(timezone.utc).isoformat()

    res_str = f"{resolution[0]}"
    config_file = output_dir / f"{case}_{res_str}.config.yaml"
    with open(config_file, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False)
```

**Step 2: Commit**

```bash
git add validation/utils/agate_runner.py
git commit -m "feat(agate): save snapshot_times in config YAML"
```

---

## Task 4: Add Snapshot Loading Functions

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`
- Modify: `validation/cases/regression/gem_reconnection.py`

**Step 1: Add load_agate_config function**

```python
def load_agate_config(case: str, resolution: list[int]) -> dict:
    """Load AGATE config including snapshot_times."""
    loader = AgateDataLoader()
    loader.ensure_files(case, resolution[0])
    case_dir = Path(loader.cache_dir) / case / str(resolution[0])
    config_path = case_dir / f"{case}_{resolution[0]}.config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)
```

**Step 2: Add load_agate_snapshot function**

```python
def load_agate_snapshot(case: str, resolution: list[int], snapshot_idx: int) -> dict:
    """Load spatial fields from a specific AGATE snapshot.

    Args:
        case: Case identifier
        resolution: Grid resolution as [nx, ny, nz]
        snapshot_idx: Index of snapshot (0 to num_snapshots-1)

    Returns:
        Dict with keys: density, momentum, magnetic_field, pressure
    """
    loader = AgateDataLoader()
    loader.ensure_files(case, resolution[0])
    case_dir = Path(loader.cache_dir) / case / str(resolution[0])

    state_path = case_dir / f"{case}_{resolution[0]}.state_{snapshot_idx:03d}.h5"
    if not state_path.exists():
        raise FileNotFoundError(f"Snapshot {snapshot_idx} not found: {state_path}")

    with h5py.File(state_path, "r") as f:
        sub = f["subID0"]
        vec = sub["vector"][:]

    rho, p, v, B = _parse_state_vector(vec)

    # Strip ghost cells and transpose (same as load_agate_fields)
    rho = rho[2:-2, 2:-2, :]
    p = p[2:-2, 2:-2, :]
    v = v[2:-2, 2:-2, :, :]
    B = B[2:-2, 2:-2, :, :]

    rho = np.transpose(rho, (0, 2, 1))
    p = np.transpose(p, (0, 2, 1))
    v = np.transpose(v, (0, 2, 1, 3))
    B = np.transpose(B, (0, 2, 1, 3))

    v = v[..., [0, 2, 1]]
    B = B[..., [0, 2, 1]]

    mom = rho[..., None] * v

    return {
        "density": rho,
        "momentum": mom,
        "magnetic_field": B,
        "pressure": p,
    }
```

**Step 3: Commit**

```bash
git add validation/cases/regression/orszag_tang.py validation/cases/regression/gem_reconnection.py
git commit -m "feat(validation): add snapshot loading functions"
```

---

## Task 5: Add JAX-FRC Snapshot Simulation Function

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`
- Modify: `validation/cases/regression/gem_reconnection.py`

**Step 1: Add run_simulation_with_snapshots function**

```python
def run_simulation_with_snapshots(
    cfg: dict,
    snapshot_times: list[float]
) -> tuple[list, any, dict]:
    """Run JAX-FRC, capturing state at each snapshot time.

    Args:
        cfg: Configuration dict
        snapshot_times: List of times to capture state

    Returns:
        Tuple of (states_list, geometry, history)
    """
    # Setup (same as run_simulation)
    config = OrszagTangConfiguration(
        nx=cfg["nx"],
        nz=cfg["nz"],
        B0=cfg.get("B0", 1.0 / math.sqrt(4.0 * math.pi)),
        eta=0.0,
    )
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = ResistiveMHD(eta=0.0, advection_scheme="ct")
    solver = Solver.create({"type": "rk4"})

    dt = cfg["dt"]
    states = []
    history = {"times": [], "metrics": []}

    for target_time in snapshot_times:
        # Step until we reach target_time
        while state.time < target_time - 1e-12:
            step_dt = min(dt, target_time - state.time)
            state = solver.step_checked(state, step_dt, model, geometry)

        # Capture state at this snapshot
        states.append(state)
        metrics = compute_metrics(
            state.n, state.p, state.v, state.B,
            geometry.dx, geometry.dy, geometry.dz
        )
        history["times"].append(float(state.time))
        history["metrics"].append(metrics)

    return states, geometry, history
```

**Step 2: Commit**

```bash
git add validation/cases/regression/orszag_tang.py validation/cases/regression/gem_reconnection.py
git commit -m "feat(validation): add snapshot simulation function"
```

---

## Task 6: Add Per-Field Metrics Functions

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`

**Step 1: Add compute_field_metrics function**

```python
def compute_field_metrics(jax_field: np.ndarray, agate_field: np.ndarray) -> dict:
    """Compute per-field metrics: L2 error, max abs error, relative error.

    Args:
        jax_field: JAX field array
        agate_field: AGATE reference field array

    Returns:
        Dict with l2_error, max_abs_error, relative_error
    """
    from jax_frc.validation.metrics import l2_error

    # Normalize fields for comparison
    def normalize(field):
        max_val = np.max(np.abs(field))
        return field / max_val if max_val > 1e-10 else field

    jax_norm = normalize(jax_field)
    agate_norm = normalize(agate_field)

    diff = jax_norm - agate_norm
    l2 = float(l2_error(jax_norm, agate_norm))
    max_abs = float(np.max(np.abs(diff)))

    agate_mag = np.max(np.abs(agate_norm))
    rel = float(np.sqrt(np.mean(diff**2)) / agate_mag) if agate_mag > 1e-10 else 0.0

    return {
        "l2_error": l2,
        "max_abs_error": max_abs,
        "relative_error": rel,
    }
```

**Step 2: Add validate_all_snapshots function**

```python
def validate_all_snapshots(
    jax_states: list,
    case: str,
    resolution: list[int],
    snapshot_times: list[float]
) -> list[dict]:
    """Compare JAX-FRC vs AGATE at each snapshot.

    Args:
        jax_states: List of JAX states at each snapshot
        case: Case name
        resolution: Grid resolution
        snapshot_times: List of snapshot times

    Returns:
        List of dicts with time and per-field errors
    """
    all_errors = []
    for i, (jax_state, t) in enumerate(zip(jax_states, snapshot_times)):
        agate_fields = load_agate_snapshot(case, resolution, snapshot_idx=i)

        field_errors = {}
        for field_name, jax_getter, agate_key in [
            ("density", lambda s: np.asarray(s.n), "density"),
            ("momentum", lambda s: np.asarray(s.n)[..., None] * np.asarray(s.v), "momentum"),
            ("magnetic_field", lambda s: np.asarray(s.B), "magnetic_field"),
            ("pressure", lambda s: np.asarray(s.p), "pressure"),
        ]:
            jax_field = jax_getter(jax_state)
            agate_field = agate_fields[agate_key]
            field_errors[field_name] = compute_field_metrics(jax_field, agate_field)

        all_errors.append({"time": t, "errors": field_errors})

    return all_errors
```

**Step 3: Commit**

```bash
git add validation/cases/regression/orszag_tang.py
git commit -m "feat(validation): add per-field metrics and snapshot validation"
```

---

## Task 7: Add Aggregate Time-Series Metrics

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`

**Step 1: Add compute_aggregate_metrics function**

```python
def compute_aggregate_metrics(
    jax_history: dict,
    agate_times: np.ndarray,
    agate_metrics: dict
) -> dict:
    """Compute time-series comparison metrics for aggregate quantities.

    Args:
        jax_history: Dict with 'times' and 'metrics' lists from JAX
        agate_times: Array of AGATE snapshot times
        agate_metrics: Dict of metric_name -> array of values

    Returns:
        Dict of metric_name -> {mean_residual, std_residual, relative_error}
    """
    results = {}

    # Extract JAX time-series
    jax_times = np.array(jax_history["times"])
    jax_metrics_by_key = {}
    for key in ["kinetic_fraction", "magnetic_fraction", "mean_energy_density",
                "enstrophy_density", "normalized_max_current"]:
        jax_metrics_by_key[key] = np.array([m[key] for m in jax_history["metrics"]])

    for key in jax_metrics_by_key:
        jax_vals = jax_metrics_by_key[key]
        agate_vals = agate_metrics[key]

        # Interpolate to common times if needed
        if len(jax_vals) != len(agate_vals):
            # Use AGATE times as reference
            jax_interp = np.interp(agate_times, jax_times, jax_vals)
            residuals = jax_interp - agate_vals
        else:
            residuals = jax_vals - agate_vals

        mean_resid = float(np.mean(residuals))
        std_resid = float(np.std(residuals))
        agate_norm = float(np.linalg.norm(agate_vals))
        rel_error = float(np.linalg.norm(residuals) / agate_norm) if agate_norm > 1e-10 else 0.0

        results[key] = {
            "mean_residual": mean_resid,
            "std_residual": std_resid,
            "relative_error": rel_error,
        }

    return results
```

**Step 2: Commit**

```bash
git add validation/cases/regression/orszag_tang.py
git commit -m "feat(validation): add aggregate time-series metrics"
```

---

## Task 8: Update Main Validation Loop

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`

**Step 1: Update RESOLUTIONS to use [nx, ny, nz] format**

```python
RESOLUTIONS = ([128, 128, 1], [256, 256, 1], [512, 512, 1])
QUICK_RESOLUTIONS = ([128, 128, 1],)
```

**Step 2: Update main() to use snapshot validation**

Update the main loop to:
1. Load AGATE config with snapshot_times
2. Run JAX-FRC with run_simulation_with_snapshots
3. Call validate_all_snapshots
4. Compute aggregate metrics
5. Print per-field and aggregate tables

**Step 3: Commit**

```bash
git add validation/cases/regression/orszag_tang.py
git commit -m "feat(validation): update main loop for snapshot validation"
```

---

## Task 9: Add Time-Series Plots

**Files:**
- Modify: `validation/utils/plots.py`

**Step 1: Add create_timeseries_comparison_plot function**

```python
def create_timeseries_comparison_plot(
    jax_times: np.ndarray,
    jax_values: np.ndarray,
    agate_times: np.ndarray,
    agate_values: np.ndarray,
    metric_name: str,
    resolution: list[int]
) -> plt.Figure:
    """Create time-series comparison plot for an aggregate metric.

    Args:
        jax_times: JAX simulation times
        jax_values: JAX metric values
        agate_times: AGATE reference times
        agate_values: AGATE metric values
        metric_name: Name of the metric
        resolution: Grid resolution

    Returns:
        matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: JAX vs AGATE
    ax1.plot(jax_times, jax_values, 'b-', label='JAX-FRC', linewidth=2)
    ax1.plot(agate_times, agate_values, 'r--', label='AGATE', linewidth=2)
    ax1.set_ylabel(metric_name)
    ax1.legend()
    ax1.set_title(f"{metric_name} Evolution (Resolution {resolution[0]})")
    ax1.grid(True, alpha=0.3)

    # Bottom: Residual
    jax_interp = np.interp(agate_times, jax_times, jax_values)
    residuals = jax_interp - agate_values
    ax2.plot(agate_times, residuals, 'g-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Residual (JAX - AGATE)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

**Step 2: Add create_field_error_evolution_plot function**

```python
def create_field_error_evolution_plot(
    snapshot_errors: list[dict],
    threshold: float,
    resolution: list[int]
) -> plt.Figure:
    """Create plot of per-field L2 error vs snapshot time.

    Args:
        snapshot_errors: List of {time, errors} dicts
        threshold: Error threshold
        resolution: Grid resolution

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    times = [s["time"] for s in snapshot_errors]
    fields = list(snapshot_errors[0]["errors"].keys())

    for field in fields:
        errors = [s["errors"][field]["l2_error"] for s in snapshot_errors]
        ax.plot(times, errors, '-o', label=field, markersize=3)

    ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    ax.set_xlabel('Time')
    ax.set_ylabel('L2 Error')
    ax.set_title(f'Per-Field L2 Error Evolution (Resolution {resolution[0]})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
```

**Step 3: Commit**

```bash
git add validation/utils/plots.py
git commit -m "feat(plots): add time-series and error evolution plots"
```

---

## Task 10: Update Reporting Tables

**Files:**
- Modify: `validation/utils/reporting.py`

**Step 1: Add print_aggregate_metrics_table function**

```python
def print_aggregate_metrics_table(aggregate_metrics: dict, threshold: float) -> None:
    """Print aggregate time-series metrics table.

    Args:
        aggregate_metrics: Dict of metric_name -> {mean_residual, std_residual, relative_error}
        threshold: Relative error threshold
    """
    print("  Aggregate Time-Series Statistics:")
    print("    Metric           Mean Resid   Std Resid   Rel Error   Status")
    print("    " + "-" * 64)

    for name, stats in aggregate_metrics.items():
        mean_r = stats["mean_residual"]
        std_r = stats["std_residual"]
        rel_e = stats["relative_error"]
        passed = rel_e <= threshold
        status = "PASS" if passed else "FAIL"
        print(f"    {name:<16} {mean_r:>10.4f}   {std_r:>9.4f}   {rel_e:>9.3f}   {status}")
```

**Step 2: Update print_field_l2_table for new format**

Add max_abs_error and relative_error columns:

```python
def print_field_l2_table(field_errors: dict, threshold: float) -> None:
    """Print per-field metrics table.

    Args:
        field_errors: Dict of field_name -> {l2_error, max_abs_error, relative_error}
        threshold: Error threshold
    """
    print("  Per-Field Statistics:")
    print("    Field            L2 Error   Max Abs Error   Rel Error   Status")
    print("    " + "-" * 60)

    for name, stats in field_errors.items():
        if isinstance(stats, dict):
            l2 = stats["l2_error"]
            max_abs = stats["max_abs_error"]
            rel = stats["relative_error"]
        else:
            l2 = stats
            max_abs = 0.0
            rel = 0.0
        passed = l2 <= threshold
        status = "PASS" if passed else "FAIL"
        print(f"    {name:<16} {l2:>8.3f}   {max_abs:>13.3f}   {rel:>9.3f}   {status}")
```

**Step 3: Commit**

```bash
git add validation/utils/reporting.py
git commit -m "feat(reporting): add aggregate metrics table and update field table"
```

---

## Task 11: Update Quick Test Mode

**Files:**
- Modify: `validation/cases/regression/orszag_tang.py`
- Modify: `validation/cases/regression/gem_reconnection.py`

**Step 1: Add QUICK_NUM_SNAPSHOTS constant**

```python
QUICK_NUM_SNAPSHOTS = 5  # t=0, 0.25*end, 0.5*end, 0.75*end, end
```

**Step 2: Update main() to use fewer snapshots in quick mode**

```python
def get_quick_snapshot_times(end_time: float) -> list[float]:
    """Get 5 evenly-spaced snapshot times for quick test mode."""
    return [i * end_time / 4 for i in range(5)]
```

In main():
```python
if quick_test:
    snapshot_times = get_quick_snapshot_times(config["end_time"])
else:
    snapshot_times = agate_config["snapshot_times"]
```

**Step 3: Ensure all metrics still pass in quick mode**

Same thresholds apply - quick mode just uses fewer snapshots.

**Step 4: Commit**

```bash
git add validation/cases/regression/orszag_tang.py validation/cases/regression/gem_reconnection.py
git commit -m "feat(validation): update quick test mode to use 5 snapshots"
```

---

## Task 12: Apply Changes to gem_reconnection.py

**Files:**
- Modify: `validation/cases/regression/gem_reconnection.py`

Apply all the same changes from Tasks 4-8 and 11 to gem_reconnection.py:
- Add load_agate_config, load_agate_snapshot
- Add run_simulation_with_snapshots
- Add compute_field_metrics, validate_all_snapshots
- Add compute_aggregate_metrics
- Update RESOLUTIONS format
- Update main() loop
- Update quick test mode

**Commit:**

```bash
git add validation/cases/regression/gem_reconnection.py
git commit -m "feat(validation): apply snapshot validation to GEM reconnection"
```

---

## Task 13: Add Unit Tests

**Files:**
- Create: `tests/test_agate_snapshots.py`

**Step 1: Write test for snapshot_times computation**

```python
def test_snapshot_times_evenly_spaced():
    """Verify snapshot times are evenly distributed."""
    from validation.utils.agate_runner import get_expected_config

    config = get_expected_config("orszag_tang", [256, 256, 1])
    times = config["snapshot_times"]

    assert len(times) == 40
    assert times[0] == 0.0
    assert abs(times[-1] - 0.48) < 1e-10

    # Check even spacing
    dt = times[1] - times[0]
    for i in range(1, len(times)):
        assert abs(times[i] - times[i-1] - dt) < 1e-10
```

**Step 2: Write test for quick mode snapshot times**

```python
def test_quick_snapshot_times():
    """Verify quick mode uses 5 snapshots."""
    from validation.cases.regression.orszag_tang import get_quick_snapshot_times

    times = get_quick_snapshot_times(0.48)
    assert len(times) == 5
    assert times[0] == 0.0
    assert times[-1] == 0.48
    assert times[2] == 0.24  # midpoint
```

**Step 3: Commit**

```bash
git add tests/test_agate_snapshots.py
git commit -m "test(validation): add unit tests for snapshot functionality"
```

---

## Task 14: Run Full Validation and Verify

**Files:** None (verification only)

**Step 1: Run full Orszag-Tang validation**

```bash
py -m validation.cases.regression.orszag_tang
```

**Step 2: Verify console output format**

Expected:
```
Resolution [256, 256, 1]:
  Validating 40 snapshots...

  Per-Field Statistics:
    Field            L2 Error   Max Abs Error   Rel Error   Status
    ------------------------------------------------------------
    density          0.082      0.182           0.145       PASS
    ...

  Aggregate Time-Series Statistics:
    Metric           Mean Resid   Std Resid   Rel Error   Status
    ----------------------------------------------------------------
    total_energy     0.0012       0.0008      0.015       PASS
    ...

  Overall: PASS
```

**Step 3: Verify HTML report**

Open generated report and check:
- [ ] Metrics table shows all field L2 errors
- [ ] Time-series plots for aggregate metrics
- [ ] Per-field error evolution plot
- [ ] Field comparison at t=0, t_mid, t_final

**Step 4: Run quick test**

```bash
py -m validation.cases.regression.orszag_tang --quick
```

Verify it uses 5 snapshots and all metrics pass.

**Step 5: Final commit**

```bash
git add -A
git commit -m "feat(validation): complete AGATE snapshot validation implementation"
```

---

## Verification Checklist

After all tasks complete:

- [ ] AGATE runner outputs 40 snapshots per simulation
- [ ] Config YAML contains resolution as [nx, ny, nz] and snapshot_times array
- [ ] Validation tests compare at all 40 snapshots
- [ ] Per-field metrics: L2 error, max abs error, relative error
- [ ] Aggregate metrics: mean residual, std residual, relative error
- [ ] Time-series plots in HTML report
- [ ] Quick test mode uses 5 snapshots with same pass criteria
- [ ] All unit tests pass
- [ ] Both Orszag-Tang and GEM reconnection cases work

