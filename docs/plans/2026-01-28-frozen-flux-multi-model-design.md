# Frozen Flux Multi-Model Comparison Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend frozen_flux validation to compare ResistiveMHD, ExtendedMHD, and analytic solutions with comprehensive visualizations.

**Architecture:** Sequential simulation runs with combined report. Both models use same initial conditions. Report includes side-by-side metrics table, 3-panel GIF animations, and static multi-frame summary plots.

**Tech Stack:** JAX, matplotlib, numpy, FuncAnimation for GIFs

---

## Task 1: Add Multi-Model Simulation Runner

**Files:**
- Modify: `validation/cases/analytic/frozen_flux.py`

**Step 1: Add run_multi_model_simulation function**

```python
def run_multi_model_simulation(cfg: dict, n_snapshots: int = 50) -> dict:
    """Run simulation with multiple models and collect snapshots.

    Returns dict with keys: times, analytic, ResistiveMHD, ExtendedMHD
    Each model entry contains list of B field snapshots.
    """
    from jax_frc.configurations.frozen_flux import FrozenFluxConfiguration

    geometry = None
    results = {"times": [], "analytic": []}

    model_configs = {
        "ResistiveMHD": {
            "model_type": "resistive_mhd",
            "advection_scheme": "ct",
            "eta": 0.0,
        },
        "ExtendedMHD": {
            "model_type": "extended_mhd",
            "eta": 0.0,
            # Hall and electron pressure enabled by default
        },
    }

    for model_name, model_params in model_configs.items():
        config = FrozenFluxConfiguration(
            nx=cfg["nx"], ny=cfg["ny"], nz=cfg["nz"],
            domain_extent=cfg["domain_extent"],
            z_extent=cfg["z_extent"],
            loop_x0=cfg["loop_x0"],
            loop_y0=cfg["loop_y0"],
            loop_radius=cfg["loop_radius"],
            loop_amplitude=cfg["loop_amplitude"],
            omega=cfg["omega"],
            **model_params,
        )

        if geometry is None:
            geometry = config.build_geometry()

        state = config.build_initial_state(geometry)
        model = config.build_model()
        solver = RK4Solver()

        t_end = cfg["t_end"]
        dt = cfg["dt"]
        n_steps = int(t_end / dt)
        snapshot_interval = max(1, n_steps // n_snapshots)

        snapshots = [np.array(state.B)]

        # Compute analytic only once (first model)
        if not results["times"]:
            results["times"] = [0.0]
            results["analytic"] = [np.array(config.analytic_solution(geometry, 0.0))]

        for step in range(n_steps):
            state = solver.step(state, dt, model, geometry)
            if (step + 1) % snapshot_interval == 0 or step == n_steps - 1:
                snapshots.append(np.array(state.B))
                if model_name == "ResistiveMHD":  # Only record times once
                    results["times"].append(float(state.time))
                    results["analytic"].append(
                        np.array(config.analytic_solution(geometry, state.time))
                    )

        results[model_name] = snapshots

    return results, geometry, config
```

**Step 2: Run to verify function compiles**

Run: `py -c "from validation.cases.analytic.frozen_flux import run_multi_model_simulation"`

---

## Task 2: Add 3-Panel GIF Animation

**Files:**
- Modify: `validation/cases/analytic/frozen_flux.py`

**Step 1: Add create_3model_2d_animation function**

```python
def create_3model_2d_animation(times, results, geometry, save_path):
    """Create 3-panel 2D animation: Analytic | ResistiveMHD | ExtendedMHD."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    z_idx = geometry.nz // 2
    x = np.array(geometry.x_grid[:, :, z_idx])
    y = np.array(geometry.y_grid[:, :, z_idx])

    # Precompute |B| for all snapshots
    B_mag = {}
    for name in ["analytic", "ResistiveMHD", "ExtendedMHD"]:
        B_mag[name] = [
            np.sqrt(np.sum(B[:, :, z_idx, :]**2, axis=-1))
            for B in results[name]
        ]

    # Global colorbar limits
    vmin = 0
    vmax = max(np.max(B_mag["analytic"][0]),
               np.max(B_mag["ResistiveMHD"][0]),
               np.max(B_mag["ExtendedMHD"][0])) * 1.1
    levels = np.linspace(vmin, vmax, 21)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = ["Analytic", "ResistiveMHD", "ExtendedMHD"]

    plt.tight_layout()

    def update(frame):
        for i, (ax, name) in enumerate(zip(axes, ["analytic", "ResistiveMHD", "ExtendedMHD"])):
            ax.clear()
            ax.contourf(x, y, B_mag[name][frame], levels=levels, cmap='viridis')
            ax.set_title(f'{titles[i]} |B| (t={times[frame]:.3f}s)')
            ax.set_xlabel('x [m]')
            ax.set_ylabel('y [m]')
            ax.set_aspect('equal')
        return axes

    anim = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
    anim.save(save_path, writer='pillow', fps=10)
    plt.close(fig)
    print(f"  Saved 3-model 2D animation: {save_path}")
```

---

## Task 3: Add 1D Overlay GIF Animation

**Files:**
- Modify: `validation/cases/analytic/frozen_flux.py`

**Step 1: Add create_3model_1d_animation function**

```python
def create_3model_1d_animation(times, results, geometry, save_path):
    """Create 1D overlay animation: all three models on same axes."""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    z_idx = geometry.nz // 2
    j_mid = geometry.ny // 2
    x_1d = np.array(geometry.x_grid[:, j_mid, z_idx])

    # Precompute 1D profiles
    B_1d = {}
    for name in ["analytic", "ResistiveMHD", "ExtendedMHD"]:
        B_1d[name] = [
            np.sqrt(np.sum(B[:, j_mid, z_idx, :]**2, axis=-1))
            for B in results[name]
        ]

    ymax = max(np.max(B_1d["analytic"][0]),
               np.max(B_1d["ResistiveMHD"][0]),
               np.max(B_1d["ExtendedMHD"][0])) * 1.2

    fig, ax = plt.subplots(figsize=(10, 5))

    line_ana, = ax.plot(x_1d, B_1d["analytic"][0], 'b-', lw=2, label='Analytic')
    line_res, = ax.plot(x_1d, B_1d["ResistiveMHD"][0], 'r--', lw=2, label='ResistiveMHD')
    line_ext, = ax.plot(x_1d, B_1d["ExtendedMHD"][0], 'g:', lw=2, label='ExtendedMHD')

    ax.set_xlim(x_1d.min(), x_1d.max())
    ax.set_ylim(0, ymax)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('|B| [T]')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    title = ax.set_title(f'|B| along y=0 (t={times[0]:.3f}s)')

    plt.tight_layout()

    def update(frame):
        line_ana.set_ydata(B_1d["analytic"][frame])
        line_res.set_ydata(B_1d["ResistiveMHD"][frame])
        line_ext.set_ydata(B_1d["ExtendedMHD"][frame])
        title.set_text(f'|B| along y=0 (t={times[frame]:.3f}s)')
        return line_ana, line_res, line_ext, title

    anim = FuncAnimation(fig, update, frames=len(times), interval=100, blit=False)
    anim.save(save_path, writer='pillow', fps=10)
    plt.close(fig)
    print(f"  Saved 3-model 1D animation: {save_path}")
```

---

## Task 4: Add Static Multi-Frame Summary Plot

**Files:**
- Modify: `validation/cases/analytic/frozen_flux.py`

**Step 1: Add create_multi_frame_summary function**

```python
def create_multi_frame_summary(times, results, geometry, save_path):
    """Create static 3x5 grid: rows=models, cols=time frames."""
    import matplotlib.pyplot as plt

    z_idx = geometry.nz // 2
    x = np.array(geometry.x_grid[:, :, z_idx])
    y = np.array(geometry.y_grid[:, :, z_idx])

    # Select 5 key frames: t=0, T/4, T/2, 3T/4, T
    n_frames = len(times)
    frame_indices = [0, n_frames//4, n_frames//2, 3*n_frames//4, n_frames-1]

    # Precompute |B|
    B_mag = {}
    for name in ["analytic", "ResistiveMHD", "ExtendedMHD"]:
        B_mag[name] = [
            np.sqrt(np.sum(B[:, :, z_idx, :]**2, axis=-1))
            for B in results[name]
        ]

    vmin = 0
    vmax = max(np.max(B_mag["analytic"][0]),
               np.max(B_mag["ResistiveMHD"][0]),
               np.max(B_mag["ExtendedMHD"][0])) * 1.1
    levels = np.linspace(vmin, vmax, 21)

    fig, axes = plt.subplots(3, 5, figsize=(18, 10))
    model_names = ["analytic", "ResistiveMHD", "ExtendedMHD"]
    row_labels = ["Analytic", "ResistiveMHD", "ExtendedMHD"]

    for row, name in enumerate(model_names):
        for col, frame_idx in enumerate(frame_indices):
            ax = axes[row, col]
            ax.contourf(x, y, B_mag[name][frame_idx], levels=levels, cmap='viridis')
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])

            if row == 0:
                ax.set_title(f't={times[frame_idx]:.3f}s')
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=12)

    plt.suptitle('Frozen Flux Evolution: Model Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved multi-frame summary: {save_path}")
```

---

## Task 5: Update main() for Multi-Model Comparison

**Files:**
- Modify: `validation/cases/analytic/frozen_flux.py`

**Step 1: Update main() to run multi-model simulation**

Replace the existing simulation call with:
```python
print("Running multi-model simulation...")
t_start = time.time()
results, geometry, config = run_multi_model_simulation(cfg, n_snapshots=50)
t_sim = time.time() - t_start
print(f"  Completed in {t_sim:.2f}s")
```

**Step 2: Compute metrics for each model**

```python
# Compute metrics for each model
metrics = {}
B_analytic_final = jnp.array(results["analytic"][-1])
B_initial = jnp.array(results["analytic"][0])
peak_initial = float(jnp.max(jnp.sqrt(jnp.sum(B_initial**2, axis=-1))))

for model_name in ["ResistiveMHD", "ExtendedMHD"]:
    B_final = jnp.array(results[model_name][-1])
    l2_err = float(l2_error(B_final, B_analytic_final))
    peak_final = float(jnp.max(jnp.sqrt(jnp.sum(B_final**2, axis=-1))))
    peak_ratio = peak_final / peak_initial

    l2_pass = l2_err < ACCEPTANCE["l2_error"]
    peak_pass = peak_ratio > ACCEPTANCE["peak_amplitude_ratio"]

    metrics[model_name] = {
        "l2_error": {"value": l2_err, "threshold": ACCEPTANCE["l2_error"], "passed": l2_pass},
        "peak_amplitude_ratio": {"value": peak_ratio, "threshold": ACCEPTANCE["peak_amplitude_ratio"], "passed": peak_pass},
        "overall_pass": l2_pass and peak_pass,
    }

overall_pass = all(m["overall_pass"] for m in metrics.values())
```

**Step 3: Generate all visualizations**

```python
# Save report first to get directory
report = ValidationReport(...)
report_dir = report.save()

# Generate GIF animations
create_3model_2d_animation(results["times"], results, geometry,
                           Path(report_dir) / "frozen_flux_3model_2d.gif")
create_3model_1d_animation(results["times"], results, geometry,
                           Path(report_dir) / "frozen_flux_3model_1d.gif")

# Generate static summary
create_multi_frame_summary(results["times"], results, geometry,
                           Path(report_dir) / "multi_frame_summary.png")
```

---

## Task 6: Run Validation and Verify

**Step 1: Run the validation script**

Run: `py validation/cases/analytic/frozen_flux.py`

Expected output:
```
Running multi-model simulation...
  Completed in X.XXs

Metrics:
  ResistiveMHD:
    L2 error: X.XXe-XX (threshold: 0.01) - PASS
    Peak ratio: X.XXXX (threshold: 0.98) - PASS
  ExtendedMHD:
    L2 error: X.XXe-XX (threshold: 0.01) - PASS/FAIL
    Peak ratio: X.XXXX (threshold: 0.98) - PASS/FAIL

Generating visualizations...
  Saved 3-model 2D animation: ...
  Saved 3-model 1D animation: ...
  Saved multi-frame summary: ...

Report saved to: validation/reports/YYYY-MM-DD_frozen_flux/
```

**Step 2: Verify output files exist**

Run: `ls validation/reports/*/frozen_flux_3model*.gif validation/reports/*/multi_frame_summary.png`

---

## Task 7: Commit Changes

**Step 1: Stage and commit**

```bash
git add validation/cases/analytic/frozen_flux.py
git commit -m "feat(validation): add multi-model comparison to frozen flux

Compare ResistiveMHD and ExtendedMHD against analytic solution:
- Sequential runs with combined metrics table
- 3-panel GIF animation (Analytic | ResistiveMHD | ExtendedMHD)
- 1D overlay GIF with all three models
- Static 3x5 multi-frame summary plot

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```
