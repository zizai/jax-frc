# Validation Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace YAML-based validation with explicit Python scripts that produce HTML reports with comparison plots.

**Architecture:** Each validation case becomes a standalone Python script in `validation/cases/<category>/`. Scripts use shared utilities for HTML report generation and plotting. The `SlabDiffusionConfiguration` class already has an `analytic_solution()` method we can leverage.

**Tech Stack:** JAX, matplotlib, Python dataclasses, HTML/CSS for reports

---

## Task 1: Create ValidationReport Dataclass

**Files:**
- Create: `validation/utils/reporting.py`
- Test: `tests/test_validation_reporting.py`

**Step 1: Write the failing test**

```python
# tests/test_validation_reporting.py
"""Tests for validation report generation."""
import pytest
from pathlib import Path
import tempfile


def test_validation_report_creation():
    """ValidationReport can be created with required fields."""
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="test_case",
        description="A test case",
        docstring="Test docstring",
        configuration={'param': 1.0},
        metrics={'l2_error': {'value': 0.05, 'threshold': 0.1, 'passed': True}},
        overall_pass=True,
    )

    assert report.name == "test_case"
    assert report.overall_pass is True
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_reporting.py::test_validation_report_creation -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'validation.utils'"

**Step 3: Create the module structure and dataclass**

```python
# validation/utils/__init__.py
"""Validation utilities."""
from validation.utils.reporting import ValidationReport
from validation.utils.plotting import plot_comparison

__all__ = ['ValidationReport', 'plot_comparison']
```

```python
# validation/utils/reporting.py
"""HTML report generation for validation cases."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from datetime import datetime
import base64
import io


@dataclass
class ValidationReport:
    """Container for validation results that generates HTML reports."""

    name: str
    description: str
    docstring: str
    configuration: dict
    metrics: dict
    overall_pass: bool
    plots: list = field(default_factory=list)
    timing: Optional[dict] = None
    warnings: Optional[list] = None

    def add_plot(self, fig, name: str = None):
        """Add matplotlib figure, converts to base64 PNG for embedding."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        self.plots.append({
            'name': name or f'plot_{len(self.plots)}',
            'data': img_base64,
        })

    def save(self, base_dir: Path = None) -> Path:
        """Write report.html to timestamped directory."""
        if base_dir is None:
            base_dir = Path("validation/reports")

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        output_dir = base_dir / f"{timestamp}_{self.name}"
        output_dir.mkdir(parents=True, exist_ok=True)

        html = self._generate_html()
        (output_dir / "report.html").write_text(html)

        return output_dir

    def _generate_html(self) -> str:
        """Generate self-contained HTML report."""
        pass_badge = "PASS" if self.overall_pass else "FAIL"
        badge_color = "#28a745" if self.overall_pass else "#dc3545"

        # Configuration table
        config_rows = "\n".join(
            f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"
            for k, v in self.configuration.items()
        )

        # Metrics table
        metrics_rows = ""
        for name, data in self.metrics.items():
            status = "PASS" if data.get('passed', True) else "FAIL"
            status_color = "#28a745" if data.get('passed', True) else "#dc3545"
            value = data.get('value', 'N/A')
            if isinstance(value, float):
                value = f"{value:.4g}"
            threshold = data.get('threshold', 'N/A')
            desc = data.get('description', '')
            metrics_rows += f"""<tr>
                <td>{name}</td>
                <td>{value}</td>
                <td>{threshold}</td>
                <td style="color: {status_color}; font-weight: bold;">{status}</td>
                <td>{desc}</td>
            </tr>"""

        # Plots
        plots_html = ""
        for plot in self.plots:
            plots_html += f"""
            <div class="plot">
                <h3>{plot['name']}</h3>
                <img src="data:image/png;base64,{plot['data']}" alt="{plot['name']}">
            </div>"""

        # Timing
        timing_html = ""
        if self.timing:
            timing_rows = "\n".join(
                f"<tr><td>{k}</td><td>{v:.2f}s</td></tr>"
                for k, v in self.timing.items()
            )
            timing_html = f"""
            <h2>Timing</h2>
            <table><tbody>{timing_rows}</tbody></table>"""

        # Warnings
        warnings_html = ""
        if self.warnings:
            warning_items = "\n".join(f"<li>{w}</li>" for w in self.warnings)
            warnings_html = f"""
            <h2>Warnings</h2>
            <ul class="warnings">{warning_items}</ul>"""

        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{self.name} - Validation Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
               max-width: 1200px; margin: 0 auto; padding: 20px; line-height: 1.6; }}
        h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; }}
        .badge {{ display: inline-block; padding: 5px 15px; border-radius: 4px;
                  color: white; font-weight: bold; margin-left: 10px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .plot {{ margin: 20px 0; }}
        .plot img {{ max-width: 100%; border: 1px solid #ddd; }}
        .physics {{ background: #f8f9fa; padding: 15px; border-left: 4px solid #007bff; margin: 20px 0; }}
        .warnings {{ color: #856404; background: #fff3cd; padding: 15px; border-radius: 4px; }}
        code {{ background: #f5f5f5; padding: 2px 5px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>{self.name} <span class="badge" style="background: {badge_color}">{pass_badge}</span></h1>
    <p><strong>{self.description}</strong></p>

    <div class="physics">
        <h2>Physics Background</h2>
        <pre>{self.docstring}</pre>
    </div>

    <h2>Configuration</h2>
    <table>
        <thead><tr><th>Parameter</th><th>Value</th></tr></thead>
        <tbody>{config_rows}</tbody>
    </table>

    <h2>Results</h2>
    <table>
        <thead><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Status</th><th>Description</th></tr></thead>
        <tbody>{metrics_rows}</tbody>
    </table>

    <h2>Plots</h2>
    {plots_html}

    {timing_html}
    {warnings_html}

    <footer style="margin-top: 40px; color: #666; font-size: 0.9em;">
        Generated: {datetime.now().isoformat()}
    </footer>
</body>
</html>"""
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_reporting.py::test_validation_report_creation -v`
Expected: PASS

**Step 5: Commit**

```bash
git add validation/utils/__init__.py validation/utils/reporting.py tests/test_validation_reporting.py
git commit -m "feat(validation): add ValidationReport dataclass for HTML report generation"
```

---

## Task 2: Add ValidationReport.save() Test

**Files:**
- Modify: `tests/test_validation_reporting.py`

**Step 1: Write the failing test**

```python
def test_validation_report_save(tmp_path):
    """ValidationReport.save() creates HTML file in timestamped directory."""
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="test_case",
        description="A test case",
        docstring="Physics description here",
        configuration={'T_peak': 200.0, 'sigma': 0.3},
        metrics={
            'l2_error': {
                'value': 0.05,
                'threshold': 0.1,
                'passed': True,
                'description': 'Relative L2 error'
            }
        },
        overall_pass=True,
    )

    output_dir = report.save(base_dir=tmp_path)

    assert output_dir.exists()
    assert (output_dir / "report.html").exists()

    html = (output_dir / "report.html").read_text()
    assert "test_case" in html
    assert "PASS" in html
    assert "T_peak" in html
    assert "200.0" in html
```

**Step 2: Run test to verify it passes (implementation already in Task 1)**

Run: `py -m pytest tests/test_validation_reporting.py::test_validation_report_save -v`
Expected: PASS (save() was implemented in Task 1)

**Step 3: Commit**

```bash
git add tests/test_validation_reporting.py
git commit -m "test(validation): add save() test for ValidationReport"
```

---

## Task 3: Add ValidationReport.add_plot() Test

**Files:**
- Modify: `tests/test_validation_reporting.py`

**Step 1: Write the failing test**

```python
def test_validation_report_add_plot(tmp_path):
    """ValidationReport.add_plot() embeds matplotlib figure as base64."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from validation.utils.reporting import ValidationReport

    report = ValidationReport(
        name="test_case",
        description="A test case",
        docstring="Physics",
        configuration={},
        metrics={},
        overall_pass=True,
    )

    # Create a simple plot
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_title("Test Plot")

    report.add_plot(fig, name="quadratic")
    plt.close(fig)

    assert len(report.plots) == 1
    assert report.plots[0]['name'] == "quadratic"
    assert report.plots[0]['data'].startswith('iVBOR')  # PNG base64 header

    # Save and verify plot is in HTML
    output_dir = report.save(base_dir=tmp_path)
    html = (output_dir / "report.html").read_text()
    assert "data:image/png;base64," in html
    assert "quadratic" in html
```

**Step 2: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_reporting.py::test_validation_report_add_plot -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_validation_reporting.py
git commit -m "test(validation): add add_plot() test for ValidationReport"
```

---

## Task 4: Create plot_comparison Utility

**Files:**
- Create: `validation/utils/plotting.py`
- Modify: `tests/test_validation_reporting.py`

**Step 1: Write the failing test**

```python
def test_plot_comparison():
    """plot_comparison creates overlay of simulation vs expected."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from validation.utils.plotting import plot_comparison

    x = np.linspace(-1, 1, 50)
    actual = x**2 + 0.01 * np.random.randn(50)
    expected = x**2

    fig = plot_comparison(
        x, actual, expected,
        labels=['Simulation', 'Analytic'],
        title='Test Comparison',
        xlabel='x',
        ylabel='y'
    )

    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert len(ax.lines) == 2
    assert ax.get_title() == 'Test Comparison'
    plt.close(fig)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_reporting.py::test_plot_comparison -v`
Expected: FAIL with "ImportError: cannot import name 'plot_comparison'"

**Step 3: Write the implementation**

```python
# validation/utils/plotting.py
"""Plotting utilities for validation comparisons."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Sequence


def plot_comparison(
    x: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
    labels: Sequence[str] = ('Simulation', 'Expected'),
    title: str = 'Comparison',
    xlabel: str = 'x',
    ylabel: str = 'y',
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Create 1D comparison plot of actual vs expected values.

    Args:
        x: Independent variable (e.g., spatial coordinate)
        actual: Simulated values
        expected: Analytic/reference values
        labels: Legend labels for [actual, expected]
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size in inches

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(x, actual, 'b-', linewidth=2, label=labels[0])
    ax.plot(x, expected, 'r--', linewidth=2, label=labels[1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_error(
    x: np.ndarray,
    actual: np.ndarray,
    expected: np.ndarray,
    title: str = 'Error',
    xlabel: str = 'x',
    ylabel: str = 'Error',
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """Plot the difference between actual and expected values.

    Args:
        x: Independent variable
        actual: Simulated values
        expected: Analytic/reference values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        figsize: Figure size

    Returns:
        Matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    error = actual - expected
    ax.plot(x, error, 'k-', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
```

**Step 4: Update __init__.py**

```python
# validation/utils/__init__.py
"""Validation utilities."""
from validation.utils.reporting import ValidationReport
from validation.utils.plotting import plot_comparison, plot_error

__all__ = ['ValidationReport', 'plot_comparison', 'plot_error']
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_reporting.py::test_plot_comparison -v`
Expected: PASS

**Step 6: Commit**

```bash
git add validation/utils/plotting.py validation/utils/__init__.py tests/test_validation_reporting.py
git commit -m "feat(validation): add plot_comparison and plot_error utilities"
```

---

## Task 5: Create diffusion_slab.py Validation Script

**Files:**
- Create: `validation/cases/analytic/diffusion_slab.py`
- Test: Run the script directly

**Step 1: Create the directory structure**

```bash
mkdir -p validation/cases/analytic
touch validation/cases/__init__.py
touch validation/cases/analytic/__init__.py
```

**Step 2: Write the validation script**

```python
# validation/cases/analytic/diffusion_slab.py
"""
Diffusion Slab Validation
=========================
1D heat diffusion test with analytic Gaussian solution.

Physics
-------
Heat equation: dT/dt = kappa * d²T/dz²

Initial condition: Gaussian temperature profile
    T(z, 0) = T_peak * exp(-z² / (2*sigma²)) + T_base

Analytic solution: The Gaussian spreads while conserving total heat
    T(z, t) = T_peak * sqrt(sigma² / (sigma² + 2*kappa*t))
              * exp(-z² / (2*(sigma² + 2*kappa*t))) + T_base

This test validates thermal transport implementation by comparing
simulation output to the exact analytic solution.
"""
import time
import jax.numpy as jnp
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from jax_frc.configurations.analytic import SlabDiffusionConfiguration
from jax_frc.solvers import Solver
from jax_frc.validation.metrics import l2_error, linf_error
from validation.utils.reporting import ValidationReport
from validation.utils.plotting import plot_comparison, plot_error


# === CASE METADATA ===
NAME = "diffusion_slab"
DESCRIPTION = "1D heat diffusion test with analytic solution"


# === CONFIGURATION ===
def setup_configuration():
    """Define simulation parameters."""
    return {
        'T_peak': 200.0,      # Peak temperature (eV)
        'T_base': 50.0,       # Background temperature (eV)
        'sigma': 0.3,         # Initial Gaussian width
        'kappa': 1.0e-3,      # Thermal diffusivity
        'nz': 64,             # Grid points
        't_end': 1.0e-4,      # End time
        'dt': 1.0e-6,         # Time step
    }


# === ANALYTIC SOLUTION ===
def analytic_solution(z, t, cfg):
    """Exact Gaussian diffusion solution.

    T(z,t) = T_peak * sqrt(sigma² / (sigma² + 2*kappa*t))
             * exp(-z² / (2*(sigma² + 2*kappa*t))) + T_base
    """
    kappa = cfg['kappa']
    sigma = cfg['sigma']
    sigma_t_sq = sigma**2 + 2 * kappa * t
    amplitude = cfg['T_peak'] * jnp.sqrt(sigma**2 / sigma_t_sq)
    return amplitude * jnp.exp(-z**2 / (2 * sigma_t_sq)) + cfg['T_base']


# === SIMULATION ===
def run_simulation(cfg):
    """Run the diffusion simulation."""
    # Build components from configuration class
    config = SlabDiffusionConfiguration(
        T_peak=cfg['T_peak'],
        T_base=cfg['T_base'],
        sigma=cfg['sigma'],
        kappa=cfg['kappa'],
        nz=cfg['nz'],
    )

    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = config.build_model()
    solver = Solver.create({'type': 'semi_implicit'})

    # Run simulation
    t_end = cfg['t_end']
    dt = cfg['dt']
    n_steps = int(t_end / dt)

    for _ in range(n_steps):
        state = solver.step(state, dt, model, geometry)

    return state, geometry


# === ACCEPTANCE CRITERIA ===
ACCEPTANCE = {
    'l2_error': {
        'threshold': 0.1,
        'description': 'Relative L2 error vs analytic solution'
    },
    'linf_error': {
        'threshold': 0.15,
        'description': 'Max absolute error vs analytic solution'
    },
}


def main():
    """Run validation and generate report."""
    print(f"Running {NAME}...")
    start_time = time.time()

    # Setup
    cfg = setup_configuration()
    setup_time = time.time()

    # Run simulation
    state, geometry = run_simulation(cfg)
    sim_time = time.time()

    # Get temperature profile at midplane (average over r)
    T_sim = jnp.mean(state.T, axis=0)  # Average over r dimension
    z = geometry.z_centers if hasattr(geometry, 'z_centers') else geometry.z_grid[0, :]

    # Compute analytic solution
    T_analytic = analytic_solution(z, cfg['t_end'], cfg)

    # Compute metrics
    l2_err = l2_error(T_sim, T_analytic)
    linf_err = linf_error(T_sim, T_analytic)

    metrics = {
        'l2_error': l2_err,
        'linf_error': linf_err,
    }

    # Check acceptance
    results = {}
    for name, spec in ACCEPTANCE.items():
        value = metrics[name]
        passed = value <= spec['threshold']
        results[name] = {
            'value': float(value),
            'threshold': spec['threshold'],
            'passed': passed,
            'description': spec['description'],
        }

    overall_pass = all(r['passed'] for r in results.values())

    # Generate report
    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration=cfg,
        metrics=results,
        overall_pass=overall_pass,
        timing={
            'setup': setup_time - start_time,
            'simulation': sim_time - setup_time,
            'total': time.time() - start_time,
        }
    )

    # Add comparison plot
    fig_comparison = plot_comparison(
        jnp.array(z), T_sim, T_analytic,
        labels=['Simulation', 'Analytic'],
        title=f'Temperature Profile at t = {cfg["t_end"]:.2e} s',
        xlabel='z (m)',
        ylabel='T (eV)'
    )
    report.add_plot(fig_comparison, name='Temperature Comparison')

    # Add error plot
    fig_error = plot_error(
        jnp.array(z), T_sim, T_analytic,
        title='Temperature Error (Simulation - Analytic)',
        xlabel='z (m)',
        ylabel='T error (eV)'
    )
    report.add_plot(fig_error, name='Error Distribution')

    # Save report
    output_dir = report.save(base_dir=Path(__file__).parent.parent.parent / "reports")

    print(f"Report: {output_dir / 'report.html'}")
    print(f"Result: {'PASS' if overall_pass else 'FAIL'}")
    for name, data in results.items():
        status = 'PASS' if data['passed'] else 'FAIL'
        print(f"  {name}: {data['value']:.4g} (threshold: {data['threshold']}) [{status}]")

    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

**Step 3: Run the script to verify it works**

Run: `cd validation/cases/analytic && py diffusion_slab.py`
Expected: Script runs, produces report, prints PASS/FAIL

**Step 4: Commit**

```bash
git add validation/cases/__init__.py validation/cases/analytic/__init__.py validation/cases/analytic/diffusion_slab.py
git commit -m "feat(validation): add diffusion_slab.py explicit validation script"
```

---

## Task 6: Delete Old YAML File for diffusion_slab

**Files:**
- Delete: `validation/cases/analytic/diffusion_slab.yaml`

**Step 1: Verify Python script works**

Run: `py validation/cases/analytic/diffusion_slab.py`
Expected: PASS

**Step 2: Delete the YAML file**

```bash
git rm validation/cases/analytic/diffusion_slab.yaml
```

**Step 3: Commit**

```bash
git commit -m "refactor(validation): remove diffusion_slab.yaml, replaced by .py script"
```

---

## Task 7: Create belova_case1.py Qualitative Validation Script

**Files:**
- Create: `validation/cases/frc/belova_case1.py`
- Create: `validation/cases/frc/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p validation/cases/frc
touch validation/cases/frc/__init__.py
```

**Step 2: Write the validation script**

```python
# validation/cases/frc/belova_case1.py
"""
Belova Case 1: Large FRC Merging
================================
Two FRCs merge without axial compression (Belova et al., Phys. Plasmas, Figs 1-2).

Physics
-------
This simulation models the collision and merging of two counter-helicity
Field-Reversed Configurations (FRCs). The expected outcome is partial
merging resulting in a doublet configuration.

Key physics:
- Resistive MHD with anomalous resistivity (Chodura model)
- eta_0 = 1e-6 (classical), eta_anom = 1e-3 (anomalous)
- No external axial compression

Expected behavior:
- Two FRCs approach and interact at z=0
- Magnetic reconnection occurs at the X-point
- Partial merger forms doublet configuration
- System remains MHD stable throughout

This is a QUALITATIVE validation - we verify stability and conservation,
not numeric accuracy against a known analytic solution.

Reference: Belova et al., Physics of Plasmas (200X), Figures 1-2
"""
import time
import jax.numpy as jnp
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from jax_frc.configurations.frc import BelovaCase1Configuration
from jax_frc.solvers import Solver
from validation.utils.reporting import ValidationReport
from validation.utils.plotting import plot_comparison


# === CASE METADATA ===
NAME = "belova_case1"
DESCRIPTION = "Large FRC merging without compression (qualitative)"


# === CONFIGURATION ===
def setup_configuration():
    """Define simulation parameters."""
    return {
        'model_type': 'resistive_mhd',
        'eta_0': 1e-6,
        'eta_anom': 1e-3,
        't_end': 30.0,
        'dt': 0.001,
        'nr': 64,
        'nz': 128,
    }


# === SIMULATION ===
def run_simulation(cfg):
    """Run the FRC merging simulation."""
    config = BelovaCase1Configuration(
        eta_0=cfg['eta_0'],
        eta_anom=cfg['eta_anom'],
        nr=cfg['nr'],
        nz=cfg['nz'],
    )

    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)
    model = config.build_model()
    solver = Solver.create({'type': 'semi_implicit'})

    # Run with history capture at key times
    t_end = cfg['t_end']
    dt = cfg['dt']
    n_steps = int(t_end / dt)

    history = {'psi': [], 'time': [], 'total_flux': [], 'total_energy': []}
    save_interval = n_steps // 10  # Save 10 snapshots

    for i in range(n_steps):
        state = solver.step(state, dt, model, geometry)

        if i % save_interval == 0:
            history['psi'].append(state.psi.copy())
            history['time'].append(state.time)
            history['total_flux'].append(float(jnp.sum(jnp.abs(state.psi))))
            history['total_energy'].append(float(jnp.sum(state.p)))

    return state, geometry, history


# === ACCEPTANCE CRITERIA ===
ACCEPTANCE = {
    'no_numerical_instability': {
        'description': 'No NaN or Inf values in solution'
    },
    'flux_conservation': {
        'threshold': 0.1,
        'description': 'Total flux conserved within 10%'
    },
    'energy_bounded': {
        'description': 'Total energy remains bounded throughout simulation'
    },
}


def check_stability(state):
    """Check for NaN/Inf in solution."""
    has_nan = bool(jnp.any(jnp.isnan(state.psi)) or jnp.any(jnp.isinf(state.psi)))
    has_nan = has_nan or bool(jnp.any(jnp.isnan(state.p)) or jnp.any(jnp.isinf(state.p)))
    return not has_nan


def check_flux_conservation(history, threshold=0.1):
    """Check if total flux is conserved within threshold."""
    flux = jnp.array(history['total_flux'])
    initial = flux[0]
    max_deviation = float(jnp.max(jnp.abs(flux - initial) / initial))
    return max_deviation <= threshold, max_deviation


def check_energy_bounded(history):
    """Check if energy remains bounded (no exponential growth)."""
    energy = jnp.array(history['total_energy'])
    # Check that energy doesn't grow more than 10x
    return float(jnp.max(energy) / energy[0]) < 10.0


def main():
    """Run validation and generate report."""
    print(f"Running {NAME}...")
    start_time = time.time()

    # Setup
    cfg = setup_configuration()
    setup_time = time.time()

    # Run simulation
    state, geometry, history = run_simulation(cfg)
    sim_time = time.time()

    # Check acceptance criteria
    results = {}

    # Stability check
    stable = check_stability(state)
    results['no_numerical_instability'] = {
        'value': 0.0 if stable else 1.0,
        'passed': stable,
        'description': ACCEPTANCE['no_numerical_instability']['description'],
    }

    # Flux conservation
    flux_ok, flux_deviation = check_flux_conservation(history, ACCEPTANCE['flux_conservation']['threshold'])
    results['flux_conservation'] = {
        'value': flux_deviation,
        'threshold': ACCEPTANCE['flux_conservation']['threshold'],
        'passed': flux_ok,
        'description': ACCEPTANCE['flux_conservation']['description'],
    }

    # Energy bounded
    energy_ok = check_energy_bounded(history)
    results['energy_bounded'] = {
        'value': float(jnp.max(jnp.array(history['total_energy'])) / history['total_energy'][0]),
        'passed': energy_ok,
        'description': ACCEPTANCE['energy_bounded']['description'],
    }

    overall_pass = all(r['passed'] for r in results.values())

    # Generate report
    report = ValidationReport(
        name=NAME,
        description=DESCRIPTION,
        docstring=__doc__,
        configuration=cfg,
        metrics=results,
        overall_pass=overall_pass,
        timing={
            'setup': setup_time - start_time,
            'simulation': sim_time - setup_time,
            'total': time.time() - start_time,
        }
    )

    # Add flux evolution plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['time'], history['total_flux'], 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Flux')
    ax.set_title('Flux Evolution During Merging')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    report.add_plot(fig, name='Flux Evolution')
    plt.close(fig)

    # Add energy evolution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(history['time'], history['total_energy'], 'r-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Total Energy')
    ax.set_title('Energy Evolution During Merging')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    report.add_plot(fig, name='Energy Evolution')
    plt.close(fig)

    # Save report
    output_dir = report.save(base_dir=Path(__file__).parent.parent.parent / "reports")

    print(f"Report: {output_dir / 'report.html'}")
    print(f"Result: {'PASS' if overall_pass else 'FAIL'}")
    for name, data in results.items():
        status = 'PASS' if data['passed'] else 'FAIL'
        value_str = f"{data['value']:.4g}" if isinstance(data.get('value'), float) else str(data.get('value', 'N/A'))
        print(f"  {name}: {value_str} [{status}]")

    return overall_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

**Step 3: Commit**

```bash
git add validation/cases/frc/__init__.py validation/cases/frc/belova_case1.py
git commit -m "feat(validation): add belova_case1.py qualitative validation script"
```

---

## Task 8: Delete Old YAML File for belova_case1

**Files:**
- Delete: `validation/cases/frc/belova_case1.yaml`

**Step 1: Delete the YAML file**

```bash
git rm validation/cases/frc/belova_case1.yaml
```

**Step 2: Commit**

```bash
git commit -m "refactor(validation): remove belova_case1.yaml, replaced by .py script"
```

---

## Task 9: Convert Remaining Cases (cylindrical_shock)

**Files:**
- Create: `validation/cases/analytic/cylindrical_shock.py`
- Delete: `validation/cases/analytic/cylindrical_shock.yaml`

Follow the same pattern as Task 5-6 for `diffusion_slab.py`. The script structure is identical; only the physics functions and configuration differ.

---

## Task 10: Convert Remaining Cases (belova_case2, belova_case4)

**Files:**
- Create: `validation/cases/frc/belova_case2.py`
- Create: `validation/cases/frc/belova_case4.py`
- Delete: `validation/cases/frc/belova_case2.yaml`
- Delete: `validation/cases/frc/belova_case4.yaml`

Follow the same pattern as Task 7-8 for `belova_case1.py`.

---

## Task 11: Convert Remaining Cases (cylindrical_gem, cylindrical_vortex)

**Files:**
- Create: `validation/cases/hall_reconnection/cylindrical_gem.py`
- Create: `validation/cases/mhd_regression/cylindrical_vortex.py`
- Delete corresponding YAML files

Follow the same pattern, adapting to each case's specific physics.

---

## Task 12: Deprecate ValidationRunner

**Files:**
- Modify: `jax_frc/validation/runner.py`

**Step 1: Add deprecation warning**

Add to the top of the `ValidationRunner.run()` method:

```python
import warnings
warnings.warn(
    "ValidationRunner is deprecated. Use the validation scripts in "
    "validation/cases/<category>/<name>.py instead.",
    DeprecationWarning,
    stacklevel=2
)
```

**Step 2: Commit**

```bash
git add jax_frc/validation/runner.py
git commit -m "deprecate(validation): add warning to ValidationRunner, prefer .py scripts"
```

---

## Task 13: Update Documentation

**Files:**
- Modify: `CLAUDE.md` (if validation section exists)
- Create: `validation/README.md`

**Step 1: Write validation README**

```markdown
# Validation Cases

This directory contains validation scripts that test JAX-FRC against known solutions.

## Running a Validation

```bash
python validation/cases/analytic/diffusion_slab.py
```

Each script produces an HTML report in `validation/reports/`.

## Case Types

- **analytic/**: Cases with exact analytic solutions (diffusion_slab, cylindrical_shock)
- **frc/**: FRC merging cases validated qualitatively (belova_case1, etc.)
- **hall_reconnection/**: Hall MHD reconnection tests
- **mhd_regression/**: Regression tests against baseline runs

## Adding a New Case

1. Create `validation/cases/<category>/<name>.py`
2. Follow the template in `diffusion_slab.py` for analytic cases or `belova_case1.py` for qualitative cases
3. Define: `setup_configuration()`, `run_simulation()`, `ACCEPTANCE` criteria
4. Run and verify report generation
```

**Step 2: Commit**

```bash
git add validation/README.md
git commit -m "docs(validation): add README explaining new validation system"
```

---

## Summary

| Task | Description | Files Changed |
|------|-------------|---------------|
| 1 | Create ValidationReport dataclass | +2 new files |
| 2 | Add save() test | +1 test |
| 3 | Add add_plot() test | +1 test |
| 4 | Create plot_comparison utility | +1 new file |
| 5 | Create diffusion_slab.py | +1 new file |
| 6 | Delete diffusion_slab.yaml | -1 file |
| 7 | Create belova_case1.py | +1 new file |
| 8 | Delete belova_case1.yaml | -1 file |
| 9-11 | Convert remaining cases | +5 new, -5 yaml |
| 12 | Deprecate ValidationRunner | modify 1 file |
| 13 | Update documentation | +1 new file |

**Total: ~10 new Python files, ~7 YAML files removed**
