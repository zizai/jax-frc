# Validation Infrastructure Design

## Overview

A framework for running validation cases against FRC reactor configurations with acceptance criteria, reference data management, and regression tracking.

**Use cases:**
- Periodic benchmarking (scheduled runs tracking trends over time)
- Manual validation campaigns (on-demand before releases, with human review)

**Not CI-blocking** — validation runs separately from the main test suite.

## Directory Structure

```
jax_frc/
  configurations/              # Reactor/benchmark configurations
    __init__.py               # Registry, AbstractConfiguration
    base.py                   # AbstractConfiguration class
    fat_cm.py                 # FATCMConfiguration
    hym_benchmark.py          # HYMBenchmarkConfiguration
    analytic.py               # SlabDiffusionConfiguration, etc.

  validation/                 # Validation infrastructure
    __init__.py
    runner.py                 # ValidationRunner
    metrics.py                # Built-in metric functions
    plotting.py               # Extends diagnostics.plotting for validation
    references.py             # ReferenceManager (analytic, file, external)
    report.py                 # HTMLReportGenerator

validation/                   # Validation data (project root)
  cases/
    analytic/
      diffusion_slab.yaml
      alfven_wave.yaml
    benchmarks/
      brio_wu.yaml
      gem_reconnection.yaml
    frc/
      fat_cm_merging.yaml
      hym_compression.yaml
  references/                 # Small reference data in repo
    gem_reconnected_flux.csv
  external/                   # Downloaded data (gitignored)
  reports/                    # Run outputs (gitignored)

scripts/
  run_validation.py           # CLI entry point
```

## Case Definition Format

Validation cases are YAML files that instantiate a Configuration class:

```yaml
name: fat_cm_merging_validation
description: "Validate FAT-CM two-FRC merge against published data"

configuration:
  class: FATCMConfiguration
  phase: merging
  overrides:  # optional parameter tweaks
    bias_field: 0.040

runtime:
  t_end: 100e-6

reference:
  type: file
  path: references/fat_cm_merge_timeline.csv

acceptance:
  quantitative:
    - metric: merge_time
      expected: 60e-6
      tolerance: 20%
    - metric: post_merge_radius
      expected: 0.22
      tolerance: 10%

  qualitative:
    - plot: excluded_flux_vs_time
      description: "Excluded flux radius evolution"
    - plot: flux_contours_at_merge
      description: "Poloidal flux contours at t=60μs"

output:
  snapshots: true
  snapshot_times: [0, 30e-6, 60e-6, 100e-6]
  plots:
    - type: time_traces
      name: flux_evolution
    - type: fields
      name: final_state
      time: -1
    - type: comparison
      name: density_vs_reference
      field: n
  html_report: true
  log_level: INFO
```

## Configuration Classes

Each reactor or benchmark is represented by a Configuration class:

```python
class AbstractConfiguration(ABC):
    """Base for all reactor/benchmark configurations."""

    name: str
    description: str

    @abstractmethod
    def build_geometry(self) -> Geometry: ...

    @abstractmethod
    def build_initial_state(self) -> State: ...

    @abstractmethod
    def build_model(self) -> PhysicsModel: ...

    @abstractmethod
    def build_boundary_conditions(self) -> BoundaryConditions: ...

    def available_phases(self) -> list[str]:
        """List valid phases for this configuration."""

    def default_runtime(self) -> dict:
        """Suggested t_end, dt for this configuration."""
```

Example implementation:

```python
class FATCMConfiguration(AbstractConfiguration):
    """FAT-CM device configuration with all phases."""

    name = "FAT-CM"
    description = "Field-reversed configuration translation and merging experiment"

    # Device constants
    chamber_radius = 0.39  # m
    flux_conserver_skin_time = 5e-3  # s
    bias_field = 0.038  # T
    reversal_field = 0.40  # T

    def __init__(self, phase: str, **overrides):
        self.phase = phase
        self.params = self.default_params() | overrides

    def build_geometry(self) -> Geometry:
        return Geometry(
            coord_system="cylindrical",
            r_min=0.01, r_max=self.chamber_radius,
            z_min=-2.0, z_max=2.0,
            nr=64, nz=128
        )

    def build_initial_state(self) -> State:
        """Initial plasma state for the specified phase."""
        if self.phase == "formation":
            return self._formation_initial_state()
        elif self.phase == "merging":
            return self._merging_initial_state()
        # ...

    def build_model(self) -> PhysicsModel:
        """Appropriate physics model for this phase."""
        # Formation/merging: resistive MHD sufficient
        # Compression: may need extended MHD or hybrid
        return ResistiveMHD(resistivity=self._resistivity_model())

    def build_boundary_conditions(self) -> BoundaryConditions:
        """Coil drives, flux conservers, etc."""
        # ...
```

Registry:

```python
CONFIGURATION_REGISTRY = {
    'FATCMConfiguration': FATCMConfiguration,
    'C2WConfiguration': C2WConfiguration,
    'HYMBenchmarkConfiguration': HYMBenchmarkConfiguration,
    'SlabDiffusionConfiguration': SlabDiffusionConfiguration,
    'GEMReconnectionConfiguration': GEMReconnectionConfiguration,
}
```

## Acceptance Criteria

### Quantitative Metrics

Built-in metrics in `jax_frc/validation/metrics.py`:

| Metric | Description |
|--------|-------------|
| `l2_error` | L2 norm relative to reference |
| `linf_error` | Max pointwise error |
| `decay_rate` | Exponential fit to timeseries |
| `growth_rate` | Same, for instabilities |
| `rmse_curve` | RMSE between two curves |
| `div_b_max` | Maximum divergence of B |
| `peak_location` | Location of field maximum |
| `conservation_drift` | Relative change in conserved quantity |
| `merge_time` | Time when FRCs merge (detected by topology) |
| `reconnection_rate` | dψ/dt at X-point |

Custom metrics via Python function path:

```yaml
acceptance:
  quantitative:
    - metric: custom
      function: "validation.cases.gem.hall_quadrupole_strength"
      threshold: 0.1
```

### Qualitative Plots

Specified in acceptance block, generated for human review:

```yaml
acceptance:
  qualitative:
    - plot: field_vs_analytic
      description: "B(x) at t=0, t_mid, t_end overlaid with analytic"
    - plot: hall_quadrupole
      description: "B_phi showing Hall quadrupole structure"
```

## Reference Data Handling

Three reference types:

### Analytic (computed at runtime)

```yaml
reference:
  type: analytic
  formula: "B0 * cos(k * x) * exp(-eta * k**2 * t)"
  variables: [B0, k, eta, x, t]
```

### Local file (in repo)

```yaml
reference:
  type: file
  path: references/brio_wu_profiles.csv
  columns:
    x: position
    rho: density
```

### External (fetched and cached)

```yaml
reference:
  type: external
  source: zenodo
  record: "15084058"
  files:
    - hallGEM512.grid.h5
    - hallGEM512.state_0100.h5
  cache_dir: external/agate
```

Reference manager:

```python
class ReferenceManager:
    def load(self, ref_config: dict, params: dict) -> ReferenceData:
        """Load reference based on type."""

    def fetch_external(self, source: str, record: str, files: list) -> Path:
        """Download if not cached, return cache path."""

    def evaluate_analytic(self, formula: str, grid, time, params) -> Array:
        """Safely evaluate analytic formula on grid."""
```

Caching:
- External files downloaded to `validation/external/`
- Cache validated by checksum in `manifest.json`
- `--refresh-cache` flag forces re-download
- `.gitignore` excludes `validation/external/`

## Report Structure

Each validation run produces a timestamped directory:

```
reports/2026-01-26T14-30-00_gem_reconnection/
  config.yaml          # Copy of input case config
  metrics.json         # All computed metrics with pass/fail
  summary.txt          # Human-readable pass/fail summary
  log.txt              # Full execution log
  plots/
    reconnected_flux_vs_time.png
    hall_quadrupole_bphi.png
  snapshots/           # Optional HDF5 checkpoints
    state_t0000.h5
    state_t0050.h5
  report.html          # Self-contained HTML with embedded plots
```

### metrics.json format

```json
{
  "case": "gem_reconnection",
  "configuration": "GEMReconnectionConfiguration",
  "timestamp": "2026-01-26T14:30:00",
  "runtime_seconds": 142.5,
  "overall_pass": false,
  "metrics": {
    "reconnection_rate_peak": {
      "value": 0.18,
      "expected": 0.17,
      "tolerance": "10%",
      "pass": true
    },
    "flux_rmse": {
      "value": 0.035,
      "threshold": 0.02,
      "pass": false,
      "message": "RMSE 0.035 exceeds threshold 0.02"
    }
  }
}
```

### HTML Report

Self-contained HTML with embedded plots (base64 PNGs):

- Header: case name, timestamp, configuration, overall pass/fail
- Configuration summary: key device parameters, phase, overrides
- Metrics table: value, expected, tolerance, pass/fail for each
- Plots: embedded, grouped by category
- Notes: description from YAML, any warnings

## Visualization Integration

Validation plotting builds on `jax_frc.diagnostics.plotting`:

```python
from jax_frc.diagnostics.plotting import (
    plot_time_traces,
    plot_fields,
    plot_profiles,
    plot_particles,
)

def generate_validation_plots(
    result: SimulationResult,
    reference: ReferenceData,
    plot_specs: list[dict],
    output_dir: Path,
) -> list[Path]:
    """Generate validation-specific plots using standard plotting utils."""

    for spec in plot_specs:
        if spec['type'] == 'time_traces':
            fig = plot_time_traces(result, save_dir=None, show=False)
            overlay_reference_timeseries(fig, reference)
            save_validation_figure(fig, spec['name'], output_dir)

        elif spec['type'] == 'comparison':
            fig = plot_field_comparison(result, reference, spec)
            save_validation_figure(fig, spec['name'], output_dir)
        # ...
```

Plot types in YAML:

```yaml
output:
  plots:
    - type: time_traces      # Standard diagnostic
    - type: fields           # Standard diagnostic
    - type: profiles         # Standard diagnostic
    - type: comparison       # Validation-specific: side-by-side
    - type: error_map        # Validation-specific: spatial error
```

## CLI Interface

Script: `scripts/run_validation.py`

```bash
# Run single case
python scripts/run_validation.py diffusion_slab

# Run all cases in a category
python scripts/run_validation.py --category analytic

# Run multiple specific cases
python scripts/run_validation.py brio_wu gem_reconnection

# Run all cases
python scripts/run_validation.py --all

# Options
python scripts/run_validation.py gem_reconnection \
  --output-dir results/nightly \
  --refresh-cache \
  --no-plots \
  --log-level DEBUG

# List available cases
python scripts/run_validation.py --list

# Dry run (validate config, don't execute)
python scripts/run_validation.py gem_reconnection --dry-run
```

Arguments:

| Argument | Description |
|----------|-------------|
| `CASES` | Positional: case names (without .yaml) |
| `--category` | Run all cases in category (analytic, benchmarks, frc) |
| `--all` | Run entire validation suite |
| `--output-dir` | Override default `validation/reports/` |
| `--refresh-cache` | Force re-download of external references |
| `--no-plots` | Skip plot generation |
| `--no-html` | Skip HTML report generation |
| `--dry-run` | Validate config only |
| `--list` | Show available cases and exit |
| `--log-level` | DEBUG, INFO, WARNING, ERROR |

Exit codes:
- `0`: All cases passed
- `1`: One or more cases failed acceptance criteria
- `2`: Configuration or runtime error

## Runner Implementation

```python
class ValidationRunner:
    def __init__(self, case_path: Path, output_dir: Path):
        self.config = self.load_config(case_path)
        self.output_dir = output_dir / self.timestamped_name()
        self.reference_mgr = ReferenceManager()

    def run(self) -> ValidationResult:
        """Execute full validation pipeline."""
        self.setup_output_dir()
        self.save_config_copy()

        # 1. Instantiate configuration
        config_cls = CONFIGURATION_REGISTRY[self.config['configuration']['class']]
        configuration = config_cls(
            phase=self.config['configuration']['phase'],
            **self.config['configuration'].get('overrides', {})
        )

        # 2. Configuration builds simulation components
        geometry = configuration.build_geometry()
        state = configuration.build_initial_state()
        model = configuration.build_model()
        boundaries = configuration.build_boundary_conditions()

        # 3. Run simulation
        history = self.execute(model, state, geometry, boundaries)

        # 4. Load/compute reference
        reference = self.reference_mgr.load(
            self.config['reference'],
            self.config['parameters']
        )

        # 5. Compute metrics
        metrics = self.compute_metrics(history, reference)

        # 6. Generate outputs
        self.save_metrics(metrics)
        if self.config.get('output', {}).get('plots', True):
            self.generate_plots(history, reference, metrics)
        if self.config.get('output', {}).get('html_report', True):
            self.generate_html_report(metrics)

        return ValidationResult(self.config['name'], metrics)
```

## Design Decisions Summary

| Aspect | Decision |
|--------|----------|
| Case definition | YAML config files instantiate Configuration classes |
| Runner | Generic, dispatches to Configuration for model/geometry/boundaries |
| Reference data | Mix: analytic computed, small files in repo, large fetched |
| Reports | Timestamped directories with metrics.json, plots, HTML |
| Plotting | Integrates with `jax_frc.diagnostics.plotting` |
| Acceptance | Quantitative thresholds + qualitative visual artifacts |
| Interface | `scripts/run_validation.py` with argparse |
| Use cases | Periodic benchmarking + manual validation campaigns |
