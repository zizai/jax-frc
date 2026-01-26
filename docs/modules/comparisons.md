# Comparisons Module

Validation framework for comparing jax-frc results against published literature.

## Overview

The `jax_frc/comparisons/` module provides infrastructure for:
- Running standardized simulation scenarios
- Comparing results against published data
- Generating validation reports

## Belova et al. FRC Merging Comparison

Primary validation case based on:
> Belova et al., "Numerical study of FRC formation and merging", Physics of Plasmas (2006)

### What's Compared

| Quantity | Description | Agreement Target |
|----------|-------------|------------------|
| Merge time | When two FRCs coalesce | Within 20% |
| Reconnection rate | Peak $d\psi/dt$ | Qualitative trend |
| Energy partition | Thermal vs magnetic | Within 30% |
| Final separatrix | Shape and position | Visual agreement |

### Physics Setup

Two counter-helicity FRCs initialized at opposite ends of domain, allowed to translate and merge:

- **Geometry**: Cylindrical $(r, z)$, typical 0.4m radius × 3m length
- **Initial state**: Hill's vortex equilibrium for each FRC
- **Boundary**: Conducting wall at $r_{max}$, periodic or open in $z$

## Implementation

### Class: `BelovaComparisonSuite`

Location: `jax_frc/comparisons/belova_merging.py`

```python
class BelovaComparisonSuite:
    def run_comparison(
        self,
        geometry: Geometry,
        resistive_config: dict,
        hybrid_config: dict,
    ) -> ComparisonReport:
        """Run both models and compare results."""
```

### Running a Comparison

```python
from jax_frc.comparisons import BelovaComparisonSuite
from jax_frc import Geometry

# Setup
suite = BelovaComparisonSuite()
geometry = suite.create_geometry(nr=64, nz=256)

# Run comparison
report = suite.run_comparison(
    geometry=geometry,
    resistive_config={'resistivity': {'type': 'chodura', 'eta_0': 1e-6}},
    hybrid_config={'n_particles': 100000}
)

# View results
print(report.summary())
```

### Output: `ComparisonReport`

```python
@dataclass
class ComparisonReport:
    resistive_result: MergingResult
    hybrid_result: MergingResult

    def merge_time_difference(self) -> float:
        """Absolute difference in merge times."""

    def merge_time_ratio(self) -> float:
        """Ratio of merge times (resistive/hybrid)."""

    def energy_partition_at_merge(self) -> dict:
        """Energy breakdown at merge time for each model."""

    def summary(self) -> str:
        """Human-readable comparison summary."""
```

### Output: `MergingResult`

```python
@dataclass
class MergingResult:
    times: Array           # Simulation time points
    psi_sep: Array         # Separatrix flux vs time
    E_magnetic: Array      # Magnetic energy vs time
    E_thermal: Array       # Thermal energy vs time
    reconnection_rate: Array  # d(psi)/dt at X-point
    merge_time: float      # Detected merge time
```

## Diagnostics Collected

The suite automatically collects:

1. **Separatrix flux** `psi_sep(t)`: Tracks FRC boundaries
2. **Energy partition** `E_mag(t)`, `E_th(t)`: Conservation check
3. **Reconnection rate** `d(psi)/dt`: Merging dynamics
4. **X-point position** `(r_x, z_x)(t)`: Merging location

## Adding New Comparisons

To add a new literature comparison:

1. Create `jax_frc/comparisons/your_comparison.py`
2. Implement comparison suite class with:
   - `create_geometry()`: Standard geometry for this case
   - `collect_diagnostics()`: What to measure
   - `run_model()`: Run simulation with given config
   - `compare()`: Generate comparison metrics
3. Add tests in `tests/test_your_comparison.py`
4. Document in this file

## Configuration Files

Pre-built configs in `examples/comparisons/`:

```yaml
# belova_resistive.yaml
model:
  type: resistive_mhd
  resistivity:
    type: chodura
    eta_0: 1.0e-6
    eta_anom: 1.0e-3

geometry:
  nr: 64
  nz: 256
  r_max: 0.4
  z_max: 1.5

time:
  t_max: 50.0e-6
  dt_max: 1.0e-8
```

## References

- Belova et al., Physics of Plasmas 13, 056115 (2006)
- Omelchenko & Schaffer, Physics of Plasmas 13, 062111 (2006)
