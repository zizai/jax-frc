# Validation Cases Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement three cylindrical validation cases (shock tube, vortex, GEM reconnection) with configurations, metrics, and tests.

**Architecture:** Each validation case follows the existing pattern: Configuration class (builds geometry, initial state, model, BCs) + YAML case file (specifies parameters, acceptance criteria) + test file (verifies configuration behavior). New metrics extend `metrics.py`.

**Tech Stack:** JAX, jax.numpy, pytest, dataclasses

---

## Task 1: Add Shock-Specific Metrics

**Files:**
- Modify: `jax_frc/validation/metrics.py`
- Test: `tests/test_validation_metrics.py`

**Step 1: Write the failing test**

Add to `tests/test_validation_metrics.py`:

```python
def test_shock_position_finds_discontinuity():
    """shock_position detects steepest gradient location."""
    from jax_frc.validation.metrics import shock_position
    import jax.numpy as jnp

    # Profile with sharp jump at z=0.3
    z = jnp.linspace(0, 1, 100)
    rho = jnp.where(z < 0.3, 1.0, 0.125)

    position = shock_position(rho, z)
    assert abs(position - 0.3) < 0.02  # Within 2 grid cells


def test_shock_position_error_computes_relative():
    """shock_position_error returns relative error vs expected."""
    from jax_frc.validation.metrics import shock_position_error
    import jax.numpy as jnp

    z = jnp.linspace(0, 1, 100)
    rho = jnp.where(z < 0.32, 1.0, 0.125)  # Shock at 0.32

    error = shock_position_error(rho, z, expected=0.3)
    assert abs(error - 0.0667) < 0.01  # (0.32 - 0.3) / 0.3 ≈ 0.067
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_metrics.py::test_shock_position_finds_discontinuity -v`
Expected: FAIL with "cannot import name 'shock_position'"

**Step 3: Write minimal implementation**

Add to `jax_frc/validation/metrics.py`:

```python
def shock_position(profile: jnp.ndarray, axis: jnp.ndarray) -> float:
    """Find position of steepest gradient (shock location).

    Args:
        profile: 1D array of field values (e.g., density)
        axis: 1D array of positions along which to search

    Returns:
        Position of maximum |gradient|
    """
    grad = jnp.abs(jnp.diff(profile))
    idx = jnp.argmax(grad)
    # Interpolate to midpoint between cells
    return float((axis[idx] + axis[idx + 1]) / 2)


def shock_position_error(
    profile: jnp.ndarray,
    axis: jnp.ndarray,
    expected: float
) -> float:
    """Compute relative error in shock position.

    Returns: |detected - expected| / |expected|
    """
    detected = shock_position(profile, axis)
    return float(jnp.abs(detected - expected) / jnp.abs(expected))
```

Also add to `METRIC_FUNCTIONS` dict:
```python
METRIC_FUNCTIONS = {
    'l2_error': l2_error,
    'linf_error': linf_error,
    'rmse_curve': rmse_curve,
    'conservation_drift': conservation_drift,
    'shock_position': shock_position,
    'shock_position_error': shock_position_error,
}
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_metrics.py::test_shock_position_finds_discontinuity tests/test_validation_metrics.py::test_shock_position_error_computes_relative -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/validation/metrics.py tests/test_validation_metrics.py
git commit -m "feat(validation): add shock_position metrics"
```

---

## Task 2: Add Reconnection Metrics

**Files:**
- Modify: `jax_frc/validation/metrics.py`
- Test: `tests/test_validation_metrics.py`

**Step 1: Write the failing test**

Add to `tests/test_validation_metrics.py`:

```python
def test_reconnection_rate_from_flux():
    """reconnection_rate computes dpsi/dt."""
    from jax_frc.validation.metrics import reconnection_rate
    import jax.numpy as jnp

    # Linear increase in reconnected flux
    times = jnp.array([0.0, 1.0, 2.0, 3.0])
    psi = jnp.array([0.0, 0.1, 0.2, 0.3])

    rate = reconnection_rate(psi, times)
    assert abs(rate - 0.1) < 0.01


def test_peak_reconnection_rate():
    """peak_reconnection_rate finds maximum rate."""
    from jax_frc.validation.metrics import peak_reconnection_rate
    import jax.numpy as jnp

    times = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
    psi = jnp.array([0.0, 0.05, 0.2, 0.25, 0.26])  # Peak rate between t=1 and t=2

    peak, t_peak = peak_reconnection_rate(psi, times)
    assert abs(peak - 0.15) < 0.02
    assert abs(t_peak - 1.5) < 0.5


def test_current_layer_thickness():
    """current_layer_thickness computes FWHM."""
    from jax_frc.validation.metrics import current_layer_thickness
    import jax.numpy as jnp

    # Gaussian current profile
    z = jnp.linspace(-2, 2, 100)
    sigma = 0.5
    J = jnp.exp(-z**2 / (2 * sigma**2))

    # FWHM = 2 * sqrt(2 * ln(2)) * sigma ≈ 2.355 * sigma
    fwhm = current_layer_thickness(J, z)
    expected_fwhm = 2.355 * sigma
    assert abs(fwhm - expected_fwhm) < 0.1
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_metrics.py::test_reconnection_rate_from_flux -v`
Expected: FAIL with "cannot import name 'reconnection_rate'"

**Step 3: Write minimal implementation**

Add to `jax_frc/validation/metrics.py`:

```python
def reconnection_rate(psi: jnp.ndarray, times: jnp.ndarray) -> float:
    """Compute average reconnection rate dpsi/dt.

    Args:
        psi: Array of reconnected flux values at each time
        times: Array of time values

    Returns:
        Average reconnection rate
    """
    dpsi = jnp.diff(psi)
    dt = jnp.diff(times)
    return float(jnp.mean(dpsi / dt))


def peak_reconnection_rate(
    psi: jnp.ndarray,
    times: jnp.ndarray
) -> tuple[float, float]:
    """Find peak reconnection rate and time.

    Returns:
        (peak_rate, time_of_peak)
    """
    dpsi = jnp.diff(psi)
    dt = jnp.diff(times)
    rates = dpsi / dt
    idx = jnp.argmax(rates)
    peak = float(rates[idx])
    t_peak = float((times[idx] + times[idx + 1]) / 2)
    return peak, t_peak


def current_layer_thickness(J: jnp.ndarray, axis: jnp.ndarray) -> float:
    """Compute FWHM of current density profile.

    Args:
        J: 1D current density profile
        axis: Position array

    Returns:
        Full width at half maximum
    """
    J_max = jnp.max(J)
    half_max = J_max / 2
    above_half = J >= half_max
    indices = jnp.where(above_half)[0]
    if len(indices) < 2:
        return 0.0
    return float(axis[indices[-1]] - axis[indices[0]])
```

Add to `METRIC_FUNCTIONS`:
```python
    'reconnection_rate': reconnection_rate,
    'peak_reconnection_rate': peak_reconnection_rate,
    'current_layer_thickness': current_layer_thickness,
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_metrics.py -k "reconnection or current_layer" -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/validation/metrics.py tests/test_validation_metrics.py
git commit -m "feat(validation): add reconnection and current layer metrics"
```

---

## Task 3: Create CylindricalShockConfiguration

**Files:**
- Create: `jax_frc/configurations/validation_benchmarks.py`
- Modify: `jax_frc/configurations/__init__.py`
- Test: `tests/test_cylindrical_shock.py`

**Step 1: Write the failing test**

Create `tests/test_cylindrical_shock.py`:

```python
"""Tests for CylindricalShockConfiguration."""
import pytest
import jax.numpy as jnp


def test_cylindrical_shock_builds_geometry():
    """Configuration creates cylindrical geometry."""
    from jax_frc.configurations import CylindricalShockConfiguration

    config = CylindricalShockConfiguration()
    geometry = config.build_geometry()

    assert geometry.coord_system == "cylindrical"
    assert geometry.nr == 16  # Minimal r resolution
    assert geometry.nz == 512
    assert geometry.z_min == -1.0
    assert geometry.z_max == 1.0


def test_cylindrical_shock_initial_conditions():
    """Initial state has Brio-Wu left/right states."""
    from jax_frc.configurations import CylindricalShockConfiguration

    config = CylindricalShockConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Left state (z < 0): rho=1.0, p=1.0
    left_idx = geometry.nz // 4
    assert jnp.allclose(state.n[0, left_idx], 1.0, rtol=0.01)
    assert jnp.allclose(state.p[0, left_idx], 1.0, rtol=0.01)

    # Right state (z > 0): rho=0.125, p=0.1
    right_idx = 3 * geometry.nz // 4
    assert jnp.allclose(state.n[0, right_idx], 0.125, rtol=0.01)
    assert jnp.allclose(state.p[0, right_idx], 0.1, rtol=0.01)


def test_cylindrical_shock_magnetic_field():
    """B field has Bz=0.75 constant, Br reverses."""
    from jax_frc.configurations import CylindricalShockConfiguration

    config = CylindricalShockConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Bz should be constant 0.75
    assert jnp.allclose(state.B[:, :, 2], 0.75, rtol=0.01)

    # Br should be +1 on left, -1 on right
    left_idx = geometry.nz // 4
    right_idx = 3 * geometry.nz // 4
    assert state.B[0, left_idx, 0] > 0.5  # Br > 0 on left
    assert state.B[0, right_idx, 0] < -0.5  # Br < 0 on right


def test_cylindrical_shock_builds_model():
    """Configuration creates ResistiveMHD model."""
    from jax_frc.configurations import CylindricalShockConfiguration
    from jax_frc.models.resistive_mhd import ResistiveMHD

    config = CylindricalShockConfiguration()
    model = config.build_model()

    # Should use resistive MHD (not extended)
    assert isinstance(model, ResistiveMHD)


def test_cylindrical_shock_in_registry():
    """Configuration is in the registry."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'CylindricalShockConfiguration' in CONFIGURATION_REGISTRY
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_cylindrical_shock.py::test_cylindrical_shock_builds_geometry -v`
Expected: FAIL with "cannot import name 'CylindricalShockConfiguration'"

**Step 3: Write minimal implementation**

Create `jax_frc/configurations/validation_benchmarks.py`:

```python
"""Validation benchmark configurations (non-FRC-specific)."""
import jax.numpy as jnp
from dataclasses import dataclass

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.resistivity import SpitzerResistivity
from .base import AbstractConfiguration


@dataclass
class CylindricalShockConfiguration(AbstractConfiguration):
    """Z-directed MHD shock tube (Brio-Wu adapted to cylindrical).

    Tests shock-capturing numerics in cylindrical coordinates.
    Initial conditions are r-independent (1D physics in z).
    """

    name: str = "cylindrical_shock"
    description: str = "Brio-Wu shock tube in cylindrical coordinates"

    # Grid parameters
    nr: int = 16  # Minimal r resolution (r-uniform problem)
    nz: int = 512
    r_min: float = 0.01
    r_max: float = 0.5
    z_min: float = -1.0
    z_max: float = 1.0

    # Left state (z < 0)
    rho_L: float = 1.0
    p_L: float = 1.0
    Br_L: float = 1.0

    # Right state (z > 0)
    rho_R: float = 0.125
    p_R: float = 0.1
    Br_R: float = -1.0

    # Common
    Bz: float = 0.75  # Guide field
    gamma: float = 2.0  # Adiabatic index

    def build_geometry(self) -> Geometry:
        return Geometry(
            coord_system="cylindrical",
            r_min=self.r_min, r_max=self.r_max,
            z_min=self.z_min, z_max=self.z_max,
            nr=self.nr, nz=self.nz
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        z = geometry.z_grid

        # Left/right states based on z
        rho = jnp.where(z < 0, self.rho_L, self.rho_R)
        p = jnp.where(z < 0, self.p_L, self.p_R)

        # Magnetic field: Br reverses, Bz constant
        Br = jnp.where(z < 0, self.Br_L, self.Br_R)
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 0].set(Br)
        B = B.at[:, :, 2].set(self.Bz)

        # Temperature from ideal gas: p = n * T (normalized)
        T = p / rho

        return State(
            psi=jnp.zeros((geometry.nr, geometry.nz)),
            n=rho,
            p=p,
            T=T,
            B=B,
            E=jnp.zeros((geometry.nr, geometry.nz, 3)),
            v=jnp.zeros((geometry.nr, geometry.nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

    def build_model(self) -> ResistiveMHD:
        return ResistiveMHD(
            resistivity=SpitzerResistivity(eta_0=1e-8),
            gamma=self.gamma
        )

    def build_boundary_conditions(self) -> list:
        return []  # Dirichlet at z boundaries (fixed states)

    def default_runtime(self) -> dict:
        # t=0.1 in Alfven time units
        return {"t_end": 0.1, "dt": 1e-4}
```

**Step 4: Add to registry**

Add to `jax_frc/configurations/__init__.py`:

After line 39 (after SlabDiffusionConfiguration import), add:
```python
from jax_frc.configurations.validation_benchmarks import (
    CylindricalShockConfiguration,
)
```

In `CONFIGURATION_REGISTRY` dict, add:
```python
    'CylindricalShockConfiguration': CylindricalShockConfiguration,
```

In `__all__` list, add:
```python
    'CylindricalShockConfiguration',
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_cylindrical_shock.py -v`
Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add jax_frc/configurations/validation_benchmarks.py jax_frc/configurations/__init__.py tests/test_cylindrical_shock.py
git commit -m "feat(configurations): add CylindricalShockConfiguration"
```

---

## Task 4: Create Shock YAML Case File

**Files:**
- Create: `validation/cases/analytic/cylindrical_shock.yaml`
- Test: (manual or via ValidationRunner)

**Step 1: Create the YAML file**

Create `validation/cases/analytic/cylindrical_shock.yaml`:

```yaml
# validation/cases/analytic/cylindrical_shock.yaml
name: cylindrical_shock
description: "Brio-Wu MHD shock tube in cylindrical coordinates"

configuration:
  class: CylindricalShockConfiguration
  overrides:
    nz: 512
    gamma: 2.0

runtime:
  t_end: 0.1
  dt: 1e-4

solver:
  type: explicit

acceptance:
  quantitative:
    - metric: shock_position_error
      field: n
      axis: z
      expected: 0.45
      tolerance: "5%"
      description: "Fast shock position within 5%"
    - metric: shock_position_error
      field: n
      axis: z
      expected: 0.28
      tolerance: "5%"
      description: "Slow shock position within 5%"
    - metric: conservation_drift
      field: total_energy
      threshold: 0.01
      description: "Energy conservation within 1%"

output:
  plots:
    - type: profiles
      name: density_profile
      axis: z
      field: n
    - type: profiles
      name: Br_profile
      axis: z
      field: B_r
  html_report: true
  log_level: INFO
```

**Step 2: Verify file exists and is valid YAML**

Run: `py -c "import yaml; yaml.safe_load(open('validation/cases/analytic/cylindrical_shock.yaml'))"`
Expected: No error

**Step 3: Commit**

```bash
git add validation/cases/analytic/cylindrical_shock.yaml
git commit -m "feat(validation): add cylindrical_shock case file"
```

---

## Task 5: Create CylindricalVortexConfiguration

**Files:**
- Modify: `jax_frc/configurations/validation_benchmarks.py`
- Modify: `jax_frc/configurations/__init__.py`
- Test: `tests/test_cylindrical_vortex.py`

**Step 1: Write the failing test**

Create `tests/test_cylindrical_vortex.py`:

```python
"""Tests for CylindricalVortexConfiguration."""
import pytest
import jax.numpy as jnp


def test_cylindrical_vortex_builds_geometry():
    """Configuration creates annulus geometry."""
    from jax_frc.configurations import CylindricalVortexConfiguration

    config = CylindricalVortexConfiguration()
    geometry = config.build_geometry()

    assert geometry.coord_system == "cylindrical"
    assert geometry.r_min == 0.2  # Annulus (avoids axis)
    assert geometry.r_max == 1.2
    assert geometry.nr == 256
    assert geometry.nz == 256


def test_cylindrical_vortex_initial_velocity():
    """Initial velocity has vortex pattern."""
    from jax_frc.configurations import CylindricalVortexConfiguration

    config = CylindricalVortexConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # vr = -v0 * sin(z), so vr should be nonzero
    assert jnp.max(jnp.abs(state.v[:, :, 0])) > 0.5


def test_cylindrical_vortex_initial_magnetic():
    """Initial B field has vortex pattern."""
    from jax_frc.configurations import CylindricalVortexConfiguration

    config = CylindricalVortexConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Br = -B0 * sin(z), so Br should be nonzero
    assert jnp.max(jnp.abs(state.B[:, :, 0])) > 0.5


def test_cylindrical_vortex_uniform_density():
    """Density should be uniform."""
    from jax_frc.configurations import CylindricalVortexConfiguration

    config = CylindricalVortexConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    assert jnp.allclose(state.n, state.n[0, 0], rtol=0.01)


def test_cylindrical_vortex_in_registry():
    """Configuration is in the registry."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'CylindricalVortexConfiguration' in CONFIGURATION_REGISTRY
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_cylindrical_vortex.py::test_cylindrical_vortex_builds_geometry -v`
Expected: FAIL with "cannot import name 'CylindricalVortexConfiguration'"

**Step 3: Write minimal implementation**

Add to `jax_frc/configurations/validation_benchmarks.py`:

```python
@dataclass
class CylindricalVortexConfiguration(AbstractConfiguration):
    """Orszag-Tang vortex adapted to cylindrical annulus.

    Tests nonlinear MHD dynamics, current sheet formation.
    Domain is an annulus to avoid axis singularity.
    """

    name: str = "cylindrical_vortex"
    description: str = "Orszag-Tang vortex in cylindrical annulus"

    # Grid parameters
    nr: int = 256
    nz: int = 256
    r_min: float = 0.2
    r_max: float = 1.2
    z_min: float = 0.0
    z_max: float = 2 * jnp.pi

    # Physics parameters (Orszag-Tang standard)
    v0: float = 1.0
    B0: float = 1.0
    rho0: float = 25.0 / (36.0 * jnp.pi)
    p0: float = 5.0 / (12.0 * jnp.pi)
    gamma: float = 5.0 / 3.0

    def build_geometry(self) -> Geometry:
        return Geometry(
            coord_system="cylindrical",
            r_min=self.r_min, r_max=self.r_max,
            z_min=self.z_min, z_max=float(self.z_max),
            nr=self.nr, nz=self.nz
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        r = geometry.r_grid
        z = geometry.z_grid

        # Normalized radial coordinate for patterns
        r_norm = (r - self.r_min) / (self.r_max - self.r_min)

        # Velocity: vr = -v0*sin(z), vz = v0*sin(2*pi*r_norm)
        vr = -self.v0 * jnp.sin(z)
        vz = self.v0 * jnp.sin(2 * jnp.pi * r_norm)
        v = jnp.zeros((geometry.nr, geometry.nz, 3))
        v = v.at[:, :, 0].set(vr)
        v = v.at[:, :, 2].set(vz)

        # Magnetic field: Br = -B0*sin(z), Bz = B0*sin(4*pi*r_norm)
        Br = -self.B0 * jnp.sin(z)
        Bz = self.B0 * jnp.sin(4 * jnp.pi * r_norm)
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 0].set(Br)
        B = B.at[:, :, 2].set(Bz)

        # Uniform density and pressure
        rho = jnp.ones((geometry.nr, geometry.nz)) * self.rho0
        p = jnp.ones((geometry.nr, geometry.nz)) * self.p0
        T = p / rho

        return State(
            psi=jnp.zeros((geometry.nr, geometry.nz)),
            n=rho,
            p=p,
            T=T,
            B=B,
            E=jnp.zeros((geometry.nr, geometry.nz, 3)),
            v=v,
            particles=None,
            time=0.0,
            step=0
        )

    def build_model(self) -> ResistiveMHD:
        return ResistiveMHD(
            resistivity=SpitzerResistivity(eta_0=1e-6),
            gamma=self.gamma
        )

    def build_boundary_conditions(self) -> list:
        # Periodic in z handled by geometry, conducting at r boundaries
        return []

    def default_runtime(self) -> dict:
        return {"t_end": 0.5, "dt": 1e-4}
```

**Step 4: Add to registry**

In `jax_frc/configurations/__init__.py`, update the import:
```python
from jax_frc.configurations.validation_benchmarks import (
    CylindricalShockConfiguration,
    CylindricalVortexConfiguration,
)
```

Add to registry and `__all__`:
```python
    'CylindricalVortexConfiguration': CylindricalVortexConfiguration,
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_cylindrical_vortex.py -v`
Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add jax_frc/configurations/validation_benchmarks.py jax_frc/configurations/__init__.py tests/test_cylindrical_vortex.py
git commit -m "feat(configurations): add CylindricalVortexConfiguration"
```

---

## Task 6: Create Vortex YAML Case File

**Files:**
- Create: `validation/cases/mhd_regression/` directory
- Create: `validation/cases/mhd_regression/cylindrical_vortex.yaml`

**Step 1: Create directory and YAML file**

```bash
mkdir -p validation/cases/mhd_regression
```

Create `validation/cases/mhd_regression/cylindrical_vortex.yaml`:

```yaml
# validation/cases/mhd_regression/cylindrical_vortex.yaml
name: cylindrical_vortex
description: "Orszag-Tang vortex in cylindrical annulus"

configuration:
  class: CylindricalVortexConfiguration
  overrides:
    nr: 256
    nz: 256

runtime:
  t_end: 0.5
  dt: 1e-4

solver:
  type: explicit

acceptance:
  quantitative:
    - metric: conservation_drift
      field: total_energy
      threshold: 0.01
      description: "Total energy conservation within 1%"
    - metric: check_tolerance
      name: peak_J_timing
      expected: 0.48
      tolerance: "10%"
      description: "Time of peak current density within 10%"

output:
  plots:
    - type: contour
      name: current_density
      field: J
      times: [0.0, 0.25, 0.5]
    - type: time_trace
      name: energy_partition
      fields: [magnetic_energy, kinetic_energy]
  html_report: true
  log_level: INFO
```

**Step 2: Verify file is valid**

Run: `py -c "import yaml; yaml.safe_load(open('validation/cases/mhd_regression/cylindrical_vortex.yaml'))"`
Expected: No error

**Step 3: Commit**

```bash
git add validation/cases/mhd_regression/cylindrical_vortex.yaml
git commit -m "feat(validation): add cylindrical_vortex case file"
```

---

## Task 7: Create CylindricalGEMConfiguration

**Files:**
- Modify: `jax_frc/configurations/validation_benchmarks.py`
- Modify: `jax_frc/configurations/__init__.py`
- Test: `tests/test_cylindrical_gem.py`

**Step 1: Write the failing test**

Create `tests/test_cylindrical_gem.py`:

```python
"""Tests for CylindricalGEMConfiguration."""
import pytest
import jax.numpy as jnp


def test_cylindrical_gem_builds_geometry():
    """Configuration creates correct domain."""
    from jax_frc.configurations import CylindricalGEMConfiguration

    config = CylindricalGEMConfiguration()
    geometry = config.build_geometry()

    assert geometry.coord_system == "cylindrical"
    assert geometry.nr == 256
    assert geometry.nz == 512


def test_cylindrical_gem_harris_sheet():
    """Initial Br follows tanh profile."""
    from jax_frc.configurations import CylindricalGEMConfiguration

    config = CylindricalGEMConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Br should be ~+B0 for z >> lambda, ~-B0 for z << -lambda
    center_r = geometry.nr // 2
    z_idx_pos = 3 * geometry.nz // 4  # z > 0
    z_idx_neg = geometry.nz // 4  # z < 0

    assert state.B[center_r, z_idx_pos, 0] > 0.5 * config.B0
    assert state.B[center_r, z_idx_neg, 0] < -0.5 * config.B0


def test_cylindrical_gem_density_profile():
    """Density has sech^2 + background profile."""
    from jax_frc.configurations import CylindricalGEMConfiguration

    config = CylindricalGEMConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Density should peak at z=0
    center_z = geometry.nz // 2
    edge_z = 0
    center_r = geometry.nr // 2

    assert state.n[center_r, center_z] > state.n[center_r, edge_z]


def test_cylindrical_gem_uses_extended_mhd():
    """Configuration uses ExtendedMHD with Hall."""
    from jax_frc.configurations import CylindricalGEMConfiguration
    from jax_frc.models.extended_mhd import ExtendedMHD

    config = CylindricalGEMConfiguration()
    model = config.build_model()

    assert isinstance(model, ExtendedMHD)


def test_cylindrical_gem_in_registry():
    """Configuration is in the registry."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'CylindricalGEMConfiguration' in CONFIGURATION_REGISTRY
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_cylindrical_gem.py::test_cylindrical_gem_builds_geometry -v`
Expected: FAIL with "cannot import name 'CylindricalGEMConfiguration'"

**Step 3: Write minimal implementation**

Add to `jax_frc/configurations/validation_benchmarks.py`:

First, add import at top:
```python
from jax_frc.models.extended_mhd import ExtendedMHD, HaloDensityModel
```

Then add the class:

```python
@dataclass
class CylindricalGEMConfiguration(AbstractConfiguration):
    """GEM reconnection challenge adapted to cylindrical coordinates.

    Harris sheet current layer with Hall MHD.
    Tests Hall reconnection physics, quadrupole signature.
    """

    name: str = "cylindrical_gem"
    description: str = "GEM magnetic reconnection in cylindrical coordinates"

    # Grid parameters
    nr: int = 256
    nz: int = 512
    r_min: float = 0.01
    r_max: float = 2.0
    z_min: float = -jnp.pi
    z_max: float = jnp.pi

    # Harris sheet parameters
    B0: float = 1.0  # Asymptotic field
    lambda_: float = 0.5  # Current sheet half-width (in d_i units)
    n0: float = 1.0  # Peak density
    n_b: float = 0.2  # Background density (fraction of n0)

    # Perturbation
    psi1: float = 0.1  # Perturbation amplitude (fraction of B0*lambda)

    def build_geometry(self) -> Geometry:
        return Geometry(
            coord_system="cylindrical",
            r_min=self.r_min, r_max=self.r_max,
            z_min=float(self.z_min), z_max=float(self.z_max),
            nr=self.nr, nz=self.nz
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        r = geometry.r_grid
        z = geometry.z_grid

        # Harris sheet: Br = B0 * tanh(z/lambda)
        Br = self.B0 * jnp.tanh(z / self.lambda_)
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 0].set(Br)

        # Density: n = n0 * sech^2(z/lambda) + n_b
        sech_sq = 1.0 / jnp.cosh(z / self.lambda_)**2
        n = self.n0 * sech_sq + self.n_b * self.n0

        # Pressure balance: p + B^2/(2*mu0) = const
        # At z=0: p_max, B=0
        # At z->inf: p_min, B=B0
        p_max = self.B0**2 / 2  # Total pressure (normalized)
        p = p_max - B[:, :, 0]**2 / 2
        p = jnp.maximum(p, 0.01)  # Floor to avoid negative pressure

        T = p / n

        # Add perturbation to seed reconnection
        Lr = self.r_max - self.r_min
        psi_pert = self.psi1 * self.B0 * self.lambda_ * (
            jnp.cos(2 * jnp.pi * (r - self.r_min) / Lr) *
            jnp.cos(z / self.lambda_)
        )

        return State(
            psi=psi_pert,
            n=n,
            p=p,
            T=T,
            B=B,
            E=jnp.zeros((geometry.nr, geometry.nz, 3)),
            v=jnp.zeros((geometry.nr, geometry.nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

    def build_model(self) -> ExtendedMHD:
        return ExtendedMHD(
            resistivity=SpitzerResistivity(eta_0=1e-4),
            halo_model=HaloDensityModel(
                halo_density=self.n_b * self.n0,
                core_density=self.n0
            ),
            # Hall term enabled by default in ExtendedMHD
        )

    def build_boundary_conditions(self) -> list:
        return []

    def default_runtime(self) -> dict:
        return {"t_end": 25.0, "dt": 0.01}  # In Alfven time units
```

**Step 4: Add to registry**

Update imports in `jax_frc/configurations/__init__.py`:
```python
from jax_frc.configurations.validation_benchmarks import (
    CylindricalShockConfiguration,
    CylindricalVortexConfiguration,
    CylindricalGEMConfiguration,
)
```

Add to registry and `__all__`:
```python
    'CylindricalGEMConfiguration': CylindricalGEMConfiguration,
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_cylindrical_gem.py -v`
Expected: PASS (5 tests)

**Step 6: Commit**

```bash
git add jax_frc/configurations/validation_benchmarks.py jax_frc/configurations/__init__.py tests/test_cylindrical_gem.py
git commit -m "feat(configurations): add CylindricalGEMConfiguration"
```

---

## Task 8: Create GEM YAML Case File

**Files:**
- Create: `validation/cases/hall_reconnection/` directory
- Create: `validation/cases/hall_reconnection/cylindrical_gem.yaml`

**Step 1: Create directory and YAML file**

```bash
mkdir -p validation/cases/hall_reconnection
```

Create `validation/cases/hall_reconnection/cylindrical_gem.yaml`:

```yaml
# validation/cases/hall_reconnection/cylindrical_gem.yaml
name: cylindrical_gem
description: "GEM magnetic reconnection challenge in cylindrical coordinates"

configuration:
  class: CylindricalGEMConfiguration
  overrides:
    nr: 256
    nz: 512
    lambda_: 0.5
    psi1: 0.1

runtime:
  t_end: 25.0
  dt: 0.01

solver:
  type: semi_implicit

acceptance:
  quantitative:
    - metric: peak_reconnection_rate
      expected: 0.1
      tolerance: "10%"
      description: "Peak reconnection rate ~0.1 B0*vA"
    - metric: check_tolerance
      name: time_to_peak
      expected: 15.0
      tolerance: "15%"
      description: "Time to peak reconnection rate"
    - metric: current_layer_thickness
      expected: 1.0
      tolerance: "20%"
      description: "Current layer thickness ~d_i"
  qualitative:
    - name: hall_quadrupole
      description: "Btheta shows quadrupole pattern"
      criterion: "max(|Btheta|) > 0.05 * B0"

output:
  plots:
    - type: contour
      name: reconnected_flux
      field: psi
      times: [0, 10, 20, 25]
    - type: contour
      name: hall_quadrupole
      field: B_theta
      times: [15, 20, 25]
    - type: time_trace
      name: reconnection_rate
      fields: [dpsi_dt_max]
  html_report: true
  log_level: INFO
```

**Step 2: Verify file is valid**

Run: `py -c "import yaml; yaml.safe_load(open('validation/cases/hall_reconnection/cylindrical_gem.yaml'))"`
Expected: No error

**Step 3: Commit**

```bash
git add validation/cases/hall_reconnection/cylindrical_gem.yaml
git commit -m "feat(validation): add cylindrical_gem case file"
```

---

## Task 9: Run All Tests and Verify

**Files:** None (verification only)

**Step 1: Run full test suite**

Run: `py -m pytest tests/ -k "not slow" -v`
Expected: All tests pass, including new ones:
- `test_cylindrical_shock.py` (5 tests)
- `test_cylindrical_vortex.py` (5 tests)
- `test_cylindrical_gem.py` (5 tests)
- `test_validation_metrics.py` (existing + 5 new)

**Step 2: Verify YAML files are loadable**

Run:
```bash
py -c "
import yaml
from pathlib import Path
for f in Path('validation/cases').rglob('*.yaml'):
    yaml.safe_load(f.open())
    print(f'OK: {f}')
"
```
Expected: All YAML files parse without error

**Step 3: Final commit if any cleanup needed**

```bash
git status
# If clean, no commit needed
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | Shock position metrics | 2 |
| 2 | Reconnection metrics | 3 |
| 3 | CylindricalShockConfiguration | 5 |
| 4 | cylindrical_shock.yaml | - |
| 5 | CylindricalVortexConfiguration | 5 |
| 6 | cylindrical_vortex.yaml | - |
| 7 | CylindricalGEMConfiguration | 5 |
| 8 | cylindrical_gem.yaml | - |
| 9 | Full verification | - |

**Total new tests:** 20
**New files:** 7 (3 configs in 1 file, 3 YAML, 3 test files)
**Modified files:** 2 (`metrics.py`, `configurations/__init__.py`)
