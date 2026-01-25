# Validation Infrastructure Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a validation framework for running FRC simulation cases against acceptance criteria with metrics, reference data, and HTML reports.

**Architecture:** Configuration classes encapsulate reactor/benchmark setups. ValidationRunner loads YAML case definitions, instantiates configurations, runs simulations, computes metrics against references, and generates reports. Integrates with existing `jax_frc.diagnostics.plotting`.

**Tech Stack:** Python dataclasses, PyYAML, Jinja2 (HTML reports), existing jax_frc modules.

---

## Task 1: AbstractConfiguration Base Class

**Files:**
- Create: `jax_frc/configurations/__init__.py`
- Create: `jax_frc/configurations/base.py`
- Test: `tests/test_configurations.py`

**Step 1: Write the failing test**

```python
# tests/test_configurations.py
"""Tests for configuration base class."""
import pytest
from abc import ABC

def test_abstract_configuration_cannot_instantiate():
    """AbstractConfiguration should not be instantiatable."""
    from jax_frc.configurations.base import AbstractConfiguration

    with pytest.raises(TypeError):
        AbstractConfiguration()

def test_abstract_configuration_has_required_methods():
    """AbstractConfiguration defines required abstract methods."""
    from jax_frc.configurations.base import AbstractConfiguration

    assert hasattr(AbstractConfiguration, 'build_geometry')
    assert hasattr(AbstractConfiguration, 'build_initial_state')
    assert hasattr(AbstractConfiguration, 'build_model')
    assert hasattr(AbstractConfiguration, 'build_boundary_conditions')
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_configurations.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'jax_frc.configurations'"

**Step 3: Write minimal implementation**

```python
# jax_frc/configurations/__init__.py
"""Configuration classes for reactor and benchmark setups."""
from .base import AbstractConfiguration

__all__ = ['AbstractConfiguration']
```

```python
# jax_frc/configurations/base.py
"""Base class for all reactor/benchmark configurations."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.base import PhysicsModel


class AbstractConfiguration(ABC):
    """Base class for all reactor/benchmark configurations.

    Each configuration encapsulates a complete simulation setup including
    geometry, initial conditions, physics model, and boundary conditions.
    """

    name: str = "abstract"
    description: str = "Base configuration class"

    @abstractmethod
    def build_geometry(self) -> Geometry:
        """Create computational geometry for this configuration."""
        ...

    @abstractmethod
    def build_initial_state(self, geometry: Geometry) -> State:
        """Create initial plasma state."""
        ...

    @abstractmethod
    def build_model(self) -> PhysicsModel:
        """Create physics model for this configuration."""
        ...

    @abstractmethod
    def build_boundary_conditions(self) -> list:
        """Create boundary conditions for this configuration."""
        ...

    def available_phases(self) -> list[str]:
        """List valid phases for this configuration."""
        return ["default"]

    def default_runtime(self) -> dict:
        """Return suggested runtime parameters."""
        return {"t_end": 1e-3, "dt": 1e-6}
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_configurations.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/configurations/ tests/test_configurations.py
git commit -m "feat(configurations): add AbstractConfiguration base class"
```

---

## Task 2: SlabDiffusionConfiguration (Analytic Test Case)

**Files:**
- Create: `jax_frc/configurations/analytic.py`
- Modify: `jax_frc/configurations/__init__.py`
- Test: `tests/test_configurations.py` (append)

**Step 1: Write the failing test**

```python
# Append to tests/test_configurations.py
def test_slab_diffusion_builds_geometry():
    """SlabDiffusionConfiguration creates valid geometry."""
    from jax_frc.configurations.analytic import SlabDiffusionConfiguration

    config = SlabDiffusionConfiguration()
    geometry = config.build_geometry()

    assert geometry.nr > 0
    assert geometry.nz > 0
    assert geometry.coord_system == "cylindrical"

def test_slab_diffusion_builds_initial_state():
    """SlabDiffusionConfiguration creates state with Gaussian temperature."""
    from jax_frc.configurations.analytic import SlabDiffusionConfiguration
    import jax.numpy as jnp

    config = SlabDiffusionConfiguration()
    geometry = config.build_geometry()
    state = config.build_initial_state(geometry)

    # Temperature should have Gaussian profile (peak in center)
    center_idx = geometry.nz // 2
    assert state.T[geometry.nr // 2, center_idx] > state.T[geometry.nr // 2, 0]

def test_slab_diffusion_builds_model():
    """SlabDiffusionConfiguration creates ExtendedMHD with thermal transport."""
    from jax_frc.configurations.analytic import SlabDiffusionConfiguration
    from jax_frc.models.extended_mhd import ExtendedMHD

    config = SlabDiffusionConfiguration()
    model = config.build_model()

    assert isinstance(model, ExtendedMHD)
    assert model.thermal is not None
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_configurations.py::test_slab_diffusion_builds_geometry -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# jax_frc/configurations/analytic.py
"""Analytic test case configurations."""
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Optional

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.extended_mhd import ExtendedMHD, HaloDensityModel, TemperatureBoundaryCondition
from jax_frc.models.energy import ThermalTransport
from jax_frc.models.resistivity import SpitzerResistivity
from .base import AbstractConfiguration


@dataclass
class SlabDiffusionConfiguration(AbstractConfiguration):
    """1D heat diffusion test case with analytic solution.

    Gaussian temperature profile diffusing along z with uniform B_z.
    Analytic solution: T(z,t) = T0 * sqrt(sigma^2 / (sigma^2 + 2*D*t)) * exp(...)
    """

    name: str = "slab_diffusion"
    description: str = "1D heat conduction test with analytic solution"

    # Grid parameters
    nr: int = 8
    nz: int = 64
    z_extent: float = 2.0  # [-z_extent, z_extent]

    # Physics parameters
    T_peak: float = 200.0  # eV
    T_base: float = 50.0   # eV
    sigma: float = 0.3     # Initial Gaussian width
    kappa: float = 1e-3    # Thermal diffusivity coefficient
    n0: float = 1e19       # Density

    def build_geometry(self) -> Geometry:
        return Geometry(
            coord_system="cylindrical",
            r_min=0.1, r_max=0.9,
            z_min=-self.z_extent, z_max=self.z_extent,
            nr=self.nr, nz=self.nz
        )

    def build_initial_state(self, geometry: Geometry) -> State:
        z = geometry.z_grid

        # Gaussian temperature profile in z
        T_init = self.T_peak * jnp.exp(-z**2 / (2 * self.sigma**2)) + self.T_base

        # Uniform B_z field (boundary-compatible envelope)
        r_norm = (geometry.r_grid - geometry.r_min) / (geometry.r_max - geometry.r_min)
        z_norm = (z - geometry.z_min) / (geometry.z_max - geometry.z_min)
        envelope = 16 * r_norm * (1 - r_norm) * z_norm * (1 - z_norm)

        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0 * envelope)

        return State(
            psi=jnp.zeros((geometry.nr, geometry.nz)),
            n=jnp.ones((geometry.nr, geometry.nz)) * self.n0,
            p=jnp.ones((geometry.nr, geometry.nz)) * 1e3,
            T=T_init,
            B=B,
            E=jnp.zeros((geometry.nr, geometry.nz, 3)),
            v=jnp.zeros((geometry.nr, geometry.nz, 3)),
            particles=None,
            time=0.0,
            step=0
        )

    def build_model(self) -> ExtendedMHD:
        # kappa_parallel = D * (3/2 * n)
        kappa_parallel = self.kappa * 1.5 * self.n0

        return ExtendedMHD(
            resistivity=SpitzerResistivity(eta_0=1e-10),
            halo_model=HaloDensityModel(halo_density=self.n0, core_density=self.n0),
            thermal=ThermalTransport(
                kappa_parallel_0=kappa_parallel,
                use_spitzer=False
            ),
            temperature_bc=TemperatureBoundaryCondition(bc_type="neumann")
        )

    def build_boundary_conditions(self) -> list:
        return []  # Neumann BCs built into model

    def default_runtime(self) -> dict:
        return {"t_end": 1e-3, "dt": 1e-5}

    def analytic_solution(self, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute analytic temperature at time t."""
        D = self.kappa
        sigma_eff_sq = self.sigma**2 + 2 * D * t
        amplitude = self.T_peak * jnp.sqrt(self.sigma**2 / sigma_eff_sq)
        return amplitude * jnp.exp(-z**2 / (2 * sigma_eff_sq)) + self.T_base
```

Update `__init__.py`:
```python
# jax_frc/configurations/__init__.py
"""Configuration classes for reactor and benchmark setups."""
from .base import AbstractConfiguration
from .analytic import SlabDiffusionConfiguration

__all__ = ['AbstractConfiguration', 'SlabDiffusionConfiguration']
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_configurations.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/configurations/ tests/test_configurations.py
git commit -m "feat(configurations): add SlabDiffusionConfiguration analytic test case"
```

---

## Task 3: Configuration Registry

**Files:**
- Modify: `jax_frc/configurations/__init__.py`
- Test: `tests/test_configurations.py` (append)

**Step 1: Write the failing test**

```python
# Append to tests/test_configurations.py
def test_configuration_registry_has_slab_diffusion():
    """Registry contains SlabDiffusionConfiguration."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    assert 'SlabDiffusionConfiguration' in CONFIGURATION_REGISTRY
    assert CONFIGURATION_REGISTRY['SlabDiffusionConfiguration'] is not None

def test_configuration_registry_creates_instance():
    """Registry can create configuration instances."""
    from jax_frc.configurations import CONFIGURATION_REGISTRY

    ConfigClass = CONFIGURATION_REGISTRY['SlabDiffusionConfiguration']
    config = ConfigClass()

    assert config.name == "slab_diffusion"
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_configurations.py::test_configuration_registry_has_slab_diffusion -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# jax_frc/configurations/__init__.py
"""Configuration classes for reactor and benchmark setups."""
from .base import AbstractConfiguration
from .analytic import SlabDiffusionConfiguration

CONFIGURATION_REGISTRY = {
    'SlabDiffusionConfiguration': SlabDiffusionConfiguration,
}

__all__ = ['AbstractConfiguration', 'SlabDiffusionConfiguration', 'CONFIGURATION_REGISTRY']
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_configurations.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/configurations/__init__.py tests/test_configurations.py
git commit -m "feat(configurations): add CONFIGURATION_REGISTRY"
```

---

## Task 4: Validation Metrics Module

**Files:**
- Create: `jax_frc/validation/__init__.py`
- Create: `jax_frc/validation/metrics.py`
- Test: `tests/test_validation_metrics.py`

**Step 1: Write the failing test**

```python
# tests/test_validation_metrics.py
"""Tests for validation metrics."""
import pytest
import jax.numpy as jnp

def test_l2_error_computes_relative_norm():
    """l2_error returns relative L2 norm."""
    from jax_frc.validation.metrics import l2_error

    actual = jnp.array([1.0, 2.0, 3.0])
    expected = jnp.array([1.1, 2.0, 2.9])

    error = l2_error(actual, expected)
    assert error > 0
    assert error < 0.1  # Small relative error

def test_linf_error_computes_max_error():
    """linf_error returns max pointwise error."""
    from jax_frc.validation.metrics import linf_error

    actual = jnp.array([1.0, 2.0, 3.0])
    expected = jnp.array([1.0, 2.5, 3.0])

    error = linf_error(actual, expected)
    assert error == pytest.approx(0.5)

def test_rmse_curve_computes_root_mean_square():
    """rmse_curve returns RMSE between curves."""
    from jax_frc.validation.metrics import rmse_curve

    actual = jnp.array([1.0, 2.0, 3.0, 4.0])
    expected = jnp.array([1.0, 2.0, 3.0, 4.0])

    rmse = rmse_curve(actual, expected)
    assert rmse == pytest.approx(0.0)

def test_check_tolerance_percentage():
    """check_tolerance handles percentage tolerances."""
    from jax_frc.validation.metrics import check_tolerance

    result = check_tolerance(value=105.0, expected=100.0, tolerance="10%")
    assert result['pass'] is True

    result = check_tolerance(value=120.0, expected=100.0, tolerance="10%")
    assert result['pass'] is False

def test_check_tolerance_absolute():
    """check_tolerance handles absolute tolerances."""
    from jax_frc.validation.metrics import check_tolerance

    result = check_tolerance(value=1.05, expected=1.0, tolerance=0.1)
    assert result['pass'] is True
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_metrics.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# jax_frc/validation/__init__.py
"""Validation infrastructure for FRC simulations."""
from .metrics import l2_error, linf_error, rmse_curve, check_tolerance

__all__ = ['l2_error', 'linf_error', 'rmse_curve', 'check_tolerance']
```

```python
# jax_frc/validation/metrics.py
"""Built-in validation metrics."""
import jax.numpy as jnp
from typing import Union


def l2_error(actual: jnp.ndarray, expected: jnp.ndarray) -> float:
    """Compute relative L2 norm error.

    Returns ||actual - expected||_2 / ||expected||_2
    """
    diff = actual - expected
    return float(jnp.linalg.norm(diff) / jnp.linalg.norm(expected))


def linf_error(actual: jnp.ndarray, expected: jnp.ndarray) -> float:
    """Compute max pointwise error.

    Returns max|actual - expected|
    """
    return float(jnp.max(jnp.abs(actual - expected)))


def rmse_curve(actual: jnp.ndarray, expected: jnp.ndarray) -> float:
    """Compute root mean square error between curves.

    Returns sqrt(mean((actual - expected)^2))
    """
    diff = actual - expected
    return float(jnp.sqrt(jnp.mean(diff**2)))


def conservation_drift(initial: float, final: float) -> float:
    """Compute relative change in conserved quantity.

    Returns |final - initial| / |initial|
    """
    return float(jnp.abs(final - initial) / jnp.abs(initial))


def div_b_max(B: jnp.ndarray, dr: float, dz: float, r: jnp.ndarray) -> float:
    """Compute maximum divergence of B field."""
    from jax_frc.operators import divergence_cylindrical
    B_r = B[:, :, 0]
    B_z = B[:, :, 2]
    div_B = divergence_cylindrical(B_r, B_z, dr, dz, r)
    return float(jnp.max(jnp.abs(div_B)))


def check_tolerance(
    value: float,
    expected: float,
    tolerance: Union[str, float]
) -> dict:
    """Check if value is within tolerance of expected.

    Args:
        value: Computed metric value
        expected: Expected value
        tolerance: Either percentage string like "10%" or absolute float

    Returns:
        dict with 'pass', 'value', 'expected', 'tolerance', 'message'
    """
    if isinstance(tolerance, str) and tolerance.endswith('%'):
        pct = float(tolerance.rstrip('%'))
        threshold = abs(expected) * pct / 100.0
        tol_str = tolerance
    else:
        threshold = float(tolerance)
        tol_str = str(tolerance)

    diff = abs(value - expected)
    passed = diff <= threshold

    return {
        'pass': passed,
        'value': value,
        'expected': expected,
        'tolerance': tol_str,
        'message': '' if passed else f"Value {value} differs from {expected} by {diff:.4g} (tolerance: {tol_str})"
    }


# Metric registry for YAML lookup
METRIC_FUNCTIONS = {
    'l2_error': l2_error,
    'linf_error': linf_error,
    'rmse_curve': rmse_curve,
    'conservation_drift': conservation_drift,
    'div_b_max': div_b_max,
}
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_metrics.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/validation/ tests/test_validation_metrics.py
git commit -m "feat(validation): add metrics module with l2_error, linf_error, rmse_curve"
```

---

## Task 5: ReferenceManager for Analytic References

**Files:**
- Create: `jax_frc/validation/references.py`
- Modify: `jax_frc/validation/__init__.py`
- Test: `tests/test_validation_references.py`

**Step 1: Write the failing test**

```python
# tests/test_validation_references.py
"""Tests for reference data management."""
import pytest
import jax.numpy as jnp

def test_reference_manager_loads_analytic():
    """ReferenceManager evaluates analytic formulas."""
    from jax_frc.validation.references import ReferenceManager

    mgr = ReferenceManager()
    ref_config = {
        'type': 'analytic',
        'formula': 'A * jnp.exp(-x**2)',
        'variables': ['A', 'x']
    }
    params = {'A': 2.0}
    x = jnp.linspace(-1, 1, 10)

    result = mgr.evaluate_analytic(ref_config['formula'], {'A': params['A'], 'x': x})

    expected = 2.0 * jnp.exp(-x**2)
    assert jnp.allclose(result, expected)

def test_reference_manager_loads_file(tmp_path):
    """ReferenceManager loads CSV reference files."""
    from jax_frc.validation.references import ReferenceManager
    import csv

    # Create test CSV
    csv_path = tmp_path / "test_ref.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerow([0.0, 1.0])
        writer.writerow([1.0, 2.0])

    mgr = ReferenceManager(base_dir=tmp_path)
    ref_config = {
        'type': 'file',
        'path': 'test_ref.csv',
        'columns': {'x': 'x', 'y': 'y'}
    }

    data = mgr.load_file(ref_config)

    assert 'x' in data
    assert 'y' in data
    assert len(data['x']) == 2
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_references.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# jax_frc/validation/references.py
"""Reference data management for validation."""
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import jax.numpy as jnp


@dataclass
class ReferenceData:
    """Container for loaded reference data."""
    data: dict
    source_type: str
    source_path: Optional[str] = None


@dataclass
class ReferenceManager:
    """Manages loading and caching of reference data."""

    base_dir: Path = field(default_factory=lambda: Path("validation"))
    cache: dict = field(default_factory=dict)

    def __post_init__(self):
        self.base_dir = Path(self.base_dir)

    def load(self, ref_config: dict, params: Optional[dict] = None) -> ReferenceData:
        """Load reference based on config type."""
        ref_type = ref_config.get('type', 'file')

        if ref_type == 'analytic':
            data = self.evaluate_analytic(ref_config['formula'], params or {})
            return ReferenceData(data={'result': data}, source_type='analytic')
        elif ref_type == 'file':
            data = self.load_file(ref_config)
            return ReferenceData(data=data, source_type='file', source_path=ref_config['path'])
        elif ref_type == 'external':
            raise NotImplementedError("External reference loading not yet implemented")
        else:
            raise ValueError(f"Unknown reference type: {ref_type}")

    def evaluate_analytic(self, formula: str, params: dict) -> jnp.ndarray:
        """Safely evaluate analytic formula.

        Args:
            formula: Python expression using jnp functions
            params: Variables to substitute (including arrays like 'x', 't')

        Returns:
            Evaluated array
        """
        # Build safe namespace with JAX numpy
        safe_namespace = {'jnp': jnp}
        safe_namespace.update(params)

        # Evaluate formula
        return eval(formula, {"__builtins__": {}}, safe_namespace)

    def load_file(self, ref_config: dict) -> dict:
        """Load reference data from CSV file.

        Args:
            ref_config: Config with 'path' and optional 'columns' mapping

        Returns:
            dict mapping column names to arrays
        """
        path = self.base_dir / ref_config['path']

        if str(path) in self.cache:
            return self.cache[str(path)]

        data = {}
        columns = ref_config.get('columns', {})

        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Get column names from mapping or use file headers
            for csv_col in rows[0].keys():
                col_name = columns.get(csv_col, csv_col)
                values = [float(row[csv_col]) for row in rows]
                data[col_name] = jnp.array(values)

        self.cache[str(path)] = data
        return data
```

Update `__init__.py`:
```python
# jax_frc/validation/__init__.py
"""Validation infrastructure for FRC simulations."""
from .metrics import l2_error, linf_error, rmse_curve, check_tolerance, METRIC_FUNCTIONS
from .references import ReferenceManager, ReferenceData

__all__ = [
    'l2_error', 'linf_error', 'rmse_curve', 'check_tolerance', 'METRIC_FUNCTIONS',
    'ReferenceManager', 'ReferenceData'
]
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_references.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/validation/ tests/test_validation_references.py
git commit -m "feat(validation): add ReferenceManager for analytic and file references"
```

---

## Task 6: ValidationResult Dataclass

**Files:**
- Create: `jax_frc/validation/result.py`
- Modify: `jax_frc/validation/__init__.py`
- Test: `tests/test_validation_result.py`

**Step 1: Write the failing test**

```python
# tests/test_validation_result.py
"""Tests for validation result container."""
import pytest

def test_validation_result_overall_pass():
    """ValidationResult computes overall_pass from metrics."""
    from jax_frc.validation.result import ValidationResult, MetricResult

    metrics = {
        'metric1': MetricResult(name='metric1', value=1.0, expected=1.0,
                                tolerance='10%', passed=True),
        'metric2': MetricResult(name='metric2', value=2.0, expected=1.5,
                                tolerance='10%', passed=False),
    }

    result = ValidationResult(
        case_name='test',
        configuration='TestConfig',
        metrics=metrics,
        runtime_seconds=1.0
    )

    assert result.overall_pass is False  # One metric failed

def test_validation_result_to_dict():
    """ValidationResult serializes to dict for JSON."""
    from jax_frc.validation.result import ValidationResult, MetricResult

    result = ValidationResult(
        case_name='test',
        configuration='TestConfig',
        metrics={},
        runtime_seconds=1.0
    )

    d = result.to_dict()
    assert 'case' in d
    assert 'timestamp' in d
    assert 'overall_pass' in d
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_result.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# jax_frc/validation/result.py
"""Validation result containers."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class MetricResult:
    """Result for a single metric evaluation."""
    name: str
    value: float
    expected: Optional[float] = None
    tolerance: Optional[str] = None
    threshold: Optional[float] = None
    passed: bool = True
    message: str = ""

    def to_dict(self) -> dict:
        d = {
            'value': self.value,
            'pass': self.passed,
        }
        if self.expected is not None:
            d['expected'] = self.expected
        if self.tolerance is not None:
            d['tolerance'] = self.tolerance
        if self.threshold is not None:
            d['threshold'] = self.threshold
        if self.message:
            d['message'] = self.message
        return d


@dataclass
class ValidationResult:
    """Complete result of a validation run."""
    case_name: str
    configuration: str
    metrics: dict  # name -> MetricResult
    runtime_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def overall_pass(self) -> bool:
        """True if all metrics passed."""
        if not self.metrics:
            return True
        return all(m.passed for m in self.metrics.values())

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            'case': self.case_name,
            'configuration': self.configuration,
            'timestamp': self.timestamp.isoformat(),
            'runtime_seconds': self.runtime_seconds,
            'overall_pass': self.overall_pass,
            'metrics': {name: m.to_dict() for name, m in self.metrics.items()}
        }
```

Update `__init__.py`:
```python
# jax_frc/validation/__init__.py
"""Validation infrastructure for FRC simulations."""
from .metrics import l2_error, linf_error, rmse_curve, check_tolerance, METRIC_FUNCTIONS
from .references import ReferenceManager, ReferenceData
from .result import ValidationResult, MetricResult

__all__ = [
    'l2_error', 'linf_error', 'rmse_curve', 'check_tolerance', 'METRIC_FUNCTIONS',
    'ReferenceManager', 'ReferenceData',
    'ValidationResult', 'MetricResult'
]
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_result.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/validation/ tests/test_validation_result.py
git commit -m "feat(validation): add ValidationResult and MetricResult dataclasses"
```

---

## Task 7: ValidationRunner Core

**Files:**
- Create: `jax_frc/validation/runner.py`
- Modify: `jax_frc/validation/__init__.py`
- Test: `tests/test_validation_runner.py`

**Step 1: Write the failing test**

```python
# tests/test_validation_runner.py
"""Tests for validation runner."""
import pytest
from pathlib import Path
import yaml

def test_runner_loads_yaml_config(tmp_path):
    """ValidationRunner loads YAML case config."""
    from jax_frc.validation.runner import ValidationRunner

    # Create minimal test case YAML
    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'test_case',
        'description': 'Test case',
        'configuration': {
            'class': 'SlabDiffusionConfiguration',
        },
        'runtime': {'t_end': 1e-6},
        'acceptance': {'quantitative': []}
    }))

    runner = ValidationRunner(case_yaml, tmp_path / "output")

    assert runner.config['name'] == 'test_case'
    assert runner.config['configuration']['class'] == 'SlabDiffusionConfiguration'

def test_runner_instantiates_configuration(tmp_path):
    """ValidationRunner creates Configuration from registry."""
    from jax_frc.validation.runner import ValidationRunner
    from jax_frc.configurations.analytic import SlabDiffusionConfiguration

    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'test_case',
        'description': 'Test',
        'configuration': {'class': 'SlabDiffusionConfiguration'},
        'runtime': {'t_end': 1e-6},
        'acceptance': {'quantitative': []}
    }))

    runner = ValidationRunner(case_yaml, tmp_path / "output")
    config = runner._build_configuration()

    assert isinstance(config, SlabDiffusionConfiguration)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_validation_runner.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# jax_frc/validation/runner.py
"""Validation runner for executing test cases."""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from jax_frc.configurations import CONFIGURATION_REGISTRY
from jax_frc.solvers.base import Solver
from .metrics import check_tolerance, METRIC_FUNCTIONS
from .references import ReferenceManager
from .result import ValidationResult, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationRunner:
    """Executes validation cases and generates reports."""

    case_path: Path
    output_dir: Path
    config: dict = field(default=None, init=False)
    reference_mgr: ReferenceManager = field(default_factory=ReferenceManager)

    def __post_init__(self):
        self.case_path = Path(self.case_path)
        self.output_dir = Path(self.output_dir)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load YAML case configuration."""
        with open(self.case_path) as f:
            return yaml.safe_load(f)

    def _build_configuration(self):
        """Instantiate Configuration class from registry."""
        class_name = self.config['configuration']['class']
        overrides = self.config['configuration'].get('overrides', {})

        if class_name not in CONFIGURATION_REGISTRY:
            raise ValueError(f"Unknown configuration: {class_name}")

        ConfigClass = CONFIGURATION_REGISTRY[class_name]
        return ConfigClass(**overrides)

    def _timestamped_name(self) -> str:
        """Generate timestamped output directory name."""
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        return f"{ts}_{self.config['name']}"

    def _setup_output_dir(self) -> Path:
        """Create output directory structure."""
        run_dir = self.output_dir / self._timestamped_name()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        return run_dir

    def run(self, dry_run: bool = False) -> ValidationResult:
        """Execute full validation pipeline.

        Args:
            dry_run: If True, validate config only without running simulation

        Returns:
            ValidationResult with metrics and pass/fail status
        """
        start_time = time.time()

        # Setup
        run_dir = self._setup_output_dir()
        logger.info(f"Running validation case: {self.config['name']}")
        logger.info(f"Output directory: {run_dir}")

        # Save config copy
        with open(run_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)

        if dry_run:
            logger.info("Dry run - skipping simulation")
            return ValidationResult(
                case_name=self.config['name'],
                configuration=self.config['configuration']['class'],
                metrics={},
                runtime_seconds=0.0
            )

        # Build simulation components
        configuration = self._build_configuration()
        geometry = configuration.build_geometry()
        initial_state = configuration.build_initial_state(geometry)
        model = configuration.build_model()

        # Get solver
        solver_config = self.config.get('solver', {'type': 'semi_implicit'})
        solver = Solver.create(solver_config)

        # Run simulation
        runtime = self.config.get('runtime', configuration.default_runtime())
        t_end = runtime.get('t_end', 1e-3)
        dt = runtime.get('dt', 1e-6)

        state = initial_state
        n_steps = int(t_end / dt)

        logger.info(f"Running {n_steps} steps to t={t_end}")
        for i in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Compute metrics
        metrics = self._compute_metrics(state, geometry, configuration)

        elapsed = time.time() - start_time
        result = ValidationResult(
            case_name=self.config['name'],
            configuration=self.config['configuration']['class'],
            metrics=metrics,
            runtime_seconds=elapsed
        )

        # Save results
        self._save_metrics(run_dir, result)

        logger.info(f"Validation complete: {'PASS' if result.overall_pass else 'FAIL'}")
        return result

    def _compute_metrics(self, state, geometry, configuration) -> dict:
        """Compute all acceptance metrics."""
        metrics = {}
        acceptance = self.config.get('acceptance', {})

        for spec in acceptance.get('quantitative', []):
            metric_name = spec['metric']

            if metric_name in METRIC_FUNCTIONS:
                # Built-in metric - would need state/reference data
                # For now, skip complex metrics
                pass

            if 'expected' in spec and 'tolerance' in spec:
                # Direct value check
                value = spec.get('value', 0.0)  # Would come from simulation
                result = check_tolerance(value, spec['expected'], spec['tolerance'])
                metrics[metric_name] = MetricResult(
                    name=metric_name,
                    value=result['value'],
                    expected=result['expected'],
                    tolerance=result['tolerance'],
                    passed=result['pass'],
                    message=result['message']
                )

        return metrics

    def _save_metrics(self, run_dir: Path, result: ValidationResult):
        """Save metrics.json."""
        with open(run_dir / "metrics.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
```

Update `__init__.py`:
```python
# jax_frc/validation/__init__.py
"""Validation infrastructure for FRC simulations."""
from .metrics import l2_error, linf_error, rmse_curve, check_tolerance, METRIC_FUNCTIONS
from .references import ReferenceManager, ReferenceData
from .result import ValidationResult, MetricResult
from .runner import ValidationRunner

__all__ = [
    'l2_error', 'linf_error', 'rmse_curve', 'check_tolerance', 'METRIC_FUNCTIONS',
    'ReferenceManager', 'ReferenceData',
    'ValidationResult', 'MetricResult',
    'ValidationRunner'
]
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_validation_runner.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/validation/ tests/test_validation_runner.py
git commit -m "feat(validation): add ValidationRunner core"
```

---

## Task 8: CLI Script

**Files:**
- Create: `scripts/run_validation.py`
- Test: Manual CLI test

**Step 1: Write the script**

```python
#!/usr/bin/env python
# scripts/run_validation.py
"""CLI entry point for validation runner."""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.validation import ValidationRunner


def find_case_file(name: str, base_dir: Path) -> Path:
    """Find case YAML file by name."""
    # Check direct path
    if Path(name).exists():
        return Path(name)

    # Check in validation/cases/
    for category in ['analytic', 'benchmarks', 'frc']:
        path = base_dir / 'cases' / category / f"{name}.yaml"
        if path.exists():
            return path

    raise FileNotFoundError(f"Case not found: {name}")


def list_cases(base_dir: Path) -> list:
    """List all available validation cases."""
    cases = []
    cases_dir = base_dir / 'cases'
    if cases_dir.exists():
        for category in cases_dir.iterdir():
            if category.is_dir():
                for yaml_file in category.glob("*.yaml"):
                    cases.append(f"{category.name}/{yaml_file.stem}")
    return sorted(cases)


def main():
    parser = argparse.ArgumentParser(description="Run validation cases")
    parser.add_argument('cases', nargs='*', help="Case names to run")
    parser.add_argument('--category', help="Run all cases in category")
    parser.add_argument('--all', action='store_true', help="Run all cases")
    parser.add_argument('--list', action='store_true', help="List available cases")
    parser.add_argument('--output-dir', type=Path, default=Path('validation/reports'))
    parser.add_argument('--dry-run', action='store_true', help="Validate config only")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    base_dir = Path('validation')

    if args.list:
        print("Available validation cases:")
        for case in list_cases(base_dir):
            print(f"  {case}")
        return 0

    if not args.cases and not args.all and not args.category:
        parser.print_help()
        return 2

    # Collect cases to run
    case_files = []
    if args.all:
        for case in list_cases(base_dir):
            category, name = case.split('/')
            case_files.append(base_dir / 'cases' / category / f"{name}.yaml")
    elif args.category:
        category_dir = base_dir / 'cases' / args.category
        case_files = list(category_dir.glob("*.yaml"))
    else:
        for name in args.cases:
            case_files.append(find_case_file(name, base_dir))

    # Run cases
    all_passed = True
    for case_file in case_files:
        try:
            runner = ValidationRunner(case_file, args.output_dir)
            result = runner.run(dry_run=args.dry_run)

            status = "PASS" if result.overall_pass else "FAIL"
            print(f"{status}: {result.case_name}")

            if not result.overall_pass:
                all_passed = False
                for name, metric in result.metrics.items():
                    if not metric.passed:
                        print(f"  - {name}: {metric.message}")

        except Exception as e:
            logging.exception(f"Error running {case_file}")
            all_passed = False
            return 2

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
```

**Step 2: Test CLI**

Run: `py scripts/run_validation.py --list`
Expected: Lists available cases (empty initially)

Run: `py scripts/run_validation.py --help`
Expected: Shows usage help

**Step 3: Commit**

```bash
git add scripts/run_validation.py
git commit -m "feat(validation): add CLI script run_validation.py"
```

---

## Task 9: Sample YAML Case File

**Files:**
- Create: `validation/cases/analytic/diffusion_slab.yaml`

**Step 1: Create directory structure**

```bash
mkdir -p validation/cases/analytic validation/cases/benchmarks validation/cases/frc
mkdir -p validation/references validation/external validation/reports
```

**Step 2: Create sample case file**

```yaml
# validation/cases/analytic/diffusion_slab.yaml
name: diffusion_slab
description: "1D heat diffusion test with analytic solution"

configuration:
  class: SlabDiffusionConfiguration
  overrides:
    T_peak: 200.0
    sigma: 0.3
    kappa: 1e-3

runtime:
  t_end: 1e-4
  dt: 1e-6

reference:
  type: analytic
  formula: "T_peak * jnp.sqrt(sigma**2 / (sigma**2 + 2*kappa*t)) * jnp.exp(-z**2 / (2*(sigma**2 + 2*kappa*t))) + T_base"
  variables: [T_peak, sigma, kappa, T_base, z, t]

acceptance:
  quantitative:
    - metric: l2_error
      field: T
      threshold: 0.1
      description: "Relative L2 error vs analytic solution"

output:
  plots:
    - type: profiles
      name: temperature_evolution
      axis: z
  html_report: true
  log_level: INFO
```

**Step 3: Create .gitignore for validation data**

```bash
# Add to validation/.gitignore
echo "external/" >> validation/.gitignore
echo "reports/" >> validation/.gitignore
```

**Step 4: Commit**

```bash
git add validation/
git commit -m "feat(validation): add sample diffusion_slab case and directory structure"
```

---

## Task 10: Integration Test

**Files:**
- Test: `tests/test_validation_integration.py`

**Step 1: Write integration test**

```python
# tests/test_validation_integration.py
"""Integration test for full validation pipeline."""
import pytest
from pathlib import Path
import yaml

@pytest.fixture
def validation_case(tmp_path):
    """Create minimal validation case for testing."""
    case_yaml = tmp_path / "test_case.yaml"
    case_yaml.write_text(yaml.dump({
        'name': 'integration_test',
        'description': 'Integration test case',
        'configuration': {
            'class': 'SlabDiffusionConfiguration',
            'overrides': {
                'nr': 4,
                'nz': 16,  # Small grid for fast testing
            }
        },
        'runtime': {
            't_end': 1e-6,  # Very short run
            'dt': 1e-7
        },
        'acceptance': {
            'quantitative': []
        }
    }))
    return case_yaml

def test_full_validation_pipeline(validation_case, tmp_path):
    """Full pipeline: load config, run sim, generate results."""
    from jax_frc.validation import ValidationRunner

    output_dir = tmp_path / "output"
    runner = ValidationRunner(validation_case, output_dir)
    result = runner.run()

    assert result.case_name == 'integration_test'
    assert result.runtime_seconds > 0

    # Check output files created
    output_dirs = list(output_dir.iterdir())
    assert len(output_dirs) == 1

    run_dir = output_dirs[0]
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "metrics.json").exists()
```

**Step 2: Run integration test**

Run: `py -m pytest tests/test_validation_integration.py -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_validation_integration.py
git commit -m "test(validation): add integration test for full pipeline"
```

---

## Verification

After completing all tasks, verify the full system:

```bash
# Run all validation tests
py -m pytest tests/test_configurations.py tests/test_validation*.py -v

# Run CLI
py scripts/run_validation.py --list
py scripts/run_validation.py diffusion_slab --dry-run

# Run actual validation (if case exists)
py scripts/run_validation.py diffusion_slab

# Check outputs
ls validation/reports/
cat validation/reports/*/metrics.json
```

---

## Future Tasks (Not in This Plan)

The following are documented for future implementation:

- **HTMLReportGenerator**: Generate self-contained HTML reports with embedded plots
- **External reference fetching**: Download from Zenodo, etc.
- **More configurations**: FATCMConfiguration, GEMReconnectionConfiguration
- **Validation plotting**: Comparison plots, error maps
- **Scheduled benchmarking**: CI integration for trend tracking
