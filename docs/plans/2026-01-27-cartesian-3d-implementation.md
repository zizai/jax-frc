# 3D Cartesian Coordinate System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the cylindrical (r, z) coordinate system with full 3D Cartesian (x, y, z) to enable non-axisymmetric physics.

**Architecture:** Direct B-field evolution via induction equation, 3D differential operators with flexible per-axis boundary conditions, dimension-by-dimension flux sweeps for neutrals.

**Tech Stack:** JAX (jnp, jit, lax.scan), pytest, existing jax_frc patterns.

---

## Task 1: 3D Geometry Class

**Files:**
- Modify: `jax_frc/core/geometry.py`
- Test: `tests/test_geometry_3d.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_geometry_3d.py
"""Tests for 3D Cartesian geometry."""

import jax.numpy as jnp
import pytest
from jax_frc.core.geometry import Geometry


class TestGeometry3D:
    """Test 3D Cartesian geometry."""

    def test_geometry_creation(self):
        """Test creating a 3D geometry."""
        geom = Geometry(
            nx=8, ny=8, nz=16,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-2.0, z_max=2.0,
            bc_x="periodic", bc_y="periodic", bc_z="dirichlet"
        )
        assert geom.nx == 8
        assert geom.ny == 8
        assert geom.nz == 16

    def test_grid_spacing(self):
        """Test grid spacing calculation."""
        geom = Geometry(
            nx=10, ny=10, nz=20,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=2.0,
        )
        assert jnp.isclose(geom.dx, 0.1)
        assert jnp.isclose(geom.dy, 0.1)
        assert jnp.isclose(geom.dz, 0.1)

    def test_coordinate_arrays(self):
        """Test 1D coordinate arrays."""
        geom = Geometry(
            nx=4, ny=4, nz=4,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
        )
        assert geom.x.shape == (4,)
        assert geom.y.shape == (4,)
        assert geom.z.shape == (4,)
        # Cell centers at 0.125, 0.375, 0.625, 0.875
        assert jnp.isclose(geom.x[0], 0.125)

    def test_3d_grids(self):
        """Test 3D meshgrid arrays."""
        geom = Geometry(
            nx=4, ny=6, nz=8,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
        )
        assert geom.x_grid.shape == (4, 6, 8)
        assert geom.y_grid.shape == (4, 6, 8)
        assert geom.z_grid.shape == (4, 6, 8)

    def test_cell_volumes(self):
        """Test cell volumes are dx * dy * dz."""
        geom = Geometry(
            nx=4, ny=4, nz=4,
            x_min=0.0, x_max=2.0,
            y_min=0.0, y_max=2.0,
            z_min=0.0, z_max=2.0,
        )
        # dx = dy = dz = 0.5, volume = 0.125
        assert geom.cell_volumes.shape == (4, 4, 4)
        assert jnp.allclose(geom.cell_volumes, 0.125)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_geometry_3d.py -v`
Expected: FAIL with import/attribute errors

**Step 3: Write minimal implementation**

Replace `jax_frc/core/geometry.py` with:

```python
"""3D Cartesian geometry for plasma simulations."""

from dataclasses import dataclass
from typing import Literal
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class Geometry:
    """3D Cartesian computational geometry.

    Attributes:
        nx, ny, nz: Number of grid cells in each direction
        x_min, x_max: Domain bounds in x
        y_min, y_max: Domain bounds in y
        z_min, z_max: Domain bounds in z
        bc_x, bc_y, bc_z: Boundary condition type per axis
    """
    nx: int
    ny: int
    nz: int
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    z_min: float = -1.0
    z_max: float = 1.0
    bc_x: Literal["periodic", "dirichlet", "neumann"] = "periodic"
    bc_y: Literal["periodic", "dirichlet", "neumann"] = "periodic"
    bc_z: Literal["periodic", "dirichlet", "neumann"] = "dirichlet"

    @property
    def dx(self) -> float:
        """Grid spacing in x."""
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        """Grid spacing in y."""
        return (self.y_max - self.y_min) / self.ny

    @property
    def dz(self) -> float:
        """Grid spacing in z."""
        return (self.z_max - self.z_min) / self.nz

    @property
    def x(self) -> Array:
        """1D array of cell-centered x coordinates."""
        return jnp.linspace(
            self.x_min + self.dx / 2,
            self.x_max - self.dx / 2,
            self.nx
        )

    @property
    def y(self) -> Array:
        """1D array of cell-centered y coordinates."""
        return jnp.linspace(
            self.y_min + self.dy / 2,
            self.y_max - self.dy / 2,
            self.ny
        )

    @property
    def z(self) -> Array:
        """1D array of cell-centered z coordinates."""
        return jnp.linspace(
            self.z_min + self.dz / 2,
            self.z_max - self.dz / 2,
            self.nz
        )

    @property
    def x_grid(self) -> Array:
        """3D array of x coordinates, shape (nx, ny, nz)."""
        x, y, z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij')
        return x

    @property
    def y_grid(self) -> Array:
        """3D array of y coordinates, shape (nx, ny, nz)."""
        x, y, z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij')
        return y

    @property
    def z_grid(self) -> Array:
        """3D array of z coordinates, shape (nx, ny, nz)."""
        x, y, z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij')
        return z

    @property
    def cell_volumes(self) -> Array:
        """Cell volumes, shape (nx, ny, nz). Simply dx * dy * dz."""
        return jnp.full((self.nx, self.ny, self.nz), self.dx * self.dy * self.dz)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_geometry_3d.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add jax_frc/core/geometry.py tests/test_geometry_3d.py
git commit -m "feat(geometry): replace cylindrical with 3D Cartesian geometry"
```

---

## Task 2: 3D Differential Operators - Gradient

**Files:**
- Modify: `jax_frc/operators.py`
- Test: `tests/test_operators_3d.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_operators_3d.py
"""Tests for 3D Cartesian differential operators."""

import jax.numpy as jnp
import pytest
from jax_frc.operators import gradient_3d
from jax_frc.core.geometry import Geometry


class TestGradient3D:
    """Test 3D gradient operator."""

    def test_gradient_constant_field(self):
        """Gradient of constant should be zero."""
        f = jnp.ones((8, 8, 8)) * 5.0
        geom = Geometry(nx=8, ny=8, nz=8)
        grad_f = gradient_3d(f, geom)
        assert grad_f.shape == (8, 8, 8, 3)
        assert jnp.allclose(grad_f, 0.0, atol=1e-10)

    def test_gradient_linear_x(self):
        """Gradient of f = x should be (1, 0, 0)."""
        geom = Geometry(
            nx=16, ny=8, nz=8,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
            bc_x="periodic", bc_y="periodic", bc_z="periodic"
        )
        f = geom.x_grid  # f = x
        grad_f = gradient_3d(f, geom)
        # df/dx = 1, df/dy = 0, df/dz = 0
        assert jnp.allclose(grad_f[..., 0], 1.0, atol=1e-6)
        assert jnp.allclose(grad_f[..., 1], 0.0, atol=1e-10)
        assert jnp.allclose(grad_f[..., 2], 0.0, atol=1e-10)

    def test_gradient_shape(self):
        """Gradient output shape is (nx, ny, nz, 3)."""
        f = jnp.ones((4, 6, 8))
        geom = Geometry(nx=4, ny=6, nz=8)
        grad_f = gradient_3d(f, geom)
        assert grad_f.shape == (4, 6, 8, 3)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_3d.py::TestGradient3D -v`
Expected: FAIL with "cannot import name 'gradient_3d'"

**Step 3: Write minimal implementation**

Add to `jax_frc/operators.py`:

```python
from jax import jit
import jax.numpy as jnp
from jax import Array

@jit
def gradient_3d(f: Array, geometry: "Geometry") -> Array:
    """Compute gradient of scalar field in 3D Cartesian coordinates.

    Args:
        f: Scalar field, shape (nx, ny, nz)
        geometry: 3D Cartesian geometry

    Returns:
        Gradient vector field, shape (nx, ny, nz, 3) with [df/dx, df/dy, df/dz]
    """
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz

    # Central differences with periodic wrapping
    df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    df_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dy)
    df_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2 * dz)

    return jnp.stack([df_dx, df_dy, df_dz], axis=-1)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_operators_3d.py::TestGradient3D -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/operators.py tests/test_operators_3d.py
git commit -m "feat(operators): add gradient_3d for 3D Cartesian"
```

---

## Task 3: 3D Differential Operators - Divergence

**Files:**
- Modify: `jax_frc/operators.py`
- Test: `tests/test_operators_3d.py`

**Step 1: Write the failing test**

Add to `tests/test_operators_3d.py`:

```python
from jax_frc.operators import divergence_3d


class TestDivergence3D:
    """Test 3D divergence operator."""

    def test_divergence_constant_field(self):
        """Divergence of constant vector field should be zero."""
        F = jnp.ones((8, 8, 8, 3)) * 3.0
        geom = Geometry(nx=8, ny=8, nz=8)
        div_F = divergence_3d(F, geom)
        assert div_F.shape == (8, 8, 8)
        assert jnp.allclose(div_F, 0.0, atol=1e-10)

    def test_divergence_linear_field(self):
        """Divergence of F = (x, y, z) should be 3."""
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
            bc_x="periodic", bc_y="periodic", bc_z="periodic"
        )
        F = jnp.stack([geom.x_grid, geom.y_grid, geom.z_grid], axis=-1)
        div_F = divergence_3d(F, geom)
        # dFx/dx + dFy/dy + dFz/dz = 1 + 1 + 1 = 3
        assert jnp.allclose(div_F, 3.0, atol=1e-6)

    def test_divergence_solenoidal(self):
        """Divergence of curl should be zero (approximately)."""
        geom = Geometry(nx=16, ny=16, nz=16)
        # Create a solenoidal field: F = (-y, x, 0)
        F = jnp.stack([-geom.y_grid, geom.x_grid, jnp.zeros_like(geom.x_grid)], axis=-1)
        div_F = divergence_3d(F, geom)
        assert jnp.allclose(div_F, 0.0, atol=1e-10)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_3d.py::TestDivergence3D -v`
Expected: FAIL with "cannot import name 'divergence_3d'"

**Step 3: Write minimal implementation**

Add to `jax_frc/operators.py`:

```python
@jit
def divergence_3d(F: Array, geometry: "Geometry") -> Array:
    """Compute divergence of vector field in 3D Cartesian coordinates.

    Args:
        F: Vector field, shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry

    Returns:
        Divergence scalar field, shape (nx, ny, nz)
    """
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz

    dFx_dx = (jnp.roll(F[..., 0], -1, axis=0) - jnp.roll(F[..., 0], 1, axis=0)) / (2 * dx)
    dFy_dy = (jnp.roll(F[..., 1], -1, axis=1) - jnp.roll(F[..., 1], 1, axis=1)) / (2 * dy)
    dFz_dz = (jnp.roll(F[..., 2], -1, axis=2) - jnp.roll(F[..., 2], 1, axis=2)) / (2 * dz)

    return dFx_dx + dFy_dy + dFz_dz
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_operators_3d.py::TestDivergence3D -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/operators.py tests/test_operators_3d.py
git commit -m "feat(operators): add divergence_3d for 3D Cartesian"
```

---

## Task 4: 3D Differential Operators - Curl

**Files:**
- Modify: `jax_frc/operators.py`
- Test: `tests/test_operators_3d.py`

**Step 1: Write the failing test**

Add to `tests/test_operators_3d.py`:

```python
from jax_frc.operators import curl_3d


class TestCurl3D:
    """Test 3D curl operator."""

    def test_curl_constant_field(self):
        """Curl of constant vector field should be zero."""
        F = jnp.ones((8, 8, 8, 3)) * 5.0
        geom = Geometry(nx=8, ny=8, nz=8)
        curl_F = curl_3d(F, geom)
        assert curl_F.shape == (8, 8, 8, 3)
        assert jnp.allclose(curl_F, 0.0, atol=1e-10)

    def test_curl_gradient_is_zero(self):
        """Curl of gradient should be zero."""
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
        )
        # f = x^2 + y^2
        f = geom.x_grid**2 + geom.y_grid**2
        grad_f = gradient_3d(f, geom)
        curl_grad_f = curl_3d(grad_f, geom)
        assert jnp.allclose(curl_grad_f, 0.0, atol=1e-6)

    def test_curl_simple_field(self):
        """Curl of F = (-y, x, 0) should be (0, 0, 2)."""
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        F = jnp.stack([-geom.y_grid, geom.x_grid, jnp.zeros_like(geom.x_grid)], axis=-1)
        curl_F = curl_3d(F, geom)
        # curl(-y, x, 0) = (0 - 0, 0 - 0, 1 - (-1)) = (0, 0, 2)
        assert jnp.allclose(curl_F[..., 0], 0.0, atol=1e-10)
        assert jnp.allclose(curl_F[..., 1], 0.0, atol=1e-10)
        assert jnp.allclose(curl_F[..., 2], 2.0, atol=1e-6)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_3d.py::TestCurl3D -v`
Expected: FAIL with "cannot import name 'curl_3d'"

**Step 3: Write minimal implementation**

Add to `jax_frc/operators.py`:

```python
@jit
def curl_3d(F: Array, geometry: "Geometry") -> Array:
    """Compute curl of vector field in 3D Cartesian coordinates.

    curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx, dFy/dx - dFx/dy)

    Args:
        F: Vector field, shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry

    Returns:
        Curl vector field, shape (nx, ny, nz, 3)
    """
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz
    Fx, Fy, Fz = F[..., 0], F[..., 1], F[..., 2]

    # Partial derivatives
    dFx_dy = (jnp.roll(Fx, -1, axis=1) - jnp.roll(Fx, 1, axis=1)) / (2 * dy)
    dFx_dz = (jnp.roll(Fx, -1, axis=2) - jnp.roll(Fx, 1, axis=2)) / (2 * dz)
    dFy_dx = (jnp.roll(Fy, -1, axis=0) - jnp.roll(Fy, 1, axis=0)) / (2 * dx)
    dFy_dz = (jnp.roll(Fy, -1, axis=2) - jnp.roll(Fy, 1, axis=2)) / (2 * dz)
    dFz_dx = (jnp.roll(Fz, -1, axis=0) - jnp.roll(Fz, 1, axis=0)) / (2 * dx)
    dFz_dy = (jnp.roll(Fz, -1, axis=1) - jnp.roll(Fz, 1, axis=1)) / (2 * dy)

    curl_x = dFz_dy - dFy_dz
    curl_y = dFx_dz - dFz_dx
    curl_z = dFy_dx - dFx_dy

    return jnp.stack([curl_x, curl_y, curl_z], axis=-1)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_operators_3d.py::TestCurl3D -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/operators.py tests/test_operators_3d.py
git commit -m "feat(operators): add curl_3d for 3D Cartesian"
```

---

## Task 5: 3D Differential Operators - Laplacian

**Files:**
- Modify: `jax_frc/operators.py`
- Test: `tests/test_operators_3d.py`

**Step 1: Write the failing test**

Add to `tests/test_operators_3d.py`:

```python
from jax_frc.operators import laplacian_3d


class TestLaplacian3D:
    """Test 3D Laplacian operator."""

    def test_laplacian_constant(self):
        """Laplacian of constant should be zero."""
        f = jnp.ones((8, 8, 8)) * 7.0
        geom = Geometry(nx=8, ny=8, nz=8)
        lap_f = laplacian_3d(f, geom)
        assert lap_f.shape == (8, 8, 8)
        assert jnp.allclose(lap_f, 0.0, atol=1e-10)

    def test_laplacian_quadratic(self):
        """Laplacian of f = x^2 + y^2 + z^2 should be 6."""
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        f = geom.x_grid**2 + geom.y_grid**2 + geom.z_grid**2
        lap_f = laplacian_3d(f, geom)
        # d^2(x^2)/dx^2 = 2, same for y and z, total = 6
        # Interior points only (boundaries have wrapping artifacts)
        interior = lap_f[2:-2, 2:-2, 2:-2]
        assert jnp.allclose(interior, 6.0, atol=1e-4)

    def test_laplacian_is_div_grad(self):
        """Laplacian should equal divergence of gradient."""
        geom = Geometry(nx=16, ny=16, nz=16)
        f = jnp.sin(2 * jnp.pi * geom.x_grid) * jnp.cos(2 * jnp.pi * geom.y_grid)
        lap_f = laplacian_3d(f, geom)
        grad_f = gradient_3d(f, geom)
        div_grad_f = divergence_3d(grad_f, geom)
        assert jnp.allclose(lap_f, div_grad_f, atol=1e-10)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_operators_3d.py::TestLaplacian3D -v`
Expected: FAIL with "cannot import name 'laplacian_3d'"

**Step 3: Write minimal implementation**

Add to `jax_frc/operators.py`:

```python
@jit
def laplacian_3d(f: Array, geometry: "Geometry") -> Array:
    """Compute Laplacian of scalar field in 3D Cartesian coordinates.

    Args:
        f: Scalar field, shape (nx, ny, nz)
        geometry: 3D Cartesian geometry

    Returns:
        Laplacian scalar field, shape (nx, ny, nz)
    """
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz

    d2f_dx2 = (jnp.roll(f, -1, axis=0) - 2*f + jnp.roll(f, 1, axis=0)) / dx**2
    d2f_dy2 = (jnp.roll(f, -1, axis=1) - 2*f + jnp.roll(f, 1, axis=1)) / dy**2
    d2f_dz2 = (jnp.roll(f, -1, axis=2) - 2*f + jnp.roll(f, 1, axis=2)) / dz**2

    return d2f_dx2 + d2f_dy2 + d2f_dz2
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_operators_3d.py::TestLaplacian3D -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/operators.py tests/test_operators_3d.py
git commit -m "feat(operators): add laplacian_3d for 3D Cartesian"
```

---

## Task 6: Update State Class for 3D

**Files:**
- Modify: `jax_frc/core/state.py`
- Test: `tests/test_state_3d.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_state_3d.py
"""Tests for 3D state container."""

import jax.numpy as jnp
import pytest
from jax_frc.core.state import State


class TestState3D:
    """Test 3D state container."""

    def test_state_zeros(self):
        """Test creating zero-initialized state."""
        state = State.zeros(nx=4, ny=6, nz=8)
        assert state.B.shape == (4, 6, 8, 3)
        assert state.E.shape == (4, 6, 8, 3)
        assert state.n.shape == (4, 6, 8)
        assert state.p.shape == (4, 6, 8)

    def test_state_is_pytree(self):
        """State should be registered as JAX pytree."""
        import jax
        state = State.zeros(nx=4, ny=4, nz=4)
        leaves = jax.tree_util.tree_leaves(state)
        assert len(leaves) > 0

    def test_state_replace(self):
        """Test replacing state fields."""
        state = State.zeros(nx=4, ny=4, nz=4)
        new_n = jnp.ones((4, 4, 4)) * 1e19
        state2 = state.replace(n=new_n)
        assert jnp.allclose(state2.n, 1e19)
        assert jnp.allclose(state.n, 0.0)  # Original unchanged
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_state_3d.py -v`
Expected: FAIL (State.zeros signature wrong or shape mismatch)

**Step 3: Write minimal implementation**

Update `jax_frc/core/state.py`:

```python
"""State container for 3D plasma simulations."""

from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class ParticleState:
    """Particle state for hybrid kinetic model."""
    x: Array      # Positions, shape (n_particles, 3)
    v: Array      # Velocities, shape (n_particles, 3)
    w: Array      # Delta-f weights, shape (n_particles,)
    species: str  # Particle species identifier


@dataclass(frozen=True)
class State:
    """State container for 3D plasma simulations.

    All fields are 3D arrays with shape (nx, ny, nz) for scalars
    and (nx, ny, nz, 3) for vectors.
    """
    B: Array           # Magnetic field [T], shape (nx, ny, nz, 3)
    E: Array           # Electric field [V/m], shape (nx, ny, nz, 3)
    n: Array           # Number density [m^-3], shape (nx, ny, nz)
    p: Array           # Pressure [Pa], shape (nx, ny, nz)
    v: Optional[Array] = None  # Velocity [m/s], shape (nx, ny, nz, 3)
    Te: Optional[Array] = None # Electron temp [J], shape (nx, ny, nz)
    Ti: Optional[Array] = None # Ion temp [J], shape (nx, ny, nz)
    particles: Optional[ParticleState] = None

    @classmethod
    def zeros(cls, nx: int, ny: int, nz: int) -> "State":
        """Create zero-initialized state."""
        return cls(
            B=jnp.zeros((nx, ny, nz, 3)),
            E=jnp.zeros((nx, ny, nz, 3)),
            n=jnp.zeros((nx, ny, nz)),
            p=jnp.zeros((nx, ny, nz)),
        )

    def replace(self, **kwargs) -> "State":
        """Return new State with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)


# Register as JAX pytree
def _state_flatten(state):
    children = (state.B, state.E, state.n, state.p, state.v, state.Te, state.Ti, state.particles)
    aux_data = None
    return children, aux_data


def _state_unflatten(aux_data, children):
    B, E, n, p, v, Te, Ti, particles = children
    return State(B=B, E=E, n=n, p=p, v=v, Te=Te, Ti=Ti, particles=particles)


jax.tree_util.register_pytree_node(State, _state_flatten, _state_unflatten)


def _particle_state_flatten(state):
    return (state.x, state.v, state.w), state.species


def _particle_state_unflatten(species, children):
    x, v, w = children
    return ParticleState(x=x, v=v, w=w, species=species)


jax.tree_util.register_pytree_node(ParticleState, _particle_state_flatten, _particle_state_unflatten)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_state_3d.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/core/state.py tests/test_state_3d.py
git commit -m "feat(state): update State class for 3D arrays"
```

---

## Task 7: Resistive MHD - Direct B Evolution

**Files:**
- Modify: `jax_frc/models/resistive_mhd.py`
- Test: `tests/test_resistive_mhd_3d.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_resistive_mhd_3d.py
"""Tests for 3D resistive MHD model."""

import jax.numpy as jnp
import pytest
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State


class TestResistiveMHD3D:
    """Test 3D resistive MHD."""

    def test_model_creation(self):
        """Test creating resistive MHD model."""
        model = ResistiveMHD(eta=1e-4)
        assert model.eta == 1e-4

    def test_compute_rhs_shape(self):
        """Test RHS computation returns correct shapes."""
        model = ResistiveMHD(eta=1e-4)
        geom = Geometry(nx=8, ny=8, nz=8)
        state = State.zeros(nx=8, ny=8, nz=8)
        # Set non-zero B field
        B = jnp.zeros((8, 8, 8, 3))
        B = B.at[..., 2].set(1.0)  # Uniform Bz
        state = state.replace(B=B, n=jnp.ones((8, 8, 8)) * 1e19)

        rhs = model.compute_rhs(state, geom)
        assert rhs.B.shape == (8, 8, 8, 3)

    def test_uniform_field_no_change(self):
        """Uniform B field should have zero dB/dt (no current)."""
        model = ResistiveMHD(eta=1e-4)
        geom = Geometry(nx=16, ny=16, nz=16)
        state = State.zeros(nx=16, ny=16, nz=16)
        B = jnp.zeros((16, 16, 16, 3))
        B = B.at[..., 2].set(1.0)  # Uniform Bz
        state = state.replace(B=B, n=jnp.ones((16, 16, 16)) * 1e19)

        rhs = model.compute_rhs(state, geom)
        # Uniform field => J = curl(B)/mu0 = 0 => dB/dt = 0
        assert jnp.allclose(rhs.B, 0.0, atol=1e-10)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_resistive_mhd_3d.py -v`
Expected: FAIL (compute_rhs uses old ψ-based approach)

**Step 3: Write minimal implementation**

Rewrite `jax_frc/models/resistive_mhd.py`:

```python
"""3D Resistive MHD model with direct B-field evolution."""

from dataclasses import dataclass
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.operators import curl_3d
from jax_frc.constants import MU0


@dataclass
class ResistiveMHD(PhysicsModel):
    """Resistive MHD model evolving B directly.

    Solves: dB/dt = -curl(E)
    Where:  E = -v × B + η*J, J = curl(B)/μ₀

    For stationary plasma (v=0): E = η*J
    """
    eta: float = 1e-4  # Resistivity [Ohm·m]

    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute dB/dt from induction equation.

        Args:
            state: Current state with B field
            geometry: 3D geometry

        Returns:
            State with B field containing dB/dt
        """
        B = state.B

        # Current density: J = curl(B) / mu_0
        J = curl_3d(B, geometry) / MU0

        # Electric field: E = η*J (assuming v=0 for pure resistive case)
        # For moving plasma: E = -v × B + η*J
        if state.v is not None:
            v = state.v
            v_cross_B = jnp.stack([
                v[..., 1] * B[..., 2] - v[..., 2] * B[..., 1],
                v[..., 2] * B[..., 0] - v[..., 0] * B[..., 2],
                v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0],
            ], axis=-1)
            E = -v_cross_B + self.eta * J
        else:
            E = self.eta * J

        # Faraday's law: dB/dt = -curl(E)
        dB_dt = -curl_3d(E, geometry)

        return state.replace(B=dB_dt)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """Resistive diffusion CFL: dt < dx^2 / (2*eta/mu0)."""
        dx_min = min(geometry.dx, geometry.dy, geometry.dz)
        diffusivity = self.eta / MU0
        return 0.25 * dx_min**2 / diffusivity

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply boundary conditions based on geometry.bc_* settings."""
        # For now, return state unchanged (periodic BCs handled by operators)
        return state
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_resistive_mhd_3d.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/models/resistive_mhd.py tests/test_resistive_mhd_3d.py
git commit -m "feat(resistive_mhd): rewrite for direct B evolution in 3D"
```

---

## Task 8: Extended MHD - Update to 3D Operators

**Files:**
- Modify: `jax_frc/models/extended_mhd.py`
- Test: `tests/test_extended_mhd_3d.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_extended_mhd_3d.py
"""Tests for 3D extended MHD model."""

import jax.numpy as jnp
import pytest
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State


class TestExtendedMHD3D:
    """Test 3D extended MHD."""

    def test_model_creation(self):
        """Test creating extended MHD model."""
        model = ExtendedMHD(eta=1e-4, include_hall=True)
        assert model.eta == 1e-4
        assert model.include_hall is True

    def test_compute_rhs_shape(self):
        """Test RHS computation returns correct shapes."""
        model = ExtendedMHD(eta=1e-4)
        geom = Geometry(nx=8, ny=8, nz=8)
        state = State.zeros(nx=8, ny=8, nz=8)
        B = jnp.zeros((8, 8, 8, 3))
        B = B.at[..., 2].set(0.1)
        state = state.replace(
            B=B,
            n=jnp.ones((8, 8, 8)) * 1e19,
            p=jnp.ones((8, 8, 8)) * 1e3,
            Te=jnp.ones((8, 8, 8)) * 100 * 1.602e-19,
        )

        rhs = model.compute_rhs(state, geom)
        assert rhs.B.shape == (8, 8, 8, 3)

    def test_hall_term_present(self):
        """Hall term should contribute when include_hall=True."""
        geom = Geometry(nx=16, ny=16, nz=16)
        state = State.zeros(nx=16, ny=16, nz=16)
        # Create non-uniform B to generate J
        B = jnp.zeros((16, 16, 16, 3))
        x = geom.x_grid
        B = B.at[..., 2].set(jnp.sin(2 * jnp.pi * x))
        state = state.replace(
            B=B,
            n=jnp.ones((16, 16, 16)) * 1e19,
            p=jnp.ones((16, 16, 16)) * 1e3,
        )

        model_with_hall = ExtendedMHD(eta=0.0, include_hall=True)
        model_no_hall = ExtendedMHD(eta=0.0, include_hall=False)

        rhs_hall = model_with_hall.compute_rhs(state, geom)
        rhs_no_hall = model_no_hall.compute_rhs(state, geom)

        # With Hall term, results should differ
        assert not jnp.allclose(rhs_hall.B, rhs_no_hall.B)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_extended_mhd_3d.py -v`
Expected: FAIL (uses old cylindrical operators)

**Step 3: Write minimal implementation**

Rewrite `jax_frc/models/extended_mhd.py`:

```python
"""3D Extended MHD model with Hall physics."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jax import jit

from jax_frc.models.base import PhysicsModel
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.operators import curl_3d, gradient_3d, laplacian_3d
from jax_frc.constants import MU0, QE


@dataclass
class ExtendedMHD(PhysicsModel):
    """Extended MHD model with Hall and electron pressure terms.

    Generalized Ohm's law:
    E = -v × B + η*J + (J × B)/(ne) - ∇p_e/(ne)

    Attributes:
        eta: Resistivity [Ohm·m]
        include_hall: Include Hall term (J × B)/(ne)
        include_electron_pressure: Include ∇p_e/(ne) term
        kappa_parallel: Parallel thermal conductivity
        kappa_perp: Perpendicular thermal conductivity
    """
    eta: float = 1e-4
    include_hall: bool = True
    include_electron_pressure: bool = True
    kappa_parallel: float = 1e20
    kappa_perp: float = 1e18

    def compute_rhs(self, state: State, geometry: Geometry) -> State:
        """Compute time derivatives for extended MHD."""
        B = state.B
        n = jnp.maximum(state.n, 1e16)  # Avoid division by zero

        # Current density: J = curl(B) / mu_0
        J = curl_3d(B, geometry) / MU0

        # Start with resistive term: E = η*J
        E = self.eta * J

        # Hall term: (J × B) / (ne)
        if self.include_hall:
            J_cross_B = jnp.stack([
                J[..., 1] * B[..., 2] - J[..., 2] * B[..., 1],
                J[..., 2] * B[..., 0] - J[..., 0] * B[..., 2],
                J[..., 0] * B[..., 1] - J[..., 1] * B[..., 0],
            ], axis=-1)
            E = E + J_cross_B / (n[..., None] * QE)

        # Electron pressure term: -∇p_e / (ne)
        if self.include_electron_pressure and state.Te is not None:
            p_e = n * state.Te  # Electron pressure
            grad_pe = gradient_3d(p_e, geometry)
            E = E - grad_pe / (n[..., None] * QE)

        # Convective term: -v × B
        if state.v is not None:
            v = state.v
            v_cross_B = jnp.stack([
                v[..., 1] * B[..., 2] - v[..., 2] * B[..., 1],
                v[..., 2] * B[..., 0] - v[..., 0] * B[..., 2],
                v[..., 0] * B[..., 1] - v[..., 1] * B[..., 0],
            ], axis=-1)
            E = E - v_cross_B

        # Faraday's law: dB/dt = -curl(E)
        dB_dt = -curl_3d(E, geometry)

        # Temperature evolution with thermal conduction
        dTe_dt = None
        if state.Te is not None:
            # Simplified isotropic conduction for now
            lap_Te = laplacian_3d(state.Te, geometry)
            dTe_dt = self.kappa_perp * lap_Te / (n * 1.5)  # 3/2 * n * dT/dt

        return state.replace(B=dB_dt, Te=dTe_dt)

    def compute_stable_dt(self, state: State, geometry: Geometry) -> float:
        """CFL constraint including Hall term."""
        dx_min = min(geometry.dx, geometry.dy, geometry.dz)

        # Resistive diffusion
        dt_resistive = 0.25 * dx_min**2 * MU0 / self.eta

        # Hall wave (whistler): omega ~ k^2 * B / (mu0 * n * e)
        if self.include_hall:
            B_max = jnp.max(jnp.sqrt(jnp.sum(state.B**2, axis=-1)))
            n_min = jnp.maximum(jnp.min(state.n), 1e16)
            whistler_speed = B_max / (MU0 * n_min * QE) * (2 * jnp.pi / dx_min)
            dt_hall = 0.1 * dx_min / jnp.maximum(whistler_speed, 1e-10)
        else:
            dt_hall = jnp.inf

        return float(jnp.minimum(dt_resistive, dt_hall))

    def apply_constraints(self, state: State, geometry: Geometry) -> State:
        """Apply boundary conditions."""
        return state
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_extended_mhd_3d.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/models/extended_mhd.py tests/test_extended_mhd_3d.py
git commit -m "feat(extended_mhd): update to 3D Cartesian operators"
```

---

## Task 9: Divergence Cleaning - 3D Poisson Projection

**Files:**
- Modify: `jax_frc/solvers/divergence_cleaning.py`
- Test: `tests/test_divergence_cleaning_3d.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_divergence_cleaning_3d.py
"""Tests for 3D divergence cleaning."""

import jax.numpy as jnp
import pytest
from jax_frc.solvers.divergence_cleaning import clean_divergence
from jax_frc.core.geometry import Geometry
from jax_frc.operators import divergence_3d


class TestDivergenceCleaning3D:
    """Test 3D divergence cleaning."""

    def test_already_divergence_free(self):
        """Divergence-free field should be unchanged."""
        geom = Geometry(nx=16, ny=16, nz=16)
        # Create solenoidal field: B = curl(A) where A = (0, 0, f(x,y))
        # curl(0, 0, A_z) = (dAz/dy, -dAz/dx, 0)
        x, y = geom.x_grid, geom.y_grid
        A_z = jnp.sin(2 * jnp.pi * x) * jnp.sin(2 * jnp.pi * y)
        B_x = jnp.gradient(A_z, geom.dy, axis=1)
        B_y = -jnp.gradient(A_z, geom.dx, axis=0)
        B_z = jnp.zeros_like(A_z)
        B = jnp.stack([B_x, B_y, B_z], axis=-1)

        B_clean = clean_divergence(B, geom)
        # Should be essentially unchanged
        assert jnp.allclose(B, B_clean, atol=1e-6)

    def test_removes_divergence(self):
        """Should remove divergence from non-solenoidal field."""
        geom = Geometry(nx=16, ny=16, nz=16)
        # Create field with divergence: B = (x, y, z)
        B = jnp.stack([geom.x_grid, geom.y_grid, geom.z_grid], axis=-1)

        div_before = divergence_3d(B, geom)
        assert jnp.max(jnp.abs(div_before)) > 1.0  # Has divergence

        B_clean = clean_divergence(B, geom)
        div_after = divergence_3d(B_clean, geom)
        assert jnp.max(jnp.abs(div_after)) < 0.1  # Much smaller

    def test_preserves_shape(self):
        """Output should have same shape as input."""
        geom = Geometry(nx=8, ny=10, nz=12)
        B = jnp.ones((8, 10, 12, 3))
        B_clean = clean_divergence(B, geom)
        assert B_clean.shape == (8, 10, 12, 3)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_divergence_cleaning_3d.py -v`
Expected: FAIL (uses old cylindrical operators)

**Step 3: Write minimal implementation**

Rewrite `jax_frc/solvers/divergence_cleaning.py`:

```python
"""3D divergence cleaning via Poisson projection."""

import jax.numpy as jnp
from jax import jit, lax
from jax import Array

from jax_frc.core.geometry import Geometry
from jax_frc.operators import divergence_3d, gradient_3d, laplacian_3d


@jit
def clean_divergence(B: Array, geometry: Geometry, max_iter: int = 100, tol: float = 1e-6) -> Array:
    """Project B field to satisfy div(B) = 0.

    Uses Poisson projection:
    1. Solve: ∇²φ = ∇·B
    2. Correct: B ← B - ∇φ

    Args:
        B: Magnetic field, shape (nx, ny, nz, 3)
        geometry: 3D geometry
        max_iter: Maximum Jacobi iterations
        tol: Convergence tolerance

    Returns:
        Divergence-free B field
    """
    # Compute divergence
    div_B = divergence_3d(B, geometry)

    # Solve Poisson equation ∇²φ = div_B using Jacobi iteration
    phi = poisson_solve_jacobi(div_B, geometry, max_iter, tol)

    # Correct: B_clean = B - ∇φ
    grad_phi = gradient_3d(phi, geometry)
    B_clean = B - grad_phi

    return B_clean


@jit
def poisson_solve_jacobi(rhs: Array, geometry: Geometry, max_iter: int = 100, tol: float = 1e-6) -> Array:
    """Solve ∇²φ = rhs using Jacobi iteration.

    For 7-point stencil in 3D:
    φ_new = (φ_E + φ_W + φ_N + φ_S + φ_T + φ_B - dx²*rhs) / 6

    Args:
        rhs: Right-hand side, shape (nx, ny, nz)
        geometry: 3D geometry
        max_iter: Maximum iterations
        tol: Convergence tolerance (not used in scan, for API consistency)

    Returns:
        Solution φ
    """
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz

    # For uniform grid with dx=dy=dz
    # Jacobi update: φ_new = (sum of neighbors - h²*rhs) / 6
    # For non-uniform: use weighted average
    h2 = dx * dy  # Approximate (assume nearly uniform)

    def jacobi_step(phi, _):
        # 6 neighbors
        phi_xp = jnp.roll(phi, -1, axis=0)  # x+1
        phi_xm = jnp.roll(phi, 1, axis=0)   # x-1
        phi_yp = jnp.roll(phi, -1, axis=1)  # y+1
        phi_ym = jnp.roll(phi, 1, axis=1)   # y-1
        phi_zp = jnp.roll(phi, -1, axis=2)  # z+1
        phi_zm = jnp.roll(phi, 1, axis=2)   # z-1

        # Jacobi update for uniform grid
        phi_new = (
            (phi_xp + phi_xm) / dx**2 +
            (phi_yp + phi_ym) / dy**2 +
            (phi_zp + phi_zm) / dz**2 -
            rhs
        ) / (2/dx**2 + 2/dy**2 + 2/dz**2)

        return phi_new, None

    phi_init = jnp.zeros_like(rhs)
    phi_final, _ = lax.scan(jacobi_step, phi_init, jnp.arange(max_iter))

    return phi_final
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_divergence_cleaning_3d.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/solvers/divergence_cleaning.py tests/test_divergence_cleaning_3d.py
git commit -m "feat(divergence_cleaning): rewrite for 3D Poisson projection"
```

---

## Task 10: Neutral Fluid - 3D Extension

**Files:**
- Modify: `jax_frc/models/neutral_fluid.py`
- Test: `tests/test_neutral_fluid_3d.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_neutral_fluid_3d.py
"""Tests for 3D neutral fluid model."""

import jax.numpy as jnp
import pytest
from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState
from jax_frc.core.geometry import Geometry


class TestNeutralFluid3D:
    """Test 3D neutral fluid."""

    def test_state_creation(self):
        """Test creating neutral state with 3D arrays."""
        state = NeutralState(
            rho_n=jnp.ones((4, 6, 8)) * 1e-6,
            mom_n=jnp.zeros((4, 6, 8, 3)),
            E_n=jnp.ones((4, 6, 8)) * 1e-3,
        )
        assert state.rho_n.shape == (4, 6, 8)
        assert state.mom_n.shape == (4, 6, 8, 3)

    def test_derived_properties(self):
        """Test velocity and pressure calculations."""
        rho = jnp.ones((4, 4, 4)) * 1e-6
        mom = jnp.zeros((4, 4, 4, 3))
        mom = mom.at[..., 0].set(1e-6)  # v_x = 1 m/s
        E = jnp.ones((4, 4, 4)) * 1e-3

        state = NeutralState(rho_n=rho, mom_n=mom, E_n=E)
        assert state.v_n.shape == (4, 4, 4, 3)
        assert state.p_n.shape == (4, 4, 4)

    def test_flux_divergence_shape(self):
        """Test flux divergence returns correct shapes."""
        model = NeutralFluid()
        geom = Geometry(nx=8, ny=8, nz=8)
        state = NeutralState(
            rho_n=jnp.ones((8, 8, 8)) * 1e-6,
            mom_n=jnp.zeros((8, 8, 8, 3)),
            E_n=jnp.ones((8, 8, 8)) * 1e-3,
        )

        d_rho, d_mom, d_E = model.compute_flux_divergence(state, geom)
        assert d_rho.shape == (8, 8, 8)
        assert d_mom.shape == (8, 8, 8, 3)
        assert d_E.shape == (8, 8, 8)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_neutral_fluid_3d.py -v`
Expected: FAIL (uses 2D arrays and axis=0,1 instead of 0,1,2)

**Step 3: Write minimal implementation**

Update `jax_frc/models/neutral_fluid.py` - key changes are updating array shapes and adding y-direction flux:

```python
"""3D Neutral fluid model for plasma-neutral coupling."""

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit

from jax_frc.constants import MI

GAMMA = 5.0 / 3.0


@dataclass(frozen=True)
class NeutralState:
    """Neutral fluid state variables (3D).

    All fields use SI units.
    """
    rho_n: Array  # Mass density [kg/m³], shape (nx, ny, nz)
    mom_n: Array  # Momentum density [kg/m²/s], shape (nx, ny, nz, 3)
    E_n: Array    # Total energy density [J/m³], shape (nx, ny, nz)

    @property
    def v_n(self) -> Array:
        """Velocity [m/s], shape (nx, ny, nz, 3)."""
        rho_safe = jnp.maximum(self.rho_n[..., None], 1e-20)
        return self.mom_n / rho_safe

    @property
    def p_n(self) -> Array:
        """Pressure [Pa], shape (nx, ny, nz)."""
        rho_safe = jnp.maximum(self.rho_n, 1e-20)
        ke = 0.5 * jnp.sum(self.mom_n**2, axis=-1) / rho_safe
        internal_energy = self.E_n - ke
        return (GAMMA - 1) * jnp.maximum(internal_energy, 0.0)

    @property
    def T_n(self) -> Array:
        """Temperature [J], shape (nx, ny, nz)."""
        n_n = self.rho_n / MI
        n_safe = jnp.maximum(n_n, 1e-10)
        return self.p_n / n_safe

    def replace(self, **kwargs) -> "NeutralState":
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)


def _neutral_state_flatten(state):
    return (state.rho_n, state.mom_n, state.E_n), None


def _neutral_state_unflatten(aux_data, children):
    rho_n, mom_n, E_n = children
    return NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)


jax.tree_util.register_pytree_node(
    NeutralState, _neutral_state_flatten, _neutral_state_unflatten
)


@jit
def hlle_flux_1d(
    rho_L: Array, rho_R: Array,
    v_L: Array, v_R: Array,
    p_L: Array, p_R: Array,
    E_L: Array, E_R: Array,
    gamma: float = GAMMA
) -> Tuple[Array, Array, Array]:
    """HLLE approximate Riemann solver for 1D Euler equations."""
    rho_L_safe = jnp.maximum(rho_L, 1e-20)
    rho_R_safe = jnp.maximum(rho_R, 1e-20)
    p_L_safe = jnp.maximum(p_L, 1e-10)
    p_R_safe = jnp.maximum(p_R, 1e-10)

    c_L = jnp.sqrt(gamma * p_L_safe / rho_L_safe)
    c_R = jnp.sqrt(gamma * p_R_safe / rho_R_safe)

    S_L = jnp.minimum(v_L - c_L, v_R - c_R)
    S_R = jnp.maximum(v_L + c_L, v_R + c_R)

    F_rho_L = rho_L * v_L
    F_rho_R = rho_R * v_R
    F_mom_L = rho_L * v_L**2 + p_L
    F_mom_R = rho_R * v_R**2 + p_R
    F_E_L = (E_L + p_L) * v_L
    F_E_R = (E_R + p_R) * v_R

    dS = S_R - S_L
    dS_safe = jnp.where(jnp.abs(dS) < 1e-10, 1e-10, dS)

    def hlle_component(F_L, F_R, U_L, U_R):
        return jnp.where(
            S_L >= 0, F_L,
            jnp.where(
                S_R <= 0, F_R,
                (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / dS_safe
            )
        )

    F_rho = hlle_component(F_rho_L, F_rho_R, rho_L, rho_R)
    F_mom = hlle_component(F_mom_L, F_mom_R, rho_L * v_L, rho_R * v_R)
    F_E = hlle_component(F_E_L, F_E_R, E_L, E_R)

    return F_rho, F_mom, F_E


@dataclass
class NeutralFluid:
    """3D Hydrodynamic neutral fluid model."""
    gamma: float = GAMMA

    def compute_flux_divergence(
        self, state: NeutralState, geometry: "Geometry"
    ) -> Tuple[Array, Array, Array]:
        """Compute -div(F) for 3D Euler equations using dimension-by-dimension HLLE."""
        dx, dy, dz = geometry.dx, geometry.dy, geometry.dz
        rho = state.rho_n
        v = state.v_n
        p = state.p_n
        E = state.E_n

        # X-direction flux
        d_rho_x, d_mom_x, d_E_x = self._compute_flux_dir(rho, v, p, E, dx, axis=0, vel_idx=0)

        # Y-direction flux
        d_rho_y, d_mom_y, d_E_y = self._compute_flux_dir(rho, v, p, E, dy, axis=1, vel_idx=1)

        # Z-direction flux
        d_rho_z, d_mom_z, d_E_z = self._compute_flux_dir(rho, v, p, E, dz, axis=2, vel_idx=2)

        d_rho = d_rho_x + d_rho_y + d_rho_z
        d_mom = d_mom_x + d_mom_y + d_mom_z
        d_E = d_E_x + d_E_y + d_E_z

        return d_rho, d_mom, d_E

    def _compute_flux_dir(self, rho, v, p, E, dx, axis, vel_idx):
        """Compute flux divergence in one direction."""
        v_dir = v[..., vel_idx]

        rho_L = jnp.roll(rho, 1, axis=axis)
        rho_R = rho
        v_L = jnp.roll(v_dir, 1, axis=axis)
        v_R = v_dir
        p_L = jnp.roll(p, 1, axis=axis)
        p_R = p
        E_L = jnp.roll(E, 1, axis=axis)
        E_R = E

        F_rho, F_mom, F_E = hlle_flux_1d(
            rho_L, rho_R, v_L, v_R, p_L, p_R, E_L, E_R, self.gamma
        )

        # Flux divergence
        d_rho = -(jnp.roll(F_rho, -1, axis=axis) - F_rho) / dx
        d_E = -(jnp.roll(F_E, -1, axis=axis) - F_E) / dx

        # Momentum: main direction gets pressure flux, others get advection
        d_mom = jnp.zeros_like(v)
        d_mom = d_mom.at[..., vel_idx].set(-(jnp.roll(F_mom, -1, axis=axis) - F_mom) / dx)

        return d_rho, d_mom, d_E

    def apply_boundary_conditions(
        self, state: NeutralState, geometry: "Geometry", bc_type: str = "reflecting"
    ) -> NeutralState:
        """Apply 3D boundary conditions."""
        # For periodic BCs (default), nothing to do
        return state
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_neutral_fluid_3d.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/models/neutral_fluid.py tests/test_neutral_fluid_3d.py
git commit -m "feat(neutral_fluid): extend to 3D Cartesian"
```

---

## Task 11: Hybrid Kinetic - 3D Field Interpolation

**Files:**
- Modify: `jax_frc/models/hybrid_kinetic.py`
- Modify: `jax_frc/models/particle_pusher.py`
- Test: `tests/test_hybrid_kinetic_3d.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_hybrid_kinetic_3d.py
"""Tests for 3D hybrid kinetic model."""

import jax.numpy as jnp
import jax.random as random
import pytest
from jax_frc.models.hybrid_kinetic import HybridKinetic
from jax_frc.models.particle_pusher import interpolate_field_to_particles_3d
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State


class TestInterpolation3D:
    """Test 3D field interpolation."""

    def test_interpolate_uniform_field(self):
        """Uniform field should interpolate to same value everywhere."""
        geom = Geometry(nx=8, ny=8, nz=8)
        field = jnp.ones((8, 8, 8, 3)) * 5.0

        # Particles at random positions
        key = random.PRNGKey(0)
        x = random.uniform(key, (100, 3),
                          minval=jnp.array([geom.x_min, geom.y_min, geom.z_min]),
                          maxval=jnp.array([geom.x_max, geom.y_max, geom.z_max]))

        field_p = interpolate_field_to_particles_3d(field, x, geom)
        assert field_p.shape == (100, 3)
        assert jnp.allclose(field_p, 5.0, atol=1e-6)

    def test_interpolate_linear_field(self):
        """Linear field should interpolate correctly."""
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
        )
        # Field = (x, y, z)
        field = jnp.stack([geom.x_grid, geom.y_grid, geom.z_grid], axis=-1)

        # Single particle at center
        x = jnp.array([[0.5, 0.5, 0.5]])
        field_p = interpolate_field_to_particles_3d(field, x, geom)
        assert jnp.allclose(field_p[0], jnp.array([0.5, 0.5, 0.5]), atol=0.1)


class TestHybridKinetic3D:
    """Test 3D hybrid kinetic model."""

    def test_model_creation(self):
        """Test creating hybrid model."""
        model = HybridKinetic.from_config({
            "eta": 1e-6,
            "equilibrium": {"n0": 1e19, "T0": 1000.0}
        })
        assert model.eta == 1e-6

    def test_compute_rhs_shape(self):
        """Test RHS has correct shape."""
        model = HybridKinetic.from_config({})
        geom = Geometry(nx=8, ny=8, nz=8)
        state = State.zeros(nx=8, ny=8, nz=8)
        B = jnp.zeros((8, 8, 8, 3))
        B = B.at[..., 2].set(0.1)
        state = state.replace(
            B=B,
            n=jnp.ones((8, 8, 8)) * 1e19,
            p=jnp.ones((8, 8, 8)) * 1e3,
        )

        rhs = model.compute_rhs(state, geom)
        assert rhs.B.shape == (8, 8, 8, 3)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_hybrid_kinetic_3d.py -v`
Expected: FAIL (missing interpolate_field_to_particles_3d)

**Step 3: Write minimal implementation**

Update `jax_frc/models/particle_pusher.py` to add 3D trilinear interpolation:

```python
# Add to jax_frc/models/particle_pusher.py

@jit
def interpolate_field_to_particles_3d(field: Array, x: Array, geometry: "Geometry") -> Array:
    """Trilinear interpolation of 3D field to particle positions.

    Args:
        field: Field values, shape (nx, ny, nz, ncomp)
        x: Particle positions, shape (n_particles, 3) as (x, y, z)
        geometry: 3D geometry

    Returns:
        Field at particle positions, shape (n_particles, ncomp)
    """
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz

    # Normalized coordinates (0 to n-1)
    xn = (x[:, 0] - geometry.x_min - dx/2) / dx
    yn = (x[:, 1] - geometry.y_min - dy/2) / dy
    zn = (x[:, 2] - geometry.z_min - dz/2) / dz

    # Integer indices and fractions
    i0 = jnp.floor(xn).astype(int)
    j0 = jnp.floor(yn).astype(int)
    k0 = jnp.floor(zn).astype(int)

    fx = xn - i0
    fy = yn - j0
    fz = zn - k0

    # Wrap indices for periodic
    i0 = i0 % nx
    j0 = j0 % ny
    k0 = k0 % nz
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    k1 = (k0 + 1) % nz

    # 8 corners of cell
    f000 = field[i0, j0, k0]
    f001 = field[i0, j0, k1]
    f010 = field[i0, j1, k0]
    f011 = field[i0, j1, k1]
    f100 = field[i1, j0, k0]
    f101 = field[i1, j0, k1]
    f110 = field[i1, j1, k0]
    f111 = field[i1, j1, k1]

    # Trilinear interpolation
    fx = fx[:, None]
    fy = fy[:, None]
    fz = fz[:, None]

    c00 = f000 * (1 - fx) + f100 * fx
    c01 = f001 * (1 - fx) + f101 * fx
    c10 = f010 * (1 - fx) + f110 * fx
    c11 = f011 * (1 - fx) + f111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    return c0 * (1 - fz) + c1 * fz
```

Update `jax_frc/models/hybrid_kinetic.py` to use 3D operators:

```python
# Key changes in hybrid_kinetic.py:
# 1. Replace _compute_current with curl_3d
# 2. Use interpolate_field_to_particles_3d
# 3. Update particle initialization for 3D Cartesian

from jax_frc.operators import curl_3d, gradient_3d
from jax_frc.models.particle_pusher import interpolate_field_to_particles_3d

# In compute_rhs:
def compute_rhs(self, state: State, geometry: Geometry) -> State:
    """Compute time derivatives for hybrid model."""
    B = state.B
    n = jnp.maximum(state.n, 1e16)

    # Current density from curl(B)
    J = curl_3d(B, geometry) / MU0

    # Hall term + resistive
    ne = n * QE
    J_cross_B = jnp.stack([
        J[..., 1] * B[..., 2] - J[..., 2] * B[..., 1],
        J[..., 2] * B[..., 0] - J[..., 0] * B[..., 2],
        J[..., 0] * B[..., 1] - J[..., 1] * B[..., 0],
    ], axis=-1)

    E = self.eta * J + J_cross_B / ne[..., None]

    # Pressure gradient
    if state.p is not None:
        grad_p = gradient_3d(state.p, geometry)
        E = E - grad_p / ne[..., None]

    # Faraday's law
    dB = -curl_3d(E, geometry)

    return state.replace(B=dB, E=E)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_hybrid_kinetic_3d.py -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
git add jax_frc/models/hybrid_kinetic.py jax_frc/models/particle_pusher.py tests/test_hybrid_kinetic_3d.py
git commit -m "feat(hybrid_kinetic): update to 3D with trilinear interpolation"
```

---

## Task 12: 3D Equilibrium Solver

**Files:**
- Modify: `jax_frc/equilibrium/grad_shafranov.py`
- Test: `tests/test_equilibrium_3d.py` (new)

**Step 1: Write the failing test**

```python
# tests/test_equilibrium_3d.py
"""Tests for 3D force-balance equilibrium solver."""

import jax.numpy as jnp
import pytest
from jax_frc.equilibrium.grad_shafranov import ForceBalanceSolver
from jax_frc.core.geometry import Geometry


class TestForceBalanceSolver:
    """Test 3D force balance equilibrium."""

    def test_solver_creation(self):
        """Test creating solver."""
        solver = ForceBalanceSolver(max_iterations=100, tolerance=1e-6)
        assert solver.max_iterations == 100

    def test_uniform_pressure_zero_current(self):
        """Uniform pressure should have zero J × B force."""
        geom = Geometry(nx=16, ny=16, nz=16)
        solver = ForceBalanceSolver()

        # Uniform pressure
        p = jnp.ones((16, 16, 16)) * 1e3

        # Uniform B field
        B_init = jnp.zeros((16, 16, 16, 3))
        B_init = B_init.at[..., 2].set(0.1)

        result = solver.compute_force_imbalance(B_init, p, geom)
        # Uniform B => J = 0 => J × B = 0, uniform p => grad(p) = 0
        # Force imbalance should be zero
        assert jnp.max(jnp.abs(result)) < 1e-10

    def test_harris_sheet_initializer(self):
        """Test Harris sheet initialization."""
        from jax_frc.equilibrium.initializers import harris_sheet_3d

        geom = Geometry(
            nx=16, ny=16, nz=32,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-2.0, z_max=2.0,
        )
        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        assert B.shape == (16, 16, 32, 3)
        # Bx should vary with y, By should be small
        assert jnp.max(jnp.abs(B[..., 0])) > 0.05
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_equilibrium_3d.py -v`
Expected: FAIL (ForceBalanceSolver not implemented)

**Step 3: Write minimal implementation**

Create/update `jax_frc/equilibrium/grad_shafranov.py`:

```python
"""3D force-balance equilibrium solver."""

from dataclasses import dataclass
from typing import Optional
import jax.numpy as jnp
from jax import jit, lax
from jax import Array

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.operators import curl_3d, gradient_3d
from jax_frc.solvers.divergence_cleaning import clean_divergence
from jax_frc.constants import MU0


@dataclass
class ForceBalanceSolver:
    """3D force-balance equilibrium solver.

    Finds B field satisfying J × B = ∇p iteratively.
    """
    max_iterations: int = 1000
    tolerance: float = 1e-6
    relaxation: float = 0.1

    def solve(self, geometry: Geometry, p_profile: Array,
              B_initial: Array) -> State:
        """Find force-balanced B field.

        Args:
            geometry: 3D geometry
            p_profile: Pressure field, shape (nx, ny, nz)
            B_initial: Initial B field guess

        Returns:
            State with equilibrium B field
        """
        def iteration_step(carry, _):
            B, _ = carry

            # Force imbalance: F = J × B - ∇p
            F = self.compute_force_imbalance(B, p_profile, geometry)

            # Relax B to reduce imbalance
            # B_new = B + dt * curl(F) (heuristic)
            curl_F = curl_3d(F, geometry)
            B_new = B + self.relaxation * curl_F

            # Clean divergence
            B_new = clean_divergence(B_new, geometry)

            error = jnp.max(jnp.abs(F))
            return (B_new, error), error

        init_carry = (B_initial, jnp.inf)
        (B_final, _), errors = lax.scan(
            iteration_step, init_carry, jnp.arange(self.max_iterations)
        )

        state = State.zeros(geometry.nx, geometry.ny, geometry.nz)
        return state.replace(B=B_final, p=p_profile)

    @staticmethod
    @jit
    def compute_force_imbalance(B: Array, p: Array, geometry: Geometry) -> Array:
        """Compute J × B - ∇p."""
        J = curl_3d(B, geometry) / MU0

        J_cross_B = jnp.stack([
            J[..., 1] * B[..., 2] - J[..., 2] * B[..., 1],
            J[..., 2] * B[..., 0] - J[..., 0] * B[..., 2],
            J[..., 0] * B[..., 1] - J[..., 1] * B[..., 0],
        ], axis=-1)

        grad_p = gradient_3d(p, geometry)

        return J_cross_B - grad_p
```

Create `jax_frc/equilibrium/initializers.py`:

```python
"""3D equilibrium initializers."""

import jax.numpy as jnp
from jax import Array
from jax_frc.core.geometry import Geometry


def harris_sheet_3d(geometry: Geometry, B0: float = 0.1, L: float = 0.1) -> Array:
    """Harris current sheet equilibrium.

    B_x = B0 * tanh(y / L)
    B_y = 0
    B_z = 0

    Args:
        geometry: 3D geometry
        B0: Asymptotic field strength
        L: Current sheet width

    Returns:
        B field, shape (nx, ny, nz, 3)
    """
    y = geometry.y_grid
    Bx = B0 * jnp.tanh(y / L)
    By = jnp.zeros_like(y)
    Bz = jnp.zeros_like(y)
    return jnp.stack([Bx, By, Bz], axis=-1)


def uniform_field_3d(geometry: Geometry, B0: float = 0.1, direction: str = "z") -> Array:
    """Uniform magnetic field.

    Args:
        geometry: 3D geometry
        B0: Field strength
        direction: "x", "y", or "z"

    Returns:
        B field, shape (nx, ny, nz, 3)
    """
    B = jnp.zeros((geometry.nx, geometry.ny, geometry.nz, 3))
    idx = {"x": 0, "y": 1, "z": 2}[direction]
    B = B.at[..., idx].set(B0)
    return B


def flux_rope_3d(geometry: Geometry, B0: float = 0.1, a: float = 0.3) -> Array:
    """Cylindrical flux rope (FRC-like).

    Axial field with twist, centered on z-axis.

    Args:
        geometry: 3D geometry
        B0: Peak field strength
        a: Flux rope radius

    Returns:
        B field, shape (nx, ny, nz, 3)
    """
    x, y = geometry.x_grid, geometry.y_grid
    r = jnp.sqrt(x**2 + y**2)

    # Axial field: Bz ~ (1 - (r/a)^2) inside, 0 outside
    Bz = B0 * jnp.maximum(1 - (r / a)**2, 0)

    # Small poloidal twist
    theta = jnp.arctan2(y, x)
    B_theta = 0.1 * B0 * (r / a) * jnp.where(r < a, 1.0, 0.0)
    Bx = -B_theta * jnp.sin(theta)
    By = B_theta * jnp.cos(theta)

    return jnp.stack([Bx, By, Bz], axis=-1)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_equilibrium_3d.py -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
git add jax_frc/equilibrium/grad_shafranov.py jax_frc/equilibrium/initializers.py tests/test_equilibrium_3d.py
git commit -m "feat(equilibrium): add 3D force-balance solver and initializers"
```

---

## Task 13: Validation Test - 3D Gaussian Diffusion

**Files:**
- Create: `tests/test_diffusion_3d.py`
- Create: `validation/diffusion_3d.py`

**Step 1: Write the failing test**

```python
# tests/test_diffusion_3d.py
"""Validation test: 3D Gaussian magnetic diffusion."""

import jax.numpy as jnp
import pytest
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.solvers.explicit import RK4Solver


def gaussian_diffusion_analytic(x, y, z, t, B0, sigma0, eta):
    """Analytic solution for 3D Gaussian diffusion.

    B_z(x,y,z,t) = B0 * (sigma0^2 / sigma(t)^2)^(3/2) * exp(-r^2 / (2*sigma(t)^2))
    where sigma(t)^2 = sigma0^2 + 2*eta*t, r^2 = x^2 + y^2 + z^2
    """
    MU0 = 1.2566e-6
    diffusivity = eta / MU0
    sigma_sq = sigma0**2 + 2 * diffusivity * t
    r_sq = x**2 + y**2 + z**2
    amplitude = B0 * (sigma0**2 / sigma_sq)**1.5
    return amplitude * jnp.exp(-r_sq / (2 * sigma_sq))


class TestGaussianDiffusion3D:
    """Test 3D Gaussian diffusion against analytic solution."""

    @pytest.mark.slow
    def test_diffusion_convergence(self):
        """Test convergence to analytic solution."""
        # Parameters
        eta = 1e-2
        B0 = 1.0
        sigma0 = 0.2
        t_final = 0.01

        # Grid
        geom = Geometry(
            nx=32, ny=32, nz=32,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-1.0, z_max=1.0,
        )

        # Initial condition
        x, y, z = geom.x_grid, geom.y_grid, geom.z_grid
        Bz_init = gaussian_diffusion_analytic(x, y, z, 0.0, B0, sigma0, eta)
        B_init = jnp.zeros((32, 32, 32, 3))
        B_init = B_init.at[..., 2].set(Bz_init)

        state = State.zeros(32, 32, 32)
        state = state.replace(B=B_init, n=jnp.ones((32, 32, 32)) * 1e19)

        # Evolve
        model = ResistiveMHD(eta=eta)
        solver = RK4Solver()
        dt = model.compute_stable_dt(state, geom) * 0.5

        n_steps = int(t_final / dt)
        for _ in range(n_steps):
            state = solver.step(model, state, geom, dt)

        # Compare to analytic
        Bz_analytic = gaussian_diffusion_analytic(x, y, z, t_final, B0, sigma0, eta)
        Bz_numeric = state.B[..., 2]

        # L2 error
        error = jnp.sqrt(jnp.mean((Bz_numeric - Bz_analytic)**2))
        rel_error = error / B0

        assert rel_error < 0.1, f"Relative error {rel_error:.3f} exceeds 10%"
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_diffusion_3d.py -v`
Expected: FAIL (may need RK4Solver or other imports)

**Step 3: Write minimal implementation**

The test uses existing components. Ensure `jax_frc/solvers/explicit.py` has RK4Solver:

```python
# jax_frc/solvers/explicit.py (if not already present)

from dataclasses import dataclass
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


@dataclass
class RK4Solver:
    """4th-order Runge-Kutta solver."""

    def step(self, model, state: State, geometry: Geometry, dt: float) -> State:
        """Single RK4 step."""
        k1 = model.compute_rhs(state, geometry)
        state_k1 = self._add_scaled(state, k1, 0.5 * dt)

        k2 = model.compute_rhs(state_k1, geometry)
        state_k2 = self._add_scaled(state, k2, 0.5 * dt)

        k3 = model.compute_rhs(state_k2, geometry)
        state_k3 = self._add_scaled(state, k3, dt)

        k4 = model.compute_rhs(state_k3, geometry)

        # Combine: state_new = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
        dB = (k1.B + 2*k2.B + 2*k3.B + k4.B) * (dt / 6)
        B_new = state.B + dB

        return state.replace(B=B_new)

    def _add_scaled(self, state: State, rhs: State, scale: float) -> State:
        """Return state + scale * rhs."""
        return state.replace(B=state.B + scale * rhs.B)
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_diffusion_3d.py -v`
Expected: PASS (1 test, marked slow)

**Step 5: Commit**

```bash
git add tests/test_diffusion_3d.py jax_frc/solvers/explicit.py
git commit -m "test: add 3D Gaussian diffusion validation test"
```

---

## Task 14: Validation Test - Alfvén Wave Propagation

**Files:**
- Create: `tests/test_alfven_wave.py`

**Step 1: Write the failing test**

```python
# tests/test_alfven_wave.py
"""Validation test: Alfvén wave propagation."""

import jax.numpy as jnp
import pytest
from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.solvers.explicit import RK4Solver
from jax_frc.constants import MU0


class TestAlfvenWave:
    """Test Alfvén wave propagation."""

    @pytest.mark.slow
    def test_wave_propagation(self):
        """Alfvén wave should propagate at v_A without damping (η=0)."""
        # Parameters
        B0 = 0.1       # Background field [T]
        rho0 = 1e-6    # Mass density [kg/m³]
        dB = 0.001     # Perturbation amplitude [T]
        eta = 0.0      # No resistivity (ideal MHD)

        # Alfvén speed
        v_A = B0 / jnp.sqrt(MU0 * rho0)

        # Domain: 1D-like (thin in x,y)
        Lz = 1.0
        geom = Geometry(
            nx=4, ny=4, nz=64,
            x_min=0.0, x_max=0.1,
            y_min=0.0, y_max=0.1,
            z_min=0.0, z_max=Lz,
            bc_x="periodic", bc_y="periodic", bc_z="periodic"
        )

        # Initial condition: B = B0*z_hat + dB*sin(kz)*y_hat
        k = 2 * jnp.pi / Lz
        z = geom.z_grid

        B_init = jnp.zeros((4, 4, 64, 3))
        B_init = B_init.at[..., 2].set(B0)           # Background Bz
        B_init = B_init.at[..., 1].set(dB * jnp.sin(k * z))  # Perturbation By

        # Need velocity for ideal MHD wave
        # For Alfvén wave: v_y = -dB * sin(kz) / sqrt(mu0 * rho0)
        v_init = jnp.zeros((4, 4, 64, 3))
        v_init = v_init.at[..., 1].set(-dB * jnp.sin(k * z) / jnp.sqrt(MU0 * rho0))

        state = State.zeros(4, 4, 64)
        n = rho0 / 1.673e-27  # Number density
        state = state.replace(B=B_init, v=v_init, n=jnp.full((4, 4, 64), n))

        # Evolve for one wave period
        T_wave = Lz / v_A
        t_final = T_wave

        model = ResistiveMHD(eta=eta)
        solver = RK4Solver()
        dt = 0.1 * geom.dz / v_A  # CFL

        t = 0.0
        while t < t_final:
            state = solver.step(model, state, geom, dt)
            t += dt

        # After one period, wave should return to initial state
        By_final = state.B[0, 0, :, 1]
        By_init = B_init[0, 0, :, 1]

        # Check phase matches (accounting for numerical diffusion)
        correlation = jnp.sum(By_final * By_init) / (
            jnp.sqrt(jnp.sum(By_final**2)) * jnp.sqrt(jnp.sum(By_init**2))
        )
        assert correlation > 0.9, f"Wave correlation {correlation:.3f} < 0.9"

        # Check amplitude preserved (no spurious damping)
        amp_ratio = jnp.max(jnp.abs(By_final)) / dB
        assert amp_ratio > 0.8, f"Amplitude ratio {amp_ratio:.3f} < 0.8"
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_alfven_wave.py -v`
Expected: May fail initially due to missing velocity handling

**Step 3: Ensure implementation handles velocity**

The ResistiveMHD model from Task 7 already handles velocity. No changes needed if implemented correctly.

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_alfven_wave.py -v`
Expected: PASS (1 test, marked slow)

**Step 5: Commit**

```bash
git add tests/test_alfven_wave.py
git commit -m "test: add Alfvén wave propagation validation test"
```

---

## Task 15: Update Existing Tests

**Files:**
- Modify: Various files in `tests/`

**Step 1: Identify failing tests**

Run: `py -m pytest tests/ -v --ignore=tests/test_diffusion_3d.py --ignore=tests/test_alfven_wave.py`

Identify which tests fail due to 2D→3D changes.

**Step 2: Update test fixtures**

For each failing test file, update array shapes from `(nr, nz)` to `(nx, ny, nz)`:

```python
# Example pattern change:
# Old: jnp.ones((64, 128))
# New: jnp.ones((8, 8, 16))

# Old: state = State.zeros(nr=64, nz=128)
# New: state = State.zeros(nx=8, ny=8, nz=16)

# Old: Geometry(nr=64, nz=128, ...)
# New: Geometry(nx=8, ny=8, nz=16, ...)
```

**Step 3: Run tests incrementally**

Fix tests one file at a time:
```bash
py -m pytest tests/test_<specific>.py -v
```

**Step 4: Run full test suite**

Run: `py -m pytest tests/ -v -k "not slow"`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add tests/
git commit -m "test: update existing tests for 3D arrays"
```

---

## Task 16: Update Module Exports

**Files:**
- Modify: `jax_frc/__init__.py`
- Modify: `jax_frc/operators.py` (exports)

**Step 1: Update `__init__.py`**

```python
# jax_frc/__init__.py
"""JAX-FRC: 3D Cartesian plasma simulation."""

from jax_frc.core.geometry import Geometry
from jax_frc.core.state import State, ParticleState
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.models.extended_mhd import ExtendedMHD
from jax_frc.models.hybrid_kinetic import HybridKinetic
from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState
from jax_frc.solvers.explicit import EulerSolver, RK4Solver
from jax_frc.solvers.divergence_cleaning import clean_divergence
from jax_frc.equilibrium.grad_shafranov import ForceBalanceSolver
from jax_frc.equilibrium.initializers import harris_sheet_3d, uniform_field_3d, flux_rope_3d
from jax_frc.operators import gradient_3d, divergence_3d, curl_3d, laplacian_3d

__all__ = [
    "Geometry",
    "State",
    "ParticleState",
    "ResistiveMHD",
    "ExtendedMHD",
    "HybridKinetic",
    "NeutralFluid",
    "NeutralState",
    "EulerSolver",
    "RK4Solver",
    "clean_divergence",
    "ForceBalanceSolver",
    "harris_sheet_3d",
    "uniform_field_3d",
    "flux_rope_3d",
    "gradient_3d",
    "divergence_3d",
    "curl_3d",
    "laplacian_3d",
]
```

**Step 2: Run import test**

```bash
python -c "from jax_frc import *; print('All imports OK')"
```

**Step 3: Commit**

```bash
git add jax_frc/__init__.py
git commit -m "chore: update module exports for 3D API"
```

---

## Task 17: Final Integration Test

**Files:**
- Create: `tests/test_integration_3d.py`

**Step 1: Write integration test**

```python
# tests/test_integration_3d.py
"""Integration test: full 3D simulation workflow."""

import jax.numpy as jnp
import pytest
from jax_frc import (
    Geometry, State, ResistiveMHD, ExtendedMHD,
    RK4Solver, clean_divergence, harris_sheet_3d
)


class TestIntegration3D:
    """Test complete 3D simulation workflow."""

    def test_resistive_mhd_workflow(self):
        """Test complete resistive MHD simulation."""
        geom = Geometry(nx=8, ny=8, nz=16)

        # Initialize Harris sheet
        B = harris_sheet_3d(geom, B0=0.1, L=0.2)
        state = State.zeros(8, 8, 16)
        state = state.replace(B=B, n=jnp.ones((8, 8, 16)) * 1e19)

        # Clean divergence
        B_clean = clean_divergence(state.B, geom)
        state = state.replace(B=B_clean)

        # Evolve
        model = ResistiveMHD(eta=1e-3)
        solver = RK4Solver()
        dt = model.compute_stable_dt(state, geom) * 0.5

        for _ in range(10):
            state = solver.step(model, state, geom, dt)

        # Check B field is still finite
        assert jnp.all(jnp.isfinite(state.B))

    def test_extended_mhd_workflow(self):
        """Test complete extended MHD simulation."""
        geom = Geometry(nx=8, ny=8, nz=16)

        state = State.zeros(8, 8, 16)
        B = jnp.zeros((8, 8, 16, 3))
        B = B.at[..., 2].set(0.1)
        state = state.replace(
            B=B,
            n=jnp.ones((8, 8, 16)) * 1e19,
            p=jnp.ones((8, 8, 16)) * 1e3,
            Te=jnp.ones((8, 8, 16)) * 100 * 1.602e-19,
        )

        model = ExtendedMHD(eta=1e-4, include_hall=True)
        solver = RK4Solver()
        dt = model.compute_stable_dt(state, geom) * 0.1

        for _ in range(5):
            state = solver.step(model, state, geom, dt)

        assert jnp.all(jnp.isfinite(state.B))
```

**Step 2: Run integration test**

Run: `py -m pytest tests/test_integration_3d.py -v`
Expected: PASS (2 tests)

**Step 3: Run full test suite**

Run: `py -m pytest tests/ -v -k "not slow"`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add tests/test_integration_3d.py
git commit -m "test: add 3D integration tests"
```

---

## Summary

This plan converts the jax-frc codebase from 2D cylindrical to 3D Cartesian coordinates through 17 tasks:

1. **Core Infrastructure** (Tasks 1-6): Geometry, operators, state
2. **Physics Models** (Tasks 7-11): Resistive MHD, Extended MHD, Divergence cleaning, Neutral fluid, Hybrid kinetic
3. **Equilibrium** (Task 12): 3D force-balance solver
4. **Validation** (Tasks 13-14): Diffusion and Alfvén wave tests
5. **Cleanup** (Tasks 15-17): Update existing tests, exports, integration

Each task follows TDD: write failing test → implement → verify pass → commit.
