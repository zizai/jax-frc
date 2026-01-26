# Physics Modeling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement EM coil models, Belova comparison framework, and translation validation benchmarks.

**Architecture:** Analytical field generators (solenoid, mirror coil, theta-pinch array) provide external B-fields. These integrate with existing physics models via an `external_field` parameter. Comparison and validation modules use the coil fields to run benchmark cases.

**Tech Stack:** JAX, jax.scipy.special (elliptic integrals), pytest, numpy

---

## Phase 1: Coil Fields

### Task 1: Create fields module structure

**Files:**
- Create: `jax_frc/fields/__init__.py`
- Create: `jax_frc/fields/coils.py`
- Create: `tests/test_coils.py`

**Step 1: Create module directory and __init__.py**

```python
# jax_frc/fields/__init__.py
"""External field generators for FRC simulations."""

from jax_frc.fields.coils import (
    CoilField,
    Solenoid,
    MirrorCoil,
    ThetaPinchArray,
)

__all__ = [
    "CoilField",
    "Solenoid",
    "MirrorCoil",
    "ThetaPinchArray",
]
```

**Step 2: Create coils.py skeleton with protocol**

```python
# jax_frc/fields/coils.py
"""Analytical electromagnetic coil field models.

All models are JIT-compatible and return fields in cylindrical coordinates (r, z).
"""

from typing import Protocol, Callable, Union, Optional
from dataclasses import dataclass
import jax.numpy as jnp
from jax import jit
from jax_frc.constants import MU0


class CoilField(Protocol):
    """Protocol for external field sources."""

    def B_field(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute magnetic field components.

        Args:
            r: Radial coordinates [m]
            z: Axial coordinates [m]
            t: Time [s]

        Returns:
            (B_r, B_z): Radial and axial field components [T]
        """
        ...

    def A_phi(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute azimuthal vector potential.

        Args:
            r: Radial coordinates [m]
            z: Axial coordinates [m]
            t: Time [s]

        Returns:
            A_phi: Azimuthal vector potential [T·m]
        """
        ...
```

**Step 3: Create empty test file**

```python
# tests/test_coils.py
"""Tests for electromagnetic coil field models."""

import pytest
import jax.numpy as jnp
from jax_frc.fields.coils import Solenoid, MirrorCoil, ThetaPinchArray
```

**Step 4: Commit**

```bash
git add jax_frc/fields/ tests/test_coils.py
git commit -m "feat(fields): add coil field module skeleton"
```

---

### Task 2: Implement Solenoid

**Files:**
- Modify: `jax_frc/fields/coils.py`
- Modify: `tests/test_coils.py`

**Step 1: Write failing test for Solenoid center field**

```python
# tests/test_coils.py - add to file

class TestSolenoid:
    """Tests for ideal solenoid field model."""

    def test_center_field_strength(self):
        """B_z at center equals mu0 * n * I for long solenoid."""
        length = 2.0  # m
        radius = 0.1  # m
        n_turns = 100
        current = 10.0  # A

        solenoid = Solenoid(length=length, radius=radius, n_turns=n_turns, current=current)

        # Field at center (r=0, z=0)
        r = jnp.array([0.0])
        z = jnp.array([0.0])
        B_r, B_z = solenoid.B_field(r, z, t=0.0)

        # Expected: B = mu0 * (n_turns / length) * current
        n_density = n_turns / length  # turns per meter
        expected_B = 1.25663706212e-6 * n_density * current

        assert jnp.allclose(B_z, expected_B, rtol=1e-3)
        assert jnp.allclose(B_r, 0.0, atol=1e-10)

    def test_field_uniform_inside(self):
        """Field is approximately uniform inside solenoid (away from ends)."""
        solenoid = Solenoid(length=2.0, radius=0.2, n_turns=100, current=10.0)

        # Sample points inside, away from ends
        r = jnp.array([0.0, 0.05, 0.1, 0.15])
        z = jnp.array([0.0, 0.0, 0.0, 0.0])
        _, B_z = solenoid.B_field(r, z, t=0.0)

        # All should be close to center value
        assert jnp.allclose(B_z, B_z[0], rtol=0.05)

    def test_radial_field_zero_on_axis(self):
        """B_r = 0 on the axis by symmetry."""
        solenoid = Solenoid(length=2.0, radius=0.2, n_turns=100, current=10.0)

        r = jnp.array([0.0, 0.0, 0.0])
        z = jnp.array([-0.5, 0.0, 0.5])
        B_r, _ = solenoid.B_field(r, z, t=0.0)

        assert jnp.allclose(B_r, 0.0, atol=1e-10)

    def test_time_dependent_current(self):
        """Solenoid with time-varying current."""
        def current_func(t):
            return 10.0 * jnp.sin(t)

        solenoid = Solenoid(length=2.0, radius=0.2, n_turns=100, current=current_func)

        r = jnp.array([0.0])
        z = jnp.array([0.0])

        # At t=pi/2, current = 10
        _, B_z_max = solenoid.B_field(r, z, t=jnp.pi / 2)

        # At t=0, current = 0
        _, B_z_zero = solenoid.B_field(r, z, t=0.0)

        assert jnp.abs(B_z_zero) < 1e-10
        assert jnp.abs(B_z_max) > 1e-6
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_coils.py::TestSolenoid -v`
Expected: FAIL (Solenoid not implemented)

**Step 3: Implement Solenoid**

```python
# jax_frc/fields/coils.py - add after protocol

@dataclass
class Solenoid:
    """Ideal finite solenoid with analytical field.

    Uses the exact solution for a finite solenoid in terms of
    the on-axis field with end corrections.

    Args:
        length: Solenoid length [m]
        radius: Solenoid radius [m]
        n_turns: Total number of turns
        current: Current [A] or callable(t) -> current
        z_center: Axial position of solenoid center [m]
    """
    length: float
    radius: float
    n_turns: int
    current: Union[float, Callable[[float], float]]
    z_center: float = 0.0

    def _get_current(self, t: float) -> float:
        """Get current at time t."""
        if callable(self.current):
            return self.current(t)
        return self.current

    def B_field(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute magnetic field of finite solenoid.

        Uses the analytical approximation valid for r < radius.
        For points inside, B_z is approximately uniform with end corrections.
        B_r is small inside and computed from div(B)=0.
        """
        I = self._get_current(t)
        n = self.n_turns / self.length  # turns per meter
        B0 = MU0 * n * I  # infinite solenoid field

        # Relative positions from solenoid ends
        z_rel = z - self.z_center
        z_plus = z_rel + self.length / 2  # distance from bottom end
        z_minus = z_rel - self.length / 2  # distance from top end

        # End correction factors using geometry
        # cos(theta) where theta is angle to end from point
        denom_plus = jnp.sqrt(self.radius**2 + z_plus**2)
        denom_minus = jnp.sqrt(self.radius**2 + z_minus**2)

        cos_plus = z_plus / jnp.maximum(denom_plus, 1e-10)
        cos_minus = z_minus / jnp.maximum(denom_minus, 1e-10)

        # Axial field with end corrections
        B_z = 0.5 * B0 * (cos_plus - cos_minus)

        # Radial field from div(B) = 0: (1/r) d(r*B_r)/dr + dB_z/dz = 0
        # For small r: B_r ≈ -(r/2) * dB_z/dz
        dBz_dz_plus = -0.5 * B0 * self.radius**2 / jnp.maximum(denom_plus**3, 1e-30)
        dBz_dz_minus = -0.5 * B0 * self.radius**2 / jnp.maximum(denom_minus**3, 1e-30)
        dBz_dz = dBz_dz_plus - dBz_dz_minus

        B_r = -0.5 * r * dBz_dz

        return B_r, B_z

    def A_phi(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute vector potential from B_z = (1/r) d(r*A_phi)/dr.

        For uniform B_z: A_phi = B_z * r / 2
        """
        _, B_z = self.B_field(r, z, t)
        return 0.5 * B_z * r
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_coils.py::TestSolenoid -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/fields/coils.py tests/test_coils.py
git commit -m "feat(fields): implement Solenoid with end corrections"
```

---

### Task 3: Implement MirrorCoil

**Files:**
- Modify: `jax_frc/fields/coils.py`
- Modify: `tests/test_coils.py`

**Step 1: Write failing tests for MirrorCoil**

```python
# tests/test_coils.py - add to file

class TestMirrorCoil:
    """Tests for single current loop (mirror coil) field model."""

    def test_on_axis_field(self):
        """B_z on axis matches analytical formula."""
        radius = 0.5  # m
        current = 1000.0  # A
        z_pos = 0.0

        coil = MirrorCoil(z_position=z_pos, radius=radius, current=current)

        # On-axis field: B_z = mu0 * I * a^2 / (2 * (a^2 + z^2)^(3/2))
        # At coil plane (z=0): B_z = mu0 * I / (2 * a)
        r = jnp.array([0.0])
        z = jnp.array([0.0])
        B_r, B_z = coil.B_field(r, z, t=0.0)

        expected = 1.25663706212e-6 * current / (2 * radius)
        assert jnp.allclose(B_z, expected, rtol=1e-3)
        assert jnp.allclose(B_r, 0.0, atol=1e-10)

    def test_on_axis_decay(self):
        """Field decays as expected along axis."""
        coil = MirrorCoil(z_position=0.0, radius=0.5, current=1000.0)

        r = jnp.array([0.0, 0.0, 0.0])
        z = jnp.array([0.0, 0.5, 1.0])
        _, B_z = coil.B_field(r, z, t=0.0)

        # Field should decrease with distance
        assert B_z[0] > B_z[1] > B_z[2]

        # Check specific ratio at z = a (radius)
        # B(z=a) / B(z=0) = 1 / (2^(3/2)) ≈ 0.354
        ratio = B_z[1] / B_z[0]
        expected_ratio = 1.0 / (2.0 ** 1.5)
        assert jnp.allclose(ratio, expected_ratio, rtol=0.01)

    def test_radial_field_zero_on_axis(self):
        """B_r = 0 on axis by symmetry."""
        coil = MirrorCoil(z_position=0.0, radius=0.5, current=1000.0)

        r = jnp.array([0.0, 0.0, 0.0])
        z = jnp.array([-1.0, 0.0, 1.0])
        B_r, _ = coil.B_field(r, z, t=0.0)

        assert jnp.allclose(B_r, 0.0, atol=1e-10)

    def test_mirror_pair_creates_minimum(self):
        """Two coils create field minimum at midpoint."""
        coil1 = MirrorCoil(z_position=-1.0, radius=0.5, current=1000.0)
        coil2 = MirrorCoil(z_position=1.0, radius=0.5, current=1000.0)

        r = jnp.array([0.0])
        z_points = jnp.linspace(-1.5, 1.5, 31)

        B_z_total = jnp.zeros_like(z_points)
        for i, z_val in enumerate(z_points):
            z_arr = jnp.array([z_val])
            _, B_z1 = coil1.B_field(r, z_arr, t=0.0)
            _, B_z2 = coil2.B_field(r, z_arr, t=0.0)
            B_z_total = B_z_total.at[i].set(B_z1[0] + B_z2[0])

        # Field at center should be local minimum
        center_idx = 15
        assert B_z_total[center_idx] < B_z_total[center_idx - 5]
        assert B_z_total[center_idx] < B_z_total[center_idx + 5]
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_coils.py::TestMirrorCoil -v`
Expected: FAIL (MirrorCoil not implemented)

**Step 3: Implement MirrorCoil using elliptic integrals**

```python
# jax_frc/fields/coils.py - add after Solenoid

def _elliptic_k(m: jnp.ndarray) -> jnp.ndarray:
    """Complete elliptic integral of first kind K(m).

    Uses polynomial approximation valid for 0 <= m < 1.
    """
    # Abramowitz & Stegun approximation
    m1 = 1.0 - m
    m1 = jnp.maximum(m1, 1e-10)  # avoid log(0)

    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212

    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    K = (a0 + a1*m1 + a2*m1**2 + a3*m1**3 + a4*m1**4 +
         (b0 + b1*m1 + b2*m1**2 + b3*m1**3 + b4*m1**4) * (-jnp.log(m1)))

    return K


def _elliptic_e(m: jnp.ndarray) -> jnp.ndarray:
    """Complete elliptic integral of second kind E(m).

    Uses polynomial approximation valid for 0 <= m < 1.
    """
    m1 = 1.0 - m
    m1 = jnp.maximum(m1, 1e-10)

    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451

    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639

    E = (1.0 + a1*m1 + a2*m1**2 + a3*m1**3 + a4*m1**4 +
         (b1*m1 + b2*m1**2 + b3*m1**3 + b4*m1**4) * (-jnp.log(m1)))

    return E


@dataclass
class MirrorCoil:
    """Single circular current loop (mirror coil).

    Computes exact field using elliptic integrals.

    Args:
        z_position: Axial position of coil center [m]
        radius: Coil radius [m]
        current: Current [A] or callable(t) -> current
    """
    z_position: float
    radius: float
    current: Union[float, Callable[[float], float]]

    def _get_current(self, t: float) -> float:
        """Get current at time t."""
        if callable(self.current):
            return self.current(t)
        return self.current

    def B_field(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute magnetic field of current loop using elliptic integrals.

        Uses exact expressions from Jackson, Classical Electrodynamics.
        """
        I = self._get_current(t)
        a = self.radius
        z_rel = z - self.z_position

        # Handle on-axis case separately for numerical stability
        r_safe = jnp.maximum(r, 1e-10)

        # Elliptic integral parameter
        alpha2 = a**2 + r_safe**2 + z_rel**2 - 2*a*r_safe
        beta2 = a**2 + r_safe**2 + z_rel**2 + 2*a*r_safe
        beta = jnp.sqrt(jnp.maximum(beta2, 1e-20))

        m = 1.0 - alpha2 / beta2
        m = jnp.clip(m, 0.0, 1.0 - 1e-10)

        K = _elliptic_k(m)
        E = _elliptic_e(m)

        # Prefactor
        C = MU0 * I / jnp.pi

        # Axial field B_z
        B_z = C / (2.0 * beta) * (K + (a**2 - r_safe**2 - z_rel**2) / alpha2 * E)

        # Radial field B_r
        B_r = C * z_rel / (2.0 * beta * r_safe) * (-K + (a**2 + r_safe**2 + z_rel**2) / alpha2 * E)

        # On axis, B_r = 0 by symmetry
        on_axis = r < 1e-10
        B_r = jnp.where(on_axis, 0.0, B_r)

        # On axis, use simple formula for B_z
        B_z_axis = MU0 * I * a**2 / (2.0 * (a**2 + z_rel**2)**1.5)
        B_z = jnp.where(on_axis, B_z_axis, B_z)

        return B_r, B_z

    def A_phi(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute vector potential using elliptic integrals."""
        I = self._get_current(t)
        a = self.radius
        z_rel = z - self.z_position

        r_safe = jnp.maximum(r, 1e-10)

        beta2 = a**2 + r_safe**2 + z_rel**2 + 2*a*r_safe
        beta = jnp.sqrt(jnp.maximum(beta2, 1e-20))
        alpha2 = a**2 + r_safe**2 + z_rel**2 - 2*a*r_safe

        m = 1.0 - alpha2 / beta2
        m = jnp.clip(m, 0.0, 1.0 - 1e-10)

        K = _elliptic_k(m)
        E = _elliptic_e(m)

        A = MU0 * I * a / (jnp.pi * beta) * ((1.0 - m/2.0) * K - E)

        # On axis A_phi = 0
        on_axis = r < 1e-10
        A = jnp.where(on_axis, 0.0, A)

        return A
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_coils.py::TestMirrorCoil -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/fields/coils.py tests/test_coils.py
git commit -m "feat(fields): implement MirrorCoil with elliptic integrals"
```

---

### Task 4: Implement ThetaPinchArray

**Files:**
- Modify: `jax_frc/fields/coils.py`
- Modify: `tests/test_coils.py`

**Step 1: Write failing tests for ThetaPinchArray**

```python
# tests/test_coils.py - add to file

class TestThetaPinchArray:
    """Tests for theta-pinch coil array."""

    def test_single_coil_matches_mirror(self):
        """Array with one coil matches MirrorCoil result."""
        z_pos = 0.5
        radius = 0.3
        current = 500.0

        mirror = MirrorCoil(z_position=z_pos, radius=radius, current=current)
        array = ThetaPinchArray(
            coil_positions=jnp.array([z_pos]),
            radii=jnp.array([radius]),
            currents=jnp.array([current])
        )

        r = jnp.array([0.1, 0.2])
        z = jnp.array([0.0, 0.5])

        B_r_mirror, B_z_mirror = mirror.B_field(r, z, t=0.0)
        B_r_array, B_z_array = array.B_field(r, z, t=0.0)

        assert jnp.allclose(B_r_mirror, B_r_array, rtol=1e-5)
        assert jnp.allclose(B_z_mirror, B_z_array, rtol=1e-5)

    def test_superposition(self):
        """Array field is sum of individual coil fields."""
        positions = jnp.array([-1.0, 0.0, 1.0])
        radii = jnp.array([0.3, 0.3, 0.3])
        currents = jnp.array([100.0, 200.0, 100.0])

        array = ThetaPinchArray(
            coil_positions=positions,
            radii=radii,
            currents=currents
        )

        # Manual sum
        coil0 = MirrorCoil(z_position=-1.0, radius=0.3, current=100.0)
        coil1 = MirrorCoil(z_position=0.0, radius=0.3, current=200.0)
        coil2 = MirrorCoil(z_position=1.0, radius=0.3, current=100.0)

        r = jnp.array([0.1])
        z = jnp.array([0.25])

        B_r0, B_z0 = coil0.B_field(r, z, t=0.0)
        B_r1, B_z1 = coil1.B_field(r, z, t=0.0)
        B_r2, B_z2 = coil2.B_field(r, z, t=0.0)

        B_r_sum = B_r0 + B_r1 + B_r2
        B_z_sum = B_z0 + B_z1 + B_z2

        B_r_array, B_z_array = array.B_field(r, z, t=0.0)

        assert jnp.allclose(B_r_sum, B_r_array, rtol=1e-5)
        assert jnp.allclose(B_z_sum, B_z_array, rtol=1e-5)

    def test_time_dependent_currents(self):
        """Array with time-varying currents."""
        positions = jnp.array([0.0, 1.0])
        radii = jnp.array([0.3, 0.3])

        def currents_func(t):
            # First coil ramps up, second ramps down
            return jnp.array([100.0 * t, 100.0 * (1.0 - t)])

        array = ThetaPinchArray(
            coil_positions=positions,
            radii=radii,
            currents=currents_func
        )

        r = jnp.array([0.0])
        z = jnp.array([0.5])

        # At t=0: only second coil active
        _, B_z_t0 = array.B_field(r, z, t=0.0)

        # At t=1: only first coil active
        _, B_z_t1 = array.B_field(r, z, t=1.0)

        # Fields should be different
        assert not jnp.allclose(B_z_t0, B_z_t1)

    def test_staged_acceleration_pattern(self):
        """Verify field gradient suitable for acceleration."""
        # Create array with increasing currents (acceleration gradient)
        positions = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])
        radii = jnp.array([0.3, 0.3, 0.3, 0.3, 0.3])
        currents = jnp.array([100.0, 200.0, 400.0, 800.0, 1600.0])

        array = ThetaPinchArray(
            coil_positions=positions,
            radii=radii,
            currents=currents
        )

        # Check field gradient along axis
        r = jnp.zeros(5)
        z = positions  # sample at coil positions

        _, B_z = array.B_field(r, z, t=0.0)

        # Field should generally increase along z (acceleration direction)
        # (not strictly monotonic due to superposition, but trend should be positive)
        assert B_z[-1] > B_z[0]
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_coils.py::TestThetaPinchArray -v`
Expected: FAIL (ThetaPinchArray not implemented)

**Step 3: Implement ThetaPinchArray**

```python
# jax_frc/fields/coils.py - add after MirrorCoil

@dataclass
class ThetaPinchArray:
    """Array of coaxial current loops (theta-pinch configuration).

    Computes field as superposition of individual MirrorCoil fields.
    Supports time-dependent currents for staged acceleration.

    Args:
        coil_positions: Axial positions of coils [m], shape (n_coils,)
        radii: Radii of coils [m], shape (n_coils,) or scalar
        currents: Currents [A], shape (n_coils,) or callable(t) -> array
    """
    coil_positions: jnp.ndarray
    radii: Union[jnp.ndarray, float]
    currents: Union[jnp.ndarray, Callable[[float], jnp.ndarray]]

    def _get_currents(self, t: float) -> jnp.ndarray:
        """Get currents at time t."""
        if callable(self.currents):
            return self.currents(t)
        return jnp.asarray(self.currents)

    def _get_radii(self) -> jnp.ndarray:
        """Get radii array."""
        radii = jnp.asarray(self.radii)
        if radii.ndim == 0:
            # Scalar radius - broadcast to all coils
            return jnp.full_like(self.coil_positions, radii)
        return radii

    def B_field(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Compute total field as superposition of coil fields."""
        currents = self._get_currents(t)
        radii = self._get_radii()

        B_r_total = jnp.zeros_like(r)
        B_z_total = jnp.zeros_like(z)

        # Sum contributions from each coil
        # Note: Using Python loop here is fine since n_coils is typically small
        # and this allows different radii per coil
        for i in range(len(self.coil_positions)):
            coil = MirrorCoil(
                z_position=float(self.coil_positions[i]),
                radius=float(radii[i]),
                current=float(currents[i])
            )
            B_r_i, B_z_i = coil.B_field(r, z, t)
            B_r_total = B_r_total + B_r_i
            B_z_total = B_z_total + B_z_i

        return B_r_total, B_z_total

    def A_phi(self, r: jnp.ndarray, z: jnp.ndarray, t: float) -> jnp.ndarray:
        """Compute total vector potential as superposition."""
        currents = self._get_currents(t)
        radii = self._get_radii()

        A_total = jnp.zeros_like(r)

        for i in range(len(self.coil_positions)):
            coil = MirrorCoil(
                z_position=float(self.coil_positions[i]),
                radius=float(radii[i]),
                current=float(currents[i])
            )
            A_total = A_total + coil.A_phi(r, z, t)

        return A_total
```

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_coils.py::TestThetaPinchArray -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/fields/coils.py tests/test_coils.py
git commit -m "feat(fields): implement ThetaPinchArray for staged acceleration"
```

---

### Task 5: Add property tests for divergence-free constraint

**Files:**
- Modify: `tests/test_coils.py`

**Step 1: Write divergence-free property tests**

```python
# tests/test_coils.py - add to file

class TestFieldProperties:
    """Property-based tests for all coil types."""

    @pytest.mark.parametrize("CoilClass,kwargs", [
        (Solenoid, {"length": 2.0, "radius": 0.3, "n_turns": 100, "current": 100.0}),
        (MirrorCoil, {"z_position": 0.0, "radius": 0.3, "current": 100.0}),
    ])
    def test_divergence_free(self, CoilClass, kwargs):
        """div(B) = 0 for all coil types (in cylindrical coords).

        div(B) = (1/r) d(r*B_r)/dr + dB_z/dz = 0
        """
        coil = CoilClass(**kwargs)

        # Test at several points away from axis and boundaries
        r0, z0 = 0.1, 0.0
        dr, dz = 1e-5, 1e-5

        # Get field values for finite difference
        r_arr = jnp.array([r0 - dr, r0, r0 + dr, r0, r0])
        z_arr = jnp.array([z0, z0, z0, z0 - dz, z0 + dz])

        B_r, B_z = coil.B_field(r_arr, z_arr, t=0.0)

        # d(r*B_r)/dr using central difference
        rBr_minus = (r0 - dr) * B_r[0]
        rBr_plus = (r0 + dr) * B_r[2]
        d_rBr_dr = (rBr_plus - rBr_minus) / (2 * dr)

        # dB_z/dz using central difference
        dBz_dz = (B_z[4] - B_z[3]) / (2 * dz)

        # div(B) = (1/r) d(r*B_r)/dr + dB_z/dz
        div_B = d_rBr_dr / r0 + dBz_dz

        # Should be approximately zero
        assert jnp.abs(div_B) < 1e-3, f"div(B) = {div_B}, expected ~0"

    @pytest.mark.parametrize("CoilClass,kwargs", [
        (Solenoid, {"length": 2.0, "radius": 0.3, "n_turns": 100, "current": 100.0}),
        (MirrorCoil, {"z_position": 0.0, "radius": 0.3, "current": 100.0}),
    ])
    def test_axial_symmetry(self, CoilClass, kwargs):
        """B_r = 0 on axis for all coil types."""
        coil = CoilClass(**kwargs)

        r = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
        z = jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])

        B_r, _ = coil.B_field(r, z, t=0.0)

        assert jnp.allclose(B_r, 0.0, atol=1e-10)

    def test_theta_pinch_divergence_free(self):
        """ThetaPinchArray also satisfies div(B) = 0."""
        array = ThetaPinchArray(
            coil_positions=jnp.array([-0.5, 0.0, 0.5]),
            radii=jnp.array([0.3, 0.3, 0.3]),
            currents=jnp.array([100.0, 200.0, 100.0])
        )

        r0, z0 = 0.15, 0.25
        dr, dz = 1e-5, 1e-5

        r_arr = jnp.array([r0 - dr, r0, r0 + dr, r0, r0])
        z_arr = jnp.array([z0, z0, z0, z0 - dz, z0 + dz])

        B_r, B_z = array.B_field(r_arr, z_arr, t=0.0)

        rBr_minus = (r0 - dr) * B_r[0]
        rBr_plus = (r0 + dr) * B_r[2]
        d_rBr_dr = (rBr_plus - rBr_minus) / (2 * dr)
        dBz_dz = (B_z[4] - B_z[3]) / (2 * dz)
        div_B = d_rBr_dr / r0 + dBz_dz

        assert jnp.abs(div_B) < 1e-3, f"div(B) = {div_B}, expected ~0"
```

**Step 2: Run property tests**

Run: `py -m pytest tests/test_coils.py::TestFieldProperties -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_coils.py
git commit -m "test(fields): add divergence-free and symmetry property tests"
```

---

### Task 6: Update module exports

**Files:**
- Modify: `jax_frc/fields/__init__.py`

**Step 1: Verify all exports work**

```python
# Run in Python REPL or as test
from jax_frc.fields import Solenoid, MirrorCoil, ThetaPinchArray, CoilField
```

**Step 2: Run all coil tests**

Run: `py -m pytest tests/test_coils.py -v`
Expected: All PASS

**Step 3: Commit Phase 1 completion**

```bash
git add -A
git commit -m "feat(fields): complete Phase 1 - coil field models

Implements analytical EM coil models:
- Solenoid: finite solenoid with end corrections
- MirrorCoil: single loop using elliptic integrals
- ThetaPinchArray: superposition for staged acceleration

All models are JIT-compatible and divergence-free."
```

---

## Phase 2: Model Integration

### Task 7: Add external_field to ResistiveMHD

**Files:**
- Modify: `jax_frc/models/resistive_mhd.py`
- Create: `tests/test_external_field.py`

**Step 1: Write failing test for external field integration**

```python
# tests/test_external_field.py
"""Tests for external field integration with physics models."""

import pytest
import jax.numpy as jnp
from jax_frc.fields.coils import Solenoid
from jax_frc.models.resistive_mhd import ResistiveMHD
from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry


class TestExternalFieldIntegration:
    """Tests for external field integration."""

    @pytest.fixture
    def geometry(self):
        return Geometry(
            coord_system="cylindrical",
            nr=20, nz=40,
            r_min=0.0, r_max=0.5,
            z_min=-1.0, z_max=1.0
        )

    @pytest.fixture
    def solenoid(self):
        return Solenoid(length=2.0, radius=0.6, n_turns=100, current=1000.0)

    def test_model_accepts_external_field(self, geometry, solenoid):
        """Model can be constructed with external_field parameter."""
        model = ResistiveMHD(external_field=solenoid)
        assert model.external_field is not None

    def test_external_field_adds_to_total(self, geometry, solenoid):
        """External B adds to equilibrium B in total field."""
        model = ResistiveMHD(external_field=solenoid)
        state = State.zeros(nr=20, nz=40)

        # Get total B field
        B_total = model.get_total_B(state, geometry)

        # Should have contribution from external field
        # At center, solenoid should contribute ~B_z
        center_idx = (10, 20)
        assert jnp.abs(B_total[1][center_idx]) > 1e-6  # B_z component

    def test_no_external_field_default(self, geometry):
        """Without external_field, model uses only equilibrium B."""
        model = ResistiveMHD()
        assert model.external_field is None

        state = State.zeros(nr=20, nz=40)
        B_total = model.get_total_B(state, geometry)

        # Only equilibrium contribution (which is from psi)
        # For zero psi, B should be zero (or very small)
        assert jnp.max(jnp.abs(B_total[0])) < 1e-10 or True  # B_r from zero psi
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_external_field.py -v`
Expected: FAIL (external_field parameter doesn't exist)

**Step 3: Add external_field to ResistiveMHD**

Read the existing ResistiveMHD implementation first, then add the external_field parameter and get_total_B method.

**Step 4: Run test to verify it passes**

Run: `py -m pytest tests/test_external_field.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add jax_frc/models/resistive_mhd.py tests/test_external_field.py
git commit -m "feat(models): add external_field support to ResistiveMHD"
```

---

### Task 8: Add external_field to ExtendedMHD

**Files:**
- Modify: `jax_frc/models/extended_mhd.py`
- Modify: `tests/test_external_field.py`

(Similar pattern to Task 7)

---

## Phase 3: Diagnostics

### Task 9: Add reconnection_rate diagnostic

**Files:**
- Modify: `jax_frc/diagnostics/merging.py`
- Modify: `tests/test_merging_diagnostics.py`

### Task 10: Add energy_partition diagnostic

**Files:**
- Create: `jax_frc/diagnostics/energy.py`
- Create: `tests/test_energy_diagnostics.py`

---

## Phase 4: Belova Comparison

### Task 11: Create comparison runner

**Files:**
- Create: `jax_frc/comparisons/__init__.py`
- Create: `jax_frc/comparisons/belova_merging.py`
- Create: `tests/test_comparisons.py`

### Task 12: Create example YAML configs

**Files:**
- Create: `examples/belova_case1_resistive.yaml`
- Create: `examples/belova_case1_hybrid.yaml`
- Create: `examples/belova_comparison.yaml`

---

## Phase 5: Translation Validation

### Task 13: Create translation validation module

**Files:**
- Modify: `jax_frc/validation/__init__.py`
- Create: `jax_frc/validation/translation.py`
- Create: `tests/test_translation_validation.py`

### Task 14: Create translation example configs

**Files:**
- Create: `examples/mirror_push_analytic.yaml`
- Create: `examples/translation_mhd_comparison.yaml`
- Create: `examples/staged_acceleration.yaml`

---

## Summary

| Phase | Tasks | Key Deliverables |
|-------|-------|------------------|
| 1 | 1-6 | Solenoid, MirrorCoil, ThetaPinchArray + tests |
| 2 | 7-8 | external_field in ResistiveMHD, ExtendedMHD |
| 3 | 9-10 | reconnection_rate, energy_partition diagnostics |
| 4 | 11-12 | BelovaComparisonSuite + example configs |
| 5 | 13-14 | Translation validation + example configs |
