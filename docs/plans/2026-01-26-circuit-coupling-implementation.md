# Circuit Coupling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add bidirectional plasma-circuit coupling with multi-coil energy extraction to the burning plasma model.

**Architecture:** New `jax_frc/circuits/` module with CircuitState, PickupCoilArray, ExternalCircuits, FluxCoupling, and CircuitSystem classes. Replaces existing DirectConversion in BurningPlasmaModel. All components are JAX pytrees using `lax.scan` for time integration.

**Tech Stack:** JAX, jax.numpy, dataclasses, lax.scan for loops, existing jax_frc.constants for MU0.

**Worktree:** `.worktrees/circuit-coupling` (branch: `feature/circuit-coupling`)

**Test command:** `py -m pytest tests/ -k "not slow" -v`

---

## Task 1: Create CircuitState and CircuitParams

**Files:**
- Create: `jax_frc/circuits/__init__.py`
- Create: `jax_frc/circuits/state.py`
- Create: `tests/test_circuit_state.py`

**Step 1: Create the circuits package init file**

Create `jax_frc/circuits/__init__.py`:

```python
"""Circuit coupling for burning plasma simulations."""

from jax_frc.circuits.state import CircuitState, CircuitParams

__all__ = ["CircuitState", "CircuitParams"]
```

**Step 2: Write the failing test for CircuitState**

Create `tests/test_circuit_state.py`:

```python
"""Tests for circuit state dataclasses."""

import jax
import jax.numpy as jnp
import pytest


class TestCircuitParams:
    """Tests for CircuitParams dataclass."""

    def test_creation(self):
        """Can create CircuitParams with arrays."""
        from jax_frc.circuits import CircuitParams

        params = CircuitParams(
            L=jnp.array([1e-3, 1e-3]),
            R=jnp.array([0.1, 0.1]),
            C=jnp.array([jnp.inf, jnp.inf]),
        )
        assert params.L.shape == (2,)
        assert params.R.shape == (2,)

    def test_tau_property(self):
        """tau = L/R gives circuit timescale."""
        from jax_frc.circuits import CircuitParams

        params = CircuitParams(
            L=jnp.array([1e-3, 2e-3]),
            R=jnp.array([0.1, 0.1]),
            C=jnp.array([jnp.inf, jnp.inf]),
        )
        tau = params.L / params.R
        assert jnp.allclose(tau, jnp.array([0.01, 0.02]))


class TestCircuitState:
    """Tests for CircuitState dataclass."""

    def test_creation(self):
        """Can create CircuitState."""
        from jax_frc.circuits import CircuitState

        state = CircuitState(
            I_pickup=jnp.zeros(3),
            Q_pickup=jnp.zeros(3),
            I_external=jnp.zeros(2),
            Q_external=jnp.zeros(2),
            Psi_pickup=jnp.zeros(3),
            Psi_external=jnp.zeros(2),
            P_extracted=0.0,
            P_dissipated=0.0,
        )
        assert state.I_pickup.shape == (3,)
        assert state.I_external.shape == (2,)

    def test_is_pytree(self):
        """CircuitState works with JAX transformations."""
        from jax_frc.circuits import CircuitState

        state = CircuitState(
            I_pickup=jnp.array([1.0, 2.0]),
            Q_pickup=jnp.zeros(2),
            I_external=jnp.array([0.5]),
            Q_external=jnp.zeros(1),
            Psi_pickup=jnp.zeros(2),
            Psi_external=jnp.zeros(1),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        @jax.jit
        def double_currents(s):
            return s.replace(I_pickup=s.I_pickup * 2)

        new_state = double_currents(state)
        assert jnp.allclose(new_state.I_pickup, jnp.array([2.0, 4.0]))

    def test_zeros_factory(self):
        """Can create zero-initialized state."""
        from jax_frc.circuits import CircuitState

        state = CircuitState.zeros(n_pickup=3, n_external=2)
        assert state.I_pickup.shape == (3,)
        assert state.I_external.shape == (2,)
        assert jnp.all(state.I_pickup == 0)
```

**Step 3: Run test to verify it fails**

Run: `py -m pytest tests/test_circuit_state.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'jax_frc.circuits'"

**Step 4: Write CircuitParams and CircuitState implementation**

Create `jax_frc/circuits/state.py`:

```python
"""Circuit state and parameter dataclasses."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class CircuitParams:
    """Parameters for a set of RLC circuits.

    Attributes:
        L: Inductance [H], shape (n_coils,)
        R: Resistance [Ω], shape (n_coils,)
        C: Capacitance [F], shape (n_coils,) - use jnp.inf for no capacitor
    """

    L: Array
    R: Array
    C: Array


@dataclass(frozen=True)
class CircuitState:
    """State of all circuits in the system.

    Attributes:
        I_pickup: Pickup coil currents [A], shape (n_pickup,)
        Q_pickup: Pickup capacitor charges [C], shape (n_pickup,)
        I_external: External coil currents [A], shape (n_external,)
        Q_external: External capacitor charges [C], shape (n_external,)
        Psi_pickup: Flux linkage through pickup coils [Wb], shape (n_pickup,)
        Psi_external: Flux linkage through external coils [Wb], shape (n_external,)
        P_extracted: Total power to loads [W]
        P_dissipated: Total power dissipated in resistance [W]
    """

    I_pickup: Array
    Q_pickup: Array
    I_external: Array
    Q_external: Array
    Psi_pickup: Array
    Psi_external: Array
    P_extracted: float
    P_dissipated: float

    def replace(self, **kwargs) -> "CircuitState":
        """Return new state with specified fields replaced."""
        from dataclasses import replace as dc_replace

        return dc_replace(self, **kwargs)

    @classmethod
    def zeros(cls, n_pickup: int, n_external: int) -> "CircuitState":
        """Create zero-initialized circuit state."""
        return cls(
            I_pickup=jnp.zeros(n_pickup),
            Q_pickup=jnp.zeros(n_pickup),
            I_external=jnp.zeros(n_external),
            Q_external=jnp.zeros(n_external),
            Psi_pickup=jnp.zeros(n_pickup),
            Psi_external=jnp.zeros(n_external),
            P_extracted=0.0,
            P_dissipated=0.0,
        )


# Register as JAX pytree
def _circuit_state_flatten(state):
    children = (
        state.I_pickup,
        state.Q_pickup,
        state.I_external,
        state.Q_external,
        state.Psi_pickup,
        state.Psi_external,
        state.P_extracted,
        state.P_dissipated,
    )
    aux_data = None
    return children, aux_data


def _circuit_state_unflatten(aux_data, children):
    return CircuitState(*children)


jax.tree_util.register_pytree_node(
    CircuitState,
    _circuit_state_flatten,
    _circuit_state_unflatten,
)
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_circuit_state.py -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add jax_frc/circuits/__init__.py jax_frc/circuits/state.py tests/test_circuit_state.py
git commit -m "feat(circuits): add CircuitState and CircuitParams dataclasses"
```

---

## Task 2: Create Waveform Functions

**Files:**
- Create: `jax_frc/circuits/waveforms.py`
- Create: `tests/test_waveforms.py`
- Modify: `jax_frc/circuits/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_waveforms.py`:

```python
"""Tests for circuit waveform functions."""

import jax.numpy as jnp
import pytest


class TestRampWaveform:
    """Tests for ramp voltage waveform."""

    def test_ramp_start(self):
        """Ramp starts at V0."""
        from jax_frc.circuits.waveforms import make_ramp

        ramp = make_ramp(V0=0.0, V1=1000.0, t_ramp=1e-4)
        assert ramp(0.0) == 0.0

    def test_ramp_end(self):
        """Ramp reaches V1 at t_ramp."""
        from jax_frc.circuits.waveforms import make_ramp

        ramp = make_ramp(V0=0.0, V1=1000.0, t_ramp=1e-4)
        assert ramp(1e-4) == 1000.0

    def test_ramp_midpoint(self):
        """Ramp is linear."""
        from jax_frc.circuits.waveforms import make_ramp

        ramp = make_ramp(V0=0.0, V1=1000.0, t_ramp=1e-4)
        assert jnp.isclose(ramp(5e-5), 500.0)

    def test_ramp_saturates(self):
        """Ramp holds at V1 after t_ramp."""
        from jax_frc.circuits.waveforms import make_ramp

        ramp = make_ramp(V0=0.0, V1=1000.0, t_ramp=1e-4)
        assert ramp(2e-4) == 1000.0


class TestSinusoidWaveform:
    """Tests for sinusoidal waveform."""

    def test_sinusoid_amplitude(self):
        """Sinusoid has correct amplitude."""
        from jax_frc.circuits.waveforms import make_sinusoid

        sin_wave = make_sinusoid(amplitude=100.0, frequency=1e3, phase=0.0)
        # At t = 1/(4*f), sin(pi/2) = 1
        t_quarter = 1.0 / (4 * 1e3)
        assert jnp.isclose(sin_wave(t_quarter), 100.0, rtol=1e-5)

    def test_sinusoid_frequency(self):
        """Sinusoid has correct period."""
        from jax_frc.circuits.waveforms import make_sinusoid

        sin_wave = make_sinusoid(amplitude=100.0, frequency=1e3, phase=0.0)
        # At t = 1/f (one period), should return to 0
        assert jnp.isclose(sin_wave(1e-3), 0.0, atol=1e-10)


class TestCrowbarWaveform:
    """Tests for crowbar (step-down) waveform."""

    def test_crowbar_before(self):
        """Crowbar is at V_initial before t_crowbar."""
        from jax_frc.circuits.waveforms import make_crowbar

        crowbar = make_crowbar(V_initial=1000.0, t_crowbar=1e-4)
        assert crowbar(5e-5) == 1000.0

    def test_crowbar_after(self):
        """Crowbar drops to 0 after t_crowbar."""
        from jax_frc.circuits.waveforms import make_crowbar

        crowbar = make_crowbar(V_initial=1000.0, t_crowbar=1e-4)
        assert crowbar(1.5e-4) == 0.0


class TestPulseWaveform:
    """Tests for pulse waveform."""

    def test_pulse_on(self):
        """Pulse is at amplitude during pulse."""
        from jax_frc.circuits.waveforms import make_pulse

        pulse = make_pulse(amplitude=500.0, t_start=1e-5, t_end=5e-5)
        assert pulse(3e-5) == 500.0

    def test_pulse_off_before(self):
        """Pulse is 0 before t_start."""
        from jax_frc.circuits.waveforms import make_pulse

        pulse = make_pulse(amplitude=500.0, t_start=1e-5, t_end=5e-5)
        assert pulse(0.5e-5) == 0.0

    def test_pulse_off_after(self):
        """Pulse is 0 after t_end."""
        from jax_frc.circuits.waveforms import make_pulse

        pulse = make_pulse(amplitude=500.0, t_start=1e-5, t_end=5e-5)
        assert pulse(6e-5) == 0.0


class TestWaveformFromConfig:
    """Tests for creating waveforms from config dicts."""

    def test_ramp_from_config(self):
        """Can create ramp from config."""
        from jax_frc.circuits.waveforms import waveform_from_config

        config = {"type": "ramp", "V0": 0, "V1": 1000, "t_ramp": 1e-4}
        waveform = waveform_from_config(config)
        assert waveform(1e-4) == 1000.0

    def test_constant_from_config(self):
        """Can create constant voltage from config."""
        from jax_frc.circuits.waveforms import waveform_from_config

        config = {"type": "constant", "value": 500.0}
        waveform = waveform_from_config(config)
        assert waveform(0.0) == 500.0
        assert waveform(1.0) == 500.0
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_waveforms.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write waveforms implementation**

Create `jax_frc/circuits/waveforms.py`:

```python
"""Waveform generators for circuit voltage/current sources.

All waveforms are JAX-compatible pure functions suitable for JIT compilation.
"""

from typing import Callable

import jax.numpy as jnp


def make_ramp(V0: float, V1: float, t_ramp: float) -> Callable[[float], float]:
    """Create linear ramp waveform.

    Args:
        V0: Initial value
        V1: Final value
        t_ramp: Time to reach V1 [s]

    Returns:
        Function t -> V that ramps from V0 to V1 over t_ramp, then holds at V1
    """

    def ramp(t: float) -> float:
        fraction = jnp.clip(t / t_ramp, 0.0, 1.0)
        return V0 + (V1 - V0) * fraction

    return ramp


def make_sinusoid(
    amplitude: float, frequency: float, phase: float = 0.0
) -> Callable[[float], float]:
    """Create sinusoidal waveform.

    Args:
        amplitude: Peak amplitude
        frequency: Frequency [Hz]
        phase: Phase offset [rad]

    Returns:
        Function t -> A * sin(2*pi*f*t + phase)
    """

    def sinusoid(t: float) -> float:
        return amplitude * jnp.sin(2 * jnp.pi * frequency * t + phase)

    return sinusoid


def make_crowbar(V_initial: float, t_crowbar: float) -> Callable[[float], float]:
    """Create crowbar (step-down) waveform.

    Args:
        V_initial: Voltage before crowbar
        t_crowbar: Time at which voltage drops to zero [s]

    Returns:
        Function t -> V that is V_initial before t_crowbar, 0 after
    """

    def crowbar(t: float) -> float:
        return jnp.where(t < t_crowbar, V_initial, 0.0)

    return crowbar


def make_pulse(
    amplitude: float, t_start: float, t_end: float
) -> Callable[[float], float]:
    """Create square pulse waveform.

    Args:
        amplitude: Pulse amplitude
        t_start: Pulse start time [s]
        t_end: Pulse end time [s]

    Returns:
        Function t -> V that is amplitude during [t_start, t_end], 0 otherwise
    """

    def pulse(t: float) -> float:
        in_pulse = (t >= t_start) & (t < t_end)
        return jnp.where(in_pulse, amplitude, 0.0)

    return pulse


def make_constant(value: float) -> Callable[[float], float]:
    """Create constant waveform.

    Args:
        value: Constant value

    Returns:
        Function t -> value
    """

    def constant(t: float) -> float:
        return value

    return constant


def waveform_from_config(config: dict) -> Callable[[float], float]:
    """Create waveform from configuration dictionary.

    Args:
        config: Dictionary with 'type' key and type-specific parameters:
            - type: "ramp" -> V0, V1, t_ramp
            - type: "sinusoid" -> amplitude, frequency, phase (optional)
            - type: "crowbar" -> V_initial, t_crowbar
            - type: "pulse" -> amplitude, t_start, t_end
            - type: "constant" -> value

    Returns:
        Waveform function t -> V
    """
    waveform_type = config["type"]

    if waveform_type == "ramp":
        return make_ramp(
            V0=config["V0"],
            V1=config["V1"],
            t_ramp=config["t_ramp"],
        )
    elif waveform_type == "sinusoid":
        return make_sinusoid(
            amplitude=config["amplitude"],
            frequency=config["frequency"],
            phase=config.get("phase", 0.0),
        )
    elif waveform_type == "crowbar":
        return make_crowbar(
            V_initial=config["V_initial"],
            t_crowbar=config["t_crowbar"],
        )
    elif waveform_type == "pulse":
        return make_pulse(
            amplitude=config["amplitude"],
            t_start=config["t_start"],
            t_end=config["t_end"],
        )
    elif waveform_type == "constant":
        return make_constant(value=config["value"])
    else:
        raise ValueError(f"Unknown waveform type: {waveform_type}")
```

**Step 4: Update __init__.py**

Update `jax_frc/circuits/__init__.py`:

```python
"""Circuit coupling for burning plasma simulations."""

from jax_frc.circuits.state import CircuitState, CircuitParams
from jax_frc.circuits.waveforms import (
    make_ramp,
    make_sinusoid,
    make_crowbar,
    make_pulse,
    make_constant,
    waveform_from_config,
)

__all__ = [
    "CircuitState",
    "CircuitParams",
    "make_ramp",
    "make_sinusoid",
    "make_crowbar",
    "make_pulse",
    "make_constant",
    "waveform_from_config",
]
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_waveforms.py -v`
Expected: PASS (11 tests)

**Step 6: Commit**

```bash
git add jax_frc/circuits/waveforms.py jax_frc/circuits/__init__.py tests/test_waveforms.py
git commit -m "feat(circuits): add waveform generators for voltage sources"
```

---

## Task 3: Create PickupCoilArray with Flux Computation

**Files:**
- Create: `jax_frc/circuits/pickup.py`
- Create: `tests/test_pickup_coils.py`
- Modify: `jax_frc/circuits/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_pickup_coils.py`:

```python
"""Tests for pickup coil array."""

import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.5,
        z_min=-1.0,
        z_max=1.0,
    )


class TestPickupCoilArray:
    """Tests for PickupCoilArray."""

    def test_creation(self):
        """Can create PickupCoilArray."""
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.pickup import PickupCoilArray

        params = CircuitParams(
            L=jnp.array([1e-3, 1e-3, 1e-3]),
            R=jnp.array([0.1, 0.1, 0.1]),
            C=jnp.array([jnp.inf, jnp.inf, jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([-0.5, 0.0, 0.5]),
            radii=jnp.array([0.4, 0.4, 0.4]),
            n_turns=jnp.array([100, 100, 100]),
            params=params,
            load_resistance=jnp.array([1.0, 1.0, 1.0]),
        )
        assert pickup.n_coils == 3

    def test_flux_uniform_field(self, geometry):
        """Flux through coil in uniform Bz field."""
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.pickup import PickupCoilArray

        params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([0.1]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([1.0]),
        )

        # Uniform Bz = 1 T field
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)  # Bz = 1 T

        Psi = pickup.compute_flux_linkages(B, geometry)

        # Expected: Psi = N * Bz * pi * r^2 = 100 * 1.0 * pi * 0.4^2
        # But we integrate over cells with r < 0.4, so it's approximate
        expected = 100 * 1.0 * jnp.pi * 0.4**2
        assert Psi.shape == (1,)
        # Allow some error due to discrete grid
        assert jnp.isclose(Psi[0], expected, rtol=0.1)

    def test_flux_multiple_coils(self, geometry):
        """Different coils see different flux based on position."""
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.pickup import PickupCoilArray

        params = CircuitParams(
            L=jnp.array([1e-3, 1e-3]),
            R=jnp.array([0.1, 0.1]),
            C=jnp.array([jnp.inf, jnp.inf]),
        )
        # Two coils at different z positions
        pickup = PickupCoilArray(
            z_positions=jnp.array([-0.5, 0.5]),
            radii=jnp.array([0.4, 0.4]),
            n_turns=jnp.array([100, 100]),
            params=params,
            load_resistance=jnp.array([1.0, 1.0]),
        )

        # Bz varies with z: stronger at z > 0
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        z_grid = geometry.z_grid
        Bz = jnp.where(z_grid > 0, 2.0, 1.0)
        B = B.at[:, :, 2].set(Bz)

        Psi = pickup.compute_flux_linkages(B, geometry)

        # Coil at z=0.5 should see ~2x the flux of coil at z=-0.5
        assert Psi[1] > Psi[0]
        assert jnp.isclose(Psi[1] / Psi[0], 2.0, rtol=0.2)

    def test_power_calculation(self):
        """Power extraction from currents."""
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.pickup import PickupCoilArray

        params = CircuitParams(
            L=jnp.array([1e-3, 1e-3]),
            R=jnp.array([0.1, 0.1]),
            C=jnp.array([jnp.inf, jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0, 0.5]),
            radii=jnp.array([0.4, 0.4]),
            n_turns=jnp.array([100, 100]),
            params=params,
            load_resistance=jnp.array([1.0, 1.0]),
        )

        I = jnp.array([10.0, 20.0])  # Currents in each coil

        P_load, P_dissipated = pickup.compute_power(I)

        # P_load = I^2 * R_load = 100*1 + 400*1 = 500 W
        assert jnp.isclose(jnp.sum(P_load), 500.0)
        # P_dissipated = I^2 * R = 100*0.1 + 400*0.1 = 50 W
        assert jnp.isclose(jnp.sum(P_dissipated), 50.0)
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_pickup_coils.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write PickupCoilArray implementation**

Create `jax_frc/circuits/pickup.py`:

```python
"""Pickup coil array for energy extraction."""

from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array

from jax_frc.circuits.state import CircuitParams
from jax_frc.core.geometry import Geometry


@dataclass(frozen=True)
class PickupCoilArray:
    """Array of pickup coils at different axial positions.

    Each coil has its own RLC circuit for energy extraction.

    Attributes:
        z_positions: Axial positions of coil centers [m], shape (n_coils,)
        radii: Coil radii [m], shape (n_coils,)
        n_turns: Turns per coil, shape (n_coils,)
        params: Circuit parameters (L, R, C) for each coil
        load_resistance: External load resistance [Ω], shape (n_coils,)
    """

    z_positions: Array
    radii: Array
    n_turns: Array
    params: CircuitParams
    load_resistance: Array

    @property
    def n_coils(self) -> int:
        """Number of pickup coils."""
        return self.z_positions.shape[0]

    def compute_flux_linkages(self, B: Array, geometry: Geometry) -> Array:
        """Compute flux linkage Ψ = N * ∫B·dA for each coil.

        Integrates Bz over the area within coil radius at each coil's
        z-position. Uses linear interpolation for z-positions between
        grid points.

        Args:
            B: Magnetic field (nr, nz, 3) with components (Br, Bphi, Bz)
            geometry: Computational geometry

        Returns:
            Psi: Flux linkage for each coil [Wb], shape (n_coils,)
        """
        Bz = B[:, :, 2]  # Axial component
        r = geometry.r  # 1D radial coordinates
        z = geometry.z  # 1D axial coordinates
        dr = geometry.dr

        def flux_for_coil(z_coil, radius, n_turn):
            # Find z index (interpolate between grid points)
            z_idx_float = (z_coil - geometry.z_min) / geometry.dz
            z_idx = jnp.clip(z_idx_float.astype(int), 0, geometry.nz - 2)
            z_frac = z_idx_float - z_idx

            # Interpolate Bz at coil z-position
            Bz_at_z = (1 - z_frac) * Bz[:, z_idx] + z_frac * Bz[:, z_idx + 1]

            # Mask for r < coil radius
            mask = r < radius

            # Integrate: Psi = integral(Bz * 2*pi*r * dr) for r < radius
            flux = jnp.sum(Bz_at_z * 2 * jnp.pi * r * dr * mask)

            return n_turn * flux

        # Vectorize over all coils
        Psi = jnp.array(
            [
                flux_for_coil(z_coil, radius, n_turn)
                for z_coil, radius, n_turn in zip(
                    self.z_positions, self.radii, self.n_turns
                )
            ]
        )

        return Psi

    def compute_power(self, I: Array) -> tuple[Array, Array]:
        """Compute power extracted and dissipated.

        Args:
            I: Current in each coil [A], shape (n_coils,)

        Returns:
            P_load: Power to external loads [W], shape (n_coils,)
            P_dissipated: Power dissipated in coil resistance [W], shape (n_coils,)
        """
        P_load = I**2 * self.load_resistance
        P_dissipated = I**2 * self.params.R
        return P_load, P_dissipated
```

**Step 4: Update __init__.py**

Add to `jax_frc/circuits/__init__.py`:

```python
"""Circuit coupling for burning plasma simulations."""

from jax_frc.circuits.state import CircuitState, CircuitParams
from jax_frc.circuits.waveforms import (
    make_ramp,
    make_sinusoid,
    make_crowbar,
    make_pulse,
    make_constant,
    waveform_from_config,
)
from jax_frc.circuits.pickup import PickupCoilArray

__all__ = [
    "CircuitState",
    "CircuitParams",
    "make_ramp",
    "make_sinusoid",
    "make_crowbar",
    "make_pulse",
    "make_constant",
    "waveform_from_config",
    "PickupCoilArray",
]
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_pickup_coils.py -v`
Expected: PASS (4 tests)

**Step 6: Commit**

```bash
git add jax_frc/circuits/pickup.py jax_frc/circuits/__init__.py tests/test_pickup_coils.py
git commit -m "feat(circuits): add PickupCoilArray with flux computation"
```

---

## Task 4: Create ExternalCircuits with Drivers

**Files:**
- Create: `jax_frc/circuits/external.py`
- Create: `tests/test_external_circuits.py`
- Modify: `jax_frc/circuits/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_external_circuits.py`:

```python
"""Tests for external circuit components."""

import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.8,
        z_min=-1.0,
        z_max=1.0,
    )


class TestCoilGeometry:
    """Tests for CoilGeometry dataclass."""

    def test_creation(self):
        """Can create CoilGeometry."""
        from jax_frc.circuits.external import CoilGeometry

        coil = CoilGeometry(
            z_center=0.0,
            radius=0.6,
            length=0.5,
            n_turns=50,
        )
        assert coil.radius == 0.6


class TestCircuitDriver:
    """Tests for CircuitDriver."""

    def test_voltage_mode(self):
        """Voltage mode driver returns waveform value."""
        from jax_frc.circuits.external import CircuitDriver
        from jax_frc.circuits.waveforms import make_ramp

        driver = CircuitDriver(
            mode="voltage",
            waveform=make_ramp(0.0, 1000.0, 1e-4),
        )
        assert driver.get_voltage(t=1e-4, state=None, error_integral=0.0) == 1000.0

    def test_current_mode(self):
        """Current mode driver returns waveform value."""
        from jax_frc.circuits.external import CircuitDriver
        from jax_frc.circuits.waveforms import make_constant

        driver = CircuitDriver(
            mode="current",
            waveform=make_constant(100.0),
        )
        assert driver.get_target_current(t=0.0, state=None) == 100.0

    def test_feedback_mode(self):
        """Feedback mode computes PID control voltage."""
        from jax_frc.circuits.external import CircuitDriver

        # Simple proportional control
        driver = CircuitDriver(
            mode="feedback",
            feedback_gains=(100.0, 0.0, 0.0),  # Kp only
            feedback_target=lambda state: 1.0,  # Target value
            feedback_measure=lambda state: 0.5,  # Measured value
        )
        # Error = target - measured = 1.0 - 0.5 = 0.5
        # V = Kp * error = 100 * 0.5 = 50
        V = driver.get_voltage(t=0.0, state=None, error_integral=0.0)
        assert jnp.isclose(V, 50.0)


class TestExternalCircuit:
    """Tests for ExternalCircuit."""

    def test_creation(self):
        """Can create ExternalCircuit."""
        from jax_frc.circuits.external import ExternalCircuit, CoilGeometry, CircuitDriver
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.waveforms import make_constant

        circuit = ExternalCircuit(
            name="test_coil",
            coil=CoilGeometry(z_center=0.0, radius=0.6, length=0.5, n_turns=50),
            params=CircuitParams(
                L=jnp.array([5e-3]),
                R=jnp.array([0.05]),
                C=jnp.array([jnp.inf]),
            ),
            driver=CircuitDriver(mode="voltage", waveform=make_constant(0.0)),
        )
        assert circuit.name == "test_coil"


class TestExternalCircuits:
    """Tests for ExternalCircuits collection."""

    def test_creation(self):
        """Can create ExternalCircuits."""
        from jax_frc.circuits.external import (
            ExternalCircuits,
            ExternalCircuit,
            CoilGeometry,
            CircuitDriver,
        )
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.waveforms import make_constant

        circuit1 = ExternalCircuit(
            name="coil1",
            coil=CoilGeometry(z_center=-0.5, radius=0.6, length=0.5, n_turns=50),
            params=CircuitParams(
                L=jnp.array([5e-3]),
                R=jnp.array([0.05]),
                C=jnp.array([jnp.inf]),
            ),
            driver=CircuitDriver(mode="voltage", waveform=make_constant(0.0)),
        )
        circuit2 = ExternalCircuit(
            name="coil2",
            coil=CoilGeometry(z_center=0.5, radius=0.6, length=0.5, n_turns=50),
            params=CircuitParams(
                L=jnp.array([5e-3]),
                R=jnp.array([0.05]),
                C=jnp.array([jnp.inf]),
            ),
            driver=CircuitDriver(mode="voltage", waveform=make_constant(0.0)),
        )

        external = ExternalCircuits(circuits=[circuit1, circuit2])
        assert external.n_circuits == 2

    def test_compute_b_field(self, geometry):
        """External coils contribute B-field."""
        from jax_frc.circuits.external import (
            ExternalCircuits,
            ExternalCircuit,
            CoilGeometry,
            CircuitDriver,
        )
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.waveforms import make_constant

        circuit = ExternalCircuit(
            name="solenoid",
            coil=CoilGeometry(z_center=0.0, radius=0.6, length=1.0, n_turns=100),
            params=CircuitParams(
                L=jnp.array([5e-3]),
                R=jnp.array([0.05]),
                C=jnp.array([jnp.inf]),
            ),
            driver=CircuitDriver(mode="voltage", waveform=make_constant(0.0)),
        )
        external = ExternalCircuits(circuits=[circuit])

        I = jnp.array([1000.0])  # 1000 A
        B = external.compute_b_field(I, geometry)

        assert B.shape == (geometry.nr, geometry.nz, 3)
        # Should have significant Bz on axis
        Bz_center = B[0, geometry.nz // 2, 2]
        assert Bz_center > 0.1  # Should be order 0.1 T for 1000 A
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_external_circuits.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write external circuits implementation**

Create `jax_frc/circuits/external.py`:

```python
"""External circuits with coils and drivers."""

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp
from jax import Array

from jax_frc.circuits.state import CircuitParams
from jax_frc.core.geometry import Geometry
from jax_frc.constants import MU0


@dataclass(frozen=True)
class CoilGeometry:
    """Physical geometry of an external coil.

    Attributes:
        z_center: Axial position of coil center [m]
        radius: Coil radius [m]
        length: Coil length [m] (for solenoid model)
        n_turns: Number of turns
    """

    z_center: float
    radius: float
    length: float
    n_turns: int


@dataclass(frozen=True)
class CircuitDriver:
    """Voltage or current source for external circuit.

    Supports three modes:
    - "voltage": Apply voltage waveform directly
    - "current": Target current waveform (requires high-bandwidth control)
    - "feedback": PID control based on plasma state

    Attributes:
        mode: "voltage", "current", or "feedback"
        waveform: For voltage/current mode, callable(t) -> value
        feedback_gains: For feedback mode, (Kp, Ki, Kd) gains
        feedback_target: For feedback mode, callable(state) -> target value
        feedback_measure: For feedback mode, callable(state) -> measured value
    """

    mode: str
    waveform: Optional[Callable[[float], float]] = None
    feedback_gains: Optional[tuple[float, float, float]] = None
    feedback_target: Optional[Callable] = None
    feedback_measure: Optional[Callable] = None

    def get_voltage(self, t: float, state, error_integral: float) -> float:
        """Get applied voltage at time t.

        Args:
            t: Current time [s]
            state: Current plasma/circuit state (for feedback)
            error_integral: Accumulated error for integral term

        Returns:
            Voltage to apply [V]
        """
        if self.mode == "voltage":
            return self.waveform(t)
        elif self.mode == "feedback":
            Kp, Ki, Kd = self.feedback_gains
            target = self.feedback_target(state)
            measured = self.feedback_measure(state)
            error = target - measured
            # Simple PI control (no derivative for now)
            return Kp * error + Ki * error_integral
        else:
            # Current mode doesn't directly provide voltage
            return 0.0

    def get_target_current(self, t: float, state) -> Optional[float]:
        """Get target current for current-controlled mode.

        Args:
            t: Current time [s]
            state: Current state (unused for waveform mode)

        Returns:
            Target current [A] or None if not in current mode
        """
        if self.mode == "current":
            return self.waveform(t)
        return None


@dataclass(frozen=True)
class ExternalCircuit:
    """Single external circuit with coil and driver.

    Attributes:
        name: Circuit identifier
        coil: Physical coil geometry
        params: Circuit parameters (L, R, C)
        driver: Voltage/current source
    """

    name: str
    coil: CoilGeometry
    params: CircuitParams
    driver: CircuitDriver


@dataclass
class ExternalCircuits:
    """Collection of external circuits.

    Manages multiple external coils and their circuits.
    """

    circuits: list[ExternalCircuit]

    @property
    def n_circuits(self) -> int:
        """Number of external circuits."""
        return len(self.circuits)

    def get_combined_params(self) -> CircuitParams:
        """Get combined circuit parameters as arrays."""
        L = jnp.array([c.params.L[0] for c in self.circuits])
        R = jnp.array([c.params.R[0] for c in self.circuits])
        C = jnp.array([c.params.C[0] for c in self.circuits])
        return CircuitParams(L=L, R=R, C=C)

    def compute_b_field(self, I: Array, geometry: Geometry) -> Array:
        """Compute magnetic field from all external coil currents.

        Uses a finite solenoid model for each coil.

        Args:
            I: Current in each coil [A], shape (n_circuits,)
            geometry: Computational geometry

        Returns:
            B: Magnetic field contribution (nr, nz, 3)
        """
        B_total = jnp.zeros((geometry.nr, geometry.nz, 3))

        for i, circuit in enumerate(self.circuits):
            coil = circuit.coil
            current = I[i]

            B_coil = self._solenoid_field(
                current=current,
                z_center=coil.z_center,
                radius=coil.radius,
                length=coil.length,
                n_turns=coil.n_turns,
                geometry=geometry,
            )
            B_total = B_total + B_coil

        return B_total

    def _solenoid_field(
        self,
        current: float,
        z_center: float,
        radius: float,
        length: float,
        n_turns: int,
        geometry: Geometry,
    ) -> Array:
        """Compute B-field of a finite solenoid.

        Uses analytical approximation valid for points inside the solenoid.

        Args:
            current: Coil current [A]
            z_center: Solenoid center position [m]
            radius: Solenoid radius [m]
            length: Solenoid length [m]
            n_turns: Number of turns
            geometry: Computational geometry

        Returns:
            B: Magnetic field (nr, nz, 3)
        """
        n = n_turns / length  # Turns per meter
        B0 = MU0 * n * current  # Infinite solenoid field

        r_grid = geometry.r_grid
        z_grid = geometry.z_grid

        # Relative positions from solenoid ends
        z_rel = z_grid - z_center
        z_plus = z_rel + length / 2
        z_minus = z_rel - length / 2

        # End correction factors
        denom_plus = jnp.sqrt(radius**2 + z_plus**2)
        denom_minus = jnp.sqrt(radius**2 + z_minus**2)

        cos_plus = z_plus / jnp.maximum(denom_plus, 1e-10)
        cos_minus = z_minus / jnp.maximum(denom_minus, 1e-10)

        # Axial field with end corrections
        Bz = 0.5 * B0 * (cos_plus - cos_minus)

        # Radial field from div(B) = 0
        dBz_dz_plus = -0.5 * B0 * radius**2 / jnp.maximum(denom_plus**3, 1e-30)
        dBz_dz_minus = -0.5 * B0 * radius**2 / jnp.maximum(denom_minus**3, 1e-30)
        dBz_dz = dBz_dz_plus - dBz_dz_minus
        Br = -0.5 * r_grid * dBz_dz

        # Assemble B vector
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 0].set(Br)
        B = B.at[:, :, 2].set(Bz)

        return B
```

**Step 4: Update __init__.py**

Update `jax_frc/circuits/__init__.py`:

```python
"""Circuit coupling for burning plasma simulations."""

from jax_frc.circuits.state import CircuitState, CircuitParams
from jax_frc.circuits.waveforms import (
    make_ramp,
    make_sinusoid,
    make_crowbar,
    make_pulse,
    make_constant,
    waveform_from_config,
)
from jax_frc.circuits.pickup import PickupCoilArray
from jax_frc.circuits.external import (
    CoilGeometry,
    CircuitDriver,
    ExternalCircuit,
    ExternalCircuits,
)

__all__ = [
    "CircuitState",
    "CircuitParams",
    "make_ramp",
    "make_sinusoid",
    "make_crowbar",
    "make_pulse",
    "make_constant",
    "waveform_from_config",
    "PickupCoilArray",
    "CoilGeometry",
    "CircuitDriver",
    "ExternalCircuit",
    "ExternalCircuits",
]
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_external_circuits.py -v`
Expected: PASS (6 tests)

**Step 6: Commit**

```bash
git add jax_frc/circuits/external.py jax_frc/circuits/__init__.py tests/test_external_circuits.py
git commit -m "feat(circuits): add ExternalCircuits with coil geometry and drivers"
```

---

## Task 5: Create FluxCoupling Module

**Files:**
- Create: `jax_frc/circuits/coupling.py`
- Create: `tests/test_flux_coupling.py`
- Modify: `jax_frc/circuits/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_flux_coupling.py`:

```python
"""Tests for flux coupling between plasma and circuits."""

import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.8,
        z_min=-1.0,
        z_max=1.0,
    )


class TestFluxCoupling:
    """Tests for FluxCoupling class."""

    def test_plasma_to_pickup(self, geometry):
        """Compute plasma flux through pickup coils."""
        from jax_frc.circuits import CircuitParams, PickupCoilArray
        from jax_frc.circuits.coupling import FluxCoupling

        params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([0.1]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([1.0]),
        )

        coupling = FluxCoupling()

        # Uniform Bz = 1 T
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)

        Psi_pickup, Psi_external = coupling.plasma_to_coils(
            B_plasma=B,
            geometry=geometry,
            pickup=pickup,
            external=None,
        )

        assert Psi_pickup.shape == (1,)
        assert Psi_external.shape == (0,)  # No external circuits
        assert Psi_pickup[0] > 0

    def test_plasma_to_external(self, geometry):
        """Compute plasma flux through external coils."""
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.external import (
            ExternalCircuits,
            ExternalCircuit,
            CoilGeometry,
            CircuitDriver,
        )
        from jax_frc.circuits.coupling import FluxCoupling
        from jax_frc.circuits.waveforms import make_constant

        circuit = ExternalCircuit(
            name="external",
            coil=CoilGeometry(z_center=0.0, radius=0.5, length=0.5, n_turns=50),
            params=CircuitParams(
                L=jnp.array([5e-3]),
                R=jnp.array([0.05]),
                C=jnp.array([jnp.inf]),
            ),
            driver=CircuitDriver(mode="voltage", waveform=make_constant(0.0)),
        )
        external = ExternalCircuits(circuits=[circuit])

        coupling = FluxCoupling()

        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)

        Psi_pickup, Psi_external = coupling.plasma_to_coils(
            B_plasma=B,
            geometry=geometry,
            pickup=None,
            external=external,
        )

        assert Psi_pickup.shape == (0,)
        assert Psi_external.shape == (1,)
        assert Psi_external[0] > 0

    def test_external_to_plasma(self, geometry):
        """External coil currents produce B-field."""
        from jax_frc.circuits import CircuitParams
        from jax_frc.circuits.external import (
            ExternalCircuits,
            ExternalCircuit,
            CoilGeometry,
            CircuitDriver,
        )
        from jax_frc.circuits.coupling import FluxCoupling
        from jax_frc.circuits.waveforms import make_constant

        circuit = ExternalCircuit(
            name="solenoid",
            coil=CoilGeometry(z_center=0.0, radius=0.6, length=1.0, n_turns=100),
            params=CircuitParams(
                L=jnp.array([5e-3]),
                R=jnp.array([0.05]),
                C=jnp.array([jnp.inf]),
            ),
            driver=CircuitDriver(mode="voltage", waveform=make_constant(0.0)),
        )
        external = ExternalCircuits(circuits=[circuit])

        coupling = FluxCoupling()

        I_external = jnp.array([1000.0])  # 1000 A
        B_coils = coupling.coils_to_plasma(I_external, external, geometry)

        assert B_coils.shape == (geometry.nr, geometry.nz, 3)
        # Should have Bz on axis
        assert B_coils[0, geometry.nz // 2, 2] > 0.1
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_flux_coupling.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write FluxCoupling implementation**

Create `jax_frc/circuits/coupling.py`:

```python
"""Flux coupling between plasma and circuits."""

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
from jax import Array

from jax_frc.core.geometry import Geometry
from jax_frc.circuits.pickup import PickupCoilArray
from jax_frc.circuits.external import ExternalCircuits


@dataclass(frozen=True)
class FluxCoupling:
    """Computes flux linkages between plasma and all circuits.

    Handles bidirectional coupling:
    - Plasma B-field induces flux in pickup and external coils
    - External coil currents contribute to total B-field
    """

    def plasma_to_coils(
        self,
        B_plasma: Array,
        geometry: Geometry,
        pickup: Optional[PickupCoilArray],
        external: Optional[ExternalCircuits],
    ) -> tuple[Array, Array]:
        """Compute plasma flux threading each circuit.

        Args:
            B_plasma: Plasma magnetic field (nr, nz, 3)
            geometry: Computational geometry
            pickup: Pickup coil array (optional)
            external: External circuits (optional)

        Returns:
            Psi_pickup: Flux through pickup coils [Wb], shape (n_pickup,)
            Psi_external: Flux through external coils [Wb], shape (n_external,)
        """
        # Pickup coils
        if pickup is not None:
            Psi_pickup = pickup.compute_flux_linkages(B_plasma, geometry)
        else:
            Psi_pickup = jnp.array([])

        # External coils
        if external is not None and external.n_circuits > 0:
            Psi_external = self._compute_external_flux(B_plasma, geometry, external)
        else:
            Psi_external = jnp.array([])

        return Psi_pickup, Psi_external

    def _compute_external_flux(
        self,
        B: Array,
        geometry: Geometry,
        external: ExternalCircuits,
    ) -> Array:
        """Compute flux through external coils.

        Similar to pickup coils but using external coil geometry.
        """
        Bz = B[:, :, 2]
        r = geometry.r
        dr = geometry.dr

        def flux_for_coil(circuit):
            coil = circuit.coil
            n_turns = coil.n_turns
            radius = coil.radius
            z_center = coil.z_center

            # Find z index
            z_idx_float = (z_center - geometry.z_min) / geometry.dz
            z_idx = jnp.clip(int(z_idx_float), 0, geometry.nz - 2)
            z_frac = z_idx_float - z_idx

            # Interpolate Bz
            Bz_at_z = (1 - z_frac) * Bz[:, z_idx] + z_frac * Bz[:, z_idx + 1]

            # Integrate
            mask = r < radius
            flux = jnp.sum(Bz_at_z * 2 * jnp.pi * r * dr * mask)

            return n_turns * flux

        Psi = jnp.array([flux_for_coil(c) for c in external.circuits])
        return Psi

    def coils_to_plasma(
        self,
        I_external: Array,
        external: ExternalCircuits,
        geometry: Geometry,
    ) -> Array:
        """Compute B-field contribution from external coil currents.

        Args:
            I_external: Current in each external coil [A], shape (n_external,)
            external: External circuits
            geometry: Computational geometry

        Returns:
            B_coils: Magnetic field from coils (nr, nz, 3)
        """
        if external is None or external.n_circuits == 0:
            return jnp.zeros((geometry.nr, geometry.nz, 3))

        return external.compute_b_field(I_external, geometry)
```

**Step 4: Update __init__.py**

Update `jax_frc/circuits/__init__.py`:

```python
"""Circuit coupling for burning plasma simulations."""

from jax_frc.circuits.state import CircuitState, CircuitParams
from jax_frc.circuits.waveforms import (
    make_ramp,
    make_sinusoid,
    make_crowbar,
    make_pulse,
    make_constant,
    waveform_from_config,
)
from jax_frc.circuits.pickup import PickupCoilArray
from jax_frc.circuits.external import (
    CoilGeometry,
    CircuitDriver,
    ExternalCircuit,
    ExternalCircuits,
)
from jax_frc.circuits.coupling import FluxCoupling

__all__ = [
    "CircuitState",
    "CircuitParams",
    "make_ramp",
    "make_sinusoid",
    "make_crowbar",
    "make_pulse",
    "make_constant",
    "waveform_from_config",
    "PickupCoilArray",
    "CoilGeometry",
    "CircuitDriver",
    "ExternalCircuit",
    "ExternalCircuits",
    "FluxCoupling",
]
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_flux_coupling.py -v`
Expected: PASS (3 tests)

**Step 6: Commit**

```bash
git add jax_frc/circuits/coupling.py jax_frc/circuits/__init__.py tests/test_flux_coupling.py
git commit -m "feat(circuits): add FluxCoupling for plasma-circuit interaction"
```

---

## Task 6: Create CircuitSystem with ODE Integration

**Files:**
- Create: `jax_frc/circuits/system.py`
- Create: `tests/test_circuit_system.py`
- Modify: `jax_frc/circuits/__init__.py`

**Step 1: Write the failing test**

Create `tests/test_circuit_system.py`:

```python
"""Tests for CircuitSystem ODE integration."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.8,
        z_min=-1.0,
        z_max=1.0,
    )


class TestCircuitODE:
    """Tests for circuit ODE integration."""

    def test_rl_decay(self):
        """RL circuit current decays as I = I0 * exp(-R*t/L)."""
        from jax_frc.circuits import CircuitState, CircuitParams, PickupCoilArray
        from jax_frc.circuits.system import CircuitSystem

        # Single pickup coil, R-only (no capacitor)
        params = CircuitParams(
            L=jnp.array([1e-3]),  # 1 mH
            R=jnp.array([1.0]),  # 1 Ohm -> tau = 1 ms
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.0]),  # No load, just R
        )

        system = CircuitSystem(pickup=pickup, external=None)

        # Initial state with I = 1 A
        state = CircuitState(
            I_pickup=jnp.array([1.0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.array([]),
            Q_external=jnp.array([]),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.array([]),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        # Zero B-field (no induced EMF)
        B = jnp.zeros((16, 32, 3))
        geometry = Geometry(
            coord_system="cylindrical",
            nr=16, nz=32,
            r_min=0.1, r_max=0.8,
            z_min=-1.0, z_max=1.0,
        )

        # Advance by one tau = 1 ms
        dt = 1e-3
        new_state = system.step(state, B, geometry, t=0.0, dt=dt)

        # I should decay to I0 * exp(-1) ≈ 0.368
        expected = 1.0 * jnp.exp(-1.0)
        assert jnp.isclose(new_state.I_pickup[0], expected, rtol=0.05)

    def test_lc_oscillation(self):
        """LC circuit oscillates at f = 1/(2*pi*sqrt(LC))."""
        from jax_frc.circuits import CircuitState, CircuitParams, PickupCoilArray
        from jax_frc.circuits.system import CircuitSystem

        L = 1e-3  # 1 mH
        C = 1e-6  # 1 uF
        # f = 1/(2*pi*sqrt(LC)) ≈ 5033 Hz, period ≈ 0.2 ms

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([0.0]),  # No resistance
            C=jnp.array([C]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.0]),
        )

        system = CircuitSystem(pickup=pickup, external=None)

        # Initial: I = 0, Q = Q0 (capacitor charged)
        Q0 = 1e-6  # 1 uC
        state = CircuitState(
            I_pickup=jnp.array([0.0]),
            Q_pickup=jnp.array([Q0]),
            I_external=jnp.array([]),
            Q_external=jnp.array([]),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.array([]),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        B = jnp.zeros((16, 32, 3))
        geometry = Geometry(
            coord_system="cylindrical",
            nr=16, nz=32,
            r_min=0.1, r_max=0.8,
            z_min=-1.0, z_max=1.0,
        )

        # Quarter period: Q should be ~0, I should be max
        period = 2 * jnp.pi * jnp.sqrt(L * C)
        dt = period / 4

        new_state = system.step(state, B, geometry, t=0.0, dt=dt)

        # Q ≈ 0 at quarter period
        assert jnp.abs(new_state.Q_pickup[0]) < 0.1 * Q0

    def test_induced_emf(self, geometry):
        """Changing flux induces current."""
        from jax_frc.circuits import CircuitState, CircuitParams, PickupCoilArray
        from jax_frc.circuits.system import CircuitSystem

        params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([1.0]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([1.0]),
        )

        system = CircuitSystem(pickup=pickup, external=None)

        # Initial state with known flux
        Psi_initial = 0.01  # 10 mWb
        state = CircuitState(
            I_pickup=jnp.array([0.0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.array([]),
            Q_external=jnp.array([]),
            Psi_pickup=jnp.array([Psi_initial]),
            Psi_external=jnp.array([]),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        # B-field that gives higher flux
        B = jnp.zeros((geometry.nr, geometry.nz, 3))
        B = B.at[:, :, 2].set(1.0)  # Will give Psi > Psi_initial

        dt = 1e-6
        new_state = system.step(state, B, geometry, t=0.0, dt=dt)

        # Current should be induced (sign depends on dPsi/dt)
        assert new_state.I_pickup[0] != 0.0

    def test_jit_compatible(self, geometry):
        """CircuitSystem.step is JIT-compatible."""
        from jax_frc.circuits import CircuitState, CircuitParams, PickupCoilArray
        from jax_frc.circuits.system import CircuitSystem

        params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([1.0]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([1.0]),
        )

        system = CircuitSystem(pickup=pickup, external=None)

        state = CircuitState.zeros(n_pickup=1, n_external=0)
        B = jnp.zeros((geometry.nr, geometry.nz, 3))

        @jax.jit
        def step_jit(s, b):
            return system.step(s, b, geometry, t=0.0, dt=1e-6)

        # Should not raise
        new_state = step_jit(state, B)
        assert new_state is not None
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_circuit_system.py -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write CircuitSystem implementation**

Create `jax_frc/circuits/system.py`:

```python
"""Circuit system with ODE integration."""

from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
from jax import lax, Array

from jax_frc.circuits.state import CircuitState, CircuitParams
from jax_frc.circuits.pickup import PickupCoilArray
from jax_frc.circuits.external import ExternalCircuits
from jax_frc.circuits.coupling import FluxCoupling
from jax_frc.core.geometry import Geometry


# Maximum subcycles for stiff circuits
MAX_SUBSTEPS = 100


@dataclass
class CircuitSystem:
    """Complete circuit system with ODE integration.

    Manages pickup coils and external circuits, handling:
    - Flux linkage computation
    - Circuit ODE integration with subcycling
    - Power calculation

    Attributes:
        pickup: Pickup coil array (optional)
        external: External circuits (optional)
        flux_coupling: Flux coupling calculator
    """

    pickup: Optional[PickupCoilArray] = None
    external: Optional[ExternalCircuits] = None
    flux_coupling: FluxCoupling = None

    def __post_init__(self):
        if self.flux_coupling is None:
            object.__setattr__(self, "flux_coupling", FluxCoupling())

    @property
    def n_pickup(self) -> int:
        """Number of pickup coils."""
        return self.pickup.n_coils if self.pickup else 0

    @property
    def n_external(self) -> int:
        """Number of external circuits."""
        return self.external.n_circuits if self.external else 0

    def step(
        self,
        state: CircuitState,
        B_plasma: Array,
        geometry: Geometry,
        t: float,
        dt: float,
    ) -> CircuitState:
        """Advance all circuits by dt.

        Args:
            state: Current circuit state
            B_plasma: Plasma magnetic field (nr, nz, 3)
            geometry: Computational geometry
            t: Current simulation time [s]
            dt: Timestep [s]

        Returns:
            Updated CircuitState
        """
        # 1. Compute new flux linkages
        Psi_pickup_new, Psi_external_new = self.flux_coupling.plasma_to_coils(
            B_plasma, geometry, self.pickup, self.external
        )

        # 2. Compute induced EMF from flux change
        if self.n_pickup > 0:
            dPsi_pickup = Psi_pickup_new - state.Psi_pickup
            V_induced_pickup = -self.pickup.n_turns * dPsi_pickup / dt
        else:
            V_induced_pickup = jnp.array([])

        if self.n_external > 0:
            dPsi_external = Psi_external_new - state.Psi_external
            n_turns_ext = jnp.array([c.coil.n_turns for c in self.external.circuits])
            V_induced_external = -n_turns_ext * dPsi_external / dt
        else:
            V_induced_external = jnp.array([])

        # 3. Integrate circuit ODEs with subcycling
        new_I_pickup, new_Q_pickup = self._integrate_pickup(
            state.I_pickup,
            state.Q_pickup,
            V_induced_pickup,
            dt,
        )

        new_I_external, new_Q_external = self._integrate_external(
            state.I_external,
            state.Q_external,
            V_induced_external,
            t,
            dt,
        )

        # 4. Compute power
        if self.n_pickup > 0:
            P_load, P_diss_pickup = self.pickup.compute_power(new_I_pickup)
            P_extracted = float(jnp.sum(P_load))
            P_dissipated = float(jnp.sum(P_diss_pickup))
        else:
            P_extracted = 0.0
            P_dissipated = 0.0

        # Add external circuit dissipation
        if self.n_external > 0:
            ext_params = self.external.get_combined_params()
            P_diss_ext = new_I_external**2 * ext_params.R
            P_dissipated += float(jnp.sum(P_diss_ext))

        return CircuitState(
            I_pickup=new_I_pickup,
            Q_pickup=new_Q_pickup,
            I_external=new_I_external,
            Q_external=new_Q_external,
            Psi_pickup=Psi_pickup_new,
            Psi_external=Psi_external_new,
            P_extracted=P_extracted,
            P_dissipated=P_dissipated,
        )

    def _integrate_pickup(
        self,
        I: Array,
        Q: Array,
        V_induced: Array,
        dt: float,
    ) -> tuple[Array, Array]:
        """Integrate pickup circuit ODEs.

        Uses implicit midpoint method for stability.
        Subcycles if dt > 0.1 * tau where tau = L/R.
        """
        if self.n_pickup == 0:
            return jnp.array([]), jnp.array([])

        params = self.pickup.params
        R_total = params.R + self.pickup.load_resistance

        # Determine subcycling
        tau = params.L / jnp.maximum(R_total, 1e-10)
        tau_min = jnp.min(tau)
        n_sub = jnp.clip(jnp.ceil(dt / (0.1 * tau_min)), 1, MAX_SUBSTEPS).astype(int)
        dt_sub = dt / n_sub

        def ode_step(carry, _):
            I_curr, Q_curr = carry

            # RLC ODE: L dI/dt + R*I + Q/C = V_induced
            # dI/dt = (V_induced - R*I - Q/C) / L
            # dQ/dt = I

            # Implicit midpoint: evaluate at midpoint
            # For stability with stiff systems

            # Explicit Euler for simplicity (subcycling handles stiffness)
            V_cap = jnp.where(jnp.isinf(params.C), 0.0, Q_curr / params.C)
            dI_dt = (V_induced - R_total * I_curr - V_cap) / params.L
            dQ_dt = I_curr

            I_new = I_curr + dI_dt * dt_sub
            Q_new = Q_curr + dQ_dt * dt_sub

            return (I_new, Q_new), None

        (I_final, Q_final), _ = lax.scan(ode_step, (I, Q), None, length=n_sub)

        return I_final, Q_final

    def _integrate_external(
        self,
        I: Array,
        Q: Array,
        V_induced: Array,
        t: float,
        dt: float,
    ) -> tuple[Array, Array]:
        """Integrate external circuit ODEs."""
        if self.n_external == 0:
            return jnp.array([]), jnp.array([])

        params = self.external.get_combined_params()

        # Get driver voltages
        V_driver = jnp.array(
            [
                c.driver.get_voltage(t, None, 0.0)
                for c in self.external.circuits
            ]
        )

        V_total = V_driver + V_induced

        # Subcycling
        tau = params.L / jnp.maximum(params.R, 1e-10)
        tau_min = jnp.min(tau)
        n_sub = jnp.clip(jnp.ceil(dt / (0.1 * tau_min)), 1, MAX_SUBSTEPS).astype(int)
        dt_sub = dt / n_sub

        def ode_step(carry, _):
            I_curr, Q_curr = carry

            V_cap = jnp.where(jnp.isinf(params.C), 0.0, Q_curr / params.C)
            dI_dt = (V_total - params.R * I_curr - V_cap) / params.L
            dQ_dt = I_curr

            I_new = I_curr + dI_dt * dt_sub
            Q_new = Q_curr + dQ_dt * dt_sub

            return (I_new, Q_new), None

        (I_final, Q_final), _ = lax.scan(ode_step, (I, Q), None, length=n_sub)

        return I_final, Q_final

    @classmethod
    def from_config(cls, config: dict) -> "CircuitSystem":
        """Create CircuitSystem from configuration dictionary.

        Args:
            config: Configuration with optional keys:
                - pickup_array: PickupCoilArray configuration
                - external: List of external circuit configurations

        Returns:
            Configured CircuitSystem
        """
        from jax_frc.circuits.waveforms import waveform_from_config
        from jax_frc.circuits.external import (
            ExternalCircuit,
            ExternalCircuits,
            CoilGeometry,
            CircuitDriver,
        )

        pickup = None
        external = None

        # Build pickup array
        if "pickup_array" in config:
            pc = config["pickup_array"]
            params = CircuitParams(
                L=jnp.array(pc["L"]),
                R=jnp.array(pc["R"]),
                C=jnp.array(pc.get("C", [jnp.inf] * len(pc["L"]))),
            )
            pickup = PickupCoilArray(
                z_positions=jnp.array(pc["z_positions"]),
                radii=jnp.array(pc["radii"]),
                n_turns=jnp.array(pc["n_turns"]),
                params=params,
                load_resistance=jnp.array(pc["load_resistance"]),
            )

        # Build external circuits
        if "external" in config and config["external"]:
            circuits = []
            for ec in config["external"]:
                coil = CoilGeometry(
                    z_center=ec["z_center"],
                    radius=ec["radius"],
                    length=ec["length"],
                    n_turns=ec["n_turns"],
                )
                params = CircuitParams(
                    L=jnp.array([ec["L"]]),
                    R=jnp.array([ec["R"]]),
                    C=jnp.array([ec.get("C", jnp.inf)]),
                )

                driver_cfg = ec.get("driver", {"mode": "voltage", "waveform": {"type": "constant", "value": 0.0}})
                if driver_cfg["mode"] == "voltage":
                    waveform = waveform_from_config(driver_cfg["waveform"])
                    driver = CircuitDriver(mode="voltage", waveform=waveform)
                elif driver_cfg["mode"] == "current":
                    waveform = waveform_from_config(driver_cfg["waveform"])
                    driver = CircuitDriver(mode="current", waveform=waveform)
                else:
                    # Feedback mode requires callables passed separately
                    driver = CircuitDriver(mode="voltage", waveform=lambda t: 0.0)

                circuits.append(
                    ExternalCircuit(
                        name=ec.get("name", f"coil_{len(circuits)}"),
                        coil=coil,
                        params=params,
                        driver=driver,
                    )
                )
            external = ExternalCircuits(circuits=circuits)

        return cls(pickup=pickup, external=external)
```

**Step 4: Update __init__.py**

Update `jax_frc/circuits/__init__.py`:

```python
"""Circuit coupling for burning plasma simulations."""

from jax_frc.circuits.state import CircuitState, CircuitParams
from jax_frc.circuits.waveforms import (
    make_ramp,
    make_sinusoid,
    make_crowbar,
    make_pulse,
    make_constant,
    waveform_from_config,
)
from jax_frc.circuits.pickup import PickupCoilArray
from jax_frc.circuits.external import (
    CoilGeometry,
    CircuitDriver,
    ExternalCircuit,
    ExternalCircuits,
)
from jax_frc.circuits.coupling import FluxCoupling
from jax_frc.circuits.system import CircuitSystem

__all__ = [
    "CircuitState",
    "CircuitParams",
    "make_ramp",
    "make_sinusoid",
    "make_crowbar",
    "make_pulse",
    "make_constant",
    "waveform_from_config",
    "PickupCoilArray",
    "CoilGeometry",
    "CircuitDriver",
    "ExternalCircuit",
    "ExternalCircuits",
    "FluxCoupling",
    "CircuitSystem",
]
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_circuit_system.py -v`
Expected: PASS (4 tests)

**Step 6: Commit**

```bash
git add jax_frc/circuits/system.py jax_frc/circuits/__init__.py tests/test_circuit_system.py
git commit -m "feat(circuits): add CircuitSystem with ODE integration and subcycling"
```

---

## Task 7: Integrate CircuitSystem into BurningPlasmaModel

**Files:**
- Modify: `jax_frc/models/burning_plasma.py`
- Modify: `jax_frc/burn/__init__.py`
- Create: `tests/test_burning_plasma_circuits.py`

**Step 1: Write the failing test**

Create `tests/test_burning_plasma_circuits.py`:

```python
"""Tests for burning plasma model with circuit coupling."""

import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=16,
        nz=32,
        r_min=0.1,
        r_max=0.5,
        z_min=-1.0,
        z_max=1.0,
    )


class TestBurningPlasmaWithCircuits:
    """Tests for BurningPlasmaModel with CircuitSystem."""

    def test_model_with_circuits(self, geometry):
        """Can create BurningPlasmaModel with CircuitSystem."""
        from jax_frc.models.burning_plasma import BurningPlasmaModel

        config = {
            "fuels": ["DT"],
            "circuits": {
                "pickup_array": {
                    "z_positions": [0.0],
                    "radii": [0.4],
                    "n_turns": [100],
                    "L": [1e-3],
                    "R": [0.1],
                    "load_resistance": [1.0],
                }
            },
        }

        model = BurningPlasmaModel.from_config(config)
        assert model.circuits is not None
        assert model.circuits.n_pickup == 1

    def test_step_with_circuits(self, geometry):
        """Model step updates circuit state."""
        from jax_frc.models.burning_plasma import BurningPlasmaModel, BurningPlasmaState
        from jax_frc.core.state import State
        from jax_frc.burn import SpeciesState, ReactionRates, PowerSources
        from jax_frc.circuits import CircuitState

        config = {
            "fuels": ["DT"],
            "circuits": {
                "pickup_array": {
                    "z_positions": [0.0],
                    "radii": [0.4],
                    "n_turns": [100],
                    "L": [1e-3],
                    "R": [0.1],
                    "load_resistance": [1.0],
                }
            },
        }

        model = BurningPlasmaModel.from_config(config)

        # Create initial state
        mhd = State.zeros(geometry.nr, geometry.nz)
        mhd = mhd.replace(
            T=jnp.ones((geometry.nr, geometry.nz)) * 10.0,
            B=jnp.ones((geometry.nr, geometry.nz, 3)),
        )
        species = SpeciesState(
            n_D=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_T=jnp.ones((geometry.nr, geometry.nz)) * 1e20,
            n_He3=jnp.zeros((geometry.nr, geometry.nz)),
            n_He4=jnp.zeros((geometry.nr, geometry.nz)),
            n_p=jnp.zeros((geometry.nr, geometry.nz)),
        )
        circuits = CircuitState.zeros(n_pickup=1, n_external=0)

        state = BurningPlasmaState(
            mhd=mhd,
            species=species,
            rates=ReactionRates(
                DT=jnp.zeros((geometry.nr, geometry.nz)),
                DD_T=jnp.zeros((geometry.nr, geometry.nz)),
                DD_HE3=jnp.zeros((geometry.nr, geometry.nz)),
                DHE3=jnp.zeros((geometry.nr, geometry.nz)),
            ),
            power=PowerSources(
                P_fusion=jnp.zeros((geometry.nr, geometry.nz)),
                P_alpha=jnp.zeros((geometry.nr, geometry.nz)),
                P_neutron=jnp.zeros((geometry.nr, geometry.nz)),
                P_charged=jnp.zeros((geometry.nr, geometry.nz)),
            ),
            circuits=circuits,
        )

        dt = 1e-9
        new_state = model.step(state, dt, geometry)

        # Circuit state should be updated (flux computed)
        assert new_state.circuits.Psi_pickup.shape == (1,)

    def test_backward_compatibility(self, geometry):
        """Model without circuits still works."""
        from jax_frc.models.burning_plasma import BurningPlasmaModel

        config = {
            "fuels": ["DT"],
            # No circuits config
        }

        model = BurningPlasmaModel.from_config(config)
        # Should have empty circuits
        assert model.circuits.n_pickup == 0
        assert model.circuits.n_external == 0
```

**Step 2: Run test to verify it fails**

Run: `py -m pytest tests/test_burning_plasma_circuits.py -v`
Expected: FAIL (model doesn't have circuits attribute yet)

**Step 3: Modify BurningPlasmaState and BurningPlasmaModel**

Read the current file first, then modify:

Update `jax_frc/models/burning_plasma.py` - replace `conversion: ConversionState` with `circuits: CircuitState` and update the model:

```python
"""Burning plasma model with fusion, transport, and energy recovery.

Combines MHD core with nuclear burn physics, species tracking,
anomalous transport, and circuit-coupled energy conversion.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import Array

from jax_frc.core.state import State
from jax_frc.core.geometry import Geometry
from jax_frc.burn.physics import BurnPhysics, ReactionRates, PowerSources
from jax_frc.burn.species import SpeciesState, SpeciesTracker
from jax_frc.circuits import CircuitState, CircuitSystem


@dataclass(frozen=True)
class BurningPlasmaState:
    """Complete state for burning plasma simulation.

    Attributes:
        mhd: MHD state (B, v, p, psi, etc.)
        species: Fuel and ash densities
        rates: Current reaction rates
        power: Current power sources
        circuits: Circuit system state (currents, flux linkages, power)
    """
    mhd: State
    species: SpeciesState
    rates: ReactionRates
    power: PowerSources
    circuits: CircuitState

    def replace(self, **kwargs) -> "BurningPlasmaState":
        """Return new state with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)


# Register BurningPlasmaState as JAX pytree
def _burning_plasma_state_flatten(state):
    children = (state.mhd, state.species, state.rates, state.power, state.circuits)
    aux_data = None
    return children, aux_data


def _burning_plasma_state_unflatten(aux_data, children):
    mhd, species, rates, power, circuits = children
    return BurningPlasmaState(
        mhd=mhd, species=species, rates=rates, power=power, circuits=circuits
    )


jax.tree_util.register_pytree_node(
    BurningPlasmaState,
    _burning_plasma_state_flatten,
    _burning_plasma_state_unflatten,
)


from jax_frc.transport import TransportModel
from jax_frc.models.resistive_mhd import ResistiveMHD


@dataclass
class BurningPlasmaModel:
    """Burning plasma model with fusion, transport, and energy recovery.

    Orchestrates MHD core, burn physics, species tracking,
    transport, and circuit-coupled energy conversion.
    """
    mhd_core: ResistiveMHD
    burn: BurnPhysics
    species_tracker: SpeciesTracker
    transport: TransportModel
    circuits: CircuitSystem

    def step(
        self,
        state: BurningPlasmaState,
        dt: float,
        geometry: Geometry,
    ) -> BurningPlasmaState:
        """Advance burning plasma state by one timestep.

        Args:
            state: Current burning plasma state
            dt: Timestep [s]
            geometry: Computational geometry

        Returns:
            Updated BurningPlasmaState
        """
        # 1. Get temperature in keV from MHD state
        T_keV = state.mhd.T  # Assuming T is already in keV

        # 2. Compute fusion reaction rates
        rates = self.burn.compute_rates(
            n_D=state.species.n_D,
            n_T=state.species.n_T,
            n_He3=state.species.n_He3,
            T_keV=T_keV,
        )

        # 3. Compute power sources
        power = self.burn.power_sources(rates)

        # 4. Compute burn source terms for species
        burn_sources = self.species_tracker.burn_sources(rates)

        # 5. Compute transport fluxes and divergences
        transport_div = {}
        for species_name, n in [
            ("D", state.species.n_D),
            ("T", state.species.n_T),
            ("He3", state.species.n_He3),
            ("He4", state.species.n_He4),
            ("p", state.species.n_p),
        ]:
            Gamma_r, Gamma_z = self.transport.particle_flux(n, geometry)
            div_Gamma = self.transport.flux_divergence(Gamma_r, Gamma_z, geometry)
            transport_div[species_name] = -div_Gamma  # -div(Gamma) is source

        # 6. Update species densities
        new_species = self.species_tracker.advance(
            state=state.species,
            burn_sources=burn_sources,
            transport_divergence=transport_div,
            dt=dt,
        )

        # 7. Update circuit state
        new_circuits = self.circuits.step(
            state=state.circuits,
            B_plasma=state.mhd.B,
            geometry=geometry,
            t=0.0,  # TODO: pass actual simulation time
            dt=dt,
        )

        # 8. Update MHD state (simplified - just pass through for now)
        # Full integration would add alpha heating to energy equation
        new_mhd = state.mhd

        return BurningPlasmaState(
            mhd=new_mhd,
            species=new_species,
            rates=rates,
            power=power,
            circuits=new_circuits,
        )

    @classmethod
    def from_config(cls, config: dict) -> "BurningPlasmaModel":
        """Create BurningPlasmaModel from configuration dictionary.

        Args:
            config: Configuration dictionary with keys:
                - mhd: MHD configuration (optional, defaults to Spitzer resistivity)
                - fuels: List of fuel types, e.g. ["DT", "DD"] (optional)
                - transport: Transport coefficients (optional)
                - circuits: Circuit system configuration (optional)

        Returns:
            Configured BurningPlasmaModel instance
        """
        from jax_frc.models.resistive_mhd import ResistiveMHD

        # MHD core
        mhd_config = config.get("mhd", {"resistivity": {"type": "spitzer"}})
        mhd_core = ResistiveMHD.from_config(mhd_config)

        # Burn physics
        fuels = tuple(config.get("fuels", ["DT"]))
        burn = BurnPhysics(fuels=fuels)

        # Species tracker
        species_tracker = SpeciesTracker()

        # Transport
        transport_config = config.get("transport", {})
        transport = TransportModel(
            D_particle=transport_config.get("D_particle", 1.0),
            chi_e=transport_config.get("chi_e", 5.0),
            chi_i=transport_config.get("chi_i", 2.0),
            v_pinch=transport_config.get("v_pinch", 0.0),
        )

        # Circuits
        circuits_config = config.get("circuits", {})
        circuits = CircuitSystem.from_config(circuits_config)

        return cls(
            mhd_core=mhd_core,
            burn=burn,
            species_tracker=species_tracker,
            transport=transport,
            circuits=circuits,
        )
```

**Step 4: Update burn/__init__.py to keep backward compatibility**

Update `jax_frc/burn/__init__.py`:

```python
"""Nuclear burn physics for fusion plasmas."""

from jax_frc.burn.physics import reactivity, BurnPhysics, ReactionRates, PowerSources
from jax_frc.burn.species import SpeciesState, SpeciesTracker
# Keep DirectConversion for backward compatibility, but CircuitSystem is preferred
from jax_frc.burn.conversion import DirectConversion, ConversionState

__all__ = [
    "reactivity", "BurnPhysics", "ReactionRates", "PowerSources",
    "SpeciesState", "SpeciesTracker",
    "DirectConversion", "ConversionState",  # Deprecated, use jax_frc.circuits
]
```

**Step 5: Run test to verify it passes**

Run: `py -m pytest tests/test_burning_plasma_circuits.py -v`
Expected: PASS (3 tests)

**Step 6: Run all tests to ensure no regressions**

Run: `py -m pytest tests/ -k "not slow" -v`
Expected: All tests pass

**Step 7: Commit**

```bash
git add jax_frc/models/burning_plasma.py jax_frc/burn/__init__.py tests/test_burning_plasma_circuits.py
git commit -m "feat(burning_plasma): integrate CircuitSystem replacing DirectConversion"
```

---

## Task 8: Update Existing Tests for New State Structure

**Files:**
- Modify: `tests/test_burning_plasma.py`

**Step 1: Run existing tests to see what fails**

Run: `py -m pytest tests/test_burning_plasma.py -v`
Expected: Some tests fail due to state structure change

**Step 2: Update test imports and state creation**

Update `tests/test_burning_plasma.py` to use `CircuitState` instead of `ConversionState`:

Replace all occurrences of:
```python
from jax_frc.burn import ..., ConversionState
conversion = ConversionState(P_electric=0.0, V_induced=0.0, dPsi_dt=0.0)
```

With:
```python
from jax_frc.circuits import CircuitState
circuits = CircuitState.zeros(n_pickup=0, n_external=0)
```

And update state creation:
```python
state = BurningPlasmaState(
    mhd=mhd,
    species=species,
    rates=rates,
    power=power,
    circuits=circuits,  # was: conversion=conversion
)
```

**Step 3: Run tests to verify they pass**

Run: `py -m pytest tests/test_burning_plasma.py -v`
Expected: PASS (all tests)

**Step 4: Commit**

```bash
git add tests/test_burning_plasma.py
git commit -m "test(burning_plasma): update tests for CircuitState"
```

---

## Task 9: Add Physics Invariant Tests

**Files:**
- Create: `tests/test_circuit_physics.py`

**Step 1: Write physics invariant tests**

Create `tests/test_circuit_physics.py`:

```python
"""Physics invariant tests for circuit coupling."""

import jax.numpy as jnp
import pytest

from jax_frc.core.geometry import Geometry


@pytest.fixture
def geometry():
    return Geometry(
        coord_system="cylindrical",
        nr=32,
        nz=64,
        r_min=0.1,
        r_max=0.8,
        z_min=-1.0,
        z_max=1.0,
    )


class TestEnergyConservation:
    """Test energy conservation in circuit dynamics."""

    def test_rl_energy_dissipation(self):
        """Energy dissipated equals initial magnetic energy."""
        from jax_frc.circuits import CircuitState, CircuitParams, PickupCoilArray
        from jax_frc.circuits.system import CircuitSystem

        L = 1e-3  # 1 mH
        R = 1.0   # 1 Ohm
        I0 = 10.0  # 10 A initial

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([R]),
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.0]),
        )

        system = CircuitSystem(pickup=pickup, external=None)

        state = CircuitState(
            I_pickup=jnp.array([I0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.array([]),
            Q_external=jnp.array([]),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.array([]),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        B = jnp.zeros((32, 64, 3))
        geometry = Geometry(
            coord_system="cylindrical",
            nr=32, nz=64,
            r_min=0.1, r_max=0.8,
            z_min=-1.0, z_max=1.0,
        )

        # Initial magnetic energy: E = 0.5 * L * I^2
        E_initial = 0.5 * L * I0**2

        # Run for many time constants
        dt = 1e-5
        n_steps = 1000
        total_dissipated = 0.0

        for _ in range(n_steps):
            state = system.step(state, B, geometry, t=0.0, dt=dt)
            # Approximate energy dissipated this step
            I_avg = state.I_pickup[0]
            total_dissipated += I_avg**2 * R * dt

        # Final current should be ~0
        assert jnp.abs(state.I_pickup[0]) < 0.01 * I0

        # Total dissipated should equal initial energy (within numerical error)
        assert jnp.isclose(total_dissipated, E_initial, rtol=0.1)

    def test_lc_energy_conservation(self):
        """LC circuit conserves total energy."""
        from jax_frc.circuits import CircuitState, CircuitParams, PickupCoilArray
        from jax_frc.circuits.system import CircuitSystem

        L = 1e-3
        C = 1e-6
        Q0 = 1e-6  # Initial charge

        params = CircuitParams(
            L=jnp.array([L]),
            R=jnp.array([0.0]),  # No resistance
            C=jnp.array([C]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.0]),
        )

        system = CircuitSystem(pickup=pickup, external=None)

        state = CircuitState(
            I_pickup=jnp.array([0.0]),
            Q_pickup=jnp.array([Q0]),
            I_external=jnp.array([]),
            Q_external=jnp.array([]),
            Psi_pickup=jnp.zeros(1),
            Psi_external=jnp.array([]),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        B = jnp.zeros((32, 64, 3))
        geometry = Geometry(
            coord_system="cylindrical",
            nr=32, nz=64,
            r_min=0.1, r_max=0.8,
            z_min=-1.0, z_max=1.0,
        )

        # Initial energy (all in capacitor)
        E_initial = 0.5 * Q0**2 / C

        # Evolve for one period
        period = 2 * jnp.pi * jnp.sqrt(L * C)
        dt = period / 100
        n_steps = 100

        energies = []
        for _ in range(n_steps):
            state = system.step(state, B, geometry, t=0.0, dt=dt)
            I = state.I_pickup[0]
            Q = state.Q_pickup[0]
            E_L = 0.5 * L * I**2
            E_C = 0.5 * Q**2 / C
            energies.append(E_L + E_C)

        energies = jnp.array(energies)

        # Energy should be conserved (within numerical error)
        assert jnp.allclose(energies, E_initial, rtol=0.05)


class TestFluxConservation:
    """Test flux conservation in superconducting limit."""

    def test_superconducting_flux_conservation(self):
        """In R=0 limit, total flux through circuit is conserved."""
        from jax_frc.circuits import CircuitState, CircuitParams, PickupCoilArray
        from jax_frc.circuits.system import CircuitSystem

        params = CircuitParams(
            L=jnp.array([1e-3]),
            R=jnp.array([1e-10]),  # Near-zero resistance
            C=jnp.array([jnp.inf]),
        )
        pickup = PickupCoilArray(
            z_positions=jnp.array([0.0]),
            radii=jnp.array([0.4]),
            n_turns=jnp.array([100]),
            params=params,
            load_resistance=jnp.array([0.0]),
        )

        system = CircuitSystem(pickup=pickup, external=None)

        # Initial state with some current
        I0 = 1.0
        Psi0 = 0.01  # Initial flux
        state = CircuitState(
            I_pickup=jnp.array([I0]),
            Q_pickup=jnp.zeros(1),
            I_external=jnp.array([]),
            Q_external=jnp.array([]),
            Psi_pickup=jnp.array([Psi0]),
            Psi_external=jnp.array([]),
            P_extracted=0.0,
            P_dissipated=0.0,
        )

        geometry = Geometry(
            coord_system="cylindrical",
            nr=32, nz=64,
            r_min=0.1, r_max=0.8,
            z_min=-1.0, z_max=1.0,
        )

        # Apply changing external flux
        # Total flux = Psi_external + L*I should be conserved
        L = params.L[0]
        Phi_total_initial = Psi0 + L * I0

        # B-field that gives different flux
        B = jnp.zeros((32, 64, 3))
        B = B.at[:, :, 2].set(0.5)  # Different Bz

        dt = 1e-6
        n_steps = 100

        for _ in range(n_steps):
            state = system.step(state, B, geometry, t=0.0, dt=dt)

        # Total flux should be conserved
        Phi_total_final = state.Psi_pickup[0] + L * state.I_pickup[0]

        # Allow some drift due to small R
        assert jnp.isclose(Phi_total_final, Phi_total_initial, rtol=0.05)
```

**Step 2: Run physics tests**

Run: `py -m pytest tests/test_circuit_physics.py -v`
Expected: PASS (3 tests)

**Step 3: Commit**

```bash
git add tests/test_circuit_physics.py
git commit -m "test(circuits): add physics invariant tests for energy and flux conservation"
```

---

## Task 10: Run Full Test Suite and Final Cleanup

**Step 1: Run all tests**

Run: `py -m pytest tests/ -k "not slow" -v`
Expected: All tests pass

**Step 2: Run slow tests if desired**

Run: `py -m pytest tests/ -v`
Expected: All tests pass

**Step 3: Final commit with summary**

If all tests pass:

```bash
git add -A
git status
# If there are any remaining changes:
git commit -m "chore: final cleanup for circuit coupling feature"
```

**Step 4: Summary of changes**

The circuit coupling feature is complete with:

- `jax_frc/circuits/` module with:
  - `state.py`: CircuitState, CircuitParams
  - `waveforms.py`: Voltage waveform generators
  - `pickup.py`: PickupCoilArray for energy extraction
  - `external.py`: ExternalCircuits with drivers
  - `coupling.py`: FluxCoupling for plasma-circuit interaction
  - `system.py`: CircuitSystem with ODE integration

- Updated `jax_frc/models/burning_plasma.py` to use CircuitSystem

- Comprehensive tests in:
  - `tests/test_circuit_state.py`
  - `tests/test_waveforms.py`
  - `tests/test_pickup_coils.py`
  - `tests/test_external_circuits.py`
  - `tests/test_flux_coupling.py`
  - `tests/test_circuit_system.py`
  - `tests/test_burning_plasma_circuits.py`
  - `tests/test_circuit_physics.py`
