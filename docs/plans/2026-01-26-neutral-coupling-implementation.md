# Neutral-Fluid Coupling Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add neutral fluid + atomic physics to resistive MHD for FRC formation simulations

**Architecture:** Operator-split coupling: MHD advances, then atomic sources update plasma/neutral densities and energies. Three new files: `atomic_rates.py` (rate coefficients), `neutral_fluid.py` (Euler equations), extended `resistive_mhd.py` (coupled model).

**Tech Stack:** JAX, jax.numpy, @jit decorators, dataclasses, existing `State`/`Geometry` containers

---

## Task 1: Ionization Rate Coefficient

**Files:**
- Create: `jax_frc/models/atomic_rates.py`
- Test: `tests/test_atomic_rates.py`

**Step 1: Write the failing test**

```python
# tests/test_atomic_rates.py
"""Tests for atomic rate coefficients."""

import jax.numpy as jnp
import pytest

from jax_frc.constants import QE


class TestIonizationRateCoefficient:
    """Tests for hydrogen ionization rate coefficient."""

    def test_ionization_rate_coefficient_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import ionization_rate_coefficient
        assert callable(ionization_rate_coefficient)

    def test_ionization_rate_positive(self):
        """Rate coefficient is positive for physical temperatures."""
        from jax_frc.models.atomic_rates import ionization_rate_coefficient
        Te = jnp.array([10.0, 50.0, 100.0, 500.0]) * QE  # eV to Joules
        sigma_v = ionization_rate_coefficient(Te)
        assert jnp.all(sigma_v > 0)

    def test_ionization_rate_peak_location(self):
        """Ionization peaks around 50-100 eV for hydrogen."""
        from jax_frc.models.atomic_rates import ionization_rate_coefficient
        Te_eV = jnp.logspace(0, 3, 100)  # 1 eV to 1 keV
        Te = Te_eV * QE
        sigma_v = ionization_rate_coefficient(Te)
        peak_idx = jnp.argmax(sigma_v)
        peak_Te_eV = Te_eV[peak_idx]
        assert 30 < peak_Te_eV < 150, f"Peak at {peak_Te_eV} eV, expected 30-150 eV"

    def test_ionization_rate_low_Te_small(self):
        """Rate is very small below ionization threshold (~13.6 eV)."""
        from jax_frc.models.atomic_rates import ionization_rate_coefficient
        Te_low = 1.0 * QE  # 1 eV
        Te_high = 100.0 * QE  # 100 eV
        assert ionization_rate_coefficient(Te_low) < 0.01 * ionization_rate_coefficient(Te_high)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'jax_frc.models.atomic_rates'"

**Step 3: Write minimal implementation**

```python
# jax_frc/models/atomic_rates.py
"""Atomic rate coefficients for plasma-neutral interactions.

Includes ionization, recombination, charge exchange, and radiation.
All rates use SI units and are JIT-compatible.
"""

from typing import Tuple
import jax.numpy as jnp
from jax import jit, Array

from jax_frc.constants import QE, MI


# =============================================================================
# Ionization (electron impact: H + e -> H+ + 2e)
# =============================================================================

@jit
def ionization_rate_coefficient(Te: Array) -> Array:
    """Voronov fit for hydrogen ionization <σv>_ion(Te) [m³/s].

    Reference: Voronov (1997), Atomic Data and Nuclear Data Tables 65, 1-35.

    <σv> = A * (1 + P*sqrt(U)) * U^K * exp(-U) / (X + U)
    where U = E_ion / Te, E_ion = 13.6 eV

    Args:
        Te: Electron temperature [J] (can be array)

    Returns:
        Rate coefficient [m³/s]
    """
    E_ion = 13.6 * QE  # Ionization energy in Joules

    # Clamp Te to avoid division by zero and overflow
    Te_safe = jnp.maximum(Te, 0.1 * QE)  # Min 0.1 eV

    U = E_ion / Te_safe

    # Voronov coefficients for hydrogen
    A = 2.91e-14  # m³/s
    P = 0.0
    K = 0.39
    X = 0.232

    # Clamp U to prevent overflow in exp(-U)
    U_clamped = jnp.minimum(U, 100.0)

    sigma_v = A * (1 + P * jnp.sqrt(U_clamped)) * U_clamped**K * jnp.exp(-U_clamped) / (X + U_clamped)

    return sigma_v
```

**Step 4: Run test to verify it passes**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py::TestIonizationRateCoefficient -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add jax_frc/models/atomic_rates.py tests/test_atomic_rates.py
git commit -m "feat(atomic): add ionization rate coefficient with Voronov fit"
```

---

## Task 2: Ionization Mass Rate

**Files:**
- Modify: `jax_frc/models/atomic_rates.py`
- Test: `tests/test_atomic_rates.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_atomic_rates.py

class TestIonizationRate:
    """Tests for mass ionization rate S_ion."""

    def test_ionization_rate_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import ionization_rate
        assert callable(ionization_rate)

    def test_ionization_rate_dimensions(self):
        """Output has same shape as inputs."""
        from jax_frc.models.atomic_rates import ionization_rate
        Te = jnp.ones((16, 32)) * 50 * QE  # 50 eV
        ne = jnp.ones((16, 32)) * 1e19  # m^-3
        rho_n = jnp.ones((16, 32)) * 1e-6  # kg/m³
        S_ion = ionization_rate(Te, ne, rho_n)
        assert S_ion.shape == (16, 32)

    def test_ionization_rate_positive(self):
        """Rate is positive for positive inputs."""
        from jax_frc.models.atomic_rates import ionization_rate
        Te = jnp.ones((16, 32)) * 50 * QE
        ne = jnp.ones((16, 32)) * 1e19
        rho_n = jnp.ones((16, 32)) * 1e-6
        S_ion = ionization_rate(Te, ne, rho_n)
        assert jnp.all(S_ion > 0)

    def test_ionization_rate_scales_with_density(self):
        """Doubling ne or rho_n doubles rate."""
        from jax_frc.models.atomic_rates import ionization_rate
        Te = 50 * QE
        ne = 1e19
        rho_n = 1e-6
        S1 = ionization_rate(Te, ne, rho_n)
        S2 = ionization_rate(Te, 2*ne, rho_n)
        assert jnp.isclose(S2, 2*S1, rtol=1e-5)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py::TestIonizationRate -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# Add to jax_frc/models/atomic_rates.py after ionization_rate_coefficient

@jit
def ionization_rate(Te: Array, ne: Array, rho_n: Array) -> Array:
    """Mass ionization rate S_ion [kg/m³/s].

    S_ion = m_i * ne * nn * <σv>_ion(Te)

    Args:
        Te: Electron temperature [J]
        ne: Electron density [m⁻³]
        rho_n: Neutral mass density [kg/m³]

    Returns:
        Mass ionization rate [kg/m³/s]
    """
    nn = rho_n / MI  # Neutral number density [m⁻³]
    sigma_v = ionization_rate_coefficient(Te)
    return MI * ne * nn * sigma_v
```

**Step 4: Run test to verify it passes**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py::TestIonizationRate -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add jax_frc/models/atomic_rates.py tests/test_atomic_rates.py
git commit -m "feat(atomic): add ionization mass rate function"
```

---

## Task 3: Recombination Rates

**Files:**
- Modify: `jax_frc/models/atomic_rates.py`
- Test: `tests/test_atomic_rates.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_atomic_rates.py

class TestRecombinationRate:
    """Tests for radiative recombination."""

    def test_recombination_rate_coefficient_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import recombination_rate_coefficient
        assert callable(recombination_rate_coefficient)

    def test_recombination_rate_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import recombination_rate
        assert callable(recombination_rate)

    def test_recombination_decreases_with_Te(self):
        """Recombination rate decreases as Te increases."""
        from jax_frc.models.atomic_rates import recombination_rate_coefficient
        Te_low = 1.0 * QE
        Te_high = 100.0 * QE
        assert recombination_rate_coefficient(Te_low) > recombination_rate_coefficient(Te_high)

    def test_recombination_dominates_at_low_Te(self):
        """Recombination > ionization below ~1 eV."""
        from jax_frc.models.atomic_rates import (
            ionization_rate_coefficient,
            recombination_rate_coefficient,
        )
        Te_low = 1.0 * QE  # 1 eV
        assert recombination_rate_coefficient(Te_low) > ionization_rate_coefficient(Te_low)

    def test_ionization_dominates_at_high_Te(self):
        """Ionization > recombination above ~50 eV."""
        from jax_frc.models.atomic_rates import (
            ionization_rate_coefficient,
            recombination_rate_coefficient,
        )
        Te_high = 100.0 * QE  # 100 eV
        assert ionization_rate_coefficient(Te_high) > recombination_rate_coefficient(Te_high)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py::TestRecombinationRate -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# Add to jax_frc/models/atomic_rates.py

# =============================================================================
# Recombination (radiative: H+ + e -> H + hν)
# =============================================================================

@jit
def recombination_rate_coefficient(Te: Array) -> Array:
    """Radiative recombination <σv>_rec(Te) [m³/s].

    Approximate fit: <σv>_rec ≈ 2.6e-19 * (13.6 eV / Te)^0.7
    Valid for Te > 0.1 eV.

    Args:
        Te: Electron temperature [J]

    Returns:
        Rate coefficient [m³/s]
    """
    Te_eV = Te / QE
    Te_eV_safe = jnp.maximum(Te_eV, 0.1)
    return 2.6e-19 * (13.6 / Te_eV_safe)**0.7


@jit
def recombination_rate(Te: Array, ne: Array, ni: Array) -> Array:
    """Mass recombination rate S_rec [kg/m³/s].

    S_rec = m_i * ne * ni * <σv>_rec(Te)

    Args:
        Te: Electron temperature [J]
        ne: Electron density [m⁻³]
        ni: Ion density [m⁻³]

    Returns:
        Mass recombination rate [kg/m³/s]
    """
    sigma_v = recombination_rate_coefficient(Te)
    return MI * ne * ni * sigma_v
```

**Step 4: Run test to verify it passes**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py::TestRecombinationRate -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add jax_frc/models/atomic_rates.py tests/test_atomic_rates.py
git commit -m "feat(atomic): add recombination rate coefficient and mass rate"
```

---

## Task 4: Charge Exchange Rates

**Files:**
- Modify: `jax_frc/models/atomic_rates.py`
- Test: `tests/test_atomic_rates.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_atomic_rates.py

class TestChargeExchangeRates:
    """Tests for charge exchange momentum and energy transfer."""

    def test_charge_exchange_rates_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import charge_exchange_rates
        assert callable(charge_exchange_rates)

    def test_charge_exchange_returns_tuple(self):
        """Returns (R_cx, Q_cx) tuple."""
        from jax_frc.models.atomic_rates import charge_exchange_rates
        Ti = jnp.ones((16, 32)) * 100 * QE
        ni = jnp.ones((16, 32)) * 1e19
        nn = jnp.ones((16, 32)) * 1e18
        v_i = jnp.zeros((16, 32, 3))
        v_n = jnp.zeros((16, 32, 3))
        R_cx, Q_cx = charge_exchange_rates(Ti, ni, nn, v_i, v_n)
        assert R_cx.shape == (16, 32, 3)
        assert Q_cx.shape == (16, 32)

    def test_charge_exchange_momentum_zero_when_velocities_equal(self):
        """R_cx = 0 when v_i = v_n."""
        from jax_frc.models.atomic_rates import charge_exchange_rates
        Ti = jnp.ones((16, 32)) * 100 * QE
        ni = jnp.ones((16, 32)) * 1e19
        nn = jnp.ones((16, 32)) * 1e18
        v = jnp.ones((16, 32, 3)) * 1000  # Same velocity
        R_cx, _ = charge_exchange_rates(Ti, ni, nn, v, v)
        assert jnp.allclose(R_cx, 0, atol=1e-20)

    def test_charge_exchange_energy_positive(self):
        """Q_cx > 0 for hot ions (energy flows from ions to neutrals)."""
        from jax_frc.models.atomic_rates import charge_exchange_rates
        Ti = jnp.ones((16, 32)) * 100 * QE  # Hot ions
        ni = jnp.ones((16, 32)) * 1e19
        nn = jnp.ones((16, 32)) * 1e18
        v_i = jnp.zeros((16, 32, 3))
        v_n = jnp.zeros((16, 32, 3))
        _, Q_cx = charge_exchange_rates(Ti, ni, nn, v_i, v_n)
        assert jnp.all(Q_cx > 0)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py::TestChargeExchangeRates -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# Add to jax_frc/models/atomic_rates.py

# =============================================================================
# Charge Exchange (H+ + H -> H + H+)
# =============================================================================

@jit
def charge_exchange_cross_section(Ti: Array) -> Array:
    """Charge exchange cross-section σ_cx(Ti) [m²].

    Nearly constant ~3e-19 m² for Ti < 10 keV.

    Args:
        Ti: Ion temperature [J]

    Returns:
        Cross-section [m²]
    """
    return 3.0e-19 * jnp.ones_like(Ti)


@jit
def charge_exchange_rates(
    Ti: Array, ni: Array, nn: Array, v_i: Array, v_n: Array
) -> Tuple[Array, Array]:
    """Charge exchange momentum and energy transfer rates.

    R_cx: Momentum transfer to plasma [N/m³] (add to plasma, subtract from neutrals)
    Q_cx: Energy transfer to plasma [W/m³] (add to plasma, subtract from neutrals)

    For cold neutrals (T_n << T_i), energy flows from ions to neutrals,
    so Q_cx represents energy gained by plasma from neutrals (negative for hot plasma).
    Convention: positive Q_cx means plasma gains energy.

    Args:
        Ti: Ion temperature [J]
        ni: Ion density [m⁻³]
        nn: Neutral density [m⁻³]
        v_i: Ion velocity (nr, nz, 3) [m/s]
        v_n: Neutral velocity (nr, nz, 3) [m/s]

    Returns:
        R_cx: Momentum transfer [N/m³], shape (nr, nz, 3)
        Q_cx: Energy transfer [W/m³], shape (nr, nz)
    """
    # Thermal speed for collision frequency
    v_thermal = jnp.sqrt(8 * Ti / (jnp.pi * MI))

    sigma = charge_exchange_cross_section(Ti)
    nu_cx = nn * sigma * v_thermal  # CX collision frequency [1/s]

    # Momentum transfer: R_cx = m_i * n_i * nu_cx * (v_n - v_i)
    # This is the momentum gained by plasma from neutrals
    # If v_n > v_i, plasma gains momentum (R_cx > 0 in that component)
    R_cx = MI * ni[..., None] * nu_cx[..., None] * (v_n - v_i)

    # Energy transfer: Q_cx = (3/2) * n_i * nu_cx * (T_n - T_i)
    # Assume cold neutrals: T_n << T_i, so Q_cx ≈ -(3/2) * n_i * nu_cx * T_i
    # This is energy lost by plasma to neutrals (negative)
    # Convention: return positive value representing energy loss rate
    Q_cx = 1.5 * ni * nu_cx * Ti  # Energy loss from plasma [W/m³]

    return R_cx, Q_cx
```

**Step 4: Run test to verify it passes**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py::TestChargeExchangeRates -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add jax_frc/models/atomic_rates.py tests/test_atomic_rates.py
git commit -m "feat(atomic): add charge exchange momentum and energy transfer"
```

---

## Task 5: Radiation Losses

**Files:**
- Modify: `jax_frc/models/atomic_rates.py`
- Test: `tests/test_atomic_rates.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_atomic_rates.py

class TestRadiationLoss:
    """Tests for radiation loss terms."""

    def test_bremsstrahlung_loss_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import bremsstrahlung_loss
        assert callable(bremsstrahlung_loss)

    def test_bremsstrahlung_positive(self):
        """Bremsstrahlung is always positive (energy sink)."""
        from jax_frc.models.atomic_rates import bremsstrahlung_loss
        Te = jnp.ones((16, 32)) * 100 * QE
        ne = jnp.ones((16, 32)) * 1e19
        ni = jnp.ones((16, 32)) * 1e19
        P_brem = bremsstrahlung_loss(Te, ne, ni)
        assert jnp.all(P_brem > 0)

    def test_bremsstrahlung_scales_with_density_squared(self):
        """P_brem ~ ne * ni."""
        from jax_frc.models.atomic_rates import bremsstrahlung_loss
        Te = 100 * QE
        ne = 1e19
        ni = 1e19
        P1 = bremsstrahlung_loss(Te, ne, ni)
        P2 = bremsstrahlung_loss(Te, 2*ne, 2*ni)
        assert jnp.isclose(P2, 4*P1, rtol=1e-5)

    def test_total_radiation_loss_exists(self):
        """Function is importable."""
        from jax_frc.models.atomic_rates import total_radiation_loss
        assert callable(total_radiation_loss)

    def test_total_radiation_positive(self):
        """Total radiation is always positive."""
        from jax_frc.models.atomic_rates import total_radiation_loss
        Te = jnp.ones((16, 32)) * 100 * QE
        ne = jnp.ones((16, 32)) * 1e19
        ni = jnp.ones((16, 32)) * 1e19
        n_imp = jnp.ones((16, 32)) * 1e17  # 1% impurity
        S_ion = jnp.ones((16, 32)) * 1e10  # kg/m³/s
        P_rad = total_radiation_loss(Te, ne, ni, n_imp, S_ion)
        assert jnp.all(P_rad > 0)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py::TestRadiationLoss -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# Add to jax_frc/models/atomic_rates.py

# =============================================================================
# Radiation Losses
# =============================================================================

@jit
def bremsstrahlung_loss(Te: Array, ne: Array, ni: Array, Z_eff: float = 1.0) -> Array:
    """Bremsstrahlung power loss P_brem [W/m³].

    P_brem = 1.69e-38 * Z_eff² * ne * ni * sqrt(Te_eV)

    Args:
        Te: Electron temperature [J]
        ne: Electron density [m⁻³]
        ni: Ion density [m⁻³]
        Z_eff: Effective charge (default 1.0 for hydrogen)

    Returns:
        Power loss [W/m³]
    """
    Te_eV = Te / QE
    Te_eV_safe = jnp.maximum(Te_eV, 0.1)
    return 1.69e-38 * Z_eff**2 * ne * ni * jnp.sqrt(Te_eV_safe)


@jit
def line_radiation_loss(Te: Array, ne: Array, n_impurity: Array) -> Array:
    """Line radiation from impurities [W/m³].

    Uses simplified cooling curve for carbon impurity.
    Peak around 10 eV, drops at higher Te.

    Args:
        Te: Electron temperature [J]
        ne: Electron density [m⁻³]
        n_impurity: Impurity density [m⁻³]

    Returns:
        Power loss [W/m³]
    """
    Te_eV = Te / QE
    Te_eV_safe = jnp.maximum(Te_eV, 0.1)
    # Gaussian cooling curve peaked at ~10 eV
    L_cool = 1e-31 * jnp.exp(-((jnp.log10(Te_eV_safe) - 1.0) / 0.5)**2)
    return ne * n_impurity * L_cool


@jit
def ionization_energy_loss(S_ion: Array) -> Array:
    """Energy sink from ionization events [W/m³].

    Each ionization costs E_ion = 13.6 eV.

    Args:
        S_ion: Mass ionization rate [kg/m³/s]

    Returns:
        Power loss [W/m³]
    """
    E_ion = 13.6 * QE
    # S_ion has units kg/m³/s, divide by MI to get ionizations/m³/s
    ionizations_per_volume = S_ion / MI
    return ionizations_per_volume * E_ion


@jit
def total_radiation_loss(
    Te: Array, ne: Array, ni: Array, n_impurity: Array, S_ion: Array, Z_eff: float = 1.0
) -> Array:
    """Total radiation sink for energy equation [W/m³].

    Combines bremsstrahlung, line radiation, and ionization energy loss.

    Args:
        Te: Electron temperature [J]
        ne: Electron density [m⁻³]
        ni: Ion density [m⁻³]
        n_impurity: Impurity density [m⁻³]
        S_ion: Mass ionization rate [kg/m³/s]
        Z_eff: Effective charge

    Returns:
        Total power loss [W/m³]
    """
    P_brem = bremsstrahlung_loss(Te, ne, ni, Z_eff)
    P_line = line_radiation_loss(Te, ne, n_impurity)
    P_ion = ionization_energy_loss(S_ion)
    return P_brem + P_line + P_ion
```

**Step 4: Run test to verify it passes**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py::TestRadiationLoss -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add jax_frc/models/atomic_rates.py tests/test_atomic_rates.py
git commit -m "feat(atomic): add radiation loss functions (bremsstrahlung, line, ionization)"
```

---

## Task 6: NeutralState Dataclass

**Files:**
- Create: `jax_frc/models/neutral_fluid.py`
- Test: `tests/test_neutral_fluid.py`

**Step 1: Write the failing test**

```python
# tests/test_neutral_fluid.py
"""Tests for neutral fluid model."""

import jax
import jax.numpy as jnp
import pytest

from jax_frc.constants import QE, MI


class TestNeutralState:
    """Tests for NeutralState dataclass."""

    def test_neutral_state_importable(self):
        """NeutralState is importable."""
        from jax_frc.models.neutral_fluid import NeutralState
        assert NeutralState is not None

    def test_neutral_state_creation(self):
        """Can create NeutralState with required fields."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * 1e-6
        mom_n = jnp.zeros((16, 32, 3))
        E_n = jnp.ones((16, 32)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)
        assert state.rho_n.shape == (16, 32)
        assert state.mom_n.shape == (16, 32, 3)
        assert state.E_n.shape == (16, 32)

    def test_neutral_state_velocity_property(self):
        """v_n = mom_n / rho_n."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * 1e-6
        mom_n = jnp.ones((16, 32, 3)) * 1e-6 * 1000  # 1000 m/s
        E_n = jnp.ones((16, 32)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)
        assert jnp.allclose(state.v_n, 1000.0)

    def test_neutral_state_pressure_property(self):
        """p_n from ideal gas EOS."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * MI * 1e19  # n = 1e19 m^-3
        mom_n = jnp.zeros((16, 32, 3))  # Stationary
        # E_n = p / (gamma - 1) for stationary gas
        gamma = 5/3
        p_target = 1e19 * 10 * QE  # n * T where T = 10 eV
        E_n = p_target / (gamma - 1)
        E_n = jnp.ones((16, 32)) * E_n
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)
        assert jnp.allclose(state.p_n, p_target, rtol=1e-5)

    def test_neutral_state_is_pytree(self):
        """NeutralState works with JAX transformations."""
        from jax_frc.models.neutral_fluid import NeutralState
        rho_n = jnp.ones((16, 32)) * 1e-6
        mom_n = jnp.zeros((16, 32, 3))
        E_n = jnp.ones((16, 32)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        # Should be able to use in JIT
        @jax.jit
        def get_density(s):
            return s.rho_n

        result = get_density(state)
        assert result.shape == (16, 32)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_neutral_fluid.py::TestNeutralState -v`
Expected: FAIL with "ModuleNotFoundError"

**Step 3: Write minimal implementation**

```python
# jax_frc/models/neutral_fluid.py
"""Neutral fluid model for plasma-neutral coupling.

Implements Euler equations for neutral gas with atomic source terms.
"""

from dataclasses import dataclass
from typing import Optional
import jax
import jax.numpy as jnp
from jax import Array

from jax_frc.constants import MI

# Adiabatic index for monatomic gas
GAMMA = 5.0 / 3.0


@dataclass(frozen=True)
class NeutralState:
    """Neutral fluid state variables.

    All fields use SI units.
    """

    rho_n: Array  # Mass density [kg/m³], shape (nr, nz)
    mom_n: Array  # Momentum density [kg/m²/s], shape (nr, nz, 3)
    E_n: Array    # Total energy density [J/m³], shape (nr, nz)

    @property
    def v_n(self) -> Array:
        """Velocity [m/s], shape (nr, nz, 3)."""
        rho_safe = jnp.maximum(self.rho_n[..., None], 1e-20)
        return self.mom_n / rho_safe

    @property
    def p_n(self) -> Array:
        """Pressure [Pa] from ideal gas EOS, shape (nr, nz)."""
        rho_safe = jnp.maximum(self.rho_n, 1e-20)
        ke = 0.5 * jnp.sum(self.mom_n**2, axis=-1) / rho_safe
        internal_energy = self.E_n - ke
        return (GAMMA - 1) * jnp.maximum(internal_energy, 0.0)

    @property
    def T_n(self) -> Array:
        """Temperature [J], shape (nr, nz)."""
        n_n = self.rho_n / MI
        n_safe = jnp.maximum(n_n, 1e-10)
        return self.p_n / n_safe

    def replace(self, **kwargs) -> "NeutralState":
        """Return new NeutralState with specified fields replaced."""
        from dataclasses import replace as dc_replace
        return dc_replace(self, **kwargs)


# Register NeutralState as JAX pytree
def _neutral_state_flatten(state):
    children = (state.rho_n, state.mom_n, state.E_n)
    aux_data = None
    return children, aux_data


def _neutral_state_unflatten(aux_data, children):
    rho_n, mom_n, E_n = children
    return NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)


jax.tree_util.register_pytree_node(
    NeutralState, _neutral_state_flatten, _neutral_state_unflatten
)
```

**Step 4: Run test to verify it passes**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_neutral_fluid.py::TestNeutralState -v`
Expected: PASS (5 tests)

**Step 5: Commit**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add jax_frc/models/neutral_fluid.py tests/test_neutral_fluid.py
git commit -m "feat(neutral): add NeutralState dataclass with pytree registration"
```

---

## Task 7: Euler Flux Functions

**Files:**
- Modify: `jax_frc/models/neutral_fluid.py`
- Test: `tests/test_neutral_fluid.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_neutral_fluid.py

class TestEulerFlux:
    """Tests for Euler flux computations."""

    def test_euler_flux_exists(self):
        """Function is importable."""
        from jax_frc.models.neutral_fluid import euler_flux_1d
        assert callable(euler_flux_1d)

    def test_euler_flux_mass(self):
        """Mass flux = rho * v."""
        from jax_frc.models.neutral_fluid import euler_flux_1d
        rho = 1.0
        v = 100.0
        p = 1e5
        E = p / (5/3 - 1) + 0.5 * rho * v**2
        F_rho, F_mom, F_E = euler_flux_1d(rho, v, p, E)
        assert jnp.isclose(F_rho, rho * v)

    def test_euler_flux_momentum(self):
        """Momentum flux = rho * v² + p."""
        from jax_frc.models.neutral_fluid import euler_flux_1d
        rho = 1.0
        v = 100.0
        p = 1e5
        E = p / (5/3 - 1) + 0.5 * rho * v**2
        F_rho, F_mom, F_E = euler_flux_1d(rho, v, p, E)
        assert jnp.isclose(F_mom, rho * v**2 + p)

    def test_euler_flux_energy(self):
        """Energy flux = (E + p) * v."""
        from jax_frc.models.neutral_fluid import euler_flux_1d
        rho = 1.0
        v = 100.0
        p = 1e5
        E = p / (5/3 - 1) + 0.5 * rho * v**2
        F_rho, F_mom, F_E = euler_flux_1d(rho, v, p, E)
        assert jnp.isclose(F_E, (E + p) * v)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_neutral_fluid.py::TestEulerFlux -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# Add to jax_frc/models/neutral_fluid.py after NeutralState

from jax import jit
from typing import Tuple


@jit
def euler_flux_1d(rho: Array, v: Array, p: Array, E: Array) -> Tuple[Array, Array, Array]:
    """Compute 1D Euler fluxes.

    Args:
        rho: Mass density [kg/m³]
        v: Velocity component in flux direction [m/s]
        p: Pressure [Pa]
        E: Total energy density [J/m³]

    Returns:
        F_rho: Mass flux [kg/m²/s]
        F_mom: Momentum flux [Pa]
        F_E: Energy flux [W/m²]
    """
    F_rho = rho * v
    F_mom = rho * v**2 + p
    F_E = (E + p) * v
    return F_rho, F_mom, F_E
```

**Step 4: Run test to verify it passes**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_neutral_fluid.py::TestEulerFlux -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add jax_frc/models/neutral_fluid.py tests/test_neutral_fluid.py
git commit -m "feat(neutral): add 1D Euler flux function"
```

---

## Task 8: HLLE Riemann Solver

**Files:**
- Modify: `jax_frc/models/neutral_fluid.py`
- Test: `tests/test_neutral_fluid.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_neutral_fluid.py

class TestHLLEFlux:
    """Tests for HLLE approximate Riemann solver."""

    def test_hlle_flux_exists(self):
        """Function is importable."""
        from jax_frc.models.neutral_fluid import hlle_flux_1d
        assert callable(hlle_flux_1d)

    def test_hlle_flux_uniform_state(self):
        """HLLE returns physical flux for uniform state."""
        from jax_frc.models.neutral_fluid import hlle_flux_1d, euler_flux_1d
        rho = 1.0
        v = 100.0
        p = 1e5
        E = p / (5/3 - 1) + 0.5 * rho * v**2

        # Same state on left and right
        F_hlle = hlle_flux_1d(rho, rho, v, v, p, p, E, E)
        F_exact = euler_flux_1d(rho, v, p, E)

        assert jnp.allclose(F_hlle[0], F_exact[0], rtol=1e-5)
        assert jnp.allclose(F_hlle[1], F_exact[1], rtol=1e-5)
        assert jnp.allclose(F_hlle[2], F_exact[2], rtol=1e-5)

    def test_hlle_flux_supersonic_right(self):
        """For supersonic flow to right, use left flux."""
        from jax_frc.models.neutral_fluid import hlle_flux_1d, euler_flux_1d
        # Supersonic flow: v >> c_s
        rho = 1.0
        v = 1000.0  # Much faster than sound speed ~300 m/s
        p = 1e5
        E = p / (5/3 - 1) + 0.5 * rho * v**2

        # Slight perturbation on right
        F_hlle = hlle_flux_1d(rho, rho*1.01, v, v, p, p*1.01, E, E*1.01)
        F_left = euler_flux_1d(rho, v, p, E)

        # Should be close to left flux
        assert jnp.allclose(F_hlle[0], F_left[0], rtol=0.1)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_neutral_fluid.py::TestHLLEFlux -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# Add to jax_frc/models/neutral_fluid.py

@jit
def hlle_flux_1d(
    rho_L: Array, rho_R: Array,
    v_L: Array, v_R: Array,
    p_L: Array, p_R: Array,
    E_L: Array, E_R: Array,
    gamma: float = GAMMA
) -> Tuple[Array, Array, Array]:
    """HLLE approximate Riemann solver for 1D Euler equations.

    Args:
        rho_L, rho_R: Left/right densities
        v_L, v_R: Left/right velocities (normal component)
        p_L, p_R: Left/right pressures
        E_L, E_R: Left/right total energies
        gamma: Adiabatic index

    Returns:
        F_rho, F_mom, F_E: Numerical fluxes at interface
    """
    # Sound speeds
    rho_L_safe = jnp.maximum(rho_L, 1e-20)
    rho_R_safe = jnp.maximum(rho_R, 1e-20)
    p_L_safe = jnp.maximum(p_L, 1e-10)
    p_R_safe = jnp.maximum(p_R, 1e-10)

    c_L = jnp.sqrt(gamma * p_L_safe / rho_L_safe)
    c_R = jnp.sqrt(gamma * p_R_safe / rho_R_safe)

    # Wave speed estimates (Davis)
    S_L = jnp.minimum(v_L - c_L, v_R - c_R)
    S_R = jnp.maximum(v_L + c_L, v_R + c_R)

    # Physical fluxes
    F_rho_L, F_mom_L, F_E_L = euler_flux_1d(rho_L, v_L, p_L, E_L)
    F_rho_R, F_mom_R, F_E_R = euler_flux_1d(rho_R, v_R, p_R, E_R)

    # Conserved variables
    U_rho_L, U_rho_R = rho_L, rho_R
    U_mom_L, U_mom_R = rho_L * v_L, rho_R * v_R
    U_E_L, U_E_R = E_L, E_R

    # Avoid division by zero
    dS = S_R - S_L
    dS_safe = jnp.where(jnp.abs(dS) < 1e-10, 1e-10, dS)

    # HLLE flux formula
    def hlle_component(F_L, F_R, U_L, U_R):
        return jnp.where(
            S_L >= 0, F_L,
            jnp.where(
                S_R <= 0, F_R,
                (S_R * F_L - S_L * F_R + S_L * S_R * (U_R - U_L)) / dS_safe
            )
        )

    F_rho = hlle_component(F_rho_L, F_rho_R, U_rho_L, U_rho_R)
    F_mom = hlle_component(F_mom_L, F_mom_R, U_mom_L, U_mom_R)
    F_E = hlle_component(F_E_L, F_E_R, U_E_L, U_E_R)

    return F_rho, F_mom, F_E
```

**Step 4: Run test to verify it passes**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_neutral_fluid.py::TestHLLEFlux -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add jax_frc/models/neutral_fluid.py tests/test_neutral_fluid.py
git commit -m "feat(neutral): add HLLE Riemann solver"
```

---

## Task 9: NeutralFluid Model Class

**Files:**
- Modify: `jax_frc/models/neutral_fluid.py`
- Test: `tests/test_neutral_fluid.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_neutral_fluid.py
from jax_frc.core.geometry import Geometry


class TestNeutralFluid:
    """Tests for NeutralFluid model class."""

    def test_neutral_fluid_importable(self):
        """NeutralFluid is importable."""
        from jax_frc.models.neutral_fluid import NeutralFluid
        assert NeutralFluid is not None

    def test_neutral_fluid_creation(self):
        """Can create NeutralFluid instance."""
        from jax_frc.models.neutral_fluid import NeutralFluid
        model = NeutralFluid(gamma=5/3)
        assert model.gamma == 5/3

    def test_compute_flux_divergence_shape(self):
        """Flux divergence returns correct shapes."""
        from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState

        nr, nz = 16, 32
        model = NeutralFluid()
        geometry = Geometry(nr=nr, nz=nz, r_max=1.0, z_max=2.0)

        rho_n = jnp.ones((nr, nz)) * 1e-6
        mom_n = jnp.zeros((nr, nz, 3))
        E_n = jnp.ones((nr, nz)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        d_rho, d_mom, d_E = model.compute_flux_divergence(state, geometry)

        assert d_rho.shape == (nr, nz)
        assert d_mom.shape == (nr, nz, 3)
        assert d_E.shape == (nr, nz)

    def test_uniform_state_zero_flux_divergence(self):
        """Uniform stationary state has ~zero flux divergence."""
        from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState

        nr, nz = 16, 32
        model = NeutralFluid()
        geometry = Geometry(nr=nr, nz=nz, r_max=1.0, z_max=2.0)

        # Uniform stationary state
        rho_n = jnp.ones((nr, nz)) * 1e-6
        mom_n = jnp.zeros((nr, nz, 3))
        p_n = 1e3
        E_n = jnp.ones((nr, nz)) * p_n / (5/3 - 1)
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        d_rho, d_mom, d_E = model.compute_flux_divergence(state, geometry)

        # Interior should be near zero (boundaries may have edge effects)
        assert jnp.max(jnp.abs(d_rho[2:-2, 2:-2])) < 1e-10
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_neutral_fluid.py::TestNeutralFluid -v`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# Add to jax_frc/models/neutral_fluid.py

from jax_frc.core.geometry import Geometry


@dataclass
class NeutralFluid:
    """Hydrodynamic neutral fluid model.

    Solves Euler equations with optional atomic source terms.
    """

    gamma: float = GAMMA

    def compute_flux_divergence(
        self, state: NeutralState, geometry: Geometry
    ) -> Tuple[Array, Array, Array]:
        """Compute -div(F) for Euler equations using HLLE.

        Args:
            state: Current neutral state
            geometry: Grid geometry

        Returns:
            d_rho: Mass density RHS [kg/m³/s]
            d_mom: Momentum density RHS [kg/m²/s²]
            d_E: Energy density RHS [W/m³]
        """
        dr, dz = geometry.dr, geometry.dz
        rho = state.rho_n
        v = state.v_n
        p = state.p_n
        E = state.E_n

        # Radial fluxes (r-direction, axis=0)
        F_r = self._compute_radial_flux(rho, v, p, E)

        # Axial fluxes (z-direction, axis=1)
        F_z = self._compute_axial_flux(rho, v, p, E)

        # Flux divergence: -d(F_r)/dr - d(F_z)/dz
        # Using central differences for divergence
        d_rho = -(
            (jnp.roll(F_r[0], -1, axis=0) - jnp.roll(F_r[0], 1, axis=0)) / (2 * dr) +
            (jnp.roll(F_z[0], -1, axis=1) - jnp.roll(F_z[0], 1, axis=1)) / (2 * dz)
        )

        # Momentum: handle each component
        d_mom_r = -(
            (jnp.roll(F_r[1], -1, axis=0) - jnp.roll(F_r[1], 1, axis=0)) / (2 * dr) +
            (jnp.roll(F_z[1], -1, axis=1) - jnp.roll(F_z[1], 1, axis=1)) / (2 * dz)
        )
        d_mom_theta = jnp.zeros_like(d_mom_r)  # No theta flux in axisymmetric
        d_mom_z = -(
            (jnp.roll(F_r[2], -1, axis=0) - jnp.roll(F_r[2], 1, axis=0)) / (2 * dr) +
            (jnp.roll(F_z[2], -1, axis=1) - jnp.roll(F_z[2], 1, axis=1)) / (2 * dz)
        )
        d_mom = jnp.stack([d_mom_r, d_mom_theta, d_mom_z], axis=-1)

        d_E = -(
            (jnp.roll(F_r[3], -1, axis=0) - jnp.roll(F_r[3], 1, axis=0)) / (2 * dr) +
            (jnp.roll(F_z[3], -1, axis=1) - jnp.roll(F_z[3], 1, axis=1)) / (2 * dz)
        )

        return d_rho, d_mom, d_E

    def _compute_radial_flux(self, rho, v, p, E):
        """Compute HLLE flux in r-direction at cell faces."""
        v_r = v[..., 0]  # Radial velocity

        # Left and right states (i-1/2 interface uses i-1 and i)
        rho_L = jnp.roll(rho, 1, axis=0)
        rho_R = rho
        v_L = jnp.roll(v_r, 1, axis=0)
        v_R = v_r
        p_L = jnp.roll(p, 1, axis=0)
        p_R = p
        E_L = jnp.roll(E, 1, axis=0)
        E_R = E

        F_rho, F_mom_r, F_E = hlle_flux_1d(
            rho_L, rho_R, v_L, v_R, p_L, p_R, E_L, E_R, self.gamma
        )

        # For momentum components perpendicular to flux direction,
        # flux is rho * v_r * v_perp
        mom_theta_L = jnp.roll(rho * v[..., 1], 1, axis=0)
        mom_theta_R = rho * v[..., 1]
        F_mom_theta = jnp.where(
            v_L + v_R > 0,
            v_L * mom_theta_L / jnp.maximum(rho_L, 1e-20),
            v_R * mom_theta_R / jnp.maximum(rho_R, 1e-20)
        ) * 0.5 * (rho_L + rho_R)

        mom_z_L = jnp.roll(rho * v[..., 2], 1, axis=0)
        mom_z_R = rho * v[..., 2]
        F_mom_z = jnp.where(
            v_L + v_R > 0,
            v_L * mom_z_L / jnp.maximum(rho_L, 1e-20),
            v_R * mom_z_R / jnp.maximum(rho_R, 1e-20)
        ) * 0.5 * (rho_L + rho_R)

        return (F_rho, F_mom_r, F_mom_z, F_E)

    def _compute_axial_flux(self, rho, v, p, E):
        """Compute HLLE flux in z-direction at cell faces."""
        v_z = v[..., 2]  # Axial velocity

        # Left and right states
        rho_L = jnp.roll(rho, 1, axis=1)
        rho_R = rho
        v_L = jnp.roll(v_z, 1, axis=1)
        v_R = v_z
        p_L = jnp.roll(p, 1, axis=1)
        p_R = p
        E_L = jnp.roll(E, 1, axis=1)
        E_R = E

        F_rho, F_mom_z, F_E = hlle_flux_1d(
            rho_L, rho_R, v_L, v_R, p_L, p_R, E_L, E_R, self.gamma
        )

        # Perpendicular momentum fluxes
        mom_r_L = jnp.roll(rho * v[..., 0], 1, axis=1)
        mom_r_R = rho * v[..., 0]
        F_mom_r = jnp.where(
            v_L + v_R > 0,
            v_L * mom_r_L / jnp.maximum(rho_L, 1e-20),
            v_R * mom_r_R / jnp.maximum(rho_R, 1e-20)
        ) * 0.5 * (rho_L + rho_R)

        return (F_rho, F_mom_r, F_mom_z, F_E)
```

**Step 4: Run test to verify it passes**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_neutral_fluid.py::TestNeutralFluid -v`
Expected: PASS (4 tests)

**Step 5: Commit**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add jax_frc/models/neutral_fluid.py tests/test_neutral_fluid.py
git commit -m "feat(neutral): add NeutralFluid model with HLLE flux divergence"
```

---

## Task 10: Neutral Boundary Conditions

**Files:**
- Modify: `jax_frc/models/neutral_fluid.py`
- Test: `tests/test_neutral_fluid.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_neutral_fluid.py

class TestNeutralBoundaryConditions:
    """Tests for neutral boundary conditions."""

    def test_apply_boundary_conditions_exists(self):
        """Method exists on NeutralFluid."""
        from jax_frc.models.neutral_fluid import NeutralFluid
        model = NeutralFluid()
        assert hasattr(model, 'apply_boundary_conditions')

    def test_reflecting_bc_reverses_normal_velocity(self):
        """Reflecting BC reverses velocity normal to wall."""
        from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState

        nr, nz = 16, 32
        model = NeutralFluid()
        geometry = Geometry(nr=nr, nz=nz, r_max=1.0, z_max=2.0)

        rho_n = jnp.ones((nr, nz)) * 1e-6
        # Velocity pointing outward at outer r boundary
        mom_n = jnp.zeros((nr, nz, 3))
        mom_n = mom_n.at[-1, :, 0].set(1e-6 * 100)  # v_r = 100 at outer wall
        E_n = jnp.ones((nr, nz)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        state_bc = model.apply_boundary_conditions(state, geometry, bc_type="reflecting")

        # v_r should be zero or reversed at outer boundary
        assert jnp.all(state_bc.mom_n[-1, :, 0] <= 0)

    def test_axis_symmetry(self):
        """Axis (r=0) has correct symmetry."""
        from jax_frc.models.neutral_fluid import NeutralFluid, NeutralState

        nr, nz = 16, 32
        model = NeutralFluid()
        geometry = Geometry(nr=nr, nz=nz, r_max=1.0, z_max=2.0)

        rho_n = jnp.ones((nr, nz)) * 1e-6
        mom_n = jnp.ones((nr, nz, 3)) * 1e-6 * 100
        E_n = jnp.ones((nr, nz)) * 1e3
        state = NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)

        state_bc = model.apply_boundary_conditions(state, geometry)

        # v_r = 0 at axis
        assert jnp.allclose(state_bc.mom_n[0, :, 0], 0, atol=1e-20)
```

**Step 2: Run test to verify it fails**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_neutral_fluid.py::TestNeutralBoundaryConditions -v`
Expected: FAIL with "AttributeError"

**Step 3: Write minimal implementation**

```python
# Add to NeutralFluid class in jax_frc/models/neutral_fluid.py

    def apply_boundary_conditions(
        self, state: NeutralState, geometry: Geometry, bc_type: str = "reflecting"
    ) -> NeutralState:
        """Apply boundary conditions to neutral state.

        Args:
            state: Current neutral state
            geometry: Grid geometry
            bc_type: "reflecting" or "absorbing"

        Returns:
            State with boundary conditions applied
        """
        rho_n = state.rho_n
        mom_n = state.mom_n
        E_n = state.E_n

        # Axis (r=0): symmetry
        # v_r = 0, v_theta = 0, scalars have zero gradient
        mom_n = mom_n.at[0, :, 0].set(0.0)  # v_r = 0
        mom_n = mom_n.at[0, :, 1].set(0.0)  # v_theta = 0
        rho_n = rho_n.at[0, :].set(rho_n[1, :])  # Neumann
        E_n = E_n.at[0, :].set(E_n[1, :])

        if bc_type == "reflecting":
            # Outer radial wall: reflect v_r
            mom_n = mom_n.at[-1, :, 0].set(-mom_n[-2, :, 0])
            rho_n = rho_n.at[-1, :].set(rho_n[-2, :])
            E_n = E_n.at[-1, :].set(E_n[-2, :])

            # Axial walls: reflect v_z
            mom_n = mom_n.at[:, 0, 2].set(-mom_n[:, 1, 2])
            mom_n = mom_n.at[:, -1, 2].set(-mom_n[:, -2, 2])
            rho_n = rho_n.at[:, 0].set(rho_n[:, 1])
            rho_n = rho_n.at[:, -1].set(rho_n[:, -2])
            E_n = E_n.at[:, 0].set(E_n[:, 1])
            E_n = E_n.at[:, -1].set(E_n[:, -2])

        elif bc_type == "absorbing":
            # Outflow: zero gradient
            rho_n = rho_n.at[-1, :].set(rho_n[-2, :])
            mom_n = mom_n.at[-1, :, :].set(mom_n[-2, :, :])
            E_n = E_n.at[-1, :].set(E_n[-2, :])

            rho_n = rho_n.at[:, 0].set(rho_n[:, 1])
            rho_n = rho_n.at[:, -1].set(rho_n[:, -2])
            mom_n = mom_n.at[:, 0, :].set(mom_n[:, 1, :])
            mom_n = mom_n.at[:, -1, :].set(mom_n[:, -2, :])
            E_n = E_n.at[:, 0].set(E_n[:, 1])
            E_n = E_n.at[:, -1].set(E_n[:, -2])

        return NeutralState(rho_n=rho_n, mom_n=mom_n, E_n=E_n)
```

**Step 4: Run test to verify it passes**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_neutral_fluid.py::TestNeutralBoundaryConditions -v`
Expected: PASS (3 tests)

**Step 5: Commit**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add jax_frc/models/neutral_fluid.py tests/test_neutral_fluid.py
git commit -m "feat(neutral): add boundary conditions (reflecting, absorbing)"
```

---

## Task 11: Run All Tests

**Step 1: Run full test suite**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/test_atomic_rates.py tests/test_neutral_fluid.py -v`
Expected: All tests PASS

**Step 2: Run existing tests to verify no regressions**

Run: `cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling && py -m pytest tests/ -v --tb=short`
Expected: All 204+ tests PASS

**Step 3: Commit checkpoint**

```bash
cd C:\Users\周光裕\jax-frc\.worktrees\neutral-coupling
git add -A
git commit -m "checkpoint: atomic rates and neutral fluid modules complete"
```

---

## Remaining Tasks (Summary)

Tasks 12-20 will complete the implementation:

12. **CoupledState dataclass** - Combine plasma State + NeutralState
13. **ResistiveMHDWithNeutrals class** - Extend ResistiveMHD with coupling
14. **Atomic source term wiring** - Connect rates to plasma/neutral equations
15. **Operator splitting solver** - Strang splitting for stiff sources
16. **Mass conservation test** - Verify total mass conserved
17. **Momentum conservation test** - Verify Newton's 3rd law
18. **Ionization front validation** - Analytic test case
19. **Radiative cooling validation** - Analytic test case
20. **Integration: FRC formation** - End-to-end test with neutral burnout

Each follows the same TDD pattern: failing test → minimal implementation → verify → commit.

---

## Commands Reference

```bash
# Run atomic rates tests
py -m pytest tests/test_atomic_rates.py -v

# Run neutral fluid tests
py -m pytest tests/test_neutral_fluid.py -v

# Run all new tests
py -m pytest tests/test_atomic_rates.py tests/test_neutral_fluid.py -v

# Run full test suite
py -m pytest tests/ -v

# Check for regressions
py -m pytest tests/ -v --tb=short
```
