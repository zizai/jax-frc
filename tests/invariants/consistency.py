"""Consistency invariants - mathematical relationships that must hold."""
import jax.numpy as jnp
from tests.invariants import Invariant, InvariantResult

class DivergenceFreeB(Invariant):
    """Magnetic field should be divergence-free: div(B) = 0."""

    def __init__(self, atol: float = 1e-6, dx: float = 1.0, dy: float = 1.0):
        self.atol = atol
        self.dx = dx
        self.dy = dy

    @property
    def name(self) -> str:
        return "DivergenceFreeB"

    def check(self, state_before, state_after) -> InvariantResult:
        # Expects state_after to be (b_x, b_y, b_z) tuple or stacked array
        if isinstance(state_after, tuple):
            b_x, b_y, b_z = state_after
        else:
            b_x, b_y, b_z = state_after[0], state_after[1], state_after[2]

        # Compute div(B) = dBx/dx + dBy/dy (2D)
        dbx_dx = (jnp.roll(b_x, -1, axis=0) - jnp.roll(b_x, 1, axis=0)) / (2 * self.dx)
        dby_dy = (jnp.roll(b_y, -1, axis=1) - jnp.roll(b_y, 1, axis=1)) / (2 * self.dy)

        div_b = dbx_dx + dby_dy
        max_div = float(jnp.max(jnp.abs(div_b)))

        passed = max_div <= self.atol

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=max_div,
            tolerance=self.atol,
            message=f"max|div(B)| = {max_div:.2e}"
        )

class WeightBounds(Invariant):
    """Particle weights must stay in [-1, 1] for delta-f method."""

    @property
    def name(self) -> str:
        return "WeightBounds"

    def check(self, state_before, state_after) -> InvariantResult:
        # Expects state_after to be weights array
        w = state_after
        min_w = float(jnp.min(w))
        max_w = float(jnp.max(w))

        in_bounds = (min_w >= -1.0) and (max_w <= 1.0)
        violation = max(0, -1.0 - min_w, max_w - 1.0)

        return InvariantResult(
            passed=in_bounds,
            name=self.name,
            value=violation,
            tolerance=0.0,
            message=f"Weight range: [{min_w:.4f}, {max_w:.4f}]"
        )

class DistributionPositivity(Invariant):
    """Distribution f = f0(1+w) should be positive for most particles."""

    def __init__(self, min_fraction: float = 0.99):
        self.min_fraction = min_fraction

    @property
    def name(self) -> str:
        return "DistributionPositivity"

    def check(self, state_before, state_after) -> InvariantResult:
        # Expects state_after to be (f0, w) tuple
        f0, w = state_after
        f = f0 * (1 + w)

        n_positive = jnp.sum(f > 0)
        n_total = f.size
        fraction_positive = float(n_positive / n_total)

        passed = fraction_positive >= self.min_fraction

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=1.0 - fraction_positive,
            tolerance=1.0 - self.min_fraction,
            message=f"Positive fraction: {fraction_positive:.4f} (need {self.min_fraction:.4f})"
        )

class ResistivityBounds(Invariant):
    """Resistivity should stay within physical bounds."""

    def __init__(self, eta_min: float = 0.0, eta_max: float = 1.0):
        self.eta_min = eta_min
        self.eta_max = eta_max

    @property
    def name(self) -> str:
        return "ResistivityBounds"

    def check(self, state_before, state_after) -> InvariantResult:
        # Expects state_after to be eta array
        eta = state_after
        actual_min = float(jnp.min(eta))
        actual_max = float(jnp.max(eta))

        in_bounds = (actual_min >= self.eta_min) and (actual_max <= self.eta_max)
        violation = max(0, self.eta_min - actual_min, actual_max - self.eta_max)

        return InvariantResult(
            passed=in_bounds,
            name=self.name,
            value=violation,
            tolerance=0.0,
            message=f"eta range: [{actual_min:.2e}, {actual_max:.2e}]"
        )
