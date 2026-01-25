"""Boundedness invariants - values must stay within physical/numerical limits."""
import jax.numpy as jnp
from tests.invariants import Invariant, InvariantResult

class FiniteValues(Invariant):
    """All values in state must be finite (no NaN or Inf)."""

    def __init__(self, field_name: str = "field"):
        self.field_name = field_name

    @property
    def name(self) -> str:
        return f"FiniteValues({self.field_name})"

    def check(self, state_before, state_after) -> InvariantResult:
        # state_after is the field array to check
        is_finite = jnp.all(jnp.isfinite(state_after))
        num_nonfinite = jnp.sum(~jnp.isfinite(state_after))

        return InvariantResult(
            passed=bool(is_finite),
            name=self.name,
            value=float(num_nonfinite),
            tolerance=0.0,
            message=f"Non-finite values: {int(num_nonfinite)}"
        )

class PositiveValues(Invariant):
    """All values must be positive (e.g., density, temperature)."""

    def __init__(self, field_name: str = "field", allow_zero: bool = True):
        self.field_name = field_name
        self.allow_zero = allow_zero

    @property
    def name(self) -> str:
        return f"PositiveValues({self.field_name})"

    def check(self, state_before, state_after) -> InvariantResult:
        if self.allow_zero:
            is_positive = jnp.all(state_after >= 0)
            num_negative = jnp.sum(state_after < 0)
        else:
            is_positive = jnp.all(state_after > 0)
            num_negative = jnp.sum(state_after <= 0)

        return InvariantResult(
            passed=bool(is_positive),
            name=self.name,
            value=float(num_negative),
            tolerance=0.0,
            message=f"Non-positive values: {int(num_negative)}, min={float(jnp.min(state_after)):.2e}"
        )

class BoundedRange(Invariant):
    """Values must stay within [min_val, max_val]."""

    def __init__(self, field_name: str, min_val: float, max_val: float):
        self.field_name = field_name
        self.min_val = min_val
        self.max_val = max_val

    @property
    def name(self) -> str:
        return f"BoundedRange({self.field_name})"

    def check(self, state_before, state_after) -> InvariantResult:
        actual_min = float(jnp.min(state_after))
        actual_max = float(jnp.max(state_after))
        in_range = (actual_min >= self.min_val) and (actual_max <= self.max_val)

        return InvariantResult(
            passed=in_range,
            name=self.name,
            value=max(self.min_val - actual_min, actual_max - self.max_val, 0),
            tolerance=0.0,
            message=f"Range [{actual_min:.2e}, {actual_max:.2e}], expected [{self.min_val:.2e}, {self.max_val:.2e}]"
        )

class NoExponentialGrowth(Invariant):
    """Field magnitude should not grow more than growth_factor per step."""

    def __init__(self, field_name: str, growth_factor: float = 2.0):
        self.field_name = field_name
        self.growth_factor = growth_factor

    @property
    def name(self) -> str:
        return f"NoExponentialGrowth({self.field_name})"

    def check(self, state_before, state_after) -> InvariantResult:
        mag_before = float(jnp.max(jnp.abs(state_before)))
        mag_after = float(jnp.max(jnp.abs(state_after)))

        # Avoid division by zero
        if mag_before < 1e-15:
            ratio = 1.0 if mag_after < 1e-15 else float('inf')
        else:
            ratio = mag_after / mag_before

        passed = ratio <= self.growth_factor

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=ratio,
            tolerance=self.growth_factor,
            message=f"Growth ratio: {ratio:.2e}, limit: {self.growth_factor:.2e}"
        )
