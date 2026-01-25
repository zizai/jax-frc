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


class AxisRegularity(Invariant):
    """Check that fields are finite and regular at the axis (r=0).

    For cylindrical coordinates, certain quantities must be well-behaved at the axis:
    - B_r(r=0) = 0 (by symmetry)
    - B_theta(r=0) = 0 (by symmetry)
    - All fields should be finite
    """

    def __init__(self, atol: float = 1e-6):
        self.atol = atol

    @property
    def name(self) -> str:
        return "AxisRegularity"

    def check(self, state_before, state_after) -> InvariantResult:
        # state_after should be a field array (nr, nz) or (nr, nz, 3)
        field = state_after

        # Check for NaN/Inf
        has_nan = jnp.any(jnp.isnan(field))
        has_inf = jnp.any(jnp.isinf(field))

        if has_nan or has_inf:
            return InvariantResult(
                passed=False,
                name=self.name,
                value=1.0,
                tolerance=0.0,
                message="Field contains NaN or Inf"
            )

        # Check axis values (first radial index)
        if field.ndim == 3:
            # Vector field: B_r and B_theta should be zero at axis
            b_r_axis = jnp.max(jnp.abs(field[0, :, 0]))
            b_theta_axis = jnp.max(jnp.abs(field[0, :, 1]))
            max_axis_violation = max(float(b_r_axis), float(b_theta_axis))
        else:
            # Scalar field: just check it's finite
            max_axis_violation = 0.0

        passed = max_axis_violation <= self.atol

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=max_axis_violation,
            tolerance=self.atol,
            message=f"Axis regularity violation: {max_axis_violation:.2e}"
        )


class CylindricalCurlTest(Invariant):
    """Verify J = curl(B)/mu0 matches analytical for a test field.

    For B_theta = r^2, the exact J_z = (1/r)*d(r*B_theta)/dr = (1/r)*d(r^3)/dr = 3r
    """

    def __init__(self, atol: float = 0.1, dr: float = 0.01):
        self.atol = atol
        self.dr = dr

    @property
    def name(self) -> str:
        return "CylindricalCurlTest"

    def check(self, state_before, state_after) -> InvariantResult:
        # state_after should be (J_z_numerical, J_z_analytical, r)
        j_z_num, j_z_exact, r = state_after

        # Compute relative error, avoiding division by zero
        denominator = jnp.maximum(jnp.abs(j_z_exact), 1e-10)
        relative_error = jnp.abs(j_z_num - j_z_exact) / denominator

        # Mask out axis region where r is very small
        mask = r.flatten() > 2 * self.dr
        if jnp.any(mask):
            max_rel_error = float(jnp.max(relative_error.flatten()[mask]))
        else:
            max_rel_error = 0.0

        passed = max_rel_error <= self.atol

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=max_rel_error,
            tolerance=self.atol,
            message=f"Max relative error in J_z: {max_rel_error:.2e}"
        )


class VelocityEvolution(Invariant):
    """Check that velocity field evolves over time (not static)."""

    def __init__(self, min_change: float = 1e-10):
        self.min_change = min_change

    @property
    def name(self) -> str:
        return "VelocityEvolution"

    def check(self, state_before, state_after) -> InvariantResult:
        # state_before and state_after are velocity fields
        v_before = state_before
        v_after = state_after

        max_change = float(jnp.max(jnp.abs(v_after - v_before)))
        passed = max_change > self.min_change

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=max_change,
            tolerance=self.min_change,
            message=f"Velocity change: {max_change:.2e}"
        )


class FieldEvolution(Invariant):
    """Check that field evolves over time (not static)."""

    def __init__(self, min_change: float = 1e-10, field_name: str = "field"):
        self.min_change = min_change
        self.field_name = field_name

    @property
    def name(self) -> str:
        return f"FieldEvolution({self.field_name})"

    def check(self, state_before, state_after) -> InvariantResult:
        f_before = state_before
        f_after = state_after

        max_change = float(jnp.max(jnp.abs(f_after - f_before)))
        passed = max_change > self.min_change

        return InvariantResult(
            passed=passed,
            name=self.name,
            value=max_change,
            tolerance=self.min_change,
            message=f"{self.field_name} change: {max_change:.2e}"
        )
