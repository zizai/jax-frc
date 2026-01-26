"""Tests for electromagnetic coil field models."""

import pytest
import jax.numpy as jnp
from jax_frc.fields.coils import Solenoid, MirrorCoil, ThetaPinchArray


class TestSolenoid:
    """Tests for ideal solenoid field model."""

    def test_center_field_strength(self):
        """B_z at center approximates mu0 * n * I with finite solenoid correction."""
        length = 2.0  # m
        radius = 0.1  # m
        n_turns = 100
        current = 10.0  # A

        solenoid = Solenoid(length=length, radius=radius, n_turns=n_turns, current=current)

        # Field at center (r=0, z=0)
        r = jnp.array([0.0])
        z = jnp.array([0.0])
        B_r, B_z = solenoid.B_field(r, z, t=0.0)

        # Expected: B = mu0 * n * I * (end correction factor)
        # For finite solenoid: B_z(center) = 0.5 * B0 * (cos_plus - cos_minus)
        # where cos terms account for the finite length
        MU0 = 1.25663706212e-6
        n_density = n_turns / length  # turns per meter
        B0 = MU0 * n_density * current  # infinite solenoid field

        # End correction factor for center of finite solenoid
        half_len = length / 2
        denom = jnp.sqrt(radius**2 + half_len**2)
        cos_factor = half_len / denom
        expected_B = B0 * cos_factor  # = 0.5 * B0 * (cos_plus - cos_minus) at z=0

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

        z_points = jnp.linspace(-1.5, 1.5, 31)
        # Use vectorized operations - MirrorCoil.B_field accepts arrays
        r_points = jnp.zeros_like(z_points)

        _, B_z1 = coil1.B_field(r_points, z_points, t=0.0)
        _, B_z2 = coil2.B_field(r_points, z_points, t=0.0)
        B_z_total = B_z1 + B_z2

        # Field at center should be local minimum
        center_idx = 15
        assert B_z_total[center_idx] < B_z_total[center_idx - 5]
        assert B_z_total[center_idx] < B_z_total[center_idx + 5]

    def test_vector_potential(self):
        """A_phi returns reasonable values and is zero on axis."""
        coil = MirrorCoil(z_position=0.0, radius=0.5, current=1000.0)

        # On axis, A_phi should be zero by symmetry
        r_axis = jnp.array([0.0, 0.0, 0.0])
        z_axis = jnp.array([-0.5, 0.0, 0.5])
        A_axis = coil.A_phi(r_axis, z_axis, t=0.0)
        assert jnp.allclose(A_axis, 0.0, atol=1e-10)

        # Off axis, A_phi should be non-zero
        r_off = jnp.array([0.1, 0.2, 0.3])
        z_off = jnp.array([0.0, 0.0, 0.0])
        A_off = coil.A_phi(r_off, z_off, t=0.0)
        assert jnp.all(A_off != 0.0)

        # A_phi should be positive for positive current (standard convention)
        assert jnp.all(A_off > 0.0)


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
        z = jnp.array([0.25])  # Sample closer to first coil to break symmetry

        # At t=0: only second coil active
        _, B_z_t0 = array.B_field(r, z, t=0.0)

        # At t=1: only first coil active
        _, B_z_t1 = array.B_field(r, z, t=1.0)

        # Fields should be different (closer to first coil, so t=1 field stronger)
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
        assert B_z[-1] > B_z[0]
