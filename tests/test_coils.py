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
