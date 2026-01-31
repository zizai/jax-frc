"""Tests for Hall-MHD wave speed utilities."""

import jax.numpy as jnp

from jax_frc.solvers.riemann.wave_speeds import fast_magnetosonic_speed, hall_signal_speed


def test_hall_signal_speed_includes_whistler():
    """Hall signal speed should include whistler contribution."""
    rho = jnp.array(1.0)
    p = jnp.array(1.0)
    Bx = jnp.array(1.0)
    By = jnp.array(1.0)
    Bz = jnp.array(1.0)
    gamma = 5.0 / 3.0
    hall_scale = 1.0
    cell_size = 0.2

    cf = fast_magnetosonic_speed(rho, p, Bx, By, Bz, gamma)
    b_mag = Bx**2 + By**2 + Bz**2
    inv_rho = 1.0 / rho
    w_alf_sq = b_mag * inv_rho
    w_whistler = jnp.pi * hall_scale * jnp.sqrt(b_mag) * inv_rho / cell_size
    w_whistler = 0.5 * w_whistler + jnp.sqrt((0.5 * w_whistler) ** 2 + w_alf_sq)
    expected = jnp.maximum(cf, w_whistler)

    actual = hall_signal_speed(rho, p, Bx, By, Bz, cell_size, gamma, hall_scale)
    assert jnp.allclose(actual, expected, atol=1e-10)
