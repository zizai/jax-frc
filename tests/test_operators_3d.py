"""Tests for 3D Cartesian differential operators."""

import jax.numpy as jnp
import pytest
from jax_frc.operators import gradient_3d, divergence_3d
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
        """Gradient of f = x should be (1, 0, 0) in interior."""
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
        # Note: boundary cells affected by periodic wrap for non-periodic f=x
        # Check interior points only (exclude first and last in x)
        assert jnp.allclose(grad_f[1:-1, :, :, 0], 1.0, atol=1e-6)
        assert jnp.allclose(grad_f[..., 1], 0.0, atol=1e-10)
        assert jnp.allclose(grad_f[..., 2], 0.0, atol=1e-10)

    def test_gradient_linear_y(self):
        """Gradient of f = y should be (0, 1, 0) in interior."""
        geom = Geometry(
            nx=8, ny=16, nz=8,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
        )
        f = geom.y_grid  # f = y
        grad_f = gradient_3d(f, geom)
        assert jnp.allclose(grad_f[..., 0], 0.0, atol=1e-10)
        assert jnp.allclose(grad_f[:, 1:-1, :, 1], 1.0, atol=1e-6)
        assert jnp.allclose(grad_f[..., 2], 0.0, atol=1e-10)

    def test_gradient_linear_z(self):
        """Gradient of f = z should be (0, 0, 1) in interior."""
        geom = Geometry(
            nx=8, ny=8, nz=16,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
        )
        f = geom.z_grid  # f = z
        grad_f = gradient_3d(f, geom)
        assert jnp.allclose(grad_f[..., 0], 0.0, atol=1e-10)
        assert jnp.allclose(grad_f[..., 1], 0.0, atol=1e-10)
        assert jnp.allclose(grad_f[:, :, 1:-1, 2], 1.0, atol=1e-6)

    def test_gradient_shape(self):
        """Gradient output shape is (nx, ny, nz, 3)."""
        f = jnp.ones((4, 6, 8))
        geom = Geometry(nx=4, ny=6, nz=8)
        grad_f = gradient_3d(f, geom)
        assert grad_f.shape == (4, 6, 8, 3)


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
        # Note: boundary cells affected by periodic wrap for non-periodic F=(x,y,z)
        # Check interior points only (exclude first and last in each dimension)
        assert jnp.allclose(div_F[1:-1, 1:-1, 1:-1], 3.0, atol=1e-6)

    def test_divergence_solenoidal(self):
        """Divergence of solenoidal field should be zero."""
        geom = Geometry(nx=16, ny=16, nz=16)
        # Create a solenoidal field: F = (-y, x, 0)
        F = jnp.stack([-geom.y_grid, geom.x_grid, jnp.zeros_like(geom.x_grid)], axis=-1)
        div_F = divergence_3d(F, geom)
        assert jnp.allclose(div_F, 0.0, atol=1e-10)
