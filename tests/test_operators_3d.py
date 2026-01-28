"""Tests for 3D Cartesian differential operators."""

import jax.numpy as jnp
import pytest
from jax_frc.operators import gradient_3d, divergence_3d, curl_3d, laplacian_3d
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
        # Check interior points only (exclude first/last two in x for 4th-order stencil)
        assert jnp.allclose(grad_f[2:-2, :, :, 0], 1.0, atol=1e-6)
        assert jnp.allclose(grad_f[..., 1], 0.0, atol=1e-7)
        assert jnp.allclose(grad_f[..., 2], 0.0, atol=1e-7)

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
        assert jnp.allclose(grad_f[..., 0], 0.0, atol=1e-7)
        # Exclude first/last two in y for 4th-order stencil
        assert jnp.allclose(grad_f[:, 2:-2, :, 1], 1.0, atol=1e-6)
        assert jnp.allclose(grad_f[..., 2], 0.0, atol=1e-7)

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
        assert jnp.allclose(grad_f[..., 0], 0.0, atol=1e-7)
        assert jnp.allclose(grad_f[..., 1], 0.0, atol=1e-7)
        assert jnp.allclose(grad_f[:, :, 1:-1, 2], 1.0, atol=1e-6)

    def test_gradient_shape(self):
        """Gradient output shape is (nx, ny, nz, 3)."""
        f = jnp.ones((4, 6, 8))
        geom = Geometry(nx=4, ny=6, nz=8)
        grad_f = gradient_3d(f, geom)
        assert grad_f.shape == (4, 6, 8, 3)

    def test_gradient_dirichlet_bc(self):
        """Gradient with Dirichlet BC uses one-sided differences at boundaries."""
        geom = Geometry(
            nx=16, ny=8, nz=8,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
            bc_x="dirichlet", bc_y="periodic", bc_z="periodic"
        )
        f = geom.x_grid  # f = x
        grad_f = gradient_3d(f, geom)
        # With Dirichlet BC, boundary gradient should use one-sided difference
        # df/dx at x=0 should be (f[1] - f[0]) / dx (forward difference)
        # df/dx at x=max should be (f[-1] - f[-2]) / dx (backward difference)
        assert jnp.allclose(grad_f[0, :, :, 0], 1.0, atol=0.1)  # Forward diff at left
        assert jnp.allclose(grad_f[-1, :, :, 0], 1.0, atol=0.1)  # Backward diff at right


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
        # Check interior points only (exclude first/last two in each dimension)
        assert jnp.allclose(div_F[2:-2, 2:-2, 2:-2], 3.0, atol=1e-6)

    def test_divergence_solenoidal(self):
        """Divergence of solenoidal field should be zero."""
        geom = Geometry(nx=16, ny=16, nz=16)
        # Create a solenoidal field: F = (-y, x, 0)
        F = jnp.stack([-geom.y_grid, geom.x_grid, jnp.zeros_like(geom.x_grid)], axis=-1)
        div_F = divergence_3d(F, geom)
        assert jnp.allclose(div_F, 0.0, atol=1e-10)

    def test_divergence_neumann_zero_gradient(self):
        """Divergence of constant field should be zero with Neumann BCs."""
        geom = Geometry(nx=8, ny=8, nz=8, bc_x="neumann", bc_y="neumann", bc_z="neumann")
        F = jnp.zeros((8, 8, 8, 3))
        F = F.at[..., 0].set(1.0)
        div_F = divergence_3d(F, geom)
        assert jnp.allclose(div_F, 0.0, atol=1e-6)

    def test_divergence_dirichlet_sine_boundary(self):
        """Dirichlet BCs should match analytic derivative at boundaries."""
        geom = Geometry(
            nx=8, ny=8, nz=8,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
            bc_x="dirichlet", bc_y="dirichlet", bc_z="dirichlet"
        )
        F = jnp.zeros((8, 8, 8, 3))
        L = geom.x_max - geom.x_min
        F = F.at[..., 0].set(jnp.sin(jnp.pi * geom.x_grid / L))
        div_F = divergence_3d(F, geom)
        expected = (jnp.pi / L) * jnp.cos(jnp.pi * geom.x_grid / L)
        assert jnp.allclose(div_F[0, :, :], expected[0, :, :], atol=1e-2)
        assert jnp.allclose(div_F[-1, :, :], expected[-1, :, :], atol=1e-2)


class TestCurl3D:
    """Test 3D curl operator."""

    def test_curl_constant_field(self):
        """Curl of constant vector field should be zero."""
        F = jnp.ones((8, 8, 8, 3)) * 5.0
        geom = Geometry(nx=8, ny=8, nz=8)
        curl_F = curl_3d(F, geom)
        assert curl_F.shape == (8, 8, 8, 3)
        assert jnp.allclose(curl_F, 0.0, atol=1e-10)

    def test_curl_gradient_is_zero(self):
        """Curl of gradient should be zero."""
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
        )
        # f = x^2 + y^2
        f = geom.x_grid**2 + geom.y_grid**2
        grad_f = gradient_3d(f, geom)
        curl_grad_f = curl_3d(grad_f, geom)
        # Check interior points (boundary is affected by periodic wrap on non-periodic function)
        assert jnp.allclose(curl_grad_f[1:-1, 1:-1, :], 0.0, atol=1e-6)

    def test_curl_simple_field(self):
        """Curl of F = (-y, x, 0) should be (0, 0, 2)."""
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        F = jnp.stack([-geom.y_grid, geom.x_grid, jnp.zeros_like(geom.x_grid)], axis=-1)
        curl_F = curl_3d(F, geom)
        # curl(-y, x, 0) = (0 - 0, 0 - 0, 1 - (-1)) = (0, 0, 2)
        # Note: boundary cells affected by periodic wrap for non-periodic F=(-y, x, 0)
        # Check interior points only (exclude first/last two in x and y for 4th-order stencil)
        assert jnp.allclose(curl_F[2:-2, 2:-2, :, 0], 0.0, atol=1e-7)
        assert jnp.allclose(curl_F[2:-2, 2:-2, :, 1], 0.0, atol=1e-7)
        assert jnp.allclose(curl_F[2:-2, 2:-2, :, 2], 2.0, atol=1e-6)

    def test_curl_dirichlet_zero_boundary(self):
        """Dirichlet BCs should match analytic derivative at boundaries."""
        geom = Geometry(
            nx=8, ny=8, nz=8,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
            bc_x="dirichlet", bc_y="dirichlet", bc_z="dirichlet"
        )
        F = jnp.zeros((8, 8, 8, 3))
        L = geom.x_max - geom.x_min
        F = F.at[..., 1].set(jnp.sin(jnp.pi * geom.x_grid / L))
        curl_F = curl_3d(F, geom)
        expected = (jnp.pi / L) * jnp.cos(jnp.pi * geom.x_grid / L)
        assert jnp.allclose(curl_F[0, :, :, 2], expected[0, :, :], atol=1e-2)
        assert jnp.allclose(curl_F[-1, :, :, 2], expected[-1, :, :], atol=1e-2)


class TestLaplacian3D:
    """Test 3D Laplacian operator."""

    def test_laplacian_constant(self):
        """Laplacian of constant should be zero."""
        f = jnp.ones((8, 8, 8)) * 7.0
        geom = Geometry(nx=8, ny=8, nz=8, bc_x="periodic", bc_y="periodic", bc_z="periodic")
        lap_f = laplacian_3d(f, geom)
        assert lap_f.shape == (8, 8, 8)
        assert jnp.allclose(lap_f, 0.0, atol=1e-10)

    def test_laplacian_quadratic(self):
        """Laplacian of f = x^2 + y^2 + z^2 should be 6."""
        geom = Geometry(
            nx=16, ny=16, nz=16,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-1.0, z_max=1.0,
        )
        f = geom.x_grid**2 + geom.y_grid**2 + geom.z_grid**2
        lap_f = laplacian_3d(f, geom)
        # d^2(x^2)/dx^2 = 2, same for y and z, total = 6
        # Interior points only (boundaries have wrapping artifacts)
        interior = lap_f[2:-2, 2:-2, 2:-2]
        assert jnp.allclose(interior, 6.0, atol=1e-4)

    def test_laplacian_is_div_grad(self):
        """Laplacian should equal divergence of gradient.

        Note: The direct Laplacian uses a 3-point stencil while div(grad)
        uses a 5-point stencil, so they differ by O(dx^2) truncation error.
        We check relative tolerance rather than absolute.
        """
        geom = Geometry(nx=16, ny=16, nz=16, bc_x="periodic", bc_y="periodic", bc_z="periodic")
        f = jnp.sin(2 * jnp.pi * geom.x_grid) * jnp.cos(2 * jnp.pi * geom.y_grid)
        lap_f = laplacian_3d(f, geom)
        grad_f = gradient_3d(f, geom)
        div_grad_f = divergence_3d(grad_f, geom)
        # Different stencils give ~15% relative difference at this resolution
        assert jnp.allclose(lap_f, div_grad_f, rtol=0.2, atol=1e-6)

    def test_laplacian_neumann_bc(self):
        """Laplacian with Neumann BC uses zero-gradient at boundaries."""
        geom = Geometry(
            nx=16, ny=16, nz=8,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
            bc_x="neumann", bc_y="neumann", bc_z="periodic"
        )
        # Constant function: laplacian should be 0 everywhere with any BC
        f = jnp.ones_like(geom.x_grid) * 5.0
        lap_f = laplacian_3d(f, geom)
        # d^2(const)/dx^2 = 0 everywhere including boundaries
        assert jnp.allclose(lap_f, 0.0, atol=1e-6)
