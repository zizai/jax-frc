"""Tests for 3D Cartesian geometry."""

import jax.numpy as jnp
import pytest
from jax_frc.core.geometry import Geometry


class TestGeometry3D:
    """Test 3D Cartesian geometry."""

    def test_geometry_creation(self):
        """Test creating a 3D geometry."""
        geom = Geometry(
            nx=8, ny=8, nz=16,
            x_min=-1.0, x_max=1.0,
            y_min=-1.0, y_max=1.0,
            z_min=-2.0, z_max=2.0,
            bc_x="periodic", bc_y="periodic", bc_z="dirichlet"
        )
        assert geom.nx == 8
        assert geom.ny == 8
        assert geom.nz == 16

    def test_grid_spacing(self):
        """Test grid spacing calculation."""
        geom = Geometry(
            nx=10, ny=10, nz=20,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=2.0,
        )
        assert jnp.isclose(geom.dx, 0.1)
        assert jnp.isclose(geom.dy, 0.1)
        assert jnp.isclose(geom.dz, 0.1)

    def test_coordinate_arrays(self):
        """Test 1D coordinate arrays."""
        geom = Geometry(
            nx=4, ny=4, nz=4,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
        )
        assert geom.x.shape == (4,)
        assert geom.y.shape == (4,)
        assert geom.z.shape == (4,)
        # Cell centers at 0.125, 0.375, 0.625, 0.875
        assert jnp.isclose(geom.x[0], 0.125)

    def test_3d_grids(self):
        """Test 3D meshgrid arrays."""
        geom = Geometry(
            nx=4, ny=6, nz=8,
            x_min=0.0, x_max=1.0,
            y_min=0.0, y_max=1.0,
            z_min=0.0, z_max=1.0,
        )
        assert geom.x_grid.shape == (4, 6, 8)
        assert geom.y_grid.shape == (4, 6, 8)
        assert geom.z_grid.shape == (4, 6, 8)

    def test_cell_volumes(self):
        """Test cell volumes are dx * dy * dz."""
        geom = Geometry(
            nx=4, ny=4, nz=4,
            x_min=0.0, x_max=2.0,
            y_min=0.0, y_max=2.0,
            z_min=0.0, z_max=2.0,
        )
        # dx = dy = dz = 0.5, volume = 0.125
        assert geom.cell_volumes.shape == (4, 4, 4)
        assert jnp.allclose(geom.cell_volumes, 0.125)
