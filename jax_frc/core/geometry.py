"""3D Cartesian geometry for plasma simulations."""

from dataclasses import dataclass
from typing import Literal
import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class Geometry:
    """3D Cartesian computational geometry.

    Attributes:
        nx, ny, nz: Number of grid cells in each direction
        x_min, x_max: Domain bounds in x
        y_min, y_max: Domain bounds in y
        z_min, z_max: Domain bounds in z
        bc_x, bc_y, bc_z: Boundary condition type per axis
    """
    nx: int
    ny: int
    nz: int
    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0
    z_min: float = -1.0
    z_max: float = 1.0
    bc_x: Literal["periodic", "dirichlet", "neumann"] = "periodic"
    bc_y: Literal["periodic", "dirichlet", "neumann"] = "periodic"
    bc_z: Literal["periodic", "dirichlet", "neumann"] = "dirichlet"

    @property
    def dx(self) -> float:
        """Grid spacing in x."""
        return (self.x_max - self.x_min) / self.nx

    @property
    def dy(self) -> float:
        """Grid spacing in y."""
        return (self.y_max - self.y_min) / self.ny

    @property
    def dz(self) -> float:
        """Grid spacing in z."""
        return (self.z_max - self.z_min) / self.nz

    @property
    def x(self) -> Array:
        """1D array of cell-centered x coordinates."""
        return jnp.linspace(
            self.x_min + self.dx / 2,
            self.x_max - self.dx / 2,
            self.nx
        )

    @property
    def y(self) -> Array:
        """1D array of cell-centered y coordinates."""
        return jnp.linspace(
            self.y_min + self.dy / 2,
            self.y_max - self.dy / 2,
            self.ny
        )

    @property
    def z(self) -> Array:
        """1D array of cell-centered z coordinates."""
        return jnp.linspace(
            self.z_min + self.dz / 2,
            self.z_max - self.dz / 2,
            self.nz
        )

    @property
    def x_grid(self) -> Array:
        """3D array of x coordinates, shape (nx, ny, nz)."""
        x, y, z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij')
        return x

    @property
    def y_grid(self) -> Array:
        """3D array of y coordinates, shape (nx, ny, nz)."""
        x, y, z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij')
        return y

    @property
    def z_grid(self) -> Array:
        """3D array of z coordinates, shape (nx, ny, nz)."""
        x, y, z = jnp.meshgrid(self.x, self.y, self.z, indexing='ij')
        return z

    @property
    def cell_volumes(self) -> Array:
        """Cell volumes, shape (nx, ny, nz). Simply dx * dy * dz."""
        return jnp.full((self.nx, self.ny, self.nz), self.dx * self.dy * self.dz)
