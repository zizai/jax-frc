"""Computational geometry and coordinate systems."""

from dataclasses import dataclass
from typing import Literal
import jax.numpy as jnp
from jax import Array

@dataclass(frozen=True)
class Geometry:
    """Defines the computational domain and coordinate system."""

    coord_system: Literal["cylindrical", "cartesian"]
    nr: int
    nz: int
    r_min: float
    r_max: float
    z_min: float
    z_max: float

    def __post_init__(self):
        # Validate inputs
        if self.r_min <= 0 and self.coord_system == "cylindrical":
            raise ValueError("r_min must be > 0 for cylindrical coordinates")
        if self.nr < 2 or self.nz < 2:
            raise ValueError("Grid must have at least 2 points in each dimension")

    @property
    def r(self) -> Array:
        """1D array of radial coordinates."""
        return jnp.linspace(self.r_min, self.r_max, self.nr)

    @property
    def z(self) -> Array:
        """1D array of axial coordinates."""
        return jnp.linspace(self.z_min, self.z_max, self.nz)

    @property
    def dr(self) -> float:
        """Radial grid spacing."""
        return (self.r_max - self.r_min) / (self.nr - 1)

    @property
    def dz(self) -> float:
        """Axial grid spacing."""
        return (self.z_max - self.z_min) / (self.nz - 1)

    @property
    def r_grid(self) -> Array:
        """2D array of radial coordinates (nr, nz)."""
        return self.r[:, None] * jnp.ones((1, self.nz))

    @property
    def z_grid(self) -> Array:
        """2D array of axial coordinates (nr, nz)."""
        return jnp.ones((self.nr, 1)) * self.z[None, :]

    @property
    def cell_volumes(self) -> Array:
        """2D array of cell volumes including 2*pi*r factor for cylindrical."""
        if self.coord_system == "cylindrical":
            return 2 * jnp.pi * self.r_grid * self.dr * self.dz
        else:
            return jnp.ones((self.nr, self.nz)) * self.dr * self.dz

    @classmethod
    def from_config(cls, config: dict) -> "Geometry":
        """Create Geometry from configuration dictionary."""
        return cls(
            coord_system=config["coord_system"],
            nr=int(config["nr"]),
            nz=int(config["nz"]),
            r_min=float(config["r_min"]),
            r_max=float(config["r_max"]),
            z_min=float(config["z_min"]),
            z_max=float(config["z_max"]),
        )
