"""Particle pushing algorithms for hybrid kinetic simulations."""

from typing import TYPE_CHECKING
import jax.numpy as jnp
from jax import jit, Array

if TYPE_CHECKING:
    from jax_frc.core.geometry import Geometry


@jit
def boris_push(x, v, E, B, q, m, dt):
    """Boris particle pusher for charged particle motion.

    Standard Boris algorithm with half-acceleration, rotation, half-acceleration.

    Args:
        x: Particle positions (n_particles, 3)
        v: Particle velocities (n_particles, 3)
        E: Electric field at particle positions (n_particles, 3)
        B: Magnetic field at particle positions (n_particles, 3)
        q: Particle charge
        m: Particle mass
        dt: Timestep

    Returns:
        x_new, v_new: Updated positions and velocities
    """
    # Half acceleration from E field
    v_minus = v + q * E * dt / (2 * m)

    # Rotation from B field
    t = q * B * dt / (2 * m)
    t_mag_sq = jnp.sum(t**2, axis=-1, keepdims=True)
    s = 2 * t / (1 + t_mag_sq)

    v_prime = v_minus + jnp.cross(v_minus, t)
    v_plus = v_minus + jnp.cross(v_prime, s)

    # Second half acceleration from E field
    v_new = v_plus + q * E * dt / (2 * m)

    # Position update
    x_new = x + v_new * dt

    return x_new, v_new


@jit
def interpolate_field_to_particles(field, x, geometry_params):
    """Interpolate grid field to particle positions using bilinear interpolation.

    Args:
        field: Grid field (nr, nz) or (nr, nz, 3)
        x: Particle positions (n_particles, 3) in (r, theta, z)
        geometry_params: tuple of (r_min, r_max, z_min, z_max, nr, nz)

    Returns:
        Interpolated field values at particle positions
    """
    r_min, r_max, z_min, z_max, nr, nz = geometry_params

    # Get r and z from particle positions
    r = x[:, 0]
    z = x[:, 2]

    # Normalize to grid indices
    r_idx = (r - r_min) / (r_max - r_min) * (nr - 1)
    z_idx = (z - z_min) / (z_max - z_min) * (nz - 1)

    # Floor indices
    r_i = jnp.floor(r_idx).astype(int)
    z_i = jnp.floor(z_idx).astype(int)

    # Clamp to valid range
    r_i = jnp.clip(r_i, 0, nr - 2)
    z_i = jnp.clip(z_i, 0, nz - 2)

    # Fractional parts
    r_frac = r_idx - r_i
    z_frac = z_idx - z_i

    # Bilinear interpolation weights
    w00 = (1 - r_frac) * (1 - z_frac)
    w01 = (1 - r_frac) * z_frac
    w10 = r_frac * (1 - z_frac)
    w11 = r_frac * z_frac

    if field.ndim == 2:
        # Scalar field
        result = (w00 * field[r_i, z_i] +
                  w01 * field[r_i, z_i + 1] +
                  w10 * field[r_i + 1, z_i] +
                  w11 * field[r_i + 1, z_i + 1])
    else:
        # Vector field (nr, nz, 3)
        result = (w00[:, None] * field[r_i, z_i, :] +
                  w01[:, None] * field[r_i, z_i + 1, :] +
                  w10[:, None] * field[r_i + 1, z_i, :] +
                  w11[:, None] * field[r_i + 1, z_i + 1, :])

    return result


@jit(static_argnums=(2,))
def interpolate_field_to_particles_3d(
    field: Array, x: Array, geometry: "Geometry"
) -> Array:
    """Trilinear interpolation of 3D field to particle positions.

    Args:
        field: Field values, shape (nx, ny, nz, ncomp) or (nx, ny, nz)
        x: Particle positions, shape (n_particles, 3) as (x, y, z)
        geometry: 3D Cartesian geometry

    Returns:
        Field at particle positions, shape (n_particles, ncomp) or (n_particles,)
    """
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz

    # Normalized coordinates (0 to n-1)
    xn = (x[:, 0] - geometry.x_min - dx / 2) / dx
    yn = (x[:, 1] - geometry.y_min - dy / 2) / dy
    zn = (x[:, 2] - geometry.z_min - dz / 2) / dz

    # Integer indices and fractions
    i0 = jnp.floor(xn).astype(int)
    j0 = jnp.floor(yn).astype(int)
    k0 = jnp.floor(zn).astype(int)

    fx = xn - i0
    fy = yn - j0
    fz = zn - k0

    # Wrap indices for periodic
    i0 = i0 % nx
    j0 = j0 % ny
    k0 = k0 % nz
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    k1 = (k0 + 1) % nz

    # 8 corners of cell
    f000 = field[i0, j0, k0]
    f001 = field[i0, j0, k1]
    f010 = field[i0, j1, k0]
    f011 = field[i0, j1, k1]
    f100 = field[i1, j0, k0]
    f101 = field[i1, j0, k1]
    f110 = field[i1, j1, k0]
    f111 = field[i1, j1, k1]

    # Trilinear interpolation
    # Handle both vector (nx, ny, nz, 3) and scalar (nx, ny, nz) fields
    if field.ndim == 4:
        fx = fx[:, None]
        fy = fy[:, None]
        fz = fz[:, None]

    c00 = f000 * (1 - fx) + f100 * fx
    c01 = f001 * (1 - fx) + f101 * fx
    c10 = f010 * (1 - fx) + f110 * fx
    c11 = f011 * (1 - fx) + f111 * fx

    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy

    return c0 * (1 - fz) + c1 * fz


def deposit_particles_to_grid(values, weights, x, geometry_params):
    """Deposit particle quantities to grid using CIC (Cloud-In-Cell).

    Note: Not JIT-compiled because grid shape depends on runtime parameters.
    For performance, wrap the calling function with jax.jit and use static_argnums.

    Args:
        values: Values to deposit (n_particles,) or (n_particles, 3)
        weights: Particle weights (n_particles,)
        x: Particle positions (n_particles, 3)
        geometry_params: tuple of (r_min, r_max, z_min, z_max, nr, nz)

    Returns:
        Grid values (nr, nz) or (nr, nz, 3)
    """
    r_min, r_max, z_min, z_max, nr, nz = geometry_params

    r = x[:, 0]
    z = x[:, 2]

    # Normalize to grid indices
    r_idx = (r - r_min) / (r_max - r_min) * (nr - 1)
    z_idx = (z - z_min) / (z_max - z_min) * (nz - 1)

    r_i = jnp.floor(r_idx).astype(int)
    z_i = jnp.floor(z_idx).astype(int)

    r_i = jnp.clip(r_i, 0, nr - 2)
    z_i = jnp.clip(z_i, 0, nz - 2)

    r_frac = r_idx - r_i
    z_frac = z_idx - z_i

    # CIC weights
    w00 = (1 - r_frac) * (1 - z_frac)
    w01 = (1 - r_frac) * z_frac
    w10 = r_frac * (1 - z_frac)
    w11 = r_frac * z_frac

    if values.ndim == 1:
        # Scalar deposition
        weighted_values = values * weights
        grid = jnp.zeros((nr, nz))
        grid = grid.at[r_i, z_i].add(w00 * weighted_values)
        grid = grid.at[r_i, z_i + 1].add(w01 * weighted_values)
        grid = grid.at[r_i + 1, z_i].add(w10 * weighted_values)
        grid = grid.at[r_i + 1, z_i + 1].add(w11 * weighted_values)
    else:
        # Vector deposition (n_particles, 3)
        weighted_values = values * weights[:, None]
        grid = jnp.zeros((nr, nz, 3))
        for c in range(3):
            grid = grid.at[r_i, z_i, c].add(w00 * weighted_values[:, c])
            grid = grid.at[r_i, z_i + 1, c].add(w01 * weighted_values[:, c])
            grid = grid.at[r_i + 1, z_i, c].add(w10 * weighted_values[:, c])
            grid = grid.at[r_i + 1, z_i + 1, c].add(w11 * weighted_values[:, c])

    return grid


@jit(static_argnums=(3,))
def deposit_particles_to_grid_3d(
    values: Array, weights: Array, x: Array, geometry: "Geometry"
) -> Array:
    """Deposit particle quantities to 3D grid using CIC (Cloud-In-Cell).

    Uses trilinear interpolation for deposition with periodic boundaries.

    Args:
        values: Values to deposit, shape (n_particles,) or (n_particles, 3)
        weights: Particle weights, shape (n_particles,)
        x: Particle positions, shape (n_particles, 3) as (x, y, z)
        geometry: 3D Cartesian geometry

    Returns:
        Grid values, shape (nx, ny, nz) or (nx, ny, nz, 3)
    """
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    dx, dy, dz = geometry.dx, geometry.dy, geometry.dz

    # Normalized coordinates (0 to n-1)
    xn = (x[:, 0] - geometry.x_min - dx / 2) / dx
    yn = (x[:, 1] - geometry.y_min - dy / 2) / dy
    zn = (x[:, 2] - geometry.z_min - dz / 2) / dz

    # Integer indices and fractions
    i0 = jnp.floor(xn).astype(int)
    j0 = jnp.floor(yn).astype(int)
    k0 = jnp.floor(zn).astype(int)

    fx = xn - i0
    fy = yn - j0
    fz = zn - k0

    # Wrap indices for periodic
    i0 = i0 % nx
    j0 = j0 % ny
    k0 = k0 % nz
    i1 = (i0 + 1) % nx
    j1 = (j0 + 1) % ny
    k1 = (k0 + 1) % nz

    # CIC weights for 8 corners
    w000 = (1 - fx) * (1 - fy) * (1 - fz)
    w001 = (1 - fx) * (1 - fy) * fz
    w010 = (1 - fx) * fy * (1 - fz)
    w011 = (1 - fx) * fy * fz
    w100 = fx * (1 - fy) * (1 - fz)
    w101 = fx * (1 - fy) * fz
    w110 = fx * fy * (1 - fz)
    w111 = fx * fy * fz

    if values.ndim == 1:
        # Scalar deposition
        weighted_values = values * weights
        grid = jnp.zeros((nx, ny, nz))
        grid = grid.at[i0, j0, k0].add(w000 * weighted_values)
        grid = grid.at[i0, j0, k1].add(w001 * weighted_values)
        grid = grid.at[i0, j1, k0].add(w010 * weighted_values)
        grid = grid.at[i0, j1, k1].add(w011 * weighted_values)
        grid = grid.at[i1, j0, k0].add(w100 * weighted_values)
        grid = grid.at[i1, j0, k1].add(w101 * weighted_values)
        grid = grid.at[i1, j1, k0].add(w110 * weighted_values)
        grid = grid.at[i1, j1, k1].add(w111 * weighted_values)
    else:
        # Vector deposition (n_particles, 3)
        weighted_values = values * weights[:, None]
        grid = jnp.zeros((nx, ny, nz, 3))
        for c in range(3):
            grid = grid.at[i0, j0, k0, c].add(w000 * weighted_values[:, c])
            grid = grid.at[i0, j0, k1, c].add(w001 * weighted_values[:, c])
            grid = grid.at[i0, j1, k0, c].add(w010 * weighted_values[:, c])
            grid = grid.at[i0, j1, k1, c].add(w011 * weighted_values[:, c])
            grid = grid.at[i1, j0, k0, c].add(w100 * weighted_values[:, c])
            grid = grid.at[i1, j0, k1, c].add(w101 * weighted_values[:, c])
            grid = grid.at[i1, j1, k0, c].add(w110 * weighted_values[:, c])
            grid = grid.at[i1, j1, k1, c].add(w111 * weighted_values[:, c])

    return grid
