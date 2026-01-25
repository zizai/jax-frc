"""Particle pushing algorithms for hybrid kinetic simulations."""

import jax.numpy as jnp
from jax import jit

MU0 = 1.2566e-6
QE = 1.602e-19
MI = 1.673e-27


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
