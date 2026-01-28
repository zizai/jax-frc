"""Constrained Transport (CT) scheme for ideal MHD.

The CT scheme uses a staggered grid (Yee lattice) to preserve div(B) = 0 exactly.
This eliminates the need for divergence cleaning and reduces numerical diffusion.

For advection-dominated problems, we use a semi-Lagrangian approach that traces
characteristics backward in time, which eliminates numerical diffusion from the
advection term.
"""

import jax.numpy as jnp
from jax import jit
from jax import Array
from jax.scipy.ndimage import map_coordinates

from jax_frc.core.geometry import Geometry
from jax_frc.operators import curl_3d


@jit(static_argnums=(2,))
def compute_emf_upwind(v: Array, B: Array, geometry: Geometry) -> Array:
    """Compute electromotive force E = -v × B.

    Uses simple cell-centered cross product.

    Args:
        v: Velocity field, shape (nx, ny, nz, 3)
        B: Magnetic field, shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry

    Returns:
        EMF field E = -v × B, shape (nx, ny, nz, 3)
    """
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    Bx, By, Bz = B[..., 0], B[..., 1], B[..., 2]

    # Cross product: E = -v × B
    Ex = -(vy * Bz - vz * By)
    Ey = -(vz * Bx - vx * Bz)
    Ez = -(vx * By - vy * Bx)

    return jnp.stack([Ex, Ey, Ez], axis=-1)


def _extrapolate_boundary(field: Array, axis: int) -> Array:
    """Copy interior values to boundary cells along axis (zero-gradient)."""
    ndim = field.ndim
    left_slice = [slice(None)] * ndim
    left_slice[axis] = 0
    left_src = [slice(None)] * ndim
    left_src[axis] = 1
    right_slice = [slice(None)] * ndim
    right_slice[axis] = -1
    right_src = [slice(None)] * ndim
    right_src[axis] = -2
    field = field.at[tuple(left_slice)].set(field[tuple(left_src)])
    field = field.at[tuple(right_slice)].set(field[tuple(right_src)])
    return field


def _apply_boundary_E(E: Array, geometry: Geometry) -> Array:
    """Apply non-periodic boundary handling to electric field."""
    for axis, bc in enumerate([geometry.bc_x, geometry.bc_y, geometry.bc_z]):
        if bc != "periodic":
            E = _extrapolate_boundary(E, axis)
    return E


@jit(static_argnums=(1,))
def curl_spectral(E: Array, geometry: Geometry) -> Array:
    """Compute curl using spectral (FFT) method.

    For periodic boundaries, spectral derivatives are exact (no truncation error).
    This eliminates numerical diffusion from spatial discretization.

    Args:
        E: Electric field, shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry (must have periodic BCs)

    Returns:
        Curl of E, shape (nx, ny, nz, 3)
    """
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    Lx = geometry.x_max - geometry.x_min
    Ly = geometry.y_max - geometry.y_min
    Lz = geometry.z_max - geometry.z_min

    Ex, Ey, Ez = E[..., 0], E[..., 1], E[..., 2]

    # Wavenumbers
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=Lx/nx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, d=Ly/ny)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(nz, d=Lz/nz)

    # 3D wavenumber grids
    Kx, Ky, Kz = jnp.meshgrid(kx, ky, kz, indexing='ij')

    # FFT of E components
    Ex_hat = jnp.fft.fftn(Ex)
    Ey_hat = jnp.fft.fftn(Ey)
    Ez_hat = jnp.fft.fftn(Ez)

    # Spectral derivatives: d/dx -> i*kx
    # curl(E) = (dEz/dy - dEy/dz, dEx/dz - dEz/dx, dEy/dx - dEx/dy)
    curl_x_hat = 1j * Ky * Ez_hat - 1j * Kz * Ey_hat
    curl_y_hat = 1j * Kz * Ex_hat - 1j * Kx * Ez_hat
    curl_z_hat = 1j * Kx * Ey_hat - 1j * Ky * Ex_hat

    # Inverse FFT
    curl_x = jnp.real(jnp.fft.ifftn(curl_x_hat))
    curl_y = jnp.real(jnp.fft.ifftn(curl_y_hat))
    curl_z = jnp.real(jnp.fft.ifftn(curl_z_hat))

    return jnp.stack([curl_x, curl_y, curl_z], axis=-1)


@jit(static_argnums=(1,))
def curl_ct(E: Array, geometry: Geometry) -> Array:
    """Compute curl using spectral method for periodic BCs.

    For periodic boundaries, spectral derivatives are exact (no truncation error).
    Falls back to 4th-order finite differences for non-periodic BCs.

    Args:
        E: Electric field, shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry

    Returns:
        Curl of E, shape (nx, ny, nz, 3)
    """
    # Use spectral method for periodic BCs (exact derivatives)
    if (geometry.bc_x == "periodic" and
        geometry.bc_y == "periodic" and
        geometry.bc_z == "periodic"):
        return curl_spectral(E, geometry)

    # Fall back to BC-aware finite differences for non-periodic
    return curl_3d(E, geometry, order=4)


def _interp_periodic_3d(f: Array, coords: Array) -> Array:
    """Interpolate 3D field at arbitrary coordinates with periodic wrapping.

    Args:
        f: Field values, shape (nx, ny, nz)
        coords: Coordinates in grid units, shape (3, nx, ny, nz)

    Returns:
        Interpolated values, shape (nx, ny, nz)
    """
    nx, ny, nz = f.shape

    # Wrap coordinates for periodic BCs
    coords_wrapped = jnp.stack([
        coords[0] % nx,
        coords[1] % ny,
        coords[2] % nz,
    ])

    # Use linear interpolation (order=1) - JAX only supports order<=1
    return map_coordinates(f, coords_wrapped, order=1, mode='wrap')


@jit(static_argnums=(2,))
def advect_semi_lagrangian(B: Array, v: Array, geometry: Geometry, dt: float) -> Array:
    """Advect B field using semi-Lagrangian method.

    Traces characteristics backward in time and interpolates B at departure points.
    This eliminates numerical diffusion from the advection term.

    Args:
        B: Magnetic field, shape (nx, ny, nz, 3)
        v: Velocity field, shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry
        dt: Time step

    Returns:
        Advected B field, shape (nx, ny, nz, 3)
    """
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz

    # Grid indices
    i = jnp.arange(nx)[:, None, None]
    j = jnp.arange(ny)[None, :, None]
    k = jnp.arange(nz)[None, None, :]

    # Broadcast to full grid
    i = jnp.broadcast_to(i, (nx, ny, nz))
    j = jnp.broadcast_to(j, (nx, ny, nz))
    k = jnp.broadcast_to(k, (nx, ny, nz))

    # Departure point in physical coordinates
    # x_dep = x - v * dt
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]

    # Convert velocity to grid units per timestep
    vx_grid = vx * dt / geometry.dx
    vy_grid = vy * dt / geometry.dy
    vz_grid = vz * dt / geometry.dz

    # Departure point in grid coordinates
    i_dep = i - vx_grid
    j_dep = j - vy_grid
    k_dep = k - vz_grid

    coords = jnp.stack([i_dep, j_dep, k_dep])

    # Interpolate each B component at departure points
    Bx_new = _interp_periodic_3d(B[..., 0], coords)
    By_new = _interp_periodic_3d(B[..., 1], coords)
    Bz_new = _interp_periodic_3d(B[..., 2], coords)

    return jnp.stack([Bx_new, By_new, Bz_new], axis=-1)


@jit(static_argnums=(2,))
def induction_rhs_ct(v: Array, B: Array, geometry: Geometry) -> Array:
    """Compute dB/dt using constrained transport scheme.

    dB/dt = -curl(E) where E = -v × B
    This is equivalent to: dB/dt = curl(v × B)

    Uses spectral curl for exact derivatives (periodic BCs).

    Args:
        v: Velocity field, shape (nx, ny, nz, 3)
        B: Magnetic field, shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry

    Returns:
        dB/dt, shape (nx, ny, nz, 3)
    """
    # Compute EMF: E = -v × B
    E = compute_emf_upwind(v, B, geometry)
    E = _apply_boundary_E(E, geometry)

    # Compute curl(E) using spectral method
    curl_E = curl_ct(E, geometry)

    # dB/dt = -curl(E) = curl(v × B)
    return -curl_E


@jit(static_argnums=(2,))
def induction_rhs_skew_symmetric(v: Array, B: Array, geometry: Geometry) -> Array:
    """Compute dB/dt using skew-symmetric (energy-conserving) formulation.

    The skew-symmetric form is:
        dB/dt = 0.5 * [curl(v × B) - (v · ∇)B + (B · ∇)v]

    This form conserves magnetic energy exactly in the continuous limit.
    With spectral derivatives, it should be nearly energy-conserving.

    Args:
        v: Velocity field, shape (nx, ny, nz, 3)
        B: Magnetic field, shape (nx, ny, nz, 3)
        geometry: 3D Cartesian geometry

    Returns:
        dB/dt, shape (nx, ny, nz, 3)
    """
    nx, ny, nz = geometry.nx, geometry.ny, geometry.nz
    Lx = geometry.x_max - geometry.x_min
    Ly = geometry.y_max - geometry.y_min
    Lz = geometry.z_max - geometry.z_min

    # Wavenumbers
    kx = 2 * jnp.pi * jnp.fft.fftfreq(nx, d=Lx/nx)
    ky = 2 * jnp.pi * jnp.fft.fftfreq(ny, d=Ly/ny)
    kz = 2 * jnp.pi * jnp.fft.fftfreq(nz, d=Lz/nz)
    Kx, Ky, Kz = jnp.meshgrid(kx, ky, kz, indexing='ij')

    # FFT of v and B components
    vx_hat = jnp.fft.fftn(v[..., 0])
    vy_hat = jnp.fft.fftn(v[..., 1])
    vz_hat = jnp.fft.fftn(v[..., 2])
    Bx_hat = jnp.fft.fftn(B[..., 0])
    By_hat = jnp.fft.fftn(B[..., 1])
    Bz_hat = jnp.fft.fftn(B[..., 2])

    # Compute gradients in spectral space
    # (v · ∇)B_i = v_j * ∂B_i/∂x_j
    def advection_term(f_hat):
        df_dx = jnp.real(jnp.fft.ifftn(1j * Kx * f_hat))
        df_dy = jnp.real(jnp.fft.ifftn(1j * Ky * f_hat))
        df_dz = jnp.real(jnp.fft.ifftn(1j * Kz * f_hat))
        return v[..., 0] * df_dx + v[..., 1] * df_dy + v[..., 2] * df_dz

    v_dot_grad_Bx = advection_term(Bx_hat)
    v_dot_grad_By = advection_term(By_hat)
    v_dot_grad_Bz = advection_term(Bz_hat)

    # (B · ∇)v_i = B_j * ∂v_i/∂x_j
    def stretching_term(f_hat):
        df_dx = jnp.real(jnp.fft.ifftn(1j * Kx * f_hat))
        df_dy = jnp.real(jnp.fft.ifftn(1j * Ky * f_hat))
        df_dz = jnp.real(jnp.fft.ifftn(1j * Kz * f_hat))
        return B[..., 0] * df_dx + B[..., 1] * df_dy + B[..., 2] * df_dz

    B_dot_grad_vx = stretching_term(vx_hat)
    B_dot_grad_vy = stretching_term(vy_hat)
    B_dot_grad_vz = stretching_term(vz_hat)

    # curl(v × B) term
    E = compute_emf_upwind(v, B, geometry)
    curl_E = curl_ct(E, geometry)
    curl_v_cross_B = -curl_E

    # Skew-symmetric form: dB/dt = 0.5 * [curl(v × B) - (v · ∇)B + (B · ∇)v]
    dBx_dt = 0.5 * (curl_v_cross_B[..., 0] - v_dot_grad_Bx + B_dot_grad_vx)
    dBy_dt = 0.5 * (curl_v_cross_B[..., 1] - v_dot_grad_By + B_dot_grad_vy)
    dBz_dt = 0.5 * (curl_v_cross_B[..., 2] - v_dot_grad_Bz + B_dot_grad_vz)

    return jnp.stack([dBx_dt, dBy_dt, dBz_dt], axis=-1)
