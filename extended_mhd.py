import jax
import jax.numpy as jnp
from jax import jit, lax, grad, vmap

from jax_frc.operators import (
    gradient_r,
    gradient_z,
    curl_cylindrical_axisymmetric,
    divergence_cylindrical,
)

MU0 = 1.2566e-6
QE = 1.602e-19
ME = 9.109e-31

@jit
def curl_2d(f_x, f_y, dx, dy):
    """
    Computes 2D curl in Cartesian: (d_fy/dx - d_fx/dy)
    DEPRECATED: Use curl_cylindrical_axisymmetric for cylindrical coordinates.
    """
    dfy_dx = (jnp.roll(f_y, -1, axis=0) - jnp.roll(f_y, 1, axis=0)) / (2 * dx)
    dfx_dy = (jnp.roll(f_x, -1, axis=1) - jnp.roll(f_x, 1, axis=1)) / (2 * dy)
    return dfy_dx - dfx_dy

@jit
def grad_2d(f, dr, dz):
    """
    Computes 2D gradient in cylindrical (r,z) using non-periodic boundaries.
    """
    df_dr = gradient_r(f, dr)
    df_dz = gradient_z(f, dz)
    return df_dr, df_dz

@jit
def cross_product_2d(a_x, a_y, b_x, b_y, a_z, b_z):
    """
    Computes 3D cross product with z-components
    """
    c_x = a_y * b_z - a_z * b_y
    c_y = a_z * b_x - a_x * b_z
    c_z = a_x * b_y - a_y * b_x
    return c_x, c_y, c_z

@jit
def extended_ohm_law(v_r, v_theta, v_z, b_r, b_theta, b_z, j_r, j_theta, j_z, n, eta, p_e, dr, dz, r):
    """
    Extended Ohm's law in cylindrical coordinates:
    E = -v x B + eta * J + (J x B) / (ne) - grad(p_e) / (ne)

    Note: This function uses cylindrical (r, theta, z) velocity and field components.
    """
    # v x B in cylindrical (r, theta, z)
    v_cross_b_r = v_theta * b_z - v_z * b_theta
    v_cross_b_theta = v_z * b_r - v_r * b_z
    v_cross_b_z = v_r * b_theta - v_theta * b_r
    term1_r, term1_theta, term1_z = -v_cross_b_r, -v_cross_b_theta, -v_cross_b_z

    # Resistive term: eta * J
    term2_r, term2_theta, term2_z = eta * j_r, eta * j_theta, eta * j_z

    # Hall term: (J x B) / (ne)
    j_cross_b_r = j_theta * b_z - j_z * b_theta
    j_cross_b_theta = j_z * b_r - j_r * b_z
    j_cross_b_z = j_r * b_theta - j_theta * b_r
    n_safe = jnp.maximum(n, 1e16)  # Avoid division by zero
    term3_r = j_cross_b_r / (n_safe * QE)
    term3_theta = j_cross_b_theta / (n_safe * QE)
    term3_z = j_cross_b_z / (n_safe * QE)

    # Electron pressure gradient: -grad(p_e) / (ne)
    dp_e_dr, dp_e_dz = grad_2d(p_e, dr, dz)
    term4_r = -dp_e_dr / (n_safe * QE)
    term4_theta = jnp.zeros_like(p_e)  # No theta gradient in axisymmetric
    term4_z = -dp_e_dz / (n_safe * QE)

    E_r = term1_r + term2_r + term3_r + term4_r
    E_theta = term1_theta + term2_theta + term3_theta + term4_theta
    E_z = term1_z + term2_z + term3_z + term4_z

    return E_r, E_theta, E_z

@jit
def hall_operator(b_r, b_theta, b_z, n, dr, dz, r):
    """
    Constructs the Hall differential operator for semi-implicit stepping.
    L_Hall ~ curl((J x B) / (ne))

    Uses correct cylindrical curl with L'Hopital at axis.
    """
    # Compute J = curl(B) / mu0 in cylindrical coordinates
    j_r, j_theta, j_z = curl_cylindrical_axisymmetric(b_r, b_theta, b_z, dr, dz, r)
    j_r = j_r / MU0
    j_theta = j_theta / MU0
    j_z = j_z / MU0

    # J x B in cylindrical
    j_cross_b_r = j_theta * b_z - j_z * b_theta
    j_cross_b_theta = j_z * b_r - j_r * b_z
    j_cross_b_z = j_r * b_theta - j_theta * b_r

    # Hall term: (J x B) / (ne)
    n_safe = jnp.maximum(n, 1e16)
    hall_r = j_cross_b_r / (n_safe * QE)
    hall_theta = j_cross_b_theta / (n_safe * QE)
    hall_z = j_cross_b_z / (n_safe * QE)

    # curl(Hall) in cylindrical coordinates
    curl_hall_r, curl_hall_theta, curl_hall_z = curl_cylindrical_axisymmetric(
        hall_r, hall_theta, hall_z, dr, dz, r
    )

    return curl_hall_r, curl_hall_theta, curl_hall_z

def compute_whistler_stable_dt(B_max, n, dr, dz):
    """
    Compute CFL-stable timestep for Whistler waves.
    Whistler speed: v_w = k * B / (mu0 * n * e) where k = pi/dx
    CFL: dt < dx / v_w

    Args:
        B_max: Maximum magnetic field strength [T]
        n: Minimum density [m^-3] (must be > 0)
        dr, dz: Grid spacings

    Returns:
        Stable timestep with 25% safety margin
    """
    dx_min = jnp.minimum(dr, dz)
    k = jnp.pi / dx_min
    v_whistler = k * B_max / (MU0 * n * QE)
    dt_cfl = dx_min / v_whistler
    return 0.25 * dt_cfl  # 25% safety margin


@jit
def compute_div_b(b_r, b_theta, b_z, dr, dz, r):
    """
    Compute divergence of magnetic field in cylindrical coordinates.
    div(B) = (1/r)*d(r*B_r)/dr + dB_z/dz
           = B_r/r + dB_r/dr + dB_z/dz

    For 2D axisymmetric: B_theta doesn't contribute to divergence.
    """
    return divergence_cylindrical(b_r, b_z, dr, dz, r)


@jit
def divergence_cleaning_step(b_r, b_theta, b_z, dr, dz, r):
    """
    Apply divergence cleaning using projection method in cylindrical coordinates.

    Solves div(B - grad(phi)) = 0 iteratively using Jacobi relaxation.
    This enforces the div(B) = 0 constraint numerically.

    Uses 50 Jacobi iterations with tolerance-based early exit via while_loop.

    Args:
        b_r, b_theta, b_z: Magnetic field components
        dr, dz: Grid spacings
        r: Radial coordinate array

    Returns:
        Cleaned magnetic field components (b_r_clean, b_theta_clean, b_z_clean)
    """
    # Compute current divergence error
    div_b = compute_div_b(b_r, b_theta, b_z, dr, dz, r)

    # Solve Laplacian(phi) = div(B) using Jacobi iteration
    phi = jnp.zeros_like(b_r)

    # Laplacian stencil coefficient (Cartesian approximation for simplicity)
    coeff = 2.0 / dr**2 + 2.0 / dz**2

    def jacobi_step(carry):
        i, phi, converged = carry
        # Jacobi update
        phi_rp = jnp.roll(phi, -1, axis=0)
        phi_rm = jnp.roll(phi, 1, axis=0)
        phi_zp = jnp.roll(phi, -1, axis=1)
        phi_zm = jnp.roll(phi, 1, axis=1)

        phi_new = ((phi_rp + phi_rm) / dr**2 + (phi_zp + phi_zm) / dz**2 - div_b) / coeff

        # Apply zero boundary conditions for phi
        phi_new = phi_new.at[0, :].set(0)
        phi_new = phi_new.at[-1, :].set(0)
        phi_new = phi_new.at[:, 0].set(0)
        phi_new = phi_new.at[:, -1].set(0)

        # Check convergence (max change in phi)
        max_change = jnp.max(jnp.abs(phi_new - phi))
        converged = max_change < 1e-8

        return (i + 1, phi_new, converged)

    def cond_fn(carry):
        i, _, converged = carry
        return (i < 50) & (~converged)

    _, phi, _ = lax.while_loop(cond_fn, jacobi_step, (0, phi, False))

    # Compute gradient of phi and subtract from B
    grad_phi_r, grad_phi_z = grad_2d(phi, dr, dz)

    b_r_clean = b_r - grad_phi_r
    b_theta_clean = b_theta  # theta-component unchanged in 2D axisymmetric
    b_z_clean = b_z - grad_phi_z

    return b_r_clean, b_theta_clean, b_z_clean

@jit
def hall_substep(b_r, b_theta, b_z, v_r, v_theta, v_z, n, p_e, dt_sub, dr, dz, r, eta):
    """Single Hall MHD substep with stable timestep using cylindrical coordinates."""
    # Compute current density J = curl(B) / mu0 in cylindrical
    j_r, j_theta, j_z = curl_cylindrical_axisymmetric(b_r, b_theta, b_z, dr, dz, r)
    j_r = j_r / MU0
    j_theta = j_theta / MU0
    j_z = j_z / MU0

    # Compute electric field from extended Ohm's law
    E_r, E_theta, E_z = extended_ohm_law(v_r, v_theta, v_z, b_r, b_theta, b_z,
                                          j_r, j_theta, j_z, n, eta, p_e, dr, dz, r)

    # Faraday's law: dB/dt = -curl(E) in cylindrical
    curl_E_r, curl_E_theta, curl_E_z = curl_cylindrical_axisymmetric(E_r, E_theta, E_z, dr, dz, r)

    b_r_new = b_r - dt_sub * curl_E_r
    b_theta_new = b_theta - dt_sub * curl_E_theta
    b_z_new = b_z - dt_sub * curl_E_z

    return b_r_new, b_theta_new, b_z_new

@jit
def semi_implicit_hall_step(b_r, b_theta, b_z, v_r, v_theta, v_z, n, p_e, dt, dr, dz, r, eta):
    """
    Subcycled time stepping to handle Whistler waves stably in cylindrical coordinates.
    Uses automatic subcycling based on Whistler CFL condition.

    For numerical stability, limits maximum substeps to 50.
    """
    # Compute stable substep based on maximum B field
    B_max = jnp.maximum(jnp.max(jnp.abs(b_z)), 1e-10)
    # Use minimum density from the actual field for worst-case Whistler speed
    n_min = jnp.maximum(jnp.min(n), 1e18)
    dt_stable = compute_whistler_stable_dt(B_max, n_min, dr, dz)

    # Number of substeps needed, capped at 50 for practicality
    n_substeps = jnp.minimum(50, jnp.maximum(1, jnp.ceil(dt / dt_stable))).astype(jnp.int32)
    dt_sub = dt / n_substeps

    # Subcycle using fori_loop (more efficient than while_loop for fixed iterations)
    def substep_body(i, carry):
        br, btheta, bz = carry
        br_new, btheta_new, bz_new = hall_substep(br, btheta, bz, v_r, v_theta, v_z,
                                                   n, p_e, dt_sub, dr, dz, r, eta)
        return (br_new, btheta_new, bz_new)

    b_r_new, b_theta_new, b_z_new = lax.fori_loop(
        0, n_substeps, substep_body, (b_r, b_theta, b_z)
    )

    return b_r_new, b_theta_new, b_z_new

@jit
def apply_halo_density(n, halo_density=1e18, core_density=1e19, r_cutoff=0.8):
    """
    Applies halo density model for vacuum handling.
    Returns array with same shape as input n.

    Note: Using halo_density=1e18 (not 1e16) for numerical stability.
    Lower halo densities cause the Hall term to dominate and produce
    unphysically fast Whistler waves that violate the CFL condition.
    """
    nr, nz = n.shape
    r = jnp.linspace(0, 1, nr)[:, None]

    # Create halo mask that broadcasts to full (nr, nz) shape
    halo_mask = 0.5 * (1 + jnp.tanh((r - r_cutoff) / 0.05))

    # Broadcast to match input shape: (nr, 1) * scalar -> (nr, nz)
    n_with_halo = halo_mask * halo_density + (1 - halo_mask) * core_density

    # Ensure output matches input shape by broadcasting
    return jnp.broadcast_to(n_with_halo, (nr, nz))

@jit
def step(state, _):
    """Single timestep of extended MHD evolution in cylindrical coordinates.

    Warning: The Hall term introduces Whistler waves with CFL constraint:
        dt < (dr * mu0 * n * e) / (pi * B)
    For typical plasma parameters, this can require thousands of substeps.
    The simulation uses subcycling with a maximum of 50 substeps per timestep.
    For stability with larger timesteps, use coarser grids or smaller dt.
    """
    b_r, b_theta, b_z, v_r, v_theta, v_z, n, p_e, t, dt, dr, dz, r, eta = state

    n_with_halo = apply_halo_density(n)

    b_r_new, b_theta_new, b_z_new = semi_implicit_hall_step(
        b_r, b_theta, b_z, v_r, v_theta, v_z, n_with_halo, p_e, dt, dr, dz, r, eta
    )

    # Apply divergence cleaning to reduce numerical div(B) errors
    b_r_new, b_theta_new, b_z_new = divergence_cleaning_step(
        b_r_new, b_theta_new, b_z_new, dr, dz, r
    )

    # Apply boundary conditions
    # Inner boundary (r=0): B_r = 0 (symmetry), B_theta = 0, B_z Neumann
    b_r_new = b_r_new.at[0, :].set(0)
    b_theta_new = b_theta_new.at[0, :].set(0)
    b_z_new = b_z_new.at[0, :].set(b_z_new[1, :])

    # Outer boundary (r=R): conducting wall, B_tangential = 0
    b_r_new = b_r_new.at[-1, :].set(0)
    b_theta_new = b_theta_new.at[-1, :].set(0)
    b_z_new = b_z_new.at[-1, :].set(0)

    # Axial boundaries (z = +/- L): conducting wall
    b_r_new = b_r_new.at[:, 0].set(0)
    b_r_new = b_r_new.at[:, -1].set(0)
    b_theta_new = b_theta_new.at[:, 0].set(0)
    b_theta_new = b_theta_new.at[:, -1].set(0)
    b_z_new = b_z_new.at[:, 0].set(0)
    b_z_new = b_z_new.at[:, -1].set(0)

    return (b_r_new, b_theta_new, b_z_new, v_r, v_theta, v_z, n, p_e, t + dt, dt, dr, dz, r, eta), (b_r, b_theta, b_z)

def run_simulation(steps=100, nr=16, nz=16, dt=1e-8, eta=1e-4):
    """Run extended MHD simulation with Hall effect in cylindrical coordinates.

    Warning: The Hall term introduces Whistler wave CFL constraint:
        dt_whistler < (dr * mu0 * n * e) / (pi * B)
    For typical parameters (B~0.1T, n~1e18 m^-3, dr~0.06), this gives
    dt_whistler ~ 2.5e-10 s. The simulation uses subcycling (up to 50
    substeps per main timestep) but may still become unstable if
    dt >> dt_whistler. For stability, use dt <= 1e-8 s.

    Args:
        steps: Number of timesteps to run
        nr: Number of radial grid points (must be >= 4)
        nz: Number of axial grid points (must be >= 4)
        dt: Timestep [s] (must be > 0). Default 1e-8 for stability.
        eta: Resistivity [Ohm*m] (must be >= 0)

    Returns:
        Tuple of (final_B_components, history)

    Raises:
        ValueError: If input parameters are invalid
    """
    # Validate inputs
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if nr < 4:
        raise ValueError(f"nr must be at least 4 for finite differences, got {nr}")
    if nz < 4:
        raise ValueError(f"nz must be at least 4 for finite differences, got {nz}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if eta < 0:
        raise ValueError(f"eta must be non-negative, got {eta}")

    dr, dz = 1.0/nr, 1.0/nz

    # Create coordinate arrays
    r = jnp.linspace(0.01, 1, nr)[:, None]  # Avoid r=0 for numerical stability
    z = jnp.linspace(-1, 1, nz)[None, :]

    # Initial magnetic field: B_r, B_theta, B_z in cylindrical
    # Using smaller B field to reduce Whistler wave speed for stability
    b_r_init = jnp.zeros((nr, nz))
    b_theta_init = jnp.zeros((nr, nz))
    b_z_init = 0.1 * jnp.exp(-r**2 - z**2)  # Reduced from 1.0 for stability

    # Initial velocity: v_r, v_theta, v_z in cylindrical
    v_r_init = jnp.zeros((nr, nz))
    v_theta_init = jnp.zeros((nr, nz))
    v_z_init = jnp.zeros((nr, nz))

    # Initial density and electron pressure
    n_init = jnp.ones((nr, nz)) * 1e19
    p_e_init = jnp.ones((nr, nz)) * 1e3

    # State: (b_r, b_theta, b_z, v_r, v_theta, v_z, n, p_e, t, dt, dr, dz, r, eta)
    state = (b_r_init, b_theta_init, b_z_init, v_r_init, v_theta_init, v_z_init,
             n_init, p_e_init, 0.0, dt, dr, dz, r, eta)

    final_state, history = lax.scan(step, state, jnp.arange(steps))
    return final_state[:3], history

if __name__ == "__main__":
    b_final, history = run_simulation(100)
    print(f"Extended MHD simulation complete. B_z max: {jnp.max(b_final[2]):.6f}")
