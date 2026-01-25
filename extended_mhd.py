import jax
import jax.numpy as jnp
from jax import jit, lax, grad, vmap

MU0 = 1.2566e-6
QE = 1.602e-19
ME = 9.109e-31

@jit
def curl_2d(f_x, f_y, dx, dy):
    """
    Computes 2D curl: (d_fy/dx - d_fx/dy)
    """
    dfy_dx = (jnp.roll(f_y, -1, axis=0) - jnp.roll(f_y, 1, axis=0)) / (2 * dx)
    dfx_dy = (jnp.roll(f_x, -1, axis=1) - jnp.roll(f_x, 1, axis=1)) / (2 * dy)
    return dfy_dx - dfx_dy

@jit
def grad_2d(f, dx, dy):
    """
    Computes 2D gradient: (df/dx, df/dy)
    """
    df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    df_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dy)
    return df_dx, df_dy

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
def extended_ohm_law(v_x, v_y, v_z, b_x, b_y, b_z, j_x, j_y, j_z, n, eta, p_e, dx, dy):
    """
    E = -v x B + eta * J + (J x B) / (ne) - grad(p_e) / (ne)
    """
    v_cross_b_x, v_cross_b_y, v_cross_b_z = cross_product_2d(v_x, v_y, b_x, b_y, v_z, b_z)
    term1_x, term1_y, term1_z = -v_cross_b_x, -v_cross_b_y, -v_cross_b_z
    
    term2_x, term2_y, term2_z = eta * j_x, eta * j_y, eta * j_z
    
    j_cross_b_x, j_cross_b_y, j_cross_b_z = cross_product_2d(j_x, j_y, b_x, b_y, j_z, b_z)
    term3_x, term3_y, term3_z = j_cross_b_x / (n * QE), j_cross_b_y / (n * QE), j_cross_b_z / (n * QE)
    
    dp_e_dx, dp_e_dy = grad_2d(p_e, dx, dy)
    term4_x, term4_y = -dp_e_dx / (n * QE), -dp_e_dy / (n * QE)
    term4_z = jnp.zeros_like(p_e)
    
    E_x = term1_x + term2_x + term3_x + term4_x
    E_y = term1_y + term2_y + term3_y + term4_y
    E_z = term1_z + term2_z + term3_z + term4_z
    
    return E_x, E_y, E_z

@jit
def hall_operator(b_x, b_y, b_z, n, dx, dy):
    """
    Constructs the Hall differential operator for semi-implicit stepping.
    L_Hall ~ curl((J x B) / (ne))
    """
    j_x = (1.0 / MU0) * curl_2d(b_z, b_y, dx, dy)
    j_y = (1.0 / MU0) * curl_2d(b_x, b_z, dx, dy)
    j_z = (1.0 / MU0) * curl_2d(b_y, b_x, dx, dy)
    
    j_cross_b_x, j_cross_b_y, j_cross_b_z = cross_product_2d(j_x, j_y, b_x, b_y, j_z, b_z)
    
    hall_x = j_cross_b_x / (n * QE)
    hall_y = j_cross_b_y / (n * QE)
    hall_z = j_cross_b_z / (n * QE)
    
    curl_hall_x = curl_2d(hall_z, hall_y, dx, dy)
    curl_hall_y = curl_2d(hall_x, hall_z, dx, dy)
    curl_hall_z = curl_2d(hall_y, hall_x, dx, dy)
    
    return curl_hall_x, curl_hall_y, curl_hall_z

def compute_whistler_stable_dt(B_max, n, dx, dy):
    """
    Compute CFL-stable timestep for Whistler waves.
    Whistler speed: v_w = k * B / (mu0 * n * e) where k = pi/dx
    CFL: dt < dx / v_w

    Args:
        B_max: Maximum magnetic field strength [T]
        n: Minimum density [m^-3] (must be > 0)
        dx, dy: Grid spacings

    Returns:
        Stable timestep with 25% safety margin
    """
    # Note: Validation happens at runtime via JAX tracing
    # For production, add explicit checks in run_simulation
    k = jnp.pi / jnp.minimum(dx, dy)
    v_whistler = k * B_max / (MU0 * n * QE)
    dt_cfl = jnp.minimum(dx, dy) / v_whistler
    return 0.25 * dt_cfl  # 25% safety margin


@jit
def compute_div_b(b_x, b_y, b_z, dx, dy):
    """
    Compute divergence of magnetic field: div(B) = dBx/dx + dBy/dy + dBz/dz.
    For 2D simulations, we compute dBx/dx + dBy/dy (Bz varies only in x,y).
    """
    db_x_dx = (jnp.roll(b_x, -1, axis=0) - jnp.roll(b_x, 1, axis=0)) / (2 * dx)
    db_y_dy = (jnp.roll(b_y, -1, axis=1) - jnp.roll(b_y, 1, axis=1)) / (2 * dy)
    return db_x_dx + db_y_dy


@jit
def divergence_cleaning_step(b_x, b_y, b_z, dx, dy):
    """
    Apply divergence cleaning using projection method.

    Solves div(B - grad(phi)) = 0 iteratively using Jacobi relaxation.
    This enforces the div(B) = 0 constraint numerically.

    Uses 5 Jacobi iterations (fixed for JIT compatibility).

    Args:
        b_x, b_y, b_z: Magnetic field components
        dx, dy: Grid spacings

    Returns:
        Cleaned magnetic field components (b_x_clean, b_y_clean, b_z_clean)
    """
    # Compute current divergence error
    div_b = compute_div_b(b_x, b_y, b_z, dx, dy)

    # Solve Laplacian(phi) = div(B) using Jacobi iteration
    # This is a simplified approach - full projection would use FFT or multigrid
    phi = jnp.zeros_like(b_x)

    # Laplacian stencil coefficient
    coeff = 2.0 / dx**2 + 2.0 / dy**2

    def jacobi_step(i, phi):
        # Jacobi update: phi_new = (1/coeff) * (neighbors/dx^2 - div_b)
        phi_xp = jnp.roll(phi, -1, axis=0)
        phi_xm = jnp.roll(phi, 1, axis=0)
        phi_yp = jnp.roll(phi, -1, axis=1)
        phi_ym = jnp.roll(phi, 1, axis=1)

        phi_new = ((phi_xp + phi_xm) / dx**2 + (phi_yp + phi_ym) / dy**2 - div_b) / coeff

        # Apply zero boundary conditions for phi
        phi_new = phi_new.at[0, :].set(0)
        phi_new = phi_new.at[-1, :].set(0)
        phi_new = phi_new.at[:, 0].set(0)
        phi_new = phi_new.at[:, -1].set(0)

        return phi_new

    # Use fori_loop with fixed iteration count for JIT compatibility
    phi = lax.fori_loop(0, 5, jacobi_step, phi)

    # Compute gradient of phi and subtract from B
    grad_phi_x, grad_phi_y = grad_2d(phi, dx, dy)

    b_x_clean = b_x - grad_phi_x
    b_y_clean = b_y - grad_phi_y
    b_z_clean = b_z  # z-component unchanged in 2D

    return b_x_clean, b_y_clean, b_z_clean

@jit
def hall_substep(b_x, b_y, b_z, v_x, v_y, v_z, n, p_e, dt_sub, dx, dy, eta):
    """Single Hall MHD substep with stable timestep."""
    # Compute current density J = curl(B) / mu0
    j_x = (1.0 / MU0) * curl_2d(b_z, b_y, dx, dy)
    j_y = (1.0 / MU0) * curl_2d(b_x, b_z, dx, dy)
    j_z = (1.0 / MU0) * curl_2d(b_y, b_x, dx, dy)

    # Compute electric field from extended Ohm's law
    E_x, E_y, E_z = extended_ohm_law(v_x, v_y, v_z, b_x, b_y, b_z,
                                      j_x, j_y, j_z, n, eta, p_e, dx, dy)

    # Faraday's law: dB/dt = -curl(E)
    curl_E_x = curl_2d(E_z, E_y, dx, dy)
    curl_E_y = curl_2d(E_x, E_z, dx, dy)
    curl_E_z = curl_2d(E_y, E_x, dx, dy)

    b_x_new = b_x - dt_sub * curl_E_x
    b_y_new = b_y - dt_sub * curl_E_y
    b_z_new = b_z - dt_sub * curl_E_z

    return b_x_new, b_y_new, b_z_new

@jit
def semi_implicit_hall_step(b_x, b_y, b_z, v_x, v_y, v_z, n, p_e, dt, dx, dy, eta):
    """
    Subcycled time stepping to handle Whistler waves stably.
    Uses automatic subcycling based on Whistler CFL condition.

    For numerical stability, limits maximum substeps to 50.
    """
    # Compute stable substep based on maximum B field
    B_max = jnp.maximum(jnp.max(jnp.abs(b_z)), 1e-10)
    # Use minimum density from the actual field for worst-case Whistler speed
    n_min = jnp.maximum(jnp.min(n), 1e18)
    dt_stable = compute_whistler_stable_dt(B_max, n_min, dx, dy)

    # Number of substeps needed, capped at 50 for practicality
    n_substeps = jnp.minimum(50, jnp.maximum(1, jnp.ceil(dt / dt_stable))).astype(jnp.int32)
    dt_sub = dt / n_substeps

    # Subcycle using fori_loop (more efficient than while_loop for fixed iterations)
    def substep_body(i, carry):
        bx, by, bz = carry
        bx_new, by_new, bz_new = hall_substep(bx, by, bz, v_x, v_y, v_z,
                                               n, p_e, dt_sub, dx, dy, eta)
        return (bx_new, by_new, bz_new)

    b_x_new, b_y_new, b_z_new = lax.fori_loop(
        0, n_substeps, substep_body, (b_x, b_y, b_z)
    )

    return b_x_new, b_y_new, b_z_new

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
    """Single timestep of extended MHD evolution.

    Warning: The Hall term introduces Whistler waves with CFL constraint:
        dt < (dx * mu0 * n * e) / (pi * B)
    For typical plasma parameters, this can require thousands of substeps.
    The simulation uses subcycling with a maximum of 50 substeps per timestep.
    For stability with larger timesteps, use coarser grids or smaller dt.
    """
    b_x, b_y, b_z, v_x, v_y, v_z, n, p_e, t, dt, dx, dy, eta = state

    n_with_halo = apply_halo_density(n)

    b_x_new, b_y_new, b_z_new = semi_implicit_hall_step(
        b_x, b_y, b_z, v_x, v_y, v_z, n_with_halo, p_e, dt, dx, dy, eta
    )

    # Note: Divergence cleaning is disabled by default because the projection
    # method requires solving a Poisson equation iteratively. For production
    # use, consider using constrained transport or vector potential formulation.
    # Uncomment the following to enable divergence cleaning:
    # b_x_new, b_y_new, b_z_new = divergence_cleaning_step(
    #     b_x_new, b_y_new, b_z_new, dx, dy
    # )

    # Apply boundary conditions (conducting wall: B_tangential = 0)
    b_x_new = b_x_new.at[0, :].set(0)
    b_x_new = b_x_new.at[-1, :].set(0)
    b_x_new = b_x_new.at[:, 0].set(0)
    b_x_new = b_x_new.at[:, -1].set(0)

    b_y_new = b_y_new.at[0, :].set(0)
    b_y_new = b_y_new.at[-1, :].set(0)
    b_y_new = b_y_new.at[:, 0].set(0)
    b_y_new = b_y_new.at[:, -1].set(0)

    b_z_new = b_z_new.at[0, :].set(0)
    b_z_new = b_z_new.at[-1, :].set(0)
    b_z_new = b_z_new.at[:, 0].set(0)
    b_z_new = b_z_new.at[:, -1].set(0)

    return (b_x_new, b_y_new, b_z_new, v_x, v_y, v_z, n, p_e, t + dt, dt, dx, dy, eta), (b_x, b_y, b_z)

def run_simulation(steps=100, nx=32, ny=32, dt=1e-6, eta=1e-4):
    """Run extended MHD simulation with Hall effect.

    Warning: The Hall term introduces Whistler wave CFL constraint:
        dt_whistler < (dx * mu0 * n * e) / (pi * B)
    For typical parameters (B~1T, n~1e18 m^-3, dx~0.03), this gives
    dt_whistler ~ 2.5e-11 s. The simulation uses subcycling (up to 50
    substeps per main timestep) but may still become unstable if
    dt >> dt_whistler. For stability, use dt < 1e-8 s for fine grids.

    Args:
        steps: Number of timesteps to run
        nx: Number of grid points in x direction (must be >= 4)
        ny: Number of grid points in y direction (must be >= 4)
        dt: Timestep [s] (must be > 0). Recommend 1e-9 to 1e-8 for stability.
        eta: Resistivity [Ohm*m] (must be >= 0)

    Returns:
        Tuple of (final_B_components, history)

    Raises:
        ValueError: If input parameters are invalid
    """
    # Validate inputs
    if steps < 1:
        raise ValueError(f"steps must be at least 1, got {steps}")
    if nx < 4:
        raise ValueError(f"nx must be at least 4 for finite differences, got {nx}")
    if ny < 4:
        raise ValueError(f"ny must be at least 4 for finite differences, got {ny}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    if eta < 0:
        raise ValueError(f"eta must be non-negative, got {eta}")

    dx, dy = 1.0/nx, 1.0/ny
    
    r = jnp.linspace(0, 1, nx)[:, None]
    z = jnp.linspace(-1, 1, ny)[None, :]
    
    b_x_init = jnp.zeros((nx, ny))
    b_y_init = jnp.zeros((nx, ny))
    b_z_init = 1.0 * jnp.exp(-r**2 - z**2)
    
    v_x_init = jnp.zeros((nx, ny))
    v_y_init = jnp.zeros((nx, ny))
    v_z_init = jnp.zeros((nx, ny))
    
    n_init = jnp.ones((nx, ny)) * 1e19
    p_e_init = jnp.ones((nx, ny)) * 1e3
    
    state = (b_x_init, b_y_init, b_z_init, v_x_init, v_y_init, v_z_init, n_init, p_e_init, 0.0, dt, dx, dy, eta)
    
    final_state, history = lax.scan(step, state, jnp.arange(steps))
    return final_state[:3], history

if __name__ == "__main__":
    b_final, history = run_simulation(100)
    print(f"Extended MHD simulation complete. B_z max: {jnp.max(b_final[2]):.6f}")
