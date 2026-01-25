import jax
import jax.numpy as jnp
from jax import jit, lax, grad, vmap
import jax.random as random

from jax_frc.operators import (
    gradient_r,
    gradient_z,
    curl_cylindrical_axisymmetric,
)

MU0 = 1.2566e-6
QE = 1.602e-19
ME = 9.109e-31
MI = 1.673e-27

@jit
def rigid_rotor_f0(r, z, vr, vz, vtheta, n0, T0, Omega):
    """
    Rigid rotor equilibrium distribution function.
    f0 = n0 * (m/(2*pi*T))^(3/2) * exp(-m/(2T) * (v_r^2 + (v_theta - Omega*r)^2 + v_z^2))
    """
    v_sq = vr**2 + (vtheta - Omega * r)**2 + vz**2
    thermal_factor = (MI / (2 * jnp.pi * T0)) ** 1.5
    return n0 * thermal_factor * jnp.exp(-MI * v_sq / (2 * T0))

@jit
def compute_electric_field(n_e, p_e, j_i, b, eta, dr, dz, r):
    """
    Computes electric field from electron fluid equation.
    E = (J_total × B) / (ne) - grad(p_e) / (ne) + eta * J_total

    In hybrid model, J_total = J_i + J_e where J_e = -ne * v_e
    Using quasi-neutrality and current balance: J_total = curl(B)/μ₀
    The Hall term (J×B)/(ne) captures the separation of ion and electron dynamics.

    Uses correct cylindrical curl with L'Hopital at axis.
    """
    # Compute total current from Ampere's law: J = curl(B)/μ₀ in cylindrical
    b_r = b[:, :, 0]
    b_theta = b[:, :, 1]
    b_z = b[:, :, 2]

    # curl(B) in cylindrical (r, theta, z) with axisymmetry:
    # J_r = -∂B_θ/∂z
    # J_θ = ∂B_r/∂z - ∂B_z/∂r
    # J_z = (1/r)*∂(r*B_θ)/∂r = B_θ/r + ∂B_θ/∂r (with L'Hopital at r=0)
    j_r_total, j_theta_total, j_z_total = curl_cylindrical_axisymmetric(
        b_r, b_theta, b_z, dr, dz, r
    )
    j_r_total = j_r_total / MU0
    j_theta_total = j_theta_total / MU0
    j_z_total = j_z_total / MU0

    j_total = jnp.stack([j_r_total, j_theta_total, j_z_total], axis=-1)

    # Hall term: (J_total × B) / (ne)
    j_cross_b = jnp.cross(j_total, b)
    n_e_safe = jnp.maximum(n_e, 1e16)  # Avoid division by zero
    hall_term = j_cross_b / (n_e_safe[:, :, None] * QE)

    # Electron pressure gradient: -grad(p_e) / (ne)
    dp_e_dr = gradient_r(p_e, dr)
    dp_e_dz = gradient_z(p_e, dz)
    grad_p_e = jnp.stack([dp_e_dr, jnp.zeros_like(dp_e_dr), dp_e_dz], axis=-1)
    pressure_term = -grad_p_e / (n_e_safe[:, :, None] * QE)

    # Resistive term: eta * J
    resistive_term = eta * j_total

    E = hall_term + pressure_term + resistive_term

    # Cap E field to prevent numerical instability
    # Typical FRC E field ~ 1000 V/m; use 1e4 V/m as conservative cap
    E_max = 1e4
    E = jnp.clip(E, -E_max, E_max)

    return E


@jit
def grad_2d(f, dr, dz):
    """
    Computes 2D gradient using non-periodic boundaries.
    """
    df_dr = gradient_r(f, dr)
    df_dz = gradient_z(f, dz)
    return df_dr, df_dz

@jit
def cylindrical_to_cartesian_vel(v_r, v_theta, v_z, theta):
    """
    Transform velocity from cylindrical (r,theta,z) to Cartesian (x,y,z).
    v_x = v_r*cos(theta) - v_theta*sin(theta)
    v_y = v_r*sin(theta) + v_theta*cos(theta)
    v_z = v_z (unchanged)
    """
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    v_x = v_r * cos_theta - v_theta * sin_theta
    v_y = v_r * sin_theta + v_theta * cos_theta
    return v_x, v_y, v_z


@jit
def cartesian_to_cylindrical_vel(v_x, v_y, v_z, theta):
    """
    Transform velocity from Cartesian (x,y,z) to cylindrical (r,theta,z).
    v_r = v_x*cos(theta) + v_y*sin(theta)
    v_theta = -v_x*sin(theta) + v_y*cos(theta)
    v_z = v_z (unchanged)
    """
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    v_r = v_x * cos_theta + v_y * sin_theta
    v_theta = -v_x * sin_theta + v_y * cos_theta
    return v_r, v_theta, v_z


@jit
def boris_push(x, v, E, B, q, m, dt):
    """
    Boris particle pusher for ion motion.

    This operates in Cartesian coordinates for correct rotation handling.
    Input/output positions x are in Cartesian (x, y, z).
    Input/output velocities v are in cylindrical (v_r, v_theta, v_z).
    Fields E, B are in cylindrical components at particle positions.
    """
    # Get theta angle for coordinate transforms
    x_cart = x[:, 0]
    y_cart = x[:, 1]
    theta = jnp.arctan2(y_cart, x_cart)

    # Convert velocity to Cartesian for Boris push
    v_r, v_theta, v_z = v[:, 0], v[:, 1], v[:, 2]
    v_x, v_y, v_z_cart = cylindrical_to_cartesian_vel(v_r, v_theta, v_z, theta)
    v_cart = jnp.stack([v_x, v_y, v_z_cart], axis=-1)

    # Convert E and B fields to Cartesian
    E_r, E_theta, E_z = E[:, 0], E[:, 1], E[:, 2]
    B_r, B_theta, B_z = B[:, 0], B[:, 1], B[:, 2]
    E_x, E_y, E_z_cart = cylindrical_to_cartesian_vel(E_r, E_theta, E_z, theta)
    B_x, B_y, B_z_cart = cylindrical_to_cartesian_vel(B_r, B_theta, B_z, theta)
    E_cart = jnp.stack([E_x, E_y, E_z_cart], axis=-1)
    B_cart = jnp.stack([B_x, B_y, B_z_cart], axis=-1)

    # Standard Boris algorithm in Cartesian
    t = q * B_cart * dt / (2 * m)
    s = 2 * t / (1 + jnp.sum(t**2, axis=-1, keepdims=True))

    v_minus = v_cart + q * E_cart * dt / (2 * m)
    v_prime = v_minus + jnp.cross(v_minus, t)
    v_plus = v_minus + jnp.cross(v_prime, s)
    v_new_cart = v_plus + q * E_cart * dt / (2 * m)

    # Update position in Cartesian
    x_new = x + v_new_cart * dt

    # Get new theta for velocity back-transform
    theta_new = jnp.arctan2(x_new[:, 1], x_new[:, 0])

    # Convert velocity back to cylindrical
    v_r_new, v_theta_new, v_z_new = cartesian_to_cylindrical_vel(
        v_new_cart[:, 0], v_new_cart[:, 1], v_new_cart[:, 2], theta_new
    )
    v_new = jnp.stack([v_r_new, v_theta_new, v_z_new], axis=-1)

    return x_new, v_new

@jit
def weight_evolution(w, x, v, E_particle, B_particle, n0, T0, Omega, dt):
    """
    Evolves particle weights using delta-f method.
    dw/dt = -(1-w) * d ln f0 / dt

    d ln f₀/dt = (1/f₀) * (∂f₀/∂r * v_r + ∂f₀/∂v · a)
    where a = (q/m)(E + v×B) is the Lorentz acceleration.
    """
    # Extract r from Cartesian coordinates
    r = jnp.sqrt(x[:, 0]**2 + x[:, 1]**2)
    z = x[:, 2]
    vr = v[:, 0]
    vtheta = v[:, 1]
    vz = v[:, 2]

    # Compute f₀ and its gradients
    f0 = rigid_rotor_f0(r, z, vr, vz, vtheta, n0, T0, Omega)

    # Common exponential factor
    v_sq = vr**2 + (vtheta - Omega * r)**2 + vz**2
    prefactor = n0 * (MI / (2 * jnp.pi * T0)) ** 1.5 * jnp.exp(-MI * v_sq / (2 * T0))

    # Gradients of f₀:
    # ∂f₀/∂r = f₀ * (M_i * Ω * (v_θ - Ω*r) / T₀)
    df0_dr = prefactor * (MI * Omega * (vtheta - Omega * r) / T0)

    # ∂f₀/∂v_r = f₀ * (-M_i * v_r / T₀)
    df0_dvr = prefactor * (-MI * vr / T0)

    # ∂f₀/∂v_z = f₀ * (-M_i * v_z / T₀)
    df0_dvz = prefactor * (-MI * vz / T0)

    # ∂f₀/∂v_θ = f₀ * (-M_i * (v_θ - Ω*r) / T₀)
    df0_dvtheta = prefactor * (-MI * (vtheta - Omega * r) / T0)

    # Compute Lorentz acceleration: a = (q/m)(E + v×B)
    # v×B in Cartesian-like (r, theta, z):
    v_cross_B = jnp.stack([
        vtheta * B_particle[:, 2] - vz * B_particle[:, 1],  # (v×B)_r
        vz * B_particle[:, 0] - vr * B_particle[:, 2],       # (v×B)_θ
        vr * B_particle[:, 1] - vtheta * B_particle[:, 0]    # (v×B)_z
    ], axis=-1)

    a = (QE / MI) * (E_particle + v_cross_B)
    a_r = a[:, 0]
    a_theta = a[:, 1]
    a_z = a[:, 2]

    # d ln f₀/dt = (1/f₀) * (∂f₀/∂r * v_r + ∂f₀/∂v_r * a_r + ∂f₀/∂v_θ * a_θ + ∂f₀/∂v_z * a_z)
    # Avoid division by zero
    f0_safe = jnp.maximum(f0, 1e-30)
    dlnf0_dt = (df0_dr * vr + df0_dvr * a_r + df0_dvtheta * a_theta + df0_dvz * a_z) / f0_safe

    # Weight evolution: dw/dt = -(1-w) * d ln f₀/dt
    dw = -(1 - w) * dlnf0_dt
    w_new = w + dw * dt

    return jnp.clip(w_new, -1.0, 1.0)

@jit
def deposit_current(x, v, w, n_e, dr, dz):
    """
    Deposits ion current onto grid using particle weights.
    Particle positions are in Cartesian: [x, y, z] where r = sqrt(x^2 + y^2)
    n_e is used as a template for the output shape.
    """
    nr, nz = n_e.shape

    # Extract r from Cartesian coordinates
    r = jnp.sqrt(x[:, 0]**2 + x[:, 1]**2)
    z = x[:, 2]
    vr = v[:, 0]
    vtheta = v[:, 1]
    vz = v[:, 2]

    r_idx = jnp.clip((r / dr).astype(jnp.int32), 0, nr - 1)
    z_idx = jnp.clip(((z + 1.0) / dz).astype(jnp.int32), 0, nz - 1)

    # Use zeros_like to create arrays with same shape as n_e
    j_r = jnp.zeros_like(n_e)
    j_theta = jnp.zeros_like(n_e)
    j_z = jnp.zeros_like(n_e)

    # Scale by charge and particle weight
    q_factor = QE * (1 + w)  # delta-f contribution
    j_r = j_r.at[r_idx, z_idx].add(q_factor * vr)
    j_theta = j_theta.at[r_idx, z_idx].add(q_factor * vtheta)
    j_z = j_z.at[r_idx, z_idx].add(q_factor * vz)

    return j_r, j_theta, j_z

@jit
def bilinear_interpolate(field, r_pos, z_pos, dr, dz, nr, nz):
    """
    Bilinear (cloud-in-cell) interpolation for PIC.

    Args:
        field: Grid field, shape (nr, nz) or (nr, nz, 3) for vectors
        r_pos: Radial particle positions
        z_pos: Axial particle positions (shifted to [0, 2])
        dr, dz: Grid spacings
        nr, nz: Grid dimensions

    Returns:
        Interpolated field values at particle positions
    """
    # Map positions to grid indices (continuous)
    r_idx = r_pos / dr
    z_idx = (z_pos + 1.0) / dz  # z in [-1, 1] -> [0, 2/dz]

    # Get lower indices
    i0 = jnp.floor(r_idx).astype(jnp.int32)
    j0 = jnp.floor(z_idx).astype(jnp.int32)

    # Get fractional positions
    dr_frac = r_idx - i0
    dz_frac = z_idx - j0

    # Clip indices to valid range
    i0 = jnp.clip(i0, 0, nr - 2)
    j0 = jnp.clip(j0, 0, nz - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    # Bilinear weights
    w00 = (1 - dr_frac) * (1 - dz_frac)
    w01 = (1 - dr_frac) * dz_frac
    w10 = dr_frac * (1 - dz_frac)
    w11 = dr_frac * dz_frac

    if field.ndim == 3:
        # Vector field: shape (nr, nz, 3)
        result = (
            w00[:, None] * field[i0, j0, :] +
            w01[:, None] * field[i0, j1, :] +
            w10[:, None] * field[i1, j0, :] +
            w11[:, None] * field[i1, j1, :]
        )
    else:
        # Scalar field: shape (nr, nz)
        result = (
            w00 * field[i0, j0] +
            w01 * field[i0, j1] +
            w10 * field[i1, j0] +
            w11 * field[i1, j1]
        )

    return result


@jit
def interpolate_field_to_particles(field, x, n_e, dr, dz):
    """
    Interpolates a grid field to particle positions using bilinear interpolation.
    field: (nr, nz, 3) array for vector fields or (nr, nz) for scalar
    x: (n_particles, 3) particle positions [x, y, z] in Cartesian
    n_e: template array for getting grid dimensions
    """
    nr, nz = n_e.shape

    # Extract r and z from Cartesian particle positions
    r = jnp.sqrt(x[:, 0]**2 + x[:, 1]**2)
    z = x[:, 2]

    return bilinear_interpolate(field, r, z, dr, dz, nr, nz)


@jit
def faraday_step(b, E, dr, dz, r, dt):
    """
    Evolve magnetic field using Faraday's law: dB/dt = -curl(E)

    Uses cylindrical curl with proper axis handling.
    """
    b_r = b[:, :, 0]
    b_theta = b[:, :, 1]
    b_z = b[:, :, 2]

    E_r = E[:, :, 0]
    E_theta = E[:, :, 1]
    E_z = E[:, :, 2]

    # curl(E) in cylindrical coordinates
    curl_E_r, curl_E_theta, curl_E_z = curl_cylindrical_axisymmetric(
        E_r, E_theta, E_z, dr, dz, r
    )

    # Faraday's law: dB/dt = -curl(E)
    b_r_new = b_r - dt * curl_E_r
    b_theta_new = b_theta - dt * curl_E_theta
    b_z_new = b_z - dt * curl_E_z

    return jnp.stack([b_r_new, b_theta_new, b_z_new], axis=-1)


@jit
def deposit_density_cic(x, w, dr, dz, n_e, n0):
    """
    Cloud-in-cell density deposition for delta-f PIC.

    The deposited density is: n_i = n0 * (1 + sum of particle contributions)
    For delta-f: contribution is proportional to weight w.

    Args:
        x: Particle positions (n_particles, 3) in Cartesian
        w: Particle weights for delta-f
        dr, dz: Grid spacings
        n_e: Template array for grid shape (nr, nz)
        n0: Background density

    Returns:
        Ion density field (nr, nz)
    """
    nr, nz = n_e.shape

    # Extract r and z from Cartesian positions
    r_pos = jnp.sqrt(x[:, 0]**2 + x[:, 1]**2)
    z_pos = x[:, 2]

    # Map to grid indices
    r_idx = r_pos / dr
    z_idx = (z_pos + 1.0) / dz

    # Lower indices
    i0 = jnp.floor(r_idx).astype(jnp.int32)
    j0 = jnp.floor(z_idx).astype(jnp.int32)

    # Fractional positions
    dr_frac = r_idx - i0
    dz_frac = z_idx - j0

    # Clip indices
    i0 = jnp.clip(i0, 0, nr - 2)
    j0 = jnp.clip(j0, 0, nz - 2)
    i1 = i0 + 1
    j1 = j0 + 1

    # CIC weights
    w00 = (1 - dr_frac) * (1 - dz_frac)
    w01 = (1 - dr_frac) * dz_frac
    w10 = dr_frac * (1 - dz_frac)
    w11 = dr_frac * dz_frac

    # Contribution from delta-f: proportional to (1 + w)
    delta_f_contrib = (1.0 + w)

    # Initialize density field using zeros_like for JIT compatibility
    n_i = jnp.zeros_like(n_e)

    # Deposit weights (CIC scatter)
    n_i = n_i.at[i0, j0].add(w00 * delta_f_contrib)
    n_i = n_i.at[i0, j1].add(w01 * delta_f_contrib)
    n_i = n_i.at[i1, j0].add(w10 * delta_f_contrib)
    n_i = n_i.at[i1, j1].add(w11 * delta_f_contrib)

    # Normalize: multiply by n0 and divide by number of particles per cell
    # This is a simplified normalization; proper PIC would account for cell volume
    n_particles = x.shape[0]
    particles_per_cell = n_particles / (nr * nz)
    n_i = n0 * n_i / jnp.maximum(particles_per_cell, 1.0)

    # Ensure minimum density
    n_i = jnp.maximum(n_i, 1e16)

    return n_i

@jit
def step(state, _):
    x, v, w, n_e, p_e, b, t, dt, dr, dz, nr, nz, r_grid, n0, T0, Omega, eta = state

    # Deposit ion current onto grid
    j_r, j_theta, j_z = deposit_current(x, v, w, n_e, dr, dz)
    j_i = jnp.stack([j_r, j_theta, j_z], axis=-1)

    # Compute electric field on grid (with r array for cylindrical operators)
    E = compute_electric_field(n_e, p_e, j_i, b, eta, dr, dz, r_grid)

    # Interpolate E and B fields to particle positions
    E_particle = interpolate_field_to_particles(E, x, n_e, dr, dz)
    B_particle = interpolate_field_to_particles(b, x, n_e, dr, dz)

    # Boris push for ion motion
    x_new, v_new = boris_push(x, v, E_particle, B_particle, QE, MI, dt)

    # Update particle weights using delta-f method
    w_new = weight_evolution(w, x_new, v_new, E_particle, B_particle, n0, T0, Omega, dt)

    # Evolve magnetic field using Faraday's law: dB/dt = -curl(E)
    b_new = faraday_step(b, E, dr, dz, r_grid, dt)

    # Apply B field boundary conditions
    # Axis (r=0): B_r = 0, B_theta = 0 by symmetry
    b_new = b_new.at[0, :, 0].set(0.0)  # B_r = 0
    b_new = b_new.at[0, :, 1].set(0.0)  # B_theta = 0
    # Outer boundary: Neumann-like
    b_new = b_new.at[-1, :, :].set(b_new[-2, :, :])
    # Axial boundaries
    b_new = b_new.at[:, 0, :].set(b_new[:, 1, :])
    b_new = b_new.at[:, -1, :].set(b_new[:, -2, :])

    # Update electron density from quasi-neutrality: n_e = n_i
    n_e_new = deposit_density_cic(x_new, w_new, dr, dz, n_e, n0)

    # Apply periodic boundary conditions in z
    z_new = x_new[:, 2]
    z_wrapped = jnp.where(z_new > 1.0, z_new - 2.0, z_new)
    z_wrapped = jnp.where(z_wrapped < -1.0, z_wrapped + 2.0, z_wrapped)
    x_new = x_new.at[:, 2].set(z_wrapped)

    # Reflect particles at radial boundaries
    r_new = jnp.sqrt(x_new[:, 0]**2 + x_new[:, 1]**2)
    mask_out = r_new > 0.9
    # Reflect radial velocity
    v_r = v_new[:, 0]
    v_new = v_new.at[:, 0].set(jnp.where(mask_out, -v_r, v_r))

    return (x_new, v_new, w_new, n_e_new, p_e, b_new, t + dt, dt, dr, dz, nr, nz, r_grid, n0, T0, Omega, eta), (x, v, w)

def initialize_particles(n_particles, key, nr, nz, n0, T0, Omega):
    """
    Initializes particle positions and velocities from rigid rotor equilibrium.
    """
    key_r, key_z, key_vr, key_vz, key_vtheta = random.split(key, 5)
    
    r = random.uniform(key_r, (n_particles,), minval=0.01, maxval=0.5)
    z = random.uniform(key_z, (n_particles,), minval=-0.5, maxval=0.5)
    theta = random.uniform(random.PRNGKey(42), (n_particles,), minval=0, maxval=2*jnp.pi)
    
    v_thermal = jnp.sqrt(T0 / MI)
    vr = random.normal(key_vr, (n_particles,)) * v_thermal
    vz = random.normal(key_vz, (n_particles,)) * v_thermal
    vtheta = random.normal(key_vtheta, (n_particles,)) * v_thermal + Omega * r
    
    x = jnp.stack([r * jnp.cos(theta), r * jnp.sin(theta), z], axis=-1)
    v = jnp.stack([vr, vtheta, vz], axis=-1)
    
    w = jnp.zeros(n_particles)
    
    return x, v, w

def run_simulation(steps=100, n_particles=10000, nr=32, nz=64, dt=1e-8, eta=1e-4):
    key = random.PRNGKey(0)
    dr, dz = 1.0/nr, 2.0/nz

    n0 = 1e19
    T0 = 100.0 * QE  # 100 eV in Joules (thermal velocity = sqrt(T0/MI))
    Omega = 1e5

    x, v, w = initialize_particles(n_particles, key, nr, nz, n0, T0, Omega)

    n_e = jnp.ones((nr, nz)) * n0
    p_e = jnp.ones((nr, nz)) * T0 * n_e

    # Create r coordinate array for cylindrical operators
    r_grid = jnp.linspace(0.01, 1, nr)[:, None]  # Avoid r=0
    z = jnp.linspace(-1, 1, nz)[None, :]

    b_r = jnp.zeros((nr, nz))
    b_theta = jnp.zeros((nr, nz))
    b_z = 1.0 * jnp.exp(-r_grid**2 - z**2)
    b = jnp.stack([b_r, b_theta, b_z], axis=-1)

    # State: (x, v, w, n_e, p_e, b, t, dt, dr, dz, nr, nz, r_grid, n0, T0, Omega, eta)
    state = (x, v, w, n_e, p_e, b, 0.0, dt, dr, dz, nr, nz, r_grid, n0, T0, Omega, eta)

    final_state, history = lax.scan(step, state, jnp.arange(steps))
    return final_state[:3], history

if __name__ == "__main__":
    (x_final, v_final, w_final), history = run_simulation(100, n_particles=1000)
    print(f"Hybrid Kinetic simulation complete. Number of particles: {x_final.shape[0]}")
