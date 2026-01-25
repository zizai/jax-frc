import jax
import jax.numpy as jnp
from jax import jit, lax, grad, vmap
import jax.random as random

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
def compute_electric_field(n_e, p_e, j_i, b, eta, dr, dz):
    """
    Computes electric field from electron fluid equation.
    E = (J_total × B) / (ne) - grad(p_e) / (ne) + eta * J_total

    In hybrid model, J_total = J_i + J_e where J_e = -ne * v_e
    Using quasi-neutrality and current balance: J_total = curl(B)/μ₀
    The Hall term (J×B)/(ne) captures the separation of ion and electron dynamics.
    """
    # Compute total current from Ampere's law: J = curl(B)/μ₀
    # In 2D (r,z), we compute the curl of B
    b_r = b[:, :, 0]
    b_theta = b[:, :, 1]
    b_z = b[:, :, 2]

    # curl(B) in cylindrical (r, theta, z):
    # J_r = (1/r)∂B_z/∂θ - ∂B_θ/∂z ≈ -∂B_θ/∂z (axisymmetric)
    # J_θ = ∂B_r/∂z - ∂B_z/∂r
    # J_z = (1/r)∂(r*B_θ)/∂r - (1/r)∂B_r/∂θ ≈ (1/r)∂(r*B_θ)/∂r

    db_theta_dz = (jnp.roll(b_theta, -1, axis=1) - jnp.roll(b_theta, 1, axis=1)) / (2 * dz)
    db_r_dz = (jnp.roll(b_r, -1, axis=1) - jnp.roll(b_r, 1, axis=1)) / (2 * dz)
    db_z_dr = (jnp.roll(b_z, -1, axis=0) - jnp.roll(b_z, 1, axis=0)) / (2 * dr)

    j_r_total = -db_theta_dz / MU0
    j_theta_total = (db_r_dz - db_z_dr) / MU0
    j_z_total = db_theta_dz / MU0  # Simplified for axisymmetry

    j_total = jnp.stack([j_r_total, j_theta_total, j_z_total], axis=-1)

    # Hall term: (J_total × B) / (ne)
    j_cross_b = jnp.cross(j_total, b)
    hall_term = j_cross_b / (n_e[:, :, None] * QE + 1e-30)

    # Electron pressure gradient: -grad(p_e) / (ne)
    dp_e_dr, dp_e_dz = grad_2d(p_e, dr, dz)
    grad_p_e = jnp.stack([dp_e_dr, jnp.zeros_like(dp_e_dr), dp_e_dz], axis=-1)
    pressure_term = -grad_p_e / (n_e[:, :, None] * QE + 1e-30)

    # Resistive term: eta * J
    resistive_term = eta * j_total

    E = hall_term + pressure_term + resistive_term
    return E

@jit
def grad_2d(f, dx, dy):
    """
    Computes 2D gradient.
    """
    df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    df_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dy)
    return df_dx, df_dy

@jit
def boris_push(x, v, E, B, q, m, dt):
    """
    Boris particle pusher for ion motion.
    """
    t = q * B * dt / (2 * m)
    s = 2 * t / (1 + jnp.sum(t**2, axis=-1, keepdims=True))
    
    v_minus = v + q * E * dt / (2 * m)
    v_prime = v_minus + jnp.cross(v_minus, t)
    v_plus = v_minus + jnp.cross(v_prime, s)
    v_new = v_plus + q * E * dt / (2 * m)
    
    x_new = x + v_new * dt
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
def interpolate_field_to_particles(field, x, n_e, dr, dz):
    """
    Interpolates a grid field to particle positions using nearest-neighbor.
    field: (nr, nz, 3) array for vector fields or (nr, nz) for scalar
    x: (n_particles, 3) particle positions [r*cos(θ), r*sin(θ), z]
    n_e: template array for getting grid dimensions
    """
    nr, nz = n_e.shape

    # Extract r and z from particle positions
    r = jnp.sqrt(x[:, 0]**2 + x[:, 1]**2)
    z = x[:, 2]

    # Map to grid indices
    r_idx = jnp.clip((r / dr).astype(jnp.int32), 0, nr - 1)
    z_idx = jnp.clip(((z + 1.0) / dz).astype(jnp.int32), 0, nz - 1)

    # Interpolate (nearest neighbor for simplicity)
    if field.ndim == 3:
        return field[r_idx, z_idx, :]
    else:
        return field[r_idx, z_idx]

@jit
def step(state, _):
    x, v, w, n_e, p_e, b, t, dt, dr, dz, nr, nz, n0, T0, Omega, eta = state

    # Deposit ion current onto grid
    j_r, j_theta, j_z = deposit_current(x, v, w, n_e, dr, dz)
    j_i = jnp.stack([j_r, j_theta, j_z], axis=-1)

    # Compute electric field on grid
    E = compute_electric_field(n_e, p_e, j_i, b, eta, dr, dz)

    # Interpolate E and B fields to particle positions
    E_particle = interpolate_field_to_particles(E, x, n_e, dr, dz)
    B_particle = interpolate_field_to_particles(b, x, n_e, dr, dz)

    # Boris push for ion motion
    x_new, v_new = boris_push(x, v, E_particle, B_particle, QE, MI, dt)

    # Update particle weights using delta-f method
    w_new = weight_evolution(w, x_new, v_new, E_particle, B_particle, n0, T0, Omega, dt)

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

    return (x_new, v_new, w_new, n_e, p_e, b, t + dt, dt, dr, dz, nr, nz, n0, T0, Omega, eta), (x, v, w)

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
    T0 = 100.0
    Omega = 1e5
    
    x, v, w = initialize_particles(n_particles, key, nr, nz, n0, T0, Omega)
    
    n_e = jnp.ones((nr, nz)) * n0
    p_e = jnp.ones((nr, nz)) * T0 * n_e
    
    r = jnp.linspace(0, 1, nr)[:, None]
    z = jnp.linspace(-1, 1, nz)[None, :]
    b_r = jnp.zeros((nr, nz))
    b_theta = jnp.zeros((nr, nz))
    b_z = 1.0 * jnp.exp(-r**2 - z**2)
    b = jnp.stack([b_r, b_theta, b_z], axis=-1)
    
    state = (x, v, w, n_e, p_e, b, 0.0, dt, dr, dz, nr, nz, n0, T0, Omega, eta)
    
    final_state, history = lax.scan(step, state, jnp.arange(steps))
    return final_state[:3], history

if __name__ == "__main__":
    (x_final, v_final, w_final), history = run_simulation(100, n_particles=1000)
    print(f"Hybrid Kinetic simulation complete. Number of particles: {x_final.shape[0]}")
