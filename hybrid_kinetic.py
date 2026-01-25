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
def compute_f0_gradient(r, z, vr, vz, vtheta, n0, T0, Omega, dr, dz):
    """
    Computes gradient of f0 for weight evolution.
    d ln f0 / dt = (1/f0) * (df0/dt)
    """
    v_sq = vr**2 + (vtheta - Omega * r)**2 + vz**2
    
    df0_dr = n0 * (MI / (2 * jnp.pi * T0)) ** 1.5 * jnp.exp(-MI * v_sq / (2 * T0)) * \
             (MI * Omega * (vtheta - Omega * r) / T0)
    
    df0_dvr = n0 * (MI / (2 * jnp.pi * T0)) ** 1.5 * jnp.exp(-MI * v_sq / (2 * T0)) * \
              (-MI * vr / T0)
    
    df0_dvz = n0 * (MI / (2 * jnp.pi * T0)) ** 1.5 * jnp.exp(-MI * v_sq / (2 * T0)) * \
              (-MI * vz / T0)
    
    df0_dvtheta = n0 * (MI / (2 * jnp.pi * T0)) ** 1.5 * jnp.exp(-MI * v_sq / (2 * T0)) * \
                  (-MI * (vtheta - Omega * r) / T0)
    
    return df0_dr, df0_dvr, df0_dvz, df0_dvtheta

@jit
def compute_electric_field(r, z, n_e, p_e, j_i, b, eta):
    """
    Computes electric field from electron fluid equation.
    E = (J_total x B - J_i,kinetic x B) / (ne) - grad(p_e) / (ne) + eta * J
    """
    j_total = j_i
    
    j_cross_b = jnp.cross(j_total, b)
    ji_cross_b = jnp.cross(j_i, b)
    
    term1 = (j_cross_b - ji_cross_b) / (n_e * QE)
    
    dp_e_dr, dp_e_dz = grad_2d(p_e, 0.01, 0.01)
    grad_p_e = jnp.array([dp_e_dr, jnp.zeros_like(dp_e_dr), dp_e_dz])
    term2 = -grad_p_e / (n_e * QE)
    
    term3 = eta * j_total
    
    E = term1 + term2 + term3
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
def weight_evolution(w, x, v, r_grid, z_grid, n0, T0, Omega, dr, dz):
    """
    Evolves particle weights using delta-f method.
    dw/dt = -(1-w) * d ln f0 / dt
    """
    r = x[:, 0]
    z = x[:, 2]
    vr = v[:, 0]
    vz = v[:, 2]
    vtheta = v[:, 1]
    
    df0_dr, df0_dvr, df0_dvz, df0_dvtheta = compute_f0_gradient(
        r, z, vr, vz, vtheta, n0, T0, Omega, dr, dz
    )
    
    f0 = rigid_rotor_f0(r, z, vr, vz, vtheta, n0, T0, Omega)
    
    dlnf0_dt = (df0_dr * vr + df0_dvr * (QE / MI) * 0 + df0_dvz * 0 + df0_dvtheta * 0) / f0
    
    dw = -(1 - w) * dlnf0_dt
    w_new = w + dw * 0.01
    
    return jnp.clip(w_new, -1.0, 1.0)

@jit
def deposit_current(x, v, w, r_grid, z_grid, nr, nz, dr, dz):
    """
    Deposits ion current onto grid using particle weights.
    """
    r = x[:, 0]
    z = x[:, 2]
    vr = v[:, 0]
    vtheta = v[:, 1]
    vz = v[:, 2]
    
    r_idx = jnp.clip((r / dr).astype(int), 0, nr - 1)
    z_idx = jnp.clip(((z + 1.0) / dz).astype(int), 0, nz - 1)
    
    j_r = jnp.zeros((nr, nz))
    j_theta = jnp.zeros((nr, nz))
    j_z = jnp.zeros((nr, nz))
    
    j_r = j_r.at[r_idx, z_idx].add(w * vr)
    j_theta = j_theta.at[r_idx, z_idx].add(w * vtheta)
    j_z = j_z.at[r_idx, z_idx].add(w * vz)
    
    return j_r, j_theta, j_z

@jit
def step(state, _):
    x, v, w, n_e, p_e, b, t, dt, dr, dz, nr, nz, n0, T0, Omega, eta = state
    
    j_r, j_theta, j_z = deposit_current(x, v, w, None, None, nr, nz, dr, dz)
    j_i = jnp.stack([j_r, j_theta, j_z], axis=-1)
    
    E = compute_electric_field(None, None, n_e, p_e, j_i, b, eta)
    
    E_broadcast = jnp.tile(E[None, None, :, :], (x.shape[0], 1, 1, 1))
    B_broadcast = jnp.tile(b[None, None, :, :], (x.shape[0], 1, 1, 1))
    
    x_new, v_new = boris_push(x, v, E_broadcast[:, 0, :, :], B_broadcast[:, 0, :, :], QE, MI, dt)
    
    w_new = weight_evolution(w, x_new, v_new, None, None, n0, T0, Omega, dr, dz)
    
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
    x_final, v_final, w_final, history = run_simulation(100, n_particles=1000)
    print(f"Hybrid Kinetic simulation complete. Number of particles: {x_final.shape[0]}")
