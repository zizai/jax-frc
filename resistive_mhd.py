import jax
import jax.numpy as jnp
from jax import jit, lax, grad, vmap
import matplotlib.pyplot as plt

MU0 = 1.2566e-6

@jit
def laplace_star(psi, dr, dz, r):
    """
    Computes the Delta* operator on psi in (r, z) coordinates.
    Delta* psi = d^2 psi / dr^2 - (1/r) * d psi / dr + d^2 psi / dz^2
    """
    psi_rr = (jnp.roll(psi, -1, axis=0) - 2 * psi + jnp.roll(psi, 1, axis=0)) / (dr**2)
    psi_zz = (jnp.roll(psi, -1, axis=1) - 2 * psi + jnp.roll(psi, 1, axis=1)) / (dz**2)
    
    psi_r = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dr)
    
    return psi_rr - (1.0 / r) * psi_r + psi_zz

@jit
def compute_j_phi(psi, dr, dz, r):
    """
    Computes toroidal current density from flux function.
    j_phi = -Delta* psi / mu0 / r
    """
    delta_star_psi = laplace_star(psi, dr, dz, r)
    return -delta_star_psi / MU0 / r

@jit
def advection_term(psi, v_r, v_z, dr, dz):
    """
    Computes the advection term v·∇ψ for the Grad-Shafranov evolution.
    This term represents the convection of magnetic flux by plasma motion.
    """
    # Compute ∇ψ using central differences
    dpsi_dr = (jnp.roll(psi, -1, axis=0) - jnp.roll(psi, 1, axis=0)) / (2 * dr)
    dpsi_dz = (jnp.roll(psi, -1, axis=1) - jnp.roll(psi, 1, axis=1)) / (2 * dz)

    # v·∇ψ = v_r * ∂ψ/∂r + v_z * ∂ψ/∂z
    return v_r * dpsi_dr + v_z * dpsi_dz

@jit
def chodura_resistivity(psi, j_phi, eta_0=1e-4, eta_anom=1e-2, threshold=1e4):
    """
    Chodura-like anomalous resistivity for FRC formation.
    eta = eta_0 + eta_anom * sigmoid(|J| - threshold)
    """
    j_mag = jnp.abs(j_phi)
    anomalous_factor = 0.5 * (1 + jnp.tanh((j_mag - threshold) / (threshold * 0.1)))
    return eta_0 + eta_anom * anomalous_factor

@jit
def circuit_dynamics(I_coil, V_bank, L_coil, M_plasma_coil, dI_plasma_dt, dt):
    """
    Solves circuit equation: V_bank = L_coil * dI/dt + d/dt(M * I_plasma)
    """
    dI_coil_dt = (V_bank - M_plasma_coil * dI_plasma_dt) / L_coil
    I_coil_new = I_coil + dI_coil_dt * dt
    return I_coil_new, dI_coil_dt

def compute_stable_dt(eta_max, dr, dz):
    """
    Compute CFL-stable timestep for diffusion equation.
    For 2D diffusion: dt < min(dr, dz)^2 / (4 * D) where D = eta/mu0

    Args:
        eta_max: Maximum resistivity value (must be > 0)
        dr: Grid spacing in r direction
        dz: Grid spacing in z direction

    Returns:
        Stable timestep with 25% safety margin

    Raises:
        ValueError: If eta_max <= 0
    """
    if eta_max <= 0:
        raise ValueError(f"eta_max must be positive for stable timestep, got {eta_max}")
    D = eta_max / MU0
    dx_min = jnp.minimum(dr, dz)
    return 0.25 * dx_min**2 / D  # 25% safety margin

@jit
def diffusion_substep(psi, v_r, v_z, dr, dz, r, dt_sub):
    """
    Single substep for Grad-Shafranov evolution with advection and diffusion.
    ∂ψ/∂t = -v·∇ψ + (η/μ₀)Δ*ψ
    """
    j_phi = compute_j_phi(psi, dr, dz, r)
    eta = chodura_resistivity(psi, j_phi)

    # Diffusion term: (η/μ₀)Δ*ψ
    diffusion = (eta / MU0) * laplace_star(psi, dr, dz, r)

    # Advection term: -v·∇ψ (negative because we move flux with flow)
    advection = -advection_term(psi, v_r, v_z, dr, dz)

    d_psi = diffusion + advection
    return psi + d_psi * dt_sub, eta, j_phi

@jit
def step(state, _):
    psi, v_r, v_z, I_coil, t, dr, dz, dt, r, z, V_bank, L_coil, M_plasma_coil = state

    # Compute stable substep size based on maximum resistivity
    # eta_max = eta_0 + eta_anom = 1e-4 + 1e-2 = 0.0101
    eta_max = 0.0101
    dt_stable = compute_stable_dt(eta_max, dr, dz)

    # Number of substeps needed (at least 1)
    n_substeps = jnp.maximum(1, jnp.ceil(dt / dt_stable)).astype(jnp.int32)
    dt_sub = dt / n_substeps

    # Subcycle the evolution equation using while_loop (handles dynamic bounds)
    def cond_fn(carry):
        i, _ = carry
        return i < n_substeps

    def body_fn(carry):
        i, psi_acc = carry
        new_psi, _, _ = diffusion_substep(psi_acc, v_r, v_z, dr, dz, r, dt_sub)
        return (i + 1, new_psi)

    _, new_psi = lax.while_loop(cond_fn, body_fn, (jnp.int32(0), psi))

    # Compute j_phi for circuit coupling
    j_phi = compute_j_phi(psi, dr, dz, r)
    j_phi_new = compute_j_phi(new_psi, dr, dz, r)

    # Use total plasma current change, but limit the rate for stability
    delta_j = jnp.mean(j_phi_new - j_phi)
    # Clip to prevent runaway: limit to 10% change per step
    max_change = 0.1 * jnp.abs(jnp.mean(j_phi) + 1e-10)
    delta_j_clipped = jnp.clip(delta_j, -max_change, max_change)
    dI_plasma_dt = delta_j_clipped / dt

    I_coil_new, dI_coil_dt = circuit_dynamics(I_coil, V_bank, L_coil, M_plasma_coil, dI_plasma_dt, dt)

    # Apply boundary conditions (conducting wall: psi = 0 at boundaries)
    # Inner boundary: Neumann-like (extrapolate from interior) to avoid singularity at r=0
    new_psi = new_psi.at[0, :].set(new_psi[1, :])
    new_psi = new_psi.at[-1, :].set(0)
    new_psi = new_psi.at[:, 0].set(0)
    new_psi = new_psi.at[:, -1].set(0)

    return (new_psi, v_r, v_z, I_coil_new, t + dt, dr, dz, dt, r, z, V_bank, L_coil, M_plasma_coil), psi

def run_simulation(steps=500, nr=64, nz=128, V_bank=1000.0, L_coil=1e-6, M_plasma_coil=1e-7):
    """Run resistive MHD simulation for FRC formation.

    Args:
        steps: Number of timesteps to run
        nr: Number of radial grid points (must be >= 4)
        nz: Number of axial grid points (must be >= 4)
        V_bank: Capacitor bank voltage [V]
        L_coil: Coil inductance [H] (must be > 0)
        M_plasma_coil: Plasma-coil mutual inductance [H]

    Returns:
        Tuple of (final_psi, final_I_coil, history)

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
    if L_coil <= 0:
        raise ValueError(f"L_coil must be positive, got {L_coil}")

    dr, dz = 1.0/nr, 2.0/nz
    dt = 1e-4

    r = jnp.linspace(0.01, 1.0, nr)[:, None]
    z = jnp.linspace(-1.0, 1.0, nz)[None, :]

    # Initial flux function: FRC-like configuration
    psi_init = (1 - r**2) * jnp.exp(-z**2)

    # Initial velocity fields (small radial inflow for compression)
    # v_r < 0 represents inward radial flow during theta-pinch formation
    v_r_init = -0.01 * r * jnp.exp(-z**2)  # Weak inward radial velocity
    v_z_init = jnp.zeros((nr, nz))  # No initial axial flow

    I_coil_init = 0.0

    state = (psi_init, v_r_init, v_z_init, I_coil_init, 0.0, dr, dz, dt, r, z, V_bank, L_coil, M_plasma_coil)

    final_state, history = lax.scan(step, state, jnp.arange(steps))
    return final_state[0], final_state[3], history

if __name__ == "__main__":
    final_psi, final_I_coil, history = run_simulation(500)
    print(f"Simulation complete. Final psi max: {jnp.max(final_psi):.6f}, Final I_coil: {final_I_coil:.6f}")
