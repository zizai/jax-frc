import jax
import jax.numpy as jnp
from jax import jit, lax, grad, vmap
import matplotlib.pyplot as plt

from jax_frc.operators import (
    laplace_star_safe,
    gradient_r,
    gradient_z,
    curl_cylindrical_axisymmetric,
)

MU0 = 1.2566e-6

@jit
def compute_j_phi(psi, dr, dz, r):
    """
    Computes toroidal current density from flux function.
    j_phi = -Delta* psi / mu0 / r

    Uses axis-safe Delta* operator to avoid 1/r singularity.
    Caps j_phi magnitude to prevent numerical instability near axis.
    """
    delta_star_psi = laplace_star_safe(psi, dr, dz, r)
    # Safe division by r with L'Hopital-like handling at axis
    r_safe = jnp.where(r > 1e-10, r, 1.0)
    j_phi = jnp.where(r > 1e-10, -delta_star_psi / MU0 / r_safe, 0.0)
    # At r=0, j_phi should be finite; use limiting value
    # For smooth psi, j_phi(r=0) ~ -2*psi_rr(0)/mu0 by L'Hopital
    # But we set it to 0 for simplicity since current density is typically 0 on axis

    # Cap j_phi to prevent numerical instability (physical FRC j_phi ~ 1e6-1e7 A/m²)
    j_max = 1e7
    j_phi = jnp.clip(j_phi, -j_max, j_max)

    return j_phi

@jit
def advection_term(psi, v_r, v_z, dr, dz):
    """
    Computes the advection term v·∇ψ for the Grad-Shafranov evolution.
    This term represents the convection of magnetic flux by plasma motion.

    Uses non-periodic gradient operators for proper boundary handling.
    """
    # Compute ∇ψ using non-periodic central differences
    dpsi_dr = gradient_r(psi, dr)
    dpsi_dz = gradient_z(psi, dz)

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
def compute_magnetic_field_from_psi(psi, dr, dz, r):
    """
    Compute B_r and B_z from poloidal flux ψ.
    B_r = -(1/r) * ∂ψ/∂z
    B_z = (1/r) * ∂ψ/∂r

    Uses axis-safe operators to handle r=0 singularity.
    """
    dpsi_dr = gradient_r(psi, dr)
    dpsi_dz = gradient_z(psi, dz)

    r_safe = jnp.where(r > 1e-10, r, 1.0)
    B_r = jnp.where(r > 1e-10, -dpsi_dz / r_safe, 0.0)
    B_z = jnp.where(r > 1e-10, dpsi_dr / r_safe, 0.0)

    # At axis, B_r = 0 by symmetry, B_z = finite
    # Use L'Hopital: lim(r->0) dpsi_dr/r = d2psi/dr2 (if dpsi_dr(0)=0)
    # But simpler: B_r(0) = 0, B_z(0) = 2*d(psi_r)/dr at r=0
    B_r = B_r.at[0, :].set(0.0)
    # B_z at axis approximated from interior
    B_z = B_z.at[0, :].set(B_z[1, :])

    return B_r, B_z


@jit
def compute_lorentz_force(j_phi, B_r, B_z):
    """
    Compute Lorentz force F = J × B in axisymmetric geometry.
    With only j_phi (toroidal current) and B_r, B_z (poloidal field):
        F_r = j_phi * B_z
        F_z = -j_phi * B_r
    """
    F_r = j_phi * B_z
    F_z = -j_phi * B_r
    return F_r, F_z


@jit
def momentum_step(rho, v_r, v_z, j_phi, B_r, B_z, p, dr, dz, r, dt):
    """
    Evolve velocity field using momentum equation.
    dv/dt = (J × B - ∇p) / ρ

    Args:
        rho: Mass density (assumed uniform for now)
        v_r, v_z: Velocity components
        j_phi: Toroidal current density
        B_r, B_z: Poloidal magnetic field components
        p: Pressure field
        dr, dz: Grid spacings
        r: Radial coordinate array
        dt: Timestep

    Returns:
        Updated (v_r_new, v_z_new)
    """
    # Lorentz force
    F_r, F_z = compute_lorentz_force(j_phi, B_r, B_z)

    # Pressure gradient
    dp_dr = gradient_r(p, dr)
    dp_dz = gradient_z(p, dz)

    # Acceleration
    a_r = (F_r - dp_dr) / rho
    a_z = (F_z - dp_dz) / rho

    # Clip acceleration to prevent numerical instability
    # Max reasonable acceleration ~ Alfven speed / dt
    # Alfven speed ~ B / sqrt(mu0 * rho) ~ 1 / sqrt(1.26e-6 * 1.67e-8) ~ 2e6 m/s
    # Limit acceleration to change v by at most 1% of Alfven speed per step
    v_alfven = 1e5  # Conservative estimate
    a_max = 0.01 * v_alfven / dt
    a_r = jnp.clip(a_r, -a_max, a_max)
    a_z = jnp.clip(a_z, -a_max, a_max)

    # Update velocity (simple forward Euler)
    v_r_new = v_r + a_r * dt
    v_z_new = v_z + a_z * dt

    # Clip velocities to prevent runaway (max ~ 0.1 * Alfven speed)
    v_max = 0.1 * v_alfven
    v_r_new = jnp.clip(v_r_new, -v_max, v_max)
    v_z_new = jnp.clip(v_z_new, -v_max, v_max)

    # Boundary conditions: no-slip at walls, symmetry at axis
    v_r_new = v_r_new.at[0, :].set(0.0)  # v_r = 0 at axis
    v_r_new = v_r_new.at[-1, :].set(0.0)  # v_r = 0 at outer wall
    v_z_new = v_z_new.at[:, 0].set(0.0)  # no flow through z boundaries
    v_z_new = v_z_new.at[:, -1].set(0.0)

    return v_r_new, v_z_new

@jit
def diffusion_substep(psi, v_r, v_z, dr, dz, r, dt_sub):
    """
    Single substep for Grad-Shafranov evolution with advection and diffusion.
    ∂ψ/∂t = -v·∇ψ + η*j_φ

    where j_φ = -Δ*ψ/(μ₀r) is the toroidal current density.
    The resistive term η*j_φ causes magnetic reconnection.

    Uses axis-safe Delta* operator.
    """
    j_phi = compute_j_phi(psi, dr, dz, r)
    eta = chodura_resistivity(psi, j_phi)

    # Resistive diffusion term: η*j_φ (this is the correct physics)
    # j_phi = -Delta*psi / (mu0 * r), so eta*j_phi causes flux to diffuse/reconnect
    diffusion = eta * j_phi

    # Advection term: -v·∇ψ (negative because we move flux with flow)
    advection = -advection_term(psi, v_r, v_z, dr, dz)

    d_psi = diffusion + advection
    return psi + d_psi * dt_sub, eta, j_phi

@jit
def step(state, _):
    psi, v_r, v_z, rho, p, I_coil, t, dr, dz, dt, r, z, V_bank, L_coil, M_plasma_coil = state

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

    # Compute current density and magnetic field for momentum equation
    j_phi = compute_j_phi(psi, dr, dz, r)
    j_phi_new = compute_j_phi(new_psi, dr, dz, r)
    B_r, B_z = compute_magnetic_field_from_psi(new_psi, dr, dz, r)

    # Evolve velocity using momentum equation
    v_r_new, v_z_new = momentum_step(rho, v_r, v_z, j_phi_new, B_r, B_z, p, dr, dz, r, dt)

    # Volume-weighted integral for circuit coupling: I_plasma = ∫ j_phi * r * dr * dz
    # This is more physically correct than simple mean
    volume_element = r * dr * dz
    I_plasma = jnp.sum(j_phi * volume_element)
    I_plasma_new = jnp.sum(j_phi_new * volume_element)
    dI_plasma_dt = (I_plasma_new - I_plasma) / dt

    # Clip to prevent runaway: limit to 10% change per step
    max_change = 0.1 * jnp.abs(I_plasma + 1e-10)
    dI_plasma_dt_clipped = jnp.clip(dI_plasma_dt, -max_change / dt, max_change / dt)

    I_coil_new, dI_coil_dt = circuit_dynamics(I_coil, V_bank, L_coil, M_plasma_coil, dI_plasma_dt_clipped, dt)

    # Apply boundary conditions (conducting wall: psi = 0 at boundaries)
    # Inner boundary: Neumann-like (extrapolate from interior) to avoid singularity at r=0
    new_psi = new_psi.at[0, :].set(new_psi[1, :])
    new_psi = new_psi.at[-1, :].set(0)
    new_psi = new_psi.at[:, 0].set(0)
    new_psi = new_psi.at[:, -1].set(0)

    return (new_psi, v_r_new, v_z_new, rho, p, I_coil_new, t + dt, dr, dz, dt, r, z, V_bank, L_coil, M_plasma_coil), psi

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
    # Use smaller timestep to avoid excessive subcycling
    # With eta_max=0.0101, stable dt ~ 3e-8, so we use 1e-6 for ~33 substeps
    dt = 1e-6

    r = jnp.linspace(0.01, 1.0, nr)[:, None]
    z = jnp.linspace(-1.0, 1.0, nz)[None, :]

    # Initial flux function: FRC-like configuration
    psi_init = (1 - r**2) * jnp.exp(-z**2)

    # Initial velocity fields (small radial inflow for compression)
    # v_r < 0 represents inward radial flow during theta-pinch formation
    v_r_init = -0.01 * r * jnp.exp(-z**2)  # Weak inward radial velocity
    v_z_init = jnp.zeros((nr, nz))  # No initial axial flow

    # Initial density and pressure (uniform for simplicity)
    # Using typical FRC parameters: n ~ 1e19 m^-3, T ~ 100 eV
    rho_init = jnp.ones((nr, nz)) * 1.673e-27 * 1e19  # Ion mass * density
    p_init = jnp.ones((nr, nz)) * 1e3  # Pressure in Pa (~ 100 eV * 1e19 m^-3)

    I_coil_init = 0.0

    # State: (psi, v_r, v_z, rho, p, I_coil, t, dr, dz, dt, r, z, V_bank, L_coil, M_plasma_coil)
    state = (psi_init, v_r_init, v_z_init, rho_init, p_init, I_coil_init, 0.0, dr, dz, dt, r, z, V_bank, L_coil, M_plasma_coil)

    final_state, history = lax.scan(step, state, jnp.arange(steps))
    return final_state[0], final_state[5], history  # psi, I_coil, history

if __name__ == "__main__":
    final_psi, final_I_coil, history = run_simulation(500)
    print(f"Simulation complete. Final psi max: {jnp.max(final_psi):.6f}, Final I_coil: {final_I_coil:.6f}")
