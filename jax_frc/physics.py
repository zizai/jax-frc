"""Plasma physics utility functions.

Provides common plasma parameter calculations including:
- Characteristic speeds (Alfven, sound)
- Length scales (Larmor radius, skin depth, Debye length)
- Dimensionless numbers (beta, Reynolds, Lundquist, Mach)
- FRC-specific quantities (separatrix radius, volume)
- Energy calculations (magnetic, kinetic, thermal)
"""

import jax
import jax.numpy as jnp
from jax import jit

from jax_frc.constants import MU0, QE, ME, MI, KB, EPSILON0

# Spitzer resistivity coefficient in SI units (Ohm*m*eV^(3/2))
# From classical transport theory: eta = SPITZER_COEFFICIENT * Z * ln(Lambda) / T_e^(3/2)
SPITZER_COEFFICIENT = 5.2e-5

# Small epsilon to avoid division by zero in numerical operations
NUMERICAL_EPSILON = 1e-10

@jit
def compute_alfven_speed(B, n, m_i=MI):
    """
    Computes Alfvén speed: v_A = B / sqrt(mu0 * n * m_i)
    """
    return B / jnp.sqrt(MU0 * n * m_i)

@jit
def compute_cyclotron_frequency(B, q=QE, m=MI):
    """
    Computes cyclotron frequency: omega_c = q * B / m
    """
    return q * B / m

@jit
def compute_larmor_radius(v_perp, B, q=QE, m=MI):
    """
    Computes Larmor radius: r_L = m * v_perp / (q * B)
    """
    return m * v_perp / (q * B)

@jit
def compute_skin_depth(n, q=QE, m=MI, mu0=MU0):
    """
    Computes ion skin depth: d_i = c / omega_pi = sqrt(m / (mu0 * n * q^2))
    """
    return jnp.sqrt(m / (mu0 * n * q**2))

@jit
def compute_debye_length(n, T, q=QE, epsilon0=EPSILON0):
    """
    Computes Debye length: lambda_D = sqrt(epsilon0 * T / (n * q^2))
    """
    return jnp.sqrt(epsilon0 * T / (n * q**2))

@jit
def compute_plasma_frequency(n, q=QE, m=MI, epsilon0=EPSILON0):
    """
    Computes plasma frequency: omega_p = sqrt(n * q^2 / (m * epsilon0))
    """
    return jnp.sqrt(n * q**2 / (m * epsilon0))

@jit
def compute_beta(n, T, B, mu0=MU0):
    """
    Computes plasma beta: beta = 2 * mu0 * n * k_B * T / B^2
    """
    return 2 * mu0 * n * KB * T / (B**2)

@jit
def compute_spitzer_resistivity(T_e, Z=1.0):
    """
    Computes Spitzer resistivity: eta = SPITZER_COEFFICIENT * Z * ln(Lambda) / T_e^(3/2)
    Simplified version without Coulomb logarithm.
    """
    return SPITZER_COEFFICIENT * Z / (T_e ** 1.5)

@jit
def compute_magnetic_pressure(B, mu0=MU0):
    """
    Computes magnetic pressure: p_B = B^2 / (2 * mu0)
    """
    return B**2 / (2 * mu0)

@jit
def compute_kinetic_pressure(n, T, kB=KB):
    """
    Computes kinetic pressure: p = n * k_B * T
    """
    return n * kB * T

@jit
def compute_reynolds_number(v, L, eta, mu0=MU0):
    """
    Computes magnetic Reynolds number: R_m = v * L / eta
    """
    return v * L / eta

@jit
def compute_lundquist_number(B, L, n, eta, mu0=MU0):
    """
    Computes Lundquist number: S = L * v_A / eta
    """
    v_A = compute_alfven_speed(B, n)
    return L * v_A / eta

@jit
def compute_mach_number(v, c_s, gamma=5/3):
    """
    Computes Mach number: M = v / c_s
    where c_s = sqrt(gamma * k_B * T / m)
    """
    return v / c_s

@jit
def compute_sound_speed(T, m=MI, kB=KB, gamma=5/3):
    """
    Computes ion sound speed: c_s = sqrt(gamma * k_B * T / m)
    """
    return jnp.sqrt(gamma * kB * T / m)

@jit
def compute_frc_separatrix_radius(psi_axis, psi_edge, B_ext):
    """
    Computes FRC separatrix radius from flux function.
    r_s = sqrt(2 * (psi_axis - psi_edge) / (pi * B_ext))
    """
    return jnp.sqrt(2 * (psi_axis - psi_edge) / (jnp.pi * B_ext))

@jit
def compute_frc_length(R, E):
    """
    Computes FRC length from elongation: L = 2 * R * E
    """
    return 2 * R * E

@jit
def compute_frc_volume(R, L):
    """
    Computes FRC volume: V = (2/3) * pi * R^2 * L
    """
    return (2.0 / 3.0) * jnp.pi * R**2 * L

@jit
def compute_frc_beta(p_plasma, p_magnetic):
    """
    Computes FRC beta: beta = p_plasma / p_magnetic
    """
    return p_plasma / p_magnetic

@jit
def normalize_field(field, field_max):
    """
    Normalizes a field to [0, 1] range.
    """
    return field / (field_max + NUMERICAL_EPSILON)

@jit
def compute_gradient(f, dx, dy, dz=None):
    """
    Computes gradient of a scalar field.
    """
    df_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2 * dx)
    df_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2 * dy)
    
    if dz is not None:
        df_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2 * dz)
        return df_dx, df_dy, df_dz
    else:
        return df_dx, df_dy

@jit
def compute_divergence(f_x, f_y, f_z, dx, dy, dz):
    """
    Computes divergence of a vector field.
    """
    dfx_dx = (jnp.roll(f_x, -1, axis=0) - jnp.roll(f_x, 1, axis=0)) / (2 * dx)
    dfy_dy = (jnp.roll(f_y, -1, axis=1) - jnp.roll(f_y, 1, axis=1)) / (2 * dy)
    dfz_dz = (jnp.roll(f_z, -1, axis=2) - jnp.roll(f_z, 1, axis=2)) / (2 * dz)
    
    return dfx_dx + dfy_dy + dfz_dz

@jit
def compute_curl(f_x, f_y, f_z, dx, dy, dz):
    """
    Computes curl of a vector field.
    """
    dfz_dy = (jnp.roll(f_z, -1, axis=1) - jnp.roll(f_z, 1, axis=1)) / (2 * dy)
    dfy_dz = (jnp.roll(f_y, -1, axis=2) - jnp.roll(f_y, 1, axis=2)) / (2 * dz)
    curl_x = dfz_dy - dfy_dz
    
    dfx_dz = (jnp.roll(f_x, -1, axis=2) - jnp.roll(f_x, 1, axis=2)) / (2 * dz)
    dfz_dx = (jnp.roll(f_z, -1, axis=0) - jnp.roll(f_z, 1, axis=0)) / (2 * dx)
    curl_y = dfx_dz - dfz_dx
    
    dfy_dx = (jnp.roll(f_y, -1, axis=0) - jnp.roll(f_y, 1, axis=0)) / (2 * dx)
    dfx_dy = (jnp.roll(f_x, -1, axis=1) - jnp.roll(f_x, 1, axis=1)) / (2 * dy)
    curl_z = dfy_dx - dfx_dy
    
    return curl_x, curl_y, curl_z

@jit
def compute_laplacian(f, dx, dy, dz=None):
    """
    Computes Laplacian of a scalar field.
    """
    d2f_dx2 = (jnp.roll(f, -1, axis=0) - 2 * f + jnp.roll(f, 1, axis=0)) / (dx**2)
    d2f_dy2 = (jnp.roll(f, -1, axis=1) - 2 * f + jnp.roll(f, 1, axis=1)) / (dy**2)
    
    if dz is not None:
        d2f_dz2 = (jnp.roll(f, -1, axis=2) - 2 * f + jnp.roll(f, 1, axis=2)) / (dz**2)
        return d2f_dx2 + d2f_dy2 + d2f_dz2
    else:
        return d2f_dx2 + d2f_dy2

@jit
def apply_boundary_conditions(field, bc_type='dirichlet', bc_value=0.0):
    """
    Applies boundary conditions to a field.
    """
    if bc_type == 'dirichlet':
        field = field.at[0, :].set(bc_value)
        field = field.at[-1, :].set(bc_value)
        field = field.at[:, 0].set(bc_value)
        field = field.at[:, -1].set(bc_value)
    elif bc_type == 'neumann':
        field = field.at[0, :].set(field[1, :])
        field = field.at[-1, :].set(field[-2, :])
        field = field.at[:, 0].set(field[:, 1])
        field = field.at[:, -1].set(field[:, -2])
    
    return field

@jit
def compute_energy_magnetic(B, mu0=MU0):
    """
    Computes magnetic energy density: u_B = B^2 / (2 * mu0)
    """
    return jnp.sum(B**2) / (2 * mu0)

@jit
def compute_energy_kinetic(rho, v):
    """
    Computes kinetic energy density: u_K = 0.5 * rho * v^2
    """
    return 0.5 * rho * jnp.sum(v**2)

@jit
def compute_energy_thermal(n, T, kB=KB):
    """
    Computes thermal energy density: u_T = 1.5 * n * kB * T
    """
    return 1.5 * n * kB * T

@jit
def compute_total_energy(B, rho, v, n, T, mu0=MU0, kB=KB):
    """
    Computes total energy: E = E_B + E_K + E_T
    """
    E_B = compute_energy_magnetic(B, mu0)
    E_K = compute_energy_kinetic(rho, v)
    E_T = compute_energy_thermal(n, T, kB)
    return E_B + E_K + E_T
