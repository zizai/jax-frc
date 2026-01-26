"""Atomic rate coefficients for plasma-neutral interactions.

Includes ionization, recombination, charge exchange, and radiation.
All rates use SI units and are JIT-compatible.
"""

import jax.numpy as jnp
from typing import Tuple

from jax import jit, Array

from jax_frc.constants import QE, MI


# =============================================================================
# Ionization (electron impact: H + e -> H+ + 2e)
# =============================================================================

@jit
def ionization_rate_coefficient(Te: Array) -> Array:
    """Voronov fit for hydrogen ionization <sigma*v>_ion(Te) [m^3/s].

    Reference: Voronov (1997), Atomic Data and Nuclear Data Tables 65, 1-35.

    <sigma*v> = A * (1 + P*sqrt(U)) * U^K * exp(-U) / (X + U)
    where U = E_ion / Te, E_ion = 13.6 eV

    Args:
        Te: Electron temperature [J] (can be array)

    Returns:
        Rate coefficient [m^3/s]
    """
    E_ion = 13.6 * QE  # Ionization energy in Joules

    # Clamp Te to avoid division by zero and overflow
    Te_safe = jnp.maximum(Te, 0.1 * QE)  # Min 0.1 eV

    U = E_ion / Te_safe

    # Voronov coefficients for hydrogen
    A = 2.91e-14  # m^3/s
    P = 0.0
    K = 0.39
    X = 0.232

    # Clamp U to prevent overflow in exp(-U)
    U_clamped = jnp.minimum(U, 100.0)

    sigma_v = A * (1 + P * jnp.sqrt(U_clamped)) * U_clamped**K * jnp.exp(-U_clamped) / (X + U_clamped)

    return sigma_v


@jit
def ionization_rate(Te: Array, ne: Array, rho_n: Array) -> Array:
    """Mass ionization rate S_ion [kg/m³/s].

    S_ion = m_i * ne * nn * <σv>_ion(Te)

    Args:
        Te: Electron temperature [J]
        ne: Electron density [m⁻³]
        rho_n: Neutral mass density [kg/m³]

    Returns:
        Mass ionization rate [kg/m³/s]
    """
    nn = rho_n / MI  # Neutral number density [m⁻³]
    sigma_v = ionization_rate_coefficient(Te)
    return MI * ne * nn * sigma_v


# =============================================================================
# Recombination (radiative: H+ + e -> H + hν)
# =============================================================================

@jit
def recombination_rate_coefficient(Te: Array) -> Array:
    """Radiative recombination <σv>_rec(Te) [m³/s].

    Approximate fit: <σv>_rec ≈ 2.6e-19 * (13.6 eV / Te)^0.7
    Valid for Te > 0.1 eV.

    Args:
        Te: Electron temperature [J]

    Returns:
        Rate coefficient [m³/s]
    """
    Te_eV = Te / QE
    Te_eV_safe = jnp.maximum(Te_eV, 0.1)
    return 2.6e-19 * (13.6 / Te_eV_safe)**0.7


@jit
def recombination_rate(Te: Array, ne: Array, ni: Array) -> Array:
    """Mass recombination rate S_rec [kg/m³/s].

    S_rec = m_i * ne * ni * <σv>_rec(Te)

    Args:
        Te: Electron temperature [J]
        ne: Electron density [m⁻³]
        ni: Ion density [m⁻³]

    Returns:
        Mass recombination rate [kg/m³/s]
    """
    sigma_v = recombination_rate_coefficient(Te)
    return MI * ne * ni * sigma_v


# =============================================================================
# Charge Exchange (H+ + H -> H + H+)
# =============================================================================

@jit
def charge_exchange_cross_section(Ti: Array) -> Array:
    """Charge exchange cross-section sigma_cx(Ti) [m^2].

    Nearly constant ~3e-19 m^2 for Ti < 10 keV.

    Args:
        Ti: Ion temperature [J]

    Returns:
        Cross-section [m^2]
    """
    return 3.0e-19 * jnp.ones_like(Ti)


@jit
def charge_exchange_rates(
    Ti: Array, ni: Array, nn: Array, v_i: Array, v_n: Array
) -> Tuple[Array, Array]:
    """Charge exchange momentum and energy transfer rates.

    R_cx: Momentum transfer to plasma [N/m^3] (add to plasma, subtract from neutrals)
    Q_cx: Energy loss rate from plasma [W/m^3]

    For cold neutrals (T_n << T_i), energy flows from ions to neutrals.
    Convention: positive Q_cx means plasma LOSES energy (to neutrals).
    This represents the rate of thermal energy transfer from plasma to neutrals.

    Args:
        Ti: Ion temperature [J]
        ni: Ion density [m^-3]
        nn: Neutral density [m^-3]
        v_i: Ion velocity (nr, nz, 3) [m/s]
        v_n: Neutral velocity (nr, nz, 3) [m/s]

    Returns:
        R_cx: Momentum transfer [N/m^3], shape (nr, nz, 3)
        Q_cx: Energy transfer [W/m^3], shape (nr, nz)
    """
    # Thermal speed for collision frequency
    v_thermal = jnp.sqrt(8 * Ti / (jnp.pi * MI))

    sigma = charge_exchange_cross_section(Ti)
    nu_cx = nn * sigma * v_thermal  # CX collision frequency [1/s]

    # Momentum transfer: R_cx = m_i * n_i * nu_cx * (v_n - v_i)
    # This is the momentum gained by plasma from neutrals
    # If v_n > v_i, plasma gains momentum (R_cx > 0 in that component)
    R_cx = MI * ni[..., None] * nu_cx[..., None] * (v_n - v_i)

    # Energy transfer: Q_cx = (3/2) * n_i * nu_cx * (T_n - T_i)
    # Assume cold neutrals: T_n << T_i, so Q_cx ≈ -(3/2) * n_i * nu_cx * T_i
    # This is energy lost by plasma to neutrals (negative)
    # Convention: return positive value representing energy loss rate
    Q_cx = 1.5 * ni * nu_cx * Ti  # Energy loss from plasma [W/m^3]

    return R_cx, Q_cx


# =============================================================================
# Radiation Losses
# =============================================================================

@jit
def bremsstrahlung_loss(Te: Array, ne: Array, ni: Array, Z_eff: float = 1.0) -> Array:
    """Bremsstrahlung power loss P_brem [W/m³].

    P_brem = 1.69e-38 * Z_eff² * ne * ni * sqrt(Te_eV)

    Args:
        Te: Electron temperature [J]
        ne: Electron density [m⁻³]
        ni: Ion density [m⁻³]
        Z_eff: Effective charge (default 1.0 for hydrogen)

    Returns:
        Power loss [W/m³]
    """
    Te_eV = Te / QE
    Te_eV_safe = jnp.maximum(Te_eV, 0.1)
    return 1.69e-38 * Z_eff**2 * ne * ni * jnp.sqrt(Te_eV_safe)


@jit
def line_radiation_loss(Te: Array, ne: Array, n_impurity: Array) -> Array:
    """Line radiation from impurities [W/m³].

    Uses simplified cooling curve for carbon impurity.
    Peak around 10 eV, drops at higher Te.

    Args:
        Te: Electron temperature [J]
        ne: Electron density [m⁻³]
        n_impurity: Impurity density [m⁻³]

    Returns:
        Power loss [W/m³]
    """
    Te_eV = Te / QE
    Te_eV_safe = jnp.maximum(Te_eV, 0.1)
    # Gaussian cooling curve peaked at ~10 eV
    L_cool = 1e-31 * jnp.exp(-((jnp.log10(Te_eV_safe) - 1.0) / 0.5)**2)
    return ne * n_impurity * L_cool


@jit
def ionization_energy_loss(S_ion: Array) -> Array:
    """Energy sink from ionization events [W/m³].

    Each ionization costs E_ion = 13.6 eV.

    Args:
        S_ion: Mass ionization rate [kg/m³/s]

    Returns:
        Power loss [W/m³]
    """
    E_ion = 13.6 * QE
    # S_ion has units kg/m³/s, divide by MI to get ionizations/m³/s
    ionizations_per_volume = S_ion / MI
    return ionizations_per_volume * E_ion


@jit
def total_radiation_loss(
    Te: Array, ne: Array, ni: Array, n_impurity: Array, S_ion: Array, Z_eff: float = 1.0
) -> Array:
    """Total radiation sink for energy equation [W/m³].

    Combines bremsstrahlung, line radiation, and ionization energy loss.

    Args:
        Te: Electron temperature [J]
        ne: Electron density [m⁻³]
        ni: Ion density [m⁻³]
        n_impurity: Impurity density [m⁻³]
        S_ion: Mass ionization rate [kg/m³/s]
        Z_eff: Effective charge

    Returns:
        Total power loss [W/m³]
    """
    P_brem = bremsstrahlung_loss(Te, ne, ni, Z_eff)
    P_line = line_radiation_loss(Te, ne, n_impurity)
    P_ion = ionization_energy_loss(S_ion)
    return P_brem + P_line + P_ion
