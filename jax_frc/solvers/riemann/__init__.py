"""Riemann solvers for MHD simulations.

This module provides approximate Riemann solvers for computing numerical fluxes
at cell interfaces in finite volume methods.

Available Solvers:
    - HLL: Harten-Lax-van Leer two-wave solver (robust, diffusive)
    - HLLD: Five-wave solver for MHD (accurate, resolves Alfven waves exactly)

Available Reconstruction:
    - PLM with MC-beta limiter (default beta=1.3)
    - Minmod limiter (most diffusive)

References:
    [1] Harten, Lax, van Leer (1983) "On Upstream Differencing..."
    [2] Miyoshi & Kusano (2005) "A multi-state HLL approximate Riemann solver..."
    [3] van Leer (1977) "Towards the ultimate conservative difference scheme"
"""

from jax_frc.solvers.riemann.hll import hll_flux_1d, hll_flux_3d
from jax_frc.solvers.riemann.hll_full import hll_update_full, hll_update_full_with_dedner
from jax_frc.solvers.riemann.hlld import hlld_update_full, hlld_flux_direction
from jax_frc.solvers.riemann.reconstruction import (
    reconstruct_plm,
    minmod,
    mc_limiter,
)
from jax_frc.solvers.riemann.wave_speeds import (
    mhd_wave_speeds,
    fast_magnetosonic_speed,
)
from jax_frc.solvers.riemann.mhd_state import (
    MHDConserved,
    MHDPrimitive,
    primitive_to_conserved,
    conserved_to_primitive,
)

__all__ = [
    # HLL solver
    "hll_flux_1d",
    "hll_flux_3d",
    "hll_update_full",
    "hll_update_full_with_dedner",
    # HLLD solver
    "hlld_update_full",
    "hlld_flux_direction",
    # Reconstruction
    "reconstruct_plm",
    "minmod",
    "mc_limiter",
    # Wave speeds
    "mhd_wave_speeds",
    "fast_magnetosonic_speed",
    # MHD state types
    "MHDConserved",
    "MHDPrimitive",
    "primitive_to_conserved",
    "conserved_to_primitive",
]
