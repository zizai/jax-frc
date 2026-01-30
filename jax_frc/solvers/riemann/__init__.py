"""Riemann solvers for MHD simulations.

This module provides approximate Riemann solvers for computing numerical fluxes
at cell interfaces in finite volume methods.

Available Solvers:
    - HLL: Harten-Lax-van Leer two-wave solver (robust, diffusive)
    - HLLD: Five-wave solver for MHD (accurate, resolves Alfven waves exactly) [TODO]

Available Reconstruction:
    - PLM with MC-beta limiter (default beta=1.3)
    - Minmod limiter (most diffusive)

NOTE: The HLL solver is currently experimental and may not be stable for all
problems. For production use, the CT (Constrained Transport) scheme is recommended.

References:
    [1] Harten, Lax, van Leer (1983) "On Upstream Differencing..."
    [2] Miyoshi & Kusano (2005) "A multi-state HLL approximate Riemann solver..."
    [3] van Leer (1977) "Towards the ultimate conservative difference scheme"
"""

from jax_frc.solvers.riemann.hll import hll_flux_1d, hll_flux_3d
from jax_frc.solvers.riemann.reconstruction import (
    reconstruct_plm,
    minmod,
    mc_limiter,
)
from jax_frc.solvers.riemann.wave_speeds import (
    mhd_wave_speeds,
    fast_magnetosonic_speed,
)

__all__ = [
    "hll_flux_1d",
    "hll_flux_3d",
    "reconstruct_plm",
    "minmod",
    "mc_limiter",
    "mhd_wave_speeds",
    "fast_magnetosonic_speed",
]
