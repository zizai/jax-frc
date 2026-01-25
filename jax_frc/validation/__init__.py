"""Validation infrastructure for FRC simulations."""
from .metrics import l2_error, linf_error, rmse_curve, check_tolerance

__all__ = ['l2_error', 'linf_error', 'rmse_curve', 'check_tolerance']
