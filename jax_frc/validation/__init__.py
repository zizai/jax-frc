"""Validation infrastructure for FRC simulations."""
from .metrics import l2_error, linf_error, rmse_curve, check_tolerance, METRIC_FUNCTIONS
from .references import ReferenceManager, ReferenceData
from .result import ValidationResult, MetricResult

__all__ = [
    'l2_error', 'linf_error', 'rmse_curve', 'check_tolerance', 'METRIC_FUNCTIONS',
    'ReferenceManager', 'ReferenceData',
    'ValidationResult', 'MetricResult'
]
