"""Validation infrastructure for FRC simulations."""
from .metrics import l2_error, linf_error, rmse_curve, check_tolerance, METRIC_FUNCTIONS
from .references import ReferenceManager, ReferenceData
from .result import ValidationResult, MetricResult
from .translation import (
    AnalyticTrajectory,
    TranslationResult,
    TranslationBenchmark,
    ModelComparisonResult,
    compute_field_gradient_at_point,
    create_mirror_push_benchmark,
    create_uniform_gradient_benchmark,
    create_staged_acceleration_benchmark,
    traveling_wave_timing,
)

__all__ = [
    'l2_error', 'linf_error', 'rmse_curve', 'check_tolerance', 'METRIC_FUNCTIONS',
    'ReferenceManager', 'ReferenceData',
    'ValidationResult', 'MetricResult',
    # Translation validation
    'AnalyticTrajectory',
    'TranslationResult',
    'TranslationBenchmark',
    'ModelComparisonResult',
    'compute_field_gradient_at_point',
    'create_mirror_push_benchmark',
    'create_uniform_gradient_benchmark',
    'create_staged_acceleration_benchmark',
    'traveling_wave_timing',
]
