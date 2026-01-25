"""Tests for validation result container."""
import pytest


def test_validation_result_overall_pass():
    """ValidationResult computes overall_pass from metrics."""
    from jax_frc.validation.result import ValidationResult, MetricResult

    metrics = {
        'metric1': MetricResult(name='metric1', value=1.0, expected=1.0,
                                tolerance='10%', passed=True),
        'metric2': MetricResult(name='metric2', value=2.0, expected=1.5,
                                tolerance='10%', passed=False),
    }

    result = ValidationResult(
        case_name='test',
        configuration='TestConfig',
        metrics=metrics,
        runtime_seconds=1.0
    )

    assert result.overall_pass is False  # One metric failed


def test_validation_result_overall_pass_all_pass():
    """ValidationResult overall_pass is True when all metrics pass."""
    from jax_frc.validation.result import ValidationResult, MetricResult

    metrics = {
        'metric1': MetricResult(name='metric1', value=1.0, expected=1.0,
                                tolerance='10%', passed=True),
        'metric2': MetricResult(name='metric2', value=1.5, expected=1.5,
                                tolerance='10%', passed=True),
    }

    result = ValidationResult(
        case_name='test',
        configuration='TestConfig',
        metrics=metrics,
        runtime_seconds=1.0
    )

    assert result.overall_pass is True


def test_validation_result_overall_pass_empty_metrics():
    """ValidationResult overall_pass is True when no metrics."""
    from jax_frc.validation.result import ValidationResult

    result = ValidationResult(
        case_name='test',
        configuration='TestConfig',
        metrics={},
        runtime_seconds=1.0
    )

    assert result.overall_pass is True


def test_validation_result_to_dict():
    """ValidationResult serializes to dict for JSON."""
    from jax_frc.validation.result import ValidationResult, MetricResult

    result = ValidationResult(
        case_name='test',
        configuration='TestConfig',
        metrics={},
        runtime_seconds=1.0
    )

    d = result.to_dict()
    assert 'case' in d
    assert 'timestamp' in d
    assert 'overall_pass' in d
    assert 'configuration' in d
    assert 'runtime_seconds' in d
    assert 'metrics' in d


def test_metric_result_to_dict():
    """MetricResult serializes to dict."""
    from jax_frc.validation.result import MetricResult

    metric = MetricResult(
        name='test_metric',
        value=1.5,
        expected=1.0,
        tolerance='10%',
        passed=False,
        message='Value exceeded tolerance'
    )

    d = metric.to_dict()
    assert d['value'] == 1.5
    assert d['expected'] == 1.0
    assert d['tolerance'] == '10%'
    assert d['pass'] is False
    assert d['message'] == 'Value exceeded tolerance'


def test_metric_result_to_dict_minimal():
    """MetricResult to_dict omits None/empty fields."""
    from jax_frc.validation.result import MetricResult

    metric = MetricResult(
        name='simple',
        value=1.0,
        passed=True
    )

    d = metric.to_dict()
    assert d['value'] == 1.0
    assert d['pass'] is True
    assert 'expected' not in d
    assert 'tolerance' not in d
    assert 'threshold' not in d
    assert 'message' not in d


def test_metric_result_with_threshold():
    """MetricResult can use threshold instead of tolerance."""
    from jax_frc.validation.result import MetricResult

    metric = MetricResult(
        name='threshold_metric',
        value=0.05,
        threshold=0.1,
        passed=True
    )

    d = metric.to_dict()
    assert d['value'] == 0.05
    assert d['threshold'] == 0.1
    assert d['pass'] is True
