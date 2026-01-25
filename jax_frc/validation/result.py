"""Validation result containers."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class MetricResult:
    """Result for a single metric evaluation.

    Attributes:
        name: Identifier for this metric.
        value: The computed metric value.
        expected: Expected value for comparison (optional).
        tolerance: Tolerance specification, e.g., '10%' or '0.01' (optional).
        threshold: Maximum allowed value (optional).
        passed: Whether this metric passed validation.
        message: Optional message describing the result.
    """
    name: str
    value: float
    expected: Optional[float] = None
    tolerance: Optional[str] = None
    threshold: Optional[float] = None
    passed: bool = True
    message: str = ""

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output.

        Only includes non-None/non-empty optional fields.
        """
        d = {
            'value': self.value,
            'pass': self.passed,
        }
        if self.expected is not None:
            d['expected'] = self.expected
        if self.tolerance is not None:
            d['tolerance'] = self.tolerance
        if self.threshold is not None:
            d['threshold'] = self.threshold
        if self.message:
            d['message'] = self.message
        return d


@dataclass
class ValidationResult:
    """Complete result of a validation run.

    Attributes:
        case_name: Name of the validation case.
        configuration: Configuration string describing the setup.
        metrics: Dictionary mapping metric names to MetricResult objects.
        runtime_seconds: Time taken for the validation run.
        timestamp: When the validation was performed.
    """
    case_name: str
    configuration: str
    metrics: dict  # name -> MetricResult
    runtime_seconds: float
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def overall_pass(self) -> bool:
        """True if all metrics passed."""
        if not self.metrics:
            return True
        return all(m.passed for m in self.metrics.values())

    def to_dict(self) -> dict:
        """Serialize to dict for JSON output."""
        return {
            'case': self.case_name,
            'configuration': self.configuration,
            'timestamp': self.timestamp.isoformat(),
            'runtime_seconds': self.runtime_seconds,
            'overall_pass': self.overall_pass,
            'metrics': {name: m.to_dict() for name, m in self.metrics.items()}
        }
