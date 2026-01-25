"""Invariant checking infrastructure."""
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

@dataclass
class InvariantResult:
    """Result of an invariant check."""
    passed: bool
    name: str
    value: float
    tolerance: float
    message: str

class Invariant(ABC):
    """Base class for all invariants."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this invariant."""
        pass

    @abstractmethod
    def check(self, state_before: Any, state_after: Any) -> InvariantResult:
        """Check if invariant holds between two states."""
        pass

def format_failure(result: InvariantResult, step: int) -> str:
    """Format a failed invariant result for display."""
    return (
        f"Invariant '{result.name}' violated at step {step}:\n"
        f"  {result.message}\n"
        f"  Value: {result.value:.2e}, Tolerance: {result.tolerance:.2e}"
    )

def format_failures(failures: list[tuple[int, InvariantResult]]) -> str:
    """Format multiple failures for display."""
    return "\n\n".join(format_failure(r, s) for s, r in failures)
