"""CLI progress reporting for simulations."""

import sys
from dataclasses import dataclass, field
from typing import TextIO, Optional, Dict, Any


@dataclass
class ProgressReporter:
    """Reports simulation progress to stderr.

    Produces output like:
        [Phase: merging] t=1.23e-6 / 5.00e-6 (24.6%) | step 1200 | dt=4.1e-9

    Attributes:
        t_end: Target end time for percentage calculation
        output_interval: Only report every N calls (default 1 = every call)
        enabled: If False, report() does nothing
        stream: Output stream (default stderr)
    """

    t_end: float
    output_interval: int = 1
    enabled: bool = True
    stream: TextIO = field(default_factory=lambda: sys.stderr)

    _call_count: int = field(default=0, init=False, repr=False)

    def report(
        self,
        t: float,
        step: int,
        dt: float,
        phase_name: str,
        diagnostics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Report current simulation progress.

        Args:
            t: Current simulation time
            step: Current step number
            dt: Current timestep
            phase_name: Name of current phase
            diagnostics: Optional dict of diagnostic values to display
        """
        if not self.enabled:
            return

        self._call_count += 1
        if self._call_count % self.output_interval != 0:
            return

        # Calculate progress percentage
        pct = (t / self.t_end * 100) if self.t_end > 0 else 0.0

        # Build progress string
        parts = [
            f"[Phase: {phase_name}]",
            f"t={t:.2e} / {self.t_end:.2e} ({pct:.1f}%)",
            f"| step {step}",
            f"| dt={dt:.1e}",
        ]

        # Add diagnostics if provided
        if diagnostics:
            for name, value in diagnostics.items():
                if isinstance(value, float):
                    parts.append(f"| {name}={value:.3g}")
                else:
                    parts.append(f"| {name}={value}")

        line = " ".join(parts)

        # Write with carriage return for in-place update
        self.stream.write(f"\r{line}")
        self.stream.flush()

    def finish(self) -> None:
        """Print final newline after progress reporting completes."""
        if self.enabled:
            self.stream.write("\n")
            self.stream.flush()
