"""Validation runner for executing test cases."""
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from jax_frc.configurations import CONFIGURATION_REGISTRY
from jax_frc.solvers.base import Solver
from .metrics import check_tolerance, METRIC_FUNCTIONS
from .references import ReferenceManager
from .result import ValidationResult, MetricResult

logger = logging.getLogger(__name__)


@dataclass
class ValidationRunner:
    """Executes validation cases and generates reports."""

    case_path: Path
    output_dir: Path
    config: dict = field(default=None, init=False)
    reference_mgr: ReferenceManager = field(default_factory=ReferenceManager)

    def __post_init__(self):
        self.case_path = Path(self.case_path)
        self.output_dir = Path(self.output_dir)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Load YAML case configuration."""
        with open(self.case_path) as f:
            return yaml.safe_load(f)

    def _build_configuration(self):
        """Instantiate Configuration class from registry."""
        class_name = self.config['configuration']['class']
        overrides = self.config['configuration'].get('overrides', {})

        if class_name not in CONFIGURATION_REGISTRY:
            raise ValueError(f"Unknown configuration: {class_name}")

        ConfigClass = CONFIGURATION_REGISTRY[class_name]
        return ConfigClass(**overrides)

    def _timestamped_name(self) -> str:
        """Generate timestamped output directory name."""
        ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        return f"{ts}_{self.config['name']}"

    def _setup_output_dir(self) -> Path:
        """Create output directory structure."""
        run_dir = self.output_dir / self._timestamped_name()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "plots").mkdir(exist_ok=True)
        return run_dir

    def run(self, dry_run: bool = False) -> ValidationResult:
        """Execute full validation pipeline."""
        start_time = time.time()

        # Setup
        run_dir = self._setup_output_dir()
        logger.info(f"Running validation case: {self.config['name']}")
        logger.info(f"Output directory: {run_dir}")

        # Save config copy
        with open(run_dir / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)

        if dry_run:
            logger.info("Dry run - skipping simulation")
            return ValidationResult(
                case_name=self.config['name'],
                configuration=self.config['configuration']['class'],
                metrics={},
                runtime_seconds=0.0
            )

        # Build simulation components
        configuration = self._build_configuration()
        geometry = configuration.build_geometry()
        initial_state = configuration.build_initial_state(geometry)
        model = configuration.build_model()

        # Get solver
        solver_config = self.config.get('solver', {'type': 'semi_implicit'})
        solver = Solver.create(solver_config)

        # Run simulation
        runtime = self.config.get('runtime', configuration.default_runtime())
        t_end = runtime.get('t_end', 1e-3)
        dt = runtime.get('dt', 1e-6)

        state = initial_state
        n_steps = int(t_end / dt)

        logger.info(f"Running {n_steps} steps to t={t_end}")
        for i in range(n_steps):
            state = solver.step(state, dt, model, geometry)

        # Compute metrics
        metrics = self._compute_metrics(state, geometry, configuration)

        elapsed = time.time() - start_time
        result = ValidationResult(
            case_name=self.config['name'],
            configuration=self.config['configuration']['class'],
            metrics=metrics,
            runtime_seconds=elapsed
        )

        # Save results
        self._save_metrics(run_dir, result)

        logger.info(f"Validation complete: {'PASS' if result.overall_pass else 'FAIL'}")
        return result

    def _compute_metrics(self, state, geometry, configuration) -> dict:
        """Compute all acceptance metrics."""
        metrics = {}
        acceptance = self.config.get('acceptance', {})

        for spec in acceptance.get('quantitative', []):
            metric_name = spec['metric']

            if 'expected' in spec and 'tolerance' in spec:
                # Direct value check
                value = spec.get('value', 0.0)
                result = check_tolerance(value, spec['expected'], spec['tolerance'])
                metrics[metric_name] = MetricResult(
                    name=metric_name,
                    value=result['value'],
                    expected=result['expected'],
                    tolerance=result['tolerance'],
                    passed=result['pass'],
                    message=result['message']
                )

        return metrics

    def _save_metrics(self, run_dir: Path, result: ValidationResult):
        """Save metrics.json."""
        with open(run_dir / "metrics.json", 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
