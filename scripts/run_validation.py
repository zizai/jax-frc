#!/usr/bin/env python
# scripts/run_validation.py
"""CLI entry point for validation runner."""
import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.validation import ValidationRunner


def find_case_file(name: str, base_dir: Path) -> Path:
    """Find case YAML file by name."""
    # Check direct path
    if Path(name).exists():
        return Path(name)

    # Check in validation/cases/
    for category in ['analytic', 'benchmarks', 'frc']:
        path = base_dir / 'cases' / category / f"{name}.yaml"
        if path.exists():
            return path

    raise FileNotFoundError(f"Case not found: {name}")


def list_cases(base_dir: Path) -> list:
    """List all available validation cases."""
    cases = []
    cases_dir = base_dir / 'cases'
    if cases_dir.exists():
        for category in cases_dir.iterdir():
            if category.is_dir():
                for yaml_file in category.glob("*.yaml"):
                    cases.append(f"{category.name}/{yaml_file.stem}")
    return sorted(cases)


def main():
    parser = argparse.ArgumentParser(description="Run validation cases")
    parser.add_argument('cases', nargs='*', help="Case names to run")
    parser.add_argument('--category', help="Run all cases in category")
    parser.add_argument('--all', action='store_true', help="Run all cases")
    parser.add_argument('--list', action='store_true', help="List available cases")
    parser.add_argument('--output-dir', type=Path, default=Path('validation/reports'))
    parser.add_argument('--dry-run', action='store_true', help="Validate config only")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    base_dir = Path('validation')

    if args.list:
        print("Available validation cases:")
        for case in list_cases(base_dir):
            print(f"  {case}")
        return 0

    if not args.cases and not args.all and not args.category:
        parser.print_help()
        return 2

    # Collect cases to run
    case_files = []
    if args.all:
        for case in list_cases(base_dir):
            category, name = case.split('/')
            case_files.append(base_dir / 'cases' / category / f"{name}.yaml")
    elif args.category:
        category_dir = base_dir / 'cases' / args.category
        if not category_dir.exists():
            logging.error(f"Category not found: {args.category}")
            return 2
        case_files = list(category_dir.glob("*.yaml"))
    else:
        for name in args.cases:
            case_files.append(find_case_file(name, base_dir))

    # Run cases
    all_passed = True
    for case_file in case_files:
        try:
            runner = ValidationRunner(case_file, args.output_dir)
            result = runner.run(dry_run=args.dry_run)

            status = "PASS" if result.overall_pass else "FAIL"
            print(f"{status}: {result.case_name}")

            if not result.overall_pass:
                all_passed = False
                for name, metric in result.metrics.items():
                    if not metric.passed:
                        print(f"  - {name}: {metric.message}")

        except Exception as e:
            logging.exception(f"Error running {case_file}")
            all_passed = False
            # Continue to next case

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
