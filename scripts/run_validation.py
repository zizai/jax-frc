#!/usr/bin/env python
# scripts/run_validation.py
"""CLI entry point for running validation cases.

Discovers and runs Python validation scripts in validation/cases/<category>/.
Each script must have a main() function that returns True (pass) or False (fail).
"""
import argparse
import importlib.util
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def discover_cases(base_dir: Path) -> dict[str, Path]:
    """Discover all validation case Python scripts.

    Returns:
        Dict mapping "category/name" to script path.
    """
    cases = {}
    cases_dir = base_dir / "cases"

    if not cases_dir.exists():
        return cases

    for category in cases_dir.iterdir():
        if not category.is_dir() or category.name.startswith("_"):
            continue

        for py_file in category.glob("*.py"):
            # Skip __init__.py and other private files
            if py_file.name.startswith("_"):
                continue

            case_name = f"{category.name}/{py_file.stem}"
            cases[case_name] = py_file

    return dict(sorted(cases.items()))


def load_case_module(script_path: Path):
    """Dynamically load a validation case module.

    Returns:
        The loaded module, or None if loading failed.
    """
    spec = importlib.util.spec_from_file_location(
        f"validation_case_{script_path.stem}", script_path
    )
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module

    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        logging.error(f"Failed to load {script_path}: {e}")
        return None


def run_case(script_path: Path) -> tuple[bool, str, str]:
    """Run a single validation case.

    Returns:
        Tuple of (passed, name, description).
    """
    module = load_case_module(script_path)
    if module is None:
        return False, script_path.stem, "Failed to load module"

    # Get metadata
    name = getattr(module, "NAME", script_path.stem)
    description = getattr(module, "DESCRIPTION", "")

    # Check for main function
    if not hasattr(module, "main"):
        logging.error(f"Case {name} has no main() function")
        return False, name, description

    # Run the case
    try:
        passed = module.main()
        return bool(passed), name, description
    except Exception as e:
        logging.exception(f"Error running {name}")
        return False, name, description


def main():
    parser = argparse.ArgumentParser(
        description="Run validation cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    List all available cases
  %(prog)s magnetic_diffusion        Run a specific case
  %(prog)s --category analytic       Run all cases in a category
  %(prog)s --all                     Run all validation cases
""",
    )
    parser.add_argument("cases", nargs="*", help="Case names to run (category/name)")
    parser.add_argument("--category", "-c", help="Run all cases in category")
    parser.add_argument("--all", "-a", action="store_true", help="Run all cases")
    parser.add_argument("--list", "-l", action="store_true", help="List available cases")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    base_dir = PROJECT_ROOT / "validation"
    all_cases = discover_cases(base_dir)

    if args.list:
        print("Available validation cases:")
        for case_name, script_path in all_cases.items():
            module = load_case_module(script_path)
            desc = getattr(module, "DESCRIPTION", "") if module else ""
            print(f"  {case_name}")
            if desc:
                print(f"    {desc}")
        return 0

    if not args.cases and not args.all and not args.category:
        parser.print_help()
        return 2

    # Collect cases to run
    cases_to_run: list[tuple[str, Path]] = []

    if args.all:
        cases_to_run = list(all_cases.items())
    elif args.category:
        for case_name, script_path in all_cases.items():
            if case_name.startswith(f"{args.category}/"):
                cases_to_run.append((case_name, script_path))
        if not cases_to_run:
            logging.error(f"No cases found in category: {args.category}")
            return 2
    else:
        for name in args.cases:
            # Try exact match first
            if name in all_cases:
                cases_to_run.append((name, all_cases[name]))
            else:
                # Try partial match (just the case name without category)
                matches = [
                    (k, v) for k, v in all_cases.items() if k.endswith(f"/{name}")
                ]
                if len(matches) == 1:
                    cases_to_run.append(matches[0])
                elif len(matches) > 1:
                    logging.error(
                        f"Ambiguous case name '{name}'. Matches: {[m[0] for m in matches]}"
                    )
                    return 2
                else:
                    logging.error(f"Case not found: {name}")
                    return 2

    # Run cases
    results: list[tuple[str, bool]] = []
    print(f"Running {len(cases_to_run)} validation case(s)...\n")

    for case_name, script_path in cases_to_run:
        print(f"{'=' * 60}")
        passed, name, _ = run_case(script_path)
        results.append((name, passed))
        print()

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_passed = sum(1 for _, p in results if p)
    n_failed = len(results) - n_passed

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    print()
    print(f"Passed: {n_passed}/{len(results)}")

    if n_failed > 0:
        print(f"Failed: {n_failed}/{len(results)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
