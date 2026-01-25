#!/usr/bin/env python
# scripts/run_example.py
"""CLI entry point for running examples."""
import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jax_frc.configurations import CONFIGURATION_REGISTRY


def find_example_file(name: str, base_dir: Path) -> Path:
    """Find example YAML file by name.

    Args:
        name: Example name (e.g., "belova_case2" or "frc/belova_case2")
        base_dir: Base directory for examples (examples/)

    Returns:
        Path to the YAML file

    Raises:
        FileNotFoundError: If example not found
    """
    # Check direct path
    if Path(name).exists():
        return Path(name)

    # Check with .yaml extension
    if Path(f"{name}.yaml").exists():
        return Path(f"{name}.yaml")

    # Check in examples/cases/
    cases_dir = base_dir / 'cases'

    # If name contains '/', treat as category/name
    if '/' in name:
        path = cases_dir / f"{name}.yaml"
        if path.exists():
            return path
    else:
        # Search all categories
        for category in cases_dir.iterdir():
            if category.is_dir():
                path = category / f"{name}.yaml"
                if path.exists():
                    return path

    raise FileNotFoundError(f"Example not found: {name}")


def list_examples(base_dir: Path) -> list:
    """List all available examples.

    Args:
        base_dir: Base directory for examples

    Returns:
        Sorted list of example names in category/name format
    """
    examples = []
    cases_dir = base_dir / 'cases'
    if cases_dir.exists():
        for category in cases_dir.iterdir():
            if category.is_dir():
                for yaml_file in category.glob("*.yaml"):
                    examples.append(f"{category.name}/{yaml_file.stem}")
    return sorted(examples)


def run_example(yaml_path: Path):
    """Run a single example from YAML.

    Args:
        yaml_path: Path to example YAML file

    Returns:
        ConfigurationResult from running the example
    """
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Build configuration from registry
    class_name = config['configuration']['class']
    overrides = config['configuration'].get('overrides', {})

    if class_name not in CONFIGURATION_REGISTRY:
        raise ValueError(f"Unknown configuration class: {class_name}")

    ConfigClass = CONFIGURATION_REGISTRY[class_name]
    configuration = ConfigClass(**overrides)

    # Apply runtime overrides
    if 'runtime' in config:
        if 't_end' in config['runtime']:
            configuration.timeout = config['runtime']['t_end']
        if 'dt' in config['runtime']:
            configuration.dt = config['runtime']['dt']

    # Print header
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"{'='*60}")
    if config.get('description'):
        for line in config['description'].strip().split('\n'):
            print(f"  {line}")
    print(f"\nConfiguration: {class_name}")
    print(f"Model: {configuration.model_type}")
    print(f"Runtime: t_end={configuration.timeout}, dt={configuration.dt}")
    print(f"{'='*60}\n")

    # Run simulation
    result = configuration.run()

    # Print summary
    print(f"\n{'='*60}")
    print(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
    print(f"{'='*60}")
    for pr in result.phase_results:
        print(f"  Phase '{pr.name}':")
        print(f"    Termination: {pr.termination}")
        print(f"    End time: {pr.end_time:.4f}")
        if hasattr(pr, 'steps') and pr.steps is not None:
            print(f"    Steps: {pr.steps}")
    print()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run example simulations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                    List available examples
  %(prog)s belova_case2              Run specific example
  %(prog)s frc/belova_case1          Run with category prefix
  %(prog)s --category frc            Run all FRC examples
  %(prog)s --all                     Run all examples
        """
    )
    parser.add_argument('examples', nargs='*', help="Example names to run")
    parser.add_argument('--category', help="Run all examples in category")
    parser.add_argument('--all', action='store_true', help="Run all examples")
    parser.add_argument('--list', action='store_true', help="List available examples")
    parser.add_argument('--log-level', default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    base_dir = Path(__file__).parent.parent / 'examples'

    if args.list:
        print("Available examples:")
        for example in list_examples(base_dir):
            print(f"  {example}")
        return 0

    if not args.examples and not args.all and not args.category:
        parser.print_help()
        return 2

    # Collect examples to run
    example_files = []
    if args.all:
        for example in list_examples(base_dir):
            category, name = example.split('/')
            example_files.append(base_dir / 'cases' / category / f"{name}.yaml")
    elif args.category:
        category_dir = base_dir / 'cases' / args.category
        if not category_dir.exists():
            logging.error(f"Category not found: {args.category}")
            return 2
        example_files = list(category_dir.glob("*.yaml"))
    else:
        for name in args.examples:
            try:
                example_files.append(find_example_file(name, base_dir))
            except FileNotFoundError as e:
                logging.error(str(e))
                return 2

    # Run examples
    all_success = True
    results = []
    for example_file in example_files:
        try:
            result = run_example(example_file)
            results.append((example_file.stem, result))
            if not result.success:
                all_success = False
        except Exception as e:
            logging.exception(f"Error running {example_file}")
            all_success = False

    # Print final summary if multiple examples
    if len(example_files) > 1:
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        for name, result in results:
            status = "SUCCESS" if result.success else "FAILED"
            print(f"  {name}: {status}")
        print()

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
