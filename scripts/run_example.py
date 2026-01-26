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
from jax_frc.diagnostics.output import OutputManager
from jax_frc.diagnostics.progress import ProgressReporter


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


def run_example(yaml_path: Path, args):
    """Run a single example from YAML.

    Args:
        yaml_path: Path to example YAML file
        args: Parsed command-line arguments

    Returns:
        Tuple of (ConfigurationResult, OutputManager) from running the example
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

    # Setup output manager
    output_manager = OutputManager(
        output_dir=args.output_dir,
        example_name=yaml_path.stem,
        save_checkpoint=args.checkpoint,
    )
    output_manager.setup()
    output_manager.save_config(config)

    # Setup progress reporter
    if args.progress:
        progress_reporter = ProgressReporter(
            t_end=configuration.timeout,
            enabled=True,
        )
        configuration.progress_reporter = progress_reporter

    # Print header
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"Output: {output_manager.run_dir}")
    print(f"{'='*60}\n")

    # Run simulation
    result = configuration.run()

    # Collect and save history
    combined_history = {"time": []}
    for pr in result.phase_results:
        if pr.history:
            for key, values in pr.history.items():
                if key not in combined_history:
                    combined_history[key] = []
                combined_history[key].extend(values)

    if combined_history["time"]:
        output_manager.save_history(combined_history, format=args.history_format)

    # Save final checkpoint
    if args.checkpoint:
        geometry = configuration.build_geometry()
        output_manager.save_final_checkpoint(
            result.final_state,
            geometry,
            metadata={"example": config['name'], "success": result.success},
        )

    # Print summary and outputs
    print(f"\n{'='*60}")
    print(f"Result: {'SUCCESS' if result.success else 'FAILED'}")
    summary = output_manager.get_summary()
    print("Outputs saved:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()

    return result, output_manager


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

    # New output options
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'),
                        help="Base directory for outputs (default: outputs/)")
    parser.add_argument('--progress', dest='progress', action='store_true', default=True,
                        help="Show progress bar (default)")
    parser.add_argument('--no-progress', dest='progress', action='store_false',
                        help="Disable progress bar")
    parser.add_argument('--plots', dest='plots', action='store_true', default=True,
                        help="Generate plots (default)")
    parser.add_argument('--no-plots', dest='plots', action='store_false',
                        help="Skip plot generation")
    parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', default=True,
                        help="Save final checkpoint (default)")
    parser.add_argument('--no-checkpoint', dest='checkpoint', action='store_false',
                        help="Skip final checkpoint")
    parser.add_argument('--history-format', choices=['csv', 'json'], default='csv',
                        help="Format for history output (default: csv)")

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
            result, output_mgr = run_example(example_file, args)
            results.append((example_file.stem, result, output_mgr))
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
        for name, result, output_mgr in results:
            status = "SUCCESS" if result.success else "FAILED"
            print(f"  {name}: {status}")
        print()

    return 0 if all_success else 1


if __name__ == '__main__':
    sys.exit(main())
