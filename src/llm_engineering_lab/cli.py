
import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from llm_engineering_lab.config import load_experiment_config
from llm_engineering_lab.experiments.runner import initialize_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-engineering-lab",
        description="LLM engineering experiments and infrastructure.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Initialize a reproducible experiment run.",
    )
    run_parser.add_argument(
        "config",
        type=Path,
        help="Path to the experiment YAML configuration.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        config = load_experiment_config(args.config)
        run_dir = initialize_run(config)

    except (OSError, ValueError, ValidationError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Experiment run initialized.")
    print(f"Experiment: {config.experiment_id}")
    print(f"Run directory: {run_dir}")
    print("Status: initialized (experiment execution is not implemented yet)")

    return 0