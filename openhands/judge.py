"""
Judge runner for evaluating reverse engineering submissions using OpenHands.

This script runs the judge evaluation on unevaluated submissions for a given task.
It can be run via: uv run examples/agents/openhands/judge.py [args]
"""

import logging
import os
from pathlib import Path
from simple_parsing import ArgumentGenerationMode, ArgumentParser

from cybergym.judge.openhands_judge import evaluate_task_with_openhands

# Setup logger
logger = logging.getLogger(__name__)


def main(raw_args=None):
    """Main entry point for judge runner."""
    parser = ArgumentParser(argument_generation_mode=ArgumentGenerationMode.BOTH)

    parser.add_argument(
        "--db",
        type=str,
        required=True,
        help="Path to SQLite database (e.g., server_poc/poc.db)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to task data directory (e.g., ./cybergym_data/data)"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task ID to evaluate (e.g., arvo:10400)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-5-20250929",
        help="Judge LLM model (default: claude-sonnet-4-5-20250929)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for database commits (default: 10)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per submission in seconds (default: 600)"
    )

    args = parser.parse_args(raw_args)

    logger.info(f"Starting judge evaluation for task {args.task}")
    logger.info(f"Database: {args.db}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Model: {args.model}")

    result = evaluate_task_with_openhands(
        db_path=args.db,
        data_dir=Path(args.data_dir),
        task_id=args.task,
        model=args.model,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )

    print(f"\n✓ Evaluation complete:")
    print(f"  Evaluated: {result['evaluated_count']}")
    print(f"  Failed: {result['failed_count']}")
    print(f"  Total: {result['total']}")
    if result['errors']:
        print(f"\nErrors:")
        for error in result['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")

    return 0 if result['failed_count'] == 0 else 1


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s"
    )
    exit(main())
