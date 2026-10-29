"""
CLI runner for the VeritasAI Evaluation Framework.
"""
import sys
import os
import argparse
import json

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.runner import EvaluationRunner


def main():
    parser = argparse.ArgumentParser(description="VeritasAI Golden Dataset Evaluation Runner")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom golden dataset JSON (default: src/evaluation/golden_dataset.json)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Limit number of evaluation queries to run",
    )
    args = parser.parse_args()

    print("=" * 65)
    print(" VERITAS-AI AUTOMATED EVALUATION SUITE ")
    print("=" * 65)

    runner = EvaluationRunner()
    items = runner.load_dataset(args.dataset)
    print(f"Loaded {len(items)} benchmark evaluation items.")
    if args.sample_size:
        print(f"Running sample of {args.sample_size} queries.")

    print("\nExecuting evaluation queries across retrieval & generation...")
    results = runner.run_evaluations(items=items, sample_size=args.sample_size)

    print("\n" + "=" * 65)
    print(f" BENCHMARK RESULTS (Run ID: {results.eval_run_id}) ")
    print("=" * 65)
    print(f"Status: {results.pass_fail_status}")
    print(f"Total Queries Evaluated: {results.total_queries}")
    print("\nMetric Breakdown:")
    for metric_name, val in results.metrics.items():
        percentage = f"{val * 100:.2f}%"
        print(f"  - {metric_name.ljust(28)}: {percentage.rjust(8)}")

    print("=" * 65)
    if results.pass_fail_status == "PASSED":
        print("ALL CI/CD EVALUATION GATES PASSED! Ready for deployment.\n")
        sys.exit(0)
    else:
        print("EVALUATION QUALITY GATE FAILED! Regression detected.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
