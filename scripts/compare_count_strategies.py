#!/usr/bin/env python3
"""Compare policies by orchestrating the shared EvaluateCountPolicy app."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess
import sys


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate basic strategy, Hi-Lo Illustrious 18 when applicable, "
            "and full deviations through one authoritative evaluator."
        ),
        epilog=(
            "All unrecognized count, game-rule, training, seed, and evaluation "
            "options, including --max-total-wager-fraction, are forwarded "
            "unchanged to EvaluateCountPolicy."
        ),
    )
    parser.add_argument("--game", default="blackjack")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--count", default="")
    parser.add_argument("--count-name", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--evaluator", default="")
    parser.add_argument(
        "--full-policy",
        default="full-deviations",
        help="full-deviations, a checkpoint Pk label, or a strategy JSON path",
    )
    parser.add_argument("--skip-illustrious18", action="store_true")
    parser.add_argument("--skip-full-deviations", action="store_true")
    parser.add_argument(
        "--graph-rounds",
        default="",
        help="Accepted for old command compatibility; no EV/count graph is produced",
    )
    return parser.parse_known_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def count_arguments(args: argparse.Namespace) -> tuple[list[str], bool]:
    if args.count and args.count_name:
        raise RuntimeError("Choose --count or --count-name, not both")
    if args.count:
        if re.fullmatch(r"W[0-9]+", args.count):
            if not args.checkpoint:
                raise RuntimeError("--count Wk requires --checkpoint")
            return ["--count", args.count], False
        return ["--count-name", args.count], args.count.lower() == "hilo"
    if args.count_name:
        return ["--count-name", args.count_name], args.count_name.lower() == "hilo"
    return [], not args.checkpoint


def write_aggregate(parent: Path, rows: list[dict]) -> None:
    aggregate = {
        "app": "compare_count_strategies.py",
        "evaluation_engine": "EvaluateCountPolicy",
        "results": rows,
    }
    with (parent / "results.json").open("w", encoding="utf-8") as output:
        json.dump(aggregate, output, indent=2)
        output.write("\n")
    with (parent / "results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "policy",
                "spread_edge",
                "spread_stddev",
                "kelly_growth_at_1",
                "kelly_growth_stddev_at_1",
                "kelly_exposure_rounds_at_1",
                "mean_gross_wager_over_bankroll_at_1",
                "max_gross_wager_over_bankroll_at_1",
                "mean_abs_fX_at_1",
                "max_abs_fX_at_1",
                "child_output",
            ]
        )
        for row in rows:
            result = row["result"]
            exposure = result["kelly_at_multiplier_1"]["exposure"]
            writer.writerow(
                [
                    row["policy"],
                    result["spread"]["edge_per_round"],
                    result["spread"]["outcome_stddev"],
                    result["kelly_at_multiplier_1"]["growth_mean"],
                    result["kelly_at_multiplier_1"]["growth_stddev"],
                    exposure["rounds"],
                    exposure["gross_exposure_summary"]["mean"],
                    exposure["gross_exposure_summary"]["maximum"],
                    exposure["absolute_return_summary"]["mean"],
                    exposure["absolute_return_summary"]["maximum"],
                    row["child_output"],
                ]
            )


def main() -> int:
    args, forwarded = parse_args()
    if args.game != "blackjack":
        raise RuntimeError("Comparison workflow currently supports --game blackjack only")
    repo = Path(__file__).resolve().parents[1]
    evaluator = Path(args.evaluator) if args.evaluator else repo / "build/bin/EvaluateCountPolicy"
    if not evaluator.is_absolute():
        evaluator = repo / evaluator
    if not evaluator.is_file():
        raise RuntimeError(f"EvaluateCountPolicy is not built: {evaluator}")
    selected_count, is_hilo = count_arguments(args)
    if any(
        option in forwarded
        for option in (
            "--count-weights",
            "--count-file",
            "--count-normalization",
            "--initial-count",
            "--initial-count-per-deck",
        )
    ):
        # I18 indices are only valid for the ordinary zero-start true Hi-Lo count.
        is_hilo = False
    source = ["--checkpoint", args.checkpoint] if args.checkpoint else []

    parent = Path(args.output_dir) if args.output_dir else (
        repo / "checkpoints/CompareCountStrategies" /
        f"{datetime.now():%Y%m%d_%H%M%S}_count_policy_comparison"
    )
    if not parent.is_absolute():
        parent = repo / parent
    parent.mkdir(parents=True, exist_ok=True)
    if args.graph_rounds:
        print(
            "WARNING: --graph-rounds is ignored; the refactored workflow evaluates "
            "spread and Kelly metrics only.",
            file=sys.stderr,
        )

    policies: list[tuple[str, list[str]]] = [("basic", ["--policy", "basic"])]
    if is_hilo and not args.skip_illustrious18:
        policies.append(("illustrious18", ["--policy", "illustrious18"]))
    if not args.skip_full_deviations:
        if re.fullmatch(r"P[0-9]+", args.full_policy):
            if not args.checkpoint:
                raise RuntimeError("--full-policy Pk requires --checkpoint")
            full_args = ["--policy", args.full_policy]
        elif args.full_policy in {"full-deviations", "train-full-deviations"}:
            full_args = ["--policy", "full-deviations"]
        else:
            full_args = ["--policy-file", args.full_policy]
        policies.append(("full_deviations", full_args))

    rows: list[dict] = []
    for label, policy_args in policies:
        child_output = parent / label
        print(f"\n===== policy={label} =====\n", flush=True)
        command = [
            str(evaluator),
            "--game", "blackjack",
            *source,
            *selected_count,
            *policy_args,
            "--output-dir", str(child_output),
            *forwarded,
        ]
        subprocess.run(command, cwd=repo, check=True)
        rows.append(
            {
                "policy": label,
                "child_output": str(child_output),
                "result": read_json(child_output / "results.json"),
            }
        )
    write_aggregate(parent, rows)
    print(f"\nCombined policy comparison: {parent}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
