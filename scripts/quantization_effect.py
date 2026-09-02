#!/usr/bin/env python3
"""Orchestrate count quantization through the shared EvaluateCountPolicy app."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime


RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
NON_TEN_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 12]


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Quantize one learned count and evaluate every version with the "
            "same fixed policy via EvaluateCountPolicy."
        ),
        epilog=(
            "All unrecognized options (for example --seed, --eval-rounds, "
            "--kelly-rounds, --kelly-measurements, --kelly-min/max/step, and "
            "--max-total-wager-fraction) "
            "are forwarded unchanged to EvaluateCountPolicy."
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--count", default="")
    parser.add_argument("--policy", default="")
    parser.add_argument(
        "--quanta", default="0,0.01,0.05,0.1,0.5,1.0",
        help="Comma-separated quantization sizes; zero is always run exactly once",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--game", default="blackjack")
    parser.add_argument(
        "--evaluator",
        default="",
        help="EvaluateCountPolicy binary override",
    )
    return parser.parse_known_args()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def resolve_checkpoint(repo: Path, value: str) -> Path:
    requested = Path(value)
    candidates = [requested] if requested.is_absolute() else [
        requested,
        repo / requested,
        repo / "checkpoints" / "alternating-checkpoints" / requested,
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "meta.json").is_file():
            return candidate.resolve()
    raise RuntimeError(f"Cannot find alternating checkpoint {value!r}")


def label_index(label: str, prefix: str) -> int:
    match = re.fullmatch(rf"{re.escape(prefix)}([0-9]+)", label)
    if not match:
        raise RuntimeError(f"Expected a label such as {prefix}3, got {label!r}")
    return int(match.group(1))


def resolve_labels(checkpoint: Path, count: str, policy: str) -> tuple[str, str]:
    if not count and policy:
        count = f"W{label_index(policy, 'P') + 1}"
    if not count:
        candidates: list[int] = []
        for path in checkpoint.glob("W*.json"):
            match = re.fullmatch(r"W([1-9][0-9]*)\.json", path.name)
            if not match:
                continue
            index = int(match.group(1))
            if (checkpoint / f"P{index - 1}_strategy.json").is_file():
                candidates.append(index)
        if not candidates:
            raise RuntimeError("Checkpoint contains no complete Wk + P(k-1) pair")
        count = f"W{max(candidates)}"
    count_index = label_index(count, "W")
    if not policy:
        policy = f"P{max(0, count_index - 1)}"
    label_index(policy, "P")
    if not (checkpoint / f"{count}.json").is_file():
        raise RuntimeError(f"Missing {count}.json")
    if not (checkpoint / f"{policy}_strategy.json").is_file():
        raise RuntimeError(f"Missing {policy}_strategy.json")
    return count, policy


def load_count(checkpoint: Path, label: str) -> dict:
    artifact = read_json(checkpoint / f"{label}.json")
    config = dict(artifact["count_config"])
    raw = artifact.get("raw_solution")
    scale = artifact.get("normalization_scale")
    if isinstance(raw, list) and len(raw) == 14 and isinstance(scale, (int, float)):
        weights = [float(raw[index]) * float(scale) for index in range(13)]
        ten_mean = sum(weights[8:12]) / 4.0
        weights[8:12] = [ten_mean] * 4
        config["weights"] = weights
    elif len(config.get("weights", [])) != 13:
        raise RuntimeError(f"{label}.json does not contain 13 count weights")
    config.setdefault("factor", 1.0)
    config.setdefault("bias", 0.0)
    config.setdefault("continuous_betting_count", False)
    config.setdefault("count_normalization", "true_count")
    config.setdefault("initial_count", 0.0)
    config.setdefault("initial_count_per_deck", 0.0)
    config.setdefault("resolution", 1.0)
    config.setdefault("min_count", -5)
    config.setdefault("max_count", 5)
    return config


def llround(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0.0 else math.ceil(value - 0.5)


def closest_integers_with_sum(values: list[float], target: int) -> list[int]:
    result = [math.floor(value) for value in values]
    current = sum(result)
    while current < target:
        costs = [
            (value + 1 - original) ** 2 - (value - original) ** 2
            for value, original in zip(result, values)
        ]
        result[min(range(len(costs)), key=costs.__getitem__)] += 1
        current += 1
    while current > target:
        costs = [
            (value - 1 - original) ** 2 - (value - original) ** 2
            for value, original in zip(result, values)
        ]
        result[min(range(len(costs)), key=costs.__getitem__)] -= 1
        current -= 1
    return result


def quantize(weights: list[float], quantum: float) -> tuple[list[float], bool, bool]:
    source_sum = sum(weights)
    source_magnitude = sum(abs(weight) for weight in weights)
    balanced = abs(source_sum) <= 1e-9 * max(1.0, source_magnitude)
    if quantum == 0.0:
        return list(weights), balanced, balanced
    scaled = [weight / quantum for weight in weights]
    if not balanced:
        return [quantum * llround(value) for value in scaled], False, False

    non_ten = [scaled[index] for index in NON_TEN_INDICES]
    center = llround(sum(scaled[8:12]) / 4.0)
    best: list[int] | None = None
    best_error = math.inf
    for ten_tag in range(center - 3, center + 4):
        candidate = [0] * 13
        candidate[8:12] = [ten_tag] * 4
        tags = closest_integers_with_sum(non_ten, -4 * ten_tag)
        for index, tag in zip(NON_TEN_INDICES, tags):
            candidate[index] = tag
        error = sum((tag - value) ** 2 for tag, value in zip(candidate, scaled))
        if (
            best is None
            or error < best_error - 1e-12
            or (abs(error - best_error) <= 1e-12 and tuple(candidate) < tuple(best))
        ):
            best = candidate
            best_error = error
    assert best is not None
    result = [0.0 if tag == 0 else quantum * tag for tag in best]
    return result, True, sum(best) == 0


def parse_quanta(value: str) -> list[float]:
    values = [float(token.strip()) for token in value.strip("[]").split(",")]
    if not values or any(not math.isfinite(item) or item < 0.0 for item in values):
        raise RuntimeError("Quantization sizes must be finite and non-negative")
    unique: list[float] = []
    for item in [0.0, *values]:
        if item not in unique:
            unique.append(item)
    return unique


def quantum_name(value: float) -> str:
    return format(value, ".12g").replace("-", "minus_").replace(".", "p")


def write_aggregate(
    parent: Path,
    checkpoint: Path,
    count_label: str,
    policy_label: str,
    rows: list[dict],
) -> None:
    reference = rows[0]
    reference_spread = reference["result"]["spread"]["edge_per_round"]
    reference_growth = reference["result"]["kelly_at_multiplier_1"]["growth_mean"]
    aggregate = {
        "app": "quantization_effect.py",
        "source_checkpoint": str(checkpoint),
        "source_count": count_label,
        "source_policy": policy_label,
        "comparison_metric": "Kelly multiplier 1.0 (no optimal-fraction mitigation)",
        "degradation_sign": "quantized minus exact; degradation is negative",
        "results": [],
    }
    for row in rows:
        result = row["result"]
        aggregate["results"].append(
            {
                "quantum": row["quantum"],
                "weights": row["weights"],
                "weight_sum": sum(row["weights"]),
                "source_was_balanced": row["source_was_balanced"],
                "balance_preserved": row["balance_preserved"],
                "spread": result["spread"],
                "spread_degradation_from_exact": (
                    result["spread"]["edge_per_round"] - reference_spread
                ),
                "kelly_at_multiplier_1": result["kelly_at_multiplier_1"],
                "kelly_growth_degradation_from_exact": (
                    result["kelly_at_multiplier_1"]["growth_mean"] - reference_growth
                ),
                "child_output": row["child_output"],
            }
        )
    with (parent / "results.json").open("w", encoding="utf-8") as output:
        json.dump(aggregate, output, indent=2)
        output.write("\n")
    with (parent / "results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "quantum",
                *RANKS,
                "weight_sum",
                "spread_edge",
                "spread_degradation",
                "kelly_growth_at_1",
                "kelly_growth_degradation_at_1",
            ]
        )
        for row in aggregate["results"]:
            writer.writerow(
                [
                    row["quantum"],
                    *row["weights"],
                    row["weight_sum"],
                    row["spread"]["edge_per_round"],
                    row["spread_degradation_from_exact"],
                    row["kelly_at_multiplier_1"]["growth_mean"],
                    row["kelly_growth_degradation_from_exact"],
                ]
            )


def main() -> int:
    args, forwarded = parse_args()
    if args.game != "blackjack":
        raise RuntimeError("Quantization workflow currently supports --game blackjack only")
    repo = Path(__file__).resolve().parents[1]
    evaluator = Path(args.evaluator) if args.evaluator else repo / "build/bin/EvaluateCountPolicy"
    if not evaluator.is_absolute():
        evaluator = repo / evaluator
    if not evaluator.is_file():
        raise RuntimeError(f"EvaluateCountPolicy is not built: {evaluator}")
    checkpoint = resolve_checkpoint(repo, args.checkpoint)
    count_label, policy_label = resolve_labels(checkpoint, args.count, args.policy)
    source = load_count(checkpoint, count_label)
    quanta = parse_quanta(args.quanta)
    parent = Path(args.output_dir) if args.output_dir else (
        repo
        / "checkpoints/QuantizationEffect"
        / f"{datetime.now():%Y%m%d_%H%M%S}_{count_label}_{policy_label}_{checkpoint.name}"
    )
    if not parent.is_absolute():
        parent = repo / parent
    counts_dir = parent / "counts"
    counts_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for quantum in quanta:
        weights, source_balanced, balance_preserved = quantize(
            [float(value) for value in source["weights"]], quantum
        )
        if quantum == 0.0 and weights != [float(value) for value in source["weights"]]:
            raise RuntimeError("Quantum zero changed the source weights")
        if source_balanced and not balance_preserved:
            raise RuntimeError(f"Quantum {quantum} did not preserve the zero-sum constraint")
        config = dict(source)
        config["weights"] = weights
        name = quantum_name(quantum)
        count_path = counts_dir / f"quantum_{name}.json"
        with count_path.open("w", encoding="utf-8") as output:
            json.dump(config, output, indent=2)
            output.write("\n")
        child_output = parent / f"quantum_{name}"
        print(f"\n===== quantum={quantum:.12g}: {count_label} + {policy_label} =====\n", flush=True)
        command = [
            str(evaluator),
            "--game", "blackjack",
            "--checkpoint", str(checkpoint),
            "--count", count_label,
            "--policy", policy_label,
            "--count-file", str(count_path),
            "--output-dir", str(child_output),
            *forwarded,
        ]
        subprocess.run(command, cwd=repo, check=True)
        result = read_json(child_output / "results.json")
        rows.append(
            {
                "quantum": quantum,
                "weights": weights,
                "source_was_balanced": source_balanced,
                "balance_preserved": balance_preserved,
                "result": result,
                "child_output": str(child_output),
            }
        )
    write_aggregate(parent, checkpoint, count_label, policy_label, rows)
    print(f"\nCombined quantization results: {parent}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
