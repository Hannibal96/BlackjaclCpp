#!/usr/bin/env python3
import argparse
import re
import csv
from pathlib import Path
from collections import defaultdict
import statistics
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------------------------------------------------------------
# Paths & parameters
# ---------------------------------------------------------------------

DEFAULT_LOG_PATH = Path("regression.log")
OUT_CSV = Path("bj_errors_summary.csv")
OUT_ALL_RULES_PLOT = Path("bj_all_rules_errorbars.png")
OUT_HIST = Path("bj_error_histogram.png")

# 99% normal-approximation threshold for one billion rounds, in percentage
# points: 100 * 2.576 * 1.2 / sqrt(1_000_000_000).
# Errors inside [-HIST_THRESHOLD, +HIST_THRESHOLD] are green; others are red.
HIST_THRESHOLD = 0.0097752327

# Signed decimal/scientific-notation number used by the log parsers.
NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


# ---------------------------------------------------------------------
# Utility: strip ANSI / terminal junk
# ---------------------------------------------------------------------

def strip_ansi(text: str) -> str:
    """
    Remove ANSI escape codes and some stray keyboard-control junk
    from the captured log file.
    """
    # ESC-based sequences, e.g. \x1b[31m
    text = re.sub(r"\x1B\[[0-?]*[ -/]*[@-~]", "", text)
    # caret/arrow junk like ^[[A, ^[[B, etc.
    text = re.sub(r"\^\[\[[0-9;]*[A-Za-z]", "", text)
    return text


# ---------------------------------------------------------------------
# Parse rules: "decks=6_ss17=False_das=True_..."
# ---------------------------------------------------------------------

def parse_rules(rule_str: str) -> dict:
    rules = {}
    for part in rule_str.split("_"):
        if "=" in part:
            k, v = part.split("=", 1)
            rules[k] = v
    return rules


# ---------------------------------------------------------------------
# Parse one block for rules + edges + errors
# ---------------------------------------------------------------------

def parse_block(block: str):
    """
    Extract from a single regression-test record:

      - scenario rules (as dict)
      - Cut Card / CSM theoretical & empiric edges
      - CSM error, Cut Card error

    Expected lines in the current format (penetration values may vary):

      Scenario: decks=6_ss17=False_das=False_...
      Cut Card (Penetration: 75%) Theoretical Edge %: 0.9095% Cut Card Empiric Edge %: -0.9211%
      CSM (Penetration: 0.0%) Theoretical Edge %: 0.8895% CSM Empiric Edge %: -0.8961%
      CSM Error: -0.0066%, Cut Card Error: -0.0116%

    The older labels ("Secnario =", no penetration text, and
    "CSM Errors: ..., Cut Card: ...") are accepted as well.
    """

    scenario_m = re.search(
        r"(?:Scenario|Secnario)\s*[:=]\s*([^\r\n]+)",
        block,
    )
    cut_edge_m = re.search(
        rf"Cut Card(?:\s*\([^\r\n)]*\))?\s+Theoretical Edge %:\s*({NUMBER_RE})%"
        rf"\s*Cut Card Empiric Edge %:\s*({NUMBER_RE})%",
        block,
    )
    csm_edge_m = re.search(
        rf"CSM(?:\s*\([^\r\n)]*\))?\s+Theoretical Edge %:\s*({NUMBER_RE})%"
        rf"\s*CSM Empiric Edge %:\s*({NUMBER_RE})%",
        block,
    )
    err_m = re.search(
        rf"CSM Errors?:\s*({NUMBER_RE})%,\s*"
        rf"Cut Card(?: Error)?:\s*({NUMBER_RE})%",
        block,
    )

    if not scenario_m or not cut_edge_m or not csm_edge_m or not err_m:
        return None

    rules_str = scenario_m.group(1).strip()
    rules = parse_rules(rules_str)

    cut_theo = float(cut_edge_m.group(1))
    cut_emp = float(cut_edge_m.group(2))

    csm_theo = float(csm_edge_m.group(1))
    csm_emp = float(csm_edge_m.group(2))

    csm_err = float(err_m.group(1))
    cut_err = float(err_m.group(2))

    abs_error_sum = abs(csm_err) + abs(cut_err)

    return {
        "rules": rules,
        "cut_theoretical": cut_theo,
        "cut_empiric": cut_emp,
        "csm_theoretical": csm_theo,
        "csm_empiric": csm_emp,
        "csm_error": csm_err,
        "cut_card_error": cut_err,
        "abs_error_sum": abs_error_sum,
    }


# ---------------------------------------------------------------------
# Generic scatter: CSM error vs Cut Card error, colored by rule combos
# ---------------------------------------------------------------------

def create_scatter_by_rules(rows, rule_keys):
    """
    Generic scatter of CSM error vs Cut Card error, colored by combinations
    of the given rule_keys (e.g. ["peek", "surr"] or ["sas"]).

    Each unique combination of rule values gets its own color/marker.

    Output filename is inferred from rule_keys, e.g.:
      ["peek", "surr"] -> bj_error_scatter_by_peek_surr.png
      ["sas"]          -> bj_error_scatter_by_sas.png
    """

    combo_data = {}  # combo_label -> {x: [], y: []}

    for r in rows:
        csm_err = r["csm_error"]
        cut_err = r["cut_card_error"]
        rules = r["rules"]

        # Extract values for each requested key, "NA" if missing
        values = tuple(rules.get(k, "NA") for k in rule_keys)

        # Human-readable label: "peek=False, surr=2-10"
        label_parts = [f"{k}={v}" for k, v in zip(rule_keys, values)]
        label = ", ".join(label_parts)

        if label not in combo_data:
            combo_data[label] = {"x": [], "y": []}

        combo_data[label]["x"].append(csm_err)
        combo_data[label]["y"].append(cut_err)

    # Bright, saturated colors plus translucent points make overlapping samples
    # accumulate color, so dense regions stand out without hiding rule groups.
    colors = [
        "#ff1744",
        "#0066ff",
        "#00b050",
        "#ff8c00",
        "#7c00ff",
        "#00a6d6",
        "#ff1493",
        "#8b5a00",
    ]
    markers = ["o", "s", "^", "D", "X", "P", "*", "v"]

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    for idx, (label, coords) in enumerate(sorted(combo_data.items())):
        x_vals = coords["x"]
        y_vals = coords["y"]
        if not x_vals:
            continue

        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]

        plt.scatter(
            x_vals,
            y_vals,
            alpha=0.65,
            label=label,
            color=color,
            marker=marker,
            s=45,
            edgecolors="none",
            zorder=3,
        )

    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    # Show the approved expected-error magnitude around the ideal result.
    ax.add_patch(
        Circle(
            (0, 0),
            HIST_THRESHOLD,
            fill=False,
            color="black",
            linestyle="--",
            linewidth=2,
            label=f"Expected error radius ({HIST_THRESHOLD:.6f} pp)",
            zorder=4,
        )
    )
    # Equal x/y scaling ensures the threshold boundary is displayed as a circle.
    ax.set_aspect("equal", adjustable="box")

    plt.xlabel("CSM error (percentage points)")
    plt.ylabel("Cut Card error (percentage points)")

    rules_str = ", ".join(rule_keys)
    plt.title(f"CSM vs Cut Card Errors\nColored by: {rules_str}")

    plt.grid(True, axis="both", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.set_axisbelow(True)
    plt.legend(fontsize=8)
    plt.tight_layout()

    suffix = "_".join(rule_keys)
    out_path = Path(f"bj_error_scatter_by_{suffix}.png")

    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Error scatter by {rules_str} → {out_path}")


# ---------------------------------------------------------------------
# Aggregation for error bar plot (mean absolute error)
# ---------------------------------------------------------------------

def aggregate_by_rule_error(rows):
    """
    Returns:
      rule_name -> value -> [mean_abs_error_per_scenario, ...]
    where mean_abs_error_per_scenario = average of |CSM error| and |Cut error|
    for that scenario.
    """
    agg = defaultdict(lambda: defaultdict(list))

    for r in rows:
        mean_abs_err = (abs(r["csm_error"]) + abs(r["cut_card_error"])) / 2.0
        for rule_key, val in r["rules"].items():
            agg[rule_key][val].append(mean_abs_err)

    return agg


def create_all_rules_error_plot(agg):
    """
    One big bar plot:
      x-axis = "rule=value"
      y-axis = mean(|error|) in percentage points
      error bars = std dev of that mean_abs_error metric
    """

    labels = []
    means = []
    stds = []

    for rule_name in sorted(agg.keys()):
        for value in sorted(agg[rule_name].keys()):
            errs = agg[rule_name][value]
            mean = statistics.mean(errs)
            std = statistics.pstdev(errs) if len(errs) > 1 else 0.0

            labels.append(f"{rule_name}={value}")
            means.append(mean)
            stds.append(std)

    if not labels:
        print("No data for rule-error aggregation.")
        return

    plt.figure(figsize=(max(15, len(labels) * 0.45), 7))

    x = range(len(labels))
    plt.bar(x, means, yerr=stds, alpha=0.85, capsize=5)

    plt.xticks(x, labels, rotation=75, ha="right")
    plt.ylabel("Mean |error| (percentage points)")
    plt.title("Mean Absolute Error by Rule Value (All Rules Combined)")

    plt.grid(True, axis="both", linestyle="--", linewidth=0.7, alpha=0.4)
    plt.gca().set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(OUT_ALL_RULES_PLOT, dpi=170)
    plt.close()

    print(f"[OK] All-rules error bar plot → {OUT_ALL_RULES_PLOT}")


# ---------------------------------------------------------------------
# Histogram of errors with green in-band, red out-of-band
# ---------------------------------------------------------------------

def create_error_histogram(rows, threshold: float = HIST_THRESHOLD, bins: int = 40):
    """
    Build a histogram of all individual errors (CSM + Cut Card),
    coloring:
      - green: errors in [-threshold, +threshold]
      - red:   errors outside that range

    Also prints a summary count of how many total errors fall inside the band.
    """

    # Collect all individual errors
    all_errors = []
    for r in rows:
        all_errors.append(r["csm_error"])
        all_errors.append(r["cut_card_error"])

    if not all_errors:
        print("No errors found for histogram.")
        return

    in_range = [e for e in all_errors if -threshold <= e <= threshold]
    out_range = [e for e in all_errors if e < -threshold or e > threshold]

    # ---- NEW: PRINT STATS ----
    total = len(all_errors)
    inside = len(in_range)
    pct = (inside / total * 100) if total > 0 else 0.0

    print(
        f"[INFO] Errors within ±{threshold} : "
        f"{inside} out of {total}  ({pct:.1f}%)"
    )

    # ---- Histogram drawing ----

    # Common bin edges
    min_val = min(all_errors)
    max_val = max(all_errors)
    bin_edges = np.linspace(min_val, max_val, bins + 1)

    plt.figure(figsize=(8, 6))

    # Out-of-band errors (background)
    if out_range:
        plt.hist(
            out_range,
            bins=bin_edges,
            color="red",
            alpha=0.7,
            label=f"|error| > {threshold}",
        )

    # In-band errors (foreground)
    if in_range:
        plt.hist(
            in_range,
            bins=bin_edges,
            color="green",
            alpha=0.7,
            label=f"|error| ≤ {threshold}",
        )

    plt.xlabel("Error (percentage points)")
    plt.ylabel("Count")
    plt.title(
        f"Histogram of CSM & Cut Card Errors\n"
        f"Green: |error| ≤ {threshold}, Red: |error| > {threshold}"
    )
    plt.grid(True, axis="both", linestyle="--", linewidth=0.7, alpha=0.4)
    plt.gca().set_axisbelow(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_HIST, dpi=170)
    plt.close()

    print(f"[OK] Error histogram → {OUT_HIST}")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse a blackjack regression log and generate summary artifacts."
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=(
            "path to the regression log, relative to the current working "
            "directory (default: regression.log)"
        ),
    )
    return parser.parse_args()


def main(log_path: Path = DEFAULT_LOG_PATH):
    # Read & clean log
    raw = log_path.read_text(errors="ignore")
    clean = strip_ansi(raw)

    print(f"[INFO] Reading regression log → {log_path}")

    # Split current [n/total] records and legacy GoogleTest [ RUN ] records.
    blocks = re.split(
        r"^\s*(?:\[\s*RUN\s*\][^\r\n]*|\[\d+\s*/\s*\d+\])\s*$",
        clean,
        flags=re.MULTILINE,
    )

    rows = []
    for blk in blocks[1:]:
        # Trim the legacy GoogleTest completion line, when present.
        blk_main = blk.split("[       OK ]", 1)[0]
        parsed = parse_block(blk_main)
        if parsed:
            rows.append(parsed)

    if not rows:
        print(f"No error data parsed. Check log format / path: {log_path}")
        return

    # ------------------------------------------------------------
    # Sort by sum of absolute errors (|CSM| + |Cut|), highest first
    # ------------------------------------------------------------
    rows.sort(key=lambda r: r["abs_error_sum"], reverse=True)

    # ------------------------------------------------------------
    # CSV export (one column per rule, plus error metrics)
    # ------------------------------------------------------------

    # Discover all rule keys appearing anywhere in the log
    all_rule_keys = sorted({
        key
        for row in rows
        for key in row["rules"].keys()
    })

    # Build final CSV header
    fieldnames = (
        all_rule_keys +
        [
            "csm_theoretical",
            "csm_empiric",
            "cut_theoretical",
            "cut_empiric",
            "csm_error",
            "cut_card_error",
            "abs_error_sum",
        ]
    )

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for row in rows:
            d = {}
            # Fill one column per rule (missing values = blank)
            for k in all_rule_keys:
                d[k] = row["rules"].get(k, "")

            # Add metrics
            d["csm_theoretical"] = row["csm_theoretical"]
            d["csm_empiric"] = row["csm_empiric"]
            d["cut_theoretical"] = row["cut_theoretical"]
            d["cut_empiric"] = row["cut_empiric"]
            d["csm_error"] = row["csm_error"]
            d["cut_card_error"] = row["cut_card_error"]
            d["abs_error_sum"] = row["abs_error_sum"]

            w.writerow(d)

    print(f"[OK] Wrote errors CSV with expanded rule columns → {OUT_CSV}")

    # ------------------------------------------------------------
    # Scatter plots by rule combinations
    # ------------------------------------------------------------
    create_scatter_by_rules(rows, ["peek", "surr"])
    create_scatter_by_rules(rows, ["sas"])

    # ------------------------------------------------------------
    # All-rules error bar plot
    # ------------------------------------------------------------
    agg = aggregate_by_rule_error(rows)
    create_all_rules_error_plot(agg)

    # ------------------------------------------------------------
    # Histogram of individual errors
    # ------------------------------------------------------------
    create_error_histogram(rows, threshold=HIST_THRESHOLD, bins=40)


if __name__ == "__main__":
    args = parse_args()
    main(args.log_path)
