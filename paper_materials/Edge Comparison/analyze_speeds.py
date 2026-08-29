#!/usr/bin/env python3
import re
import csv
from pathlib import Path
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

LOG_PATH = Path("regression.log")               # your original log file
OUT_CSV = Path("bj_speed_summary.csv")
OUT_ALL_RULES_PLOT = Path("bj_all_rules_speedbars.png")
OUT_ALL_RULES_RATIO_PLOT = Path("bj_all_rules_speedratio_bars.png")


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
# Parse one block for rules + speeds
# ---------------------------------------------------------------------

def parse_block(block: str):
    """
    Extract:
      - scenario rules
      - CSM Speed
      - Cut Card Speed

    Expected lines (example):
      Starting Regression Test with ... Secnario = decks=6_ss17=False_...
      ...
      CSM Speed: 4846088 hands/sec, Cut Card Speed: 10767160 hands/sec
    """

    scenario_m = re.search(r"Secnario\s*=\s*(.*)", block)
    speed_m = re.search(
        r"CSM Speed:\s*([0-9]+)\s*hands/sec,\s*Cut Card Speed:\s*([0-9]+)\s*hands/sec",
        block
    )

    if not scenario_m or not speed_m:
        return None

    rules_str = scenario_m.group(1).strip()
    rules = parse_rules(rules_str)

    csm_speed = int(speed_m.group(1))
    cut_speed = int(speed_m.group(2))

    total_speed = csm_speed + cut_speed
    speed_ratio = cut_speed / csm_speed if csm_speed != 0 else float("inf")

    return {
        "rules": rules,
        "csm_speed": csm_speed,
        "cut_card_speed": cut_speed,
        "total_speed": total_speed,
        "speed_ratio": speed_ratio,
    }


# ---------------------------------------------------------------------
# Generic scatter: CSM Speed vs Cut Card Speed, colored by rule combos
# ---------------------------------------------------------------------

def create_scatter_by_rules(rows, rule_keys):
    """
    Generic scatter of CSM Speed vs Cut Card Speed, colored by combinations
    of the given rule_keys (e.g. ["peek", "surr"] or ["sas"]).

    Each unique combination of rule values gets its own color/marker.

    Output filename is inferred from rule_keys, e.g.:
      ["peek", "surr"] -> bj_speed_scatter_by_peek_surr.png
      ["sas"]          -> bj_speed_scatter_by_sas.png
    """

    combo_data = {}  # combo_label -> {x: [], y: []}

    for r in rows:
        csm_speed = r["csm_speed"]
        cut_speed = r["cut_card_speed"]
        rules = r["rules"]

        # Extract values for each requested key, "NA" if missing
        values = tuple(rules.get(k, "NA") for k in rule_keys)

        # Human-readable label: "peek=False, surr=2-10"
        label_parts = [f"{k}={v}" for k, v in zip(rule_keys, values)]
        label = ", ".join(label_parts)

        if label not in combo_data:
            combo_data[label] = {"x": [], "y": []}

        combo_data[label]["x"].append(csm_speed)
        combo_data[label]["y"].append(cut_speed)

    # Colors & markers (color + shape to help visually)
    colors = [
        "tab:red",
        "tab:blue",
        "black",
        "tab:orange",
        "tab:brown",
        "tab:purple",
        "tab:gray",
        "tab:pink",
    ]
    markers = ["o", "s", "^", "D", "X", "P", "*", "v"]

    plt.figure(figsize=(8, 8))

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
            alpha=0.75,
            label=label,
            color=color,
            marker=marker,
            s=35,
        )

    # Reference lines at 0 (not that meaningful for speed, but symmetric)
    plt.axhline(0, linewidth=1)
    plt.axvline(0, linewidth=1)

    plt.xlabel("CSM Speed (hands/sec)")
    plt.ylabel("Cut Card Speed (hands/sec)")

    rules_str = ", ".join(rule_keys)
    plt.title(f"CSM vs Cut Card Speed\nColored by: {rules_str}")

    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()

    suffix = "_".join(rule_keys)
    out_path = Path(f"bj_speed_scatter_by_{suffix}.png")

    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"[OK] Speed scatter by {rules_str} → {out_path}")


# ---------------------------------------------------------------------
# Aggregators for total speed and ratio per rule=value
# ---------------------------------------------------------------------

def aggregate_by_rule_speed(rows):
    """
    Returns:
      rule_name -> value -> [total_speed, total_speed, ...]
    """
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        total_speed = r["total_speed"]
        for rule_key, val in r["rules"].items():
            agg[rule_key][val].append(total_speed)
    return agg


def aggregate_by_rule_ratio(rows):
    """
    Returns:
      rule_name -> value -> [speed_ratio, speed_ratio, ...]
      where speed_ratio = cut_card_speed / csm_speed
    """
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        ratio = r["speed_ratio"]
        for rule_key, val in r["rules"].items():
            agg[rule_key][val].append(ratio)
    return agg


# ---------------------------------------------------------------------
# Bar plot: total speed per rule=value
# ---------------------------------------------------------------------

def create_all_rules_speed_plot(agg):
    """
    One big bar plot:
      x-axis = "rule=value"
      y-axis = mean(total_speed)
      error bars = std dev(total_speed)
    """

    labels = []
    means = []
    stds = []

    for rule_name in sorted(agg.keys()):
        for value in sorted(agg[rule_name].keys()):
            speeds = agg[rule_name][value]
            mean = statistics.mean(speeds)
            std = statistics.pstdev(speeds) if len(speeds) > 1 else 0.0

            labels.append(f"{rule_name}={value}")
            means.append(mean)
            stds.append(std)

    if not labels:
        print("No data for rule-speed aggregation.")
        return

    plt.figure(figsize=(max(15, len(labels) * 0.45), 7))

    x = range(len(labels))
    plt.bar(x, means, yerr=stds, alpha=0.85, capsize=5)

    plt.xticks(x, labels, rotation=75, ha="right")
    plt.ylabel("Mean total speed (CSM + Cut Card) [hands/sec]")
    plt.title("Total Speed by Rule Value (All Rules Combined)")

    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_ALL_RULES_PLOT, dpi=170)
    plt.close()

    print(f"[OK] All-rules TOTAL speed bar plot → {OUT_ALL_RULES_PLOT}")


# ---------------------------------------------------------------------
# Bar plot: speed ratio per rule=value
# ---------------------------------------------------------------------

def create_all_rules_speedratio_plot(agg_ratio):
    """
    One big bar plot:
      x-axis = "rule=value"
      y-axis = mean(speed_ratio) where
               speed_ratio = cut_card_speed / csm_speed
      error bars = std dev(speed_ratio)
    """

    labels = []
    means = []
    stds = []

    for rule_name in sorted(agg_ratio.keys()):
        for value in sorted(agg_ratio[rule_name].keys()):
            ratios = agg_ratio[rule_name][value]

            mean = statistics.mean(ratios)
            std = statistics.pstdev(ratios) if len(ratios) > 1 else 0.0

            labels.append(f"{rule_name}={value}")
            means.append(mean)
            stds.append(std)

    if not labels:
        print("No data for rule-ratio aggregation.")
        return

    plt.figure(figsize=(max(15, len(labels) * 0.45), 7))

    x = range(len(labels))
    plt.bar(x, means, yerr=stds, alpha=0.85, capsize=5)

    plt.xticks(x, labels, rotation=75, ha="right")
    plt.ylabel("Mean speed ratio (cut_card_speed / csm_speed)")
    plt.title("Speed Ratio by Rule Value (All Rules Combined)")

    plt.grid(True, axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(OUT_ALL_RULES_RATIO_PLOT, dpi=170)
    plt.close()

    print(f"[OK] All-rules SPEED RATIO bar plot → {OUT_ALL_RULES_RATIO_PLOT}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    # Read & clean log
    raw = LOG_PATH.read_text(errors="ignore")
    clean = strip_ansi(raw)

    # Split into [ RUN ] blocks
    blocks = clean.split("[ RUN      ]")

    rows = []
    for blk in blocks[1:]:
        # Stop at the [ OK ] line for that test
        blk_main = blk.split("[       OK ]", 1)[0]
        parsed = parse_block(blk_main)
        if parsed:
            rows.append(parsed)

    if not rows:
        print("No speed data parsed. Check log format / LOG_PATH.")
        return

    # ------------------------------------------------------------
    # Sort by speed ratio (cut_card_speed / csm_speed), highest first
    # ------------------------------------------------------------
    rows.sort(key=lambda r: r["speed_ratio"], reverse=True)

    # ------------------------------------------------------------
    # CSV export (one column per rule, plus speed metrics)
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
            "csm_speed",
            "cut_card_speed",
            "total_speed",
            "speed_ratio",
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
            d["csm_speed"] = row["csm_speed"]
            d["cut_card_speed"] = row["cut_card_speed"]
            d["total_speed"] = row["total_speed"]
            d["speed_ratio"] = row["speed_ratio"]

            w.writerow(d)

    print(f"[OK] Wrote speed CSV with expanded rule columns → {OUT_CSV}")

    # ------------------------------------------------------------
    # Scatter plots by rule combinations
    # ------------------------------------------------------------

    # 1) Colored by sas
    create_scatter_by_rules(rows, ["decks"])

    # Example extra: colored by decks & sas
    # create_scatter_by_rules(rows, ["decks", "sas"])

    # ------------------------------------------------------------
    # All-rules speed plots (total speed and ratio)
    # ------------------------------------------------------------
    agg_speed = aggregate_by_rule_speed(rows)
    create_all_rules_speed_plot(agg_speed)

    agg_ratio = aggregate_by_rule_ratio(rows)
    create_all_rules_speedratio_plot(agg_ratio)


if __name__ == "__main__":
    main()
