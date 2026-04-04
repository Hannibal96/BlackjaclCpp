#!/usr/bin/env python3
"""
Hyperparameter Optimization for Q-Learning Blackjack Agent using Optuna.

Structure:
  - Section 1: Configuration  (what to run, what to tune, what to parse)
  - Section 2: Runner         (subprocess execution & CLI arg building)
  - Section 3: Parser         (output -> metrics dict)
  - Section 4: Objective      (optuna trial -> metric)
  - Section 5: Study          (optuna study orchestration)
  - Section 6: Entry point    (main / CLI)

To adapt for a different executable or metric, subclass ExeRunner / OutputParser
or replace the CONFIG dict at the bottom of this file.
"""

from __future__ import annotations

import re
import json
import time
import hashlib
import logging
import argparse
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import optuna
from optuna.samplers import TPESampler

# Project root = parent of the directory that contains this script.
# Works regardless of where you invoke Python from.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)  # keep optuna quiet


# ===========================================================================
# SECTION 1: Configuration
# ===========================================================================

@dataclass
class ParamSpec:
    """
    Describes one hyperparameter to tune.

    type:       "float" | "int" | "categorical"
    low / high: numeric bounds (used for float / int)
    log:        log-scale sampling (float / int only)
    choices:    list of options (categorical only)
    cli_flag:   the CLI flag passed to the executable (e.g. "--temp-start")
    constraint: optional lambda(params) -> bool; trial is pruned if False
    """
    cli_flag: str
    type: str  # "float" | "int" | "categorical"
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    choices: list | None = None


def _param_space_hash(param_space: dict) -> str:
    """
    Return a 6-character hex hash of the param space definition.
    Changes whenever any param type, bounds, or choices change, which
    prevents Optuna's distribution-compatibility error when resuming from DB.
    """
    fingerprint = json.dumps(
        {name: {"type": spec.type, "low": spec.low, "high": spec.high,
                "log": spec.log, "choices": spec.choices}
         for name, spec in sorted(param_space.items())},
        sort_keys=True,
    )
    return hashlib.sha1(fingerprint.encode()).hexdigest()[:6]


def _compact_number(s: str) -> str:
    """Shorten large integers: 100000000 -> 100M, 1000 -> 1K, etc."""
    try:
        n = int(s)
        if n >= 1_000_000_000:
            return f"{n // 1_000_000_000}B"
        if n >= 1_000_000:
            return f"{n // 1_000_000}M"
        if n >= 1_000:
            return f"{n // 1_000}K"
        return str(n)
    except ValueError:
        return s


@dataclass
class FixedArgs:
    """
    CLI arguments that are constant across all trials.
    key = parameter name (used as CLI flag stem), value = argument value.
    """
    args: dict[str, Any] = field(default_factory=dict)

    # Maps CLI flag -> short label used in file-name slugs.
    # Flags absent from this map are omitted from the slug.
    SLUG_LABELS: dict[str, str] = field(default_factory=lambda: {
        "--decks":        "d",
        "--ss17":         "ss17",
        "--das":          "das",
        "--peek":         "peek",
        "--surr":         "surr",
        "--penetration":  "pen",
        "--num-rounds":   "r",
        "--num-threads":  "t",
    })

    def to_cli_list(self) -> list[str]:
        parts = []
        for flag, val in self.args.items():
            parts += [flag, str(val)]
        return parts

    def to_slug(self) -> str:
        """
        Build a compact, file-system-safe identifier from the fixed args.
        Example: d6_ss17true_dastrue_peekfalse_surr2-10_pen0_r100M_t10
        Flags not listed in SLUG_LABELS are skipped.
        """
        parts = []
        for flag, label in self.SLUG_LABELS.items():
            if flag not in self.args:
                continue
            val = str(self.args[flag]).strip("[]")  # strip optional brackets
            val = _compact_number(val)
            parts.append(label + val)
        return "_".join(parts)


@dataclass
class StudyConfig:
    """Top-level configuration for an HPO study."""
    # Path to the executable
    executable: str

    # Arguments fixed for every trial
    fixed_args: FixedArgs

    # Hyperparameter search space
    param_space: dict[str, ParamSpec]

    # Cross-parameter constraints: list of (description, lambda(dict)->bool)
    # If any constraint returns False the trial is pruned.
    constraints: list[tuple[str, Any]] = field(default_factory=list)

    # Metric to optimise (key returned by the parser)
    metric_key: str = "accuracy"
    direction: str = "maximize"  # "maximize" | "minimize"

    # Optuna study settings
    n_trials: int = 50
    n_jobs: int = 1           # parallel trials (careful: each trial runs the exe)
    study_name: str = "qlearning_hpo"
    storage: str | None = None  # e.g. "sqlite:///hpo.db" for persistence

    # Where to save a JSON results file (None = don't save)
    results_path: str | None = "hpo_results.json"

    # Timeout per trial in seconds (None = no limit)
    trial_timeout: int | None = None

    # Working directory for the subprocess (None = inherit Python process cwd)
    cwd: str | None = None


# ---------------------------------------------------------------------------
# Concrete configuration for QLearningRegressionTest
# ---------------------------------------------------------------------------

QLEARNING_CONFIG = StudyConfig(
    executable="./build/release/bin/QLearningRegressionTest",
    cwd=str(_PROJECT_ROOT),

    fixed_args=FixedArgs({
        "--num-threads":  10,
        "--num-rounds":   100_000_000,
        "--decks":        "[6]",
        "--ss17":         "[true]",
        "--das":          "[true]",
        "--peek":         "[false]",
        "--surr":         "[2-10]",
        "--penetration":  0,
    }),

    param_space={
        # 1 → 100, linear steps of 5 (plus 1 and 2 as anchors)
        "temp_start": ParamSpec(
            cli_flag="--temp-start",
            type="categorical",
            choices=[1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100],
        ),
        # 0.001 → 0.1, alternating ×5 and ×2 steps (1-5 pattern on each decade)
        "temp_min": ParamSpec(
            cli_flag="--temp-min",
            type="categorical",
            choices=[0.001, 0.005, 0.01, 0.05, 0.1],
        ),
        # very close to 1; clean 9s-style values
        "temp_decay": ParamSpec(
            cli_flag="--temp-decay",
            type="categorical",
            choices=[0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999],
        ),
        # 0.001 → 0.1, same 1-5 pattern
        "alpha_start": ParamSpec(
            cli_flag="--alpha-start",
            type="categorical",
            choices=[0.001, 0.005, 0.01, 0.05, 0.1],
        ),
        # 0.00001 → 0.01, exponential steps; must stay below alpha_start
        "alpha_min": ParamSpec(
            cli_flag="--alpha-min",
            type="categorical",
            choices=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01],
        ),
        # 10 → 100 000 in decade jumps (optionally with ×5 mid-points)
        "alpha_decay": ParamSpec(
            cli_flag="--alpha-decay",
            type="categorical",
            choices=[10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000],
        ),
    },

    # Ensure min < start for temperature and alpha
    constraints=[
        ("temp_min < temp_start",   lambda p: p["temp_min"]   < p["temp_start"]),
        ("alpha_min < alpha_start", lambda p: p["alpha_min"]  < p["alpha_start"]),
    ],

    metric_key="accuracy",
    direction="maximize",

    n_trials=100,
    n_jobs=1,  # increase if you have spare CPU cores beyond --num-threads
    study_name="qlearning_hpo",
    storage=None,  # set to "sqlite:///hpo.db" to persist across runs
    results_path="hpo_results.json",
    trial_timeout=None,
)


# ===========================================================================
# SECTION 2: Runner
# ===========================================================================

class ExeRunner:
    """
    Runs an external executable with a given set of CLI arguments.
    Fixed args are always included; trial args are merged in.
    """

    def __init__(self, executable: str, fixed_args: FixedArgs,
                 timeout: int | None = None, cwd: str | None = None):
        self.executable = executable
        self.fixed_args = fixed_args
        self.timeout = timeout
        self.cwd = cwd

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, trial_args: dict[str, Any]) -> tuple[str, int]:
        """
        Run the executable with fixed + trial args.
        Returns (stdout+stderr combined, returncode).
        """
        cmd = self._build_command(trial_args)
        log.debug("CMD: %s", " ".join(cmd))

        t0 = time.perf_counter()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            cwd=self.cwd,
        )
        elapsed = time.perf_counter() - t0
        log.debug("Finished in %.1fs (rc=%d)", elapsed, result.returncode)

        combined = result.stdout + "\n" + result.stderr
        return combined, result.returncode

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_command(self, trial_args: dict[str, Any]) -> list[str]:
        cmd = [self.executable]
        cmd += self.fixed_args.to_cli_list()
        for flag, val in trial_args.items():
            cmd += [flag, str(val)]
        return cmd


# ===========================================================================
# SECTION 3: Parser
# ===========================================================================

class OutputParser(ABC):
    """Base class for parsing executable output into a metrics dict."""

    @abstractmethod
    def parse(self, output: str) -> dict[str, Any]:
        """Return a dict of metric_name -> value. Raise ValueError on failure."""


class QLearningOutputParser(OutputParser):
    """
    Parses QLearningRegressionTest output.

    Extracts:
      accuracy    (float, 0–100)        from "Accuracy: XX.XX%"
      differences (int)                 from "Differences: N / M"
      total_cells (int)                 from "Differences: N / M"
      edge_pct    (float)               from "Edge %: XX.XX%"
      training_time (float, seconds)    from "Training Time: XX.XX seconds"
      speed       (int, hands/sec)      from "Speed: N hands/sec"
    """

    # Patterns
    _RE_ACCURACY   = re.compile(r"Accuracy:\s*([\d.]+)%")
    _RE_DIFF       = re.compile(r"Differences:\s*(\d+)\s*/\s*(\d+)")
    _RE_EDGE       = re.compile(r"Edge %:\s*([-\d.]+)%")
    _RE_TIME       = re.compile(r"Training Time:\s*([\d.]+)\s*seconds")
    _RE_SPEED      = re.compile(r"Speed:\s*([\d,]+)\s*hands/sec")

    def parse(self, output: str) -> dict[str, Any]:
        metrics: dict[str, Any] = {}

        m = self._RE_ACCURACY.search(output)
        if not m:
            raise ValueError("Could not find 'Accuracy:' in output.\n" + output[-2000:])
        metrics["accuracy"] = float(m.group(1))

        m = self._RE_DIFF.search(output)
        if m:
            metrics["differences"] = int(m.group(1))
            metrics["total_cells"] = int(m.group(2))

        m = self._RE_EDGE.search(output)
        if m:
            metrics["edge_pct"] = float(m.group(1))

        m = self._RE_TIME.search(output)
        if m:
            metrics["training_time"] = float(m.group(1))

        m = self._RE_SPEED.search(output)
        if m:
            speed_str = m.group(1).replace(",", "")
            metrics["speed"] = int(speed_str)

        return metrics


# ===========================================================================
# SECTION 4: Objective
# ===========================================================================

class OptunaObjective:
    """
    Wraps runner + parser into an optuna-compatible objective function.

    Handles:
      - Suggesting parameters from the param_space
      - Optional parameter transforms (e.g. temp_decay stored as complement)
      - Cross-parameter constraints (prune early if violated)
      - Attaching all metrics as trial user attributes
    """

    def __init__(
        self,
        runner: ExeRunner,
        parser: OutputParser,
        param_space: dict[str, ParamSpec],
        constraints: list[tuple[str, Any]],
        metric_key: str,
    ):
        self.runner = runner
        self.parser = parser
        self.param_space = param_space
        self.constraints = constraints
        self.metric_key = metric_key

    # ------------------------------------------------------------------
    # Optuna entry point
    # ------------------------------------------------------------------

    def __call__(self, trial: optuna.Trial) -> float:
        # 1. Suggest raw parameter values
        raw_params = self._suggest(trial)

        # 2. Transform to actual CLI values
        cli_values, display_values = self._transform(raw_params)

        # 3. Check cross-parameter constraints
        for desc, fn in self.constraints:
            if not fn(display_values):
                log.debug("Constraint violated: %s — pruning trial %d", desc, trial.number)
                raise optuna.TrialPruned(f"Constraint violated: {desc}")

        # 4. Build CLI arg dict {flag: value}
        cli_args = {
            self.param_space[name].cli_flag: val
            for name, val in cli_values.items()
        }

        # 5. Run executable
        log.info(
            "Trial %d | %s",
            trial.number,
            "  ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                      for k, v in display_values.items()),
        )
        output, rc = self.runner.run(cli_args)

        if rc != 0:
            log.warning("Trial %d: non-zero exit code %d", trial.number, rc)

        # 6. Parse output
        try:
            metrics = self.parser.parse(output)
        except ValueError as e:
            log.warning("Trial %d: parse error: %s", trial.number, e)
            raise optuna.TrialPruned("Parse error") from e

        # 7. Attach metrics as user attributes (visible in optuna dashboard / CSV)
        for k, v in metrics.items():
            trial.set_user_attr(k, v)
        for k, v in display_values.items():
            trial.set_user_attr(f"param_{k}", v)

        score = metrics[self.metric_key]
        log.info("Trial %d → %s=%.4f", trial.number, self.metric_key, score)
        return score

    # ------------------------------------------------------------------
    # Parameter suggestion
    # ------------------------------------------------------------------

    def _suggest(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest raw values from optuna for each param in the space."""
        raw: dict[str, Any] = {}
        for name, spec in self.param_space.items():
            if spec.type == "float":
                raw[name] = trial.suggest_float(name, spec.low, spec.high, log=spec.log)
            elif spec.type == "int":
                raw[name] = trial.suggest_int(name, spec.low, spec.high, log=spec.log)
            elif spec.type == "categorical":
                raw[name] = trial.suggest_categorical(name, spec.choices)
            else:
                raise ValueError(f"Unknown param type: {spec.type!r}")
        return raw

    def _transform(
        self, raw: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        Apply any domain-specific transformations.

        Returns:
          cli_values:     values to pass to the executable
          display_values: human-readable values (also used for constraints)

        All current parameters are categorical clean values, so no
        transformation is needed. Override this method to add custom
        transforms for future continuous parameters.
        """
        cli: dict[str, Any] = dict(raw)
        display: dict[str, Any] = dict(raw)
        return cli, display


# ===========================================================================
# SECTION 5: Study
# ===========================================================================

class HPOStudy:
    """
    Orchestrates an optuna study: creates it, runs trials, and saves results.
    """

    def __init__(self, config: StudyConfig):
        self.config = config
        self.runner = ExeRunner(
            config.executable,
            config.fixed_args,
            timeout=config.trial_timeout,
            cwd=config.cwd,
        )
        self.parser = self._build_parser()
        self.objective = OptunaObjective(
            runner=self.runner,
            parser=self.parser,
            param_space=config.param_space,
            constraints=config.constraints,
            metric_key=config.metric_key,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> optuna.Study:
        study = optuna.create_study(
            study_name=self.config.study_name,
            direction=self.config.direction,
            sampler=TPESampler(seed=42),
            storage=self.config.storage,
            load_if_exists=True,
        )

        log.info(
            "Starting HPO: %d trials | direction=%s | metric=%s",
            self.config.n_trials,
            self.config.direction,
            self.config.metric_key,
        )

        study.optimize(
            self.objective,
            n_trials=self.config.n_trials,
            n_jobs=self.config.n_jobs,
            callbacks=[self._trial_callback],
            catch=(Exception,),
        )

        self._report(study)

        if self.config.results_path:
            self._save_results(study)

        return study

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_parser(self) -> OutputParser:
        # Extend here to support other executables
        return QLearningOutputParser()

    def _trial_callback(self, study: optuna.Study, trial: optuna.FrozenTrial) -> None:
        try:
            best = study.best_trial
        except ValueError:
            return  # no completed trials yet
        log.info(
            ">>> Best so far: trial %d | %s=%.4f | params=%s",
            best.number,
            self.config.metric_key,
            best.value,
            best.params,
        )

    def _report(self, study: optuna.Study) -> None:
        try:
            best = study.best_trial
        except ValueError:
            log.warning("No completed trials — nothing to report.")
            return
        print("\n" + "=" * 60)
        print("HPO COMPLETE")
        print("=" * 60)
        print(f"Best trial:  #{best.number}")
        print(f"Best {self.config.metric_key}: {best.value:.4f}")
        print("\nBest hyperparameters (raw optuna):")
        for k, v in best.params.items():
            print(f"  {k:20s} = {v}")
        print("\nBest hyperparameters (CLI flags):")
        cli_values, display_values = self.objective._transform(best.params)
        for name, val in cli_values.items():
            flag = self.config.param_space[name].cli_flag
            print(f"  {flag:20s} {val}")
        print("\nFull command:")
        cli_args = {
            self.config.param_space[name].cli_flag: val
            for name, val in cli_values.items()
        }
        full_cmd = [self.config.executable] + self.config.fixed_args.to_cli_list()
        for flag, val in cli_args.items():
            full_cmd += [flag, str(val)]
        print("  " + " ".join(full_cmd))
        print("=" * 60)

    def _save_results(self, study: optuna.Study) -> None:
        results = []
        for t in study.trials:
            if t.state != optuna.trial.TrialState.COMPLETE:
                continue
            entry = {
                "trial": t.number,
                "value": t.value,
                "params": t.params,
                "user_attrs": t.user_attrs,
            }
            results.append(entry)

        results.sort(key=lambda r: r["value"], reverse=(self.config.direction == "maximize"))

        out = Path(self.config.results_path)
        out.write_text(json.dumps(results, indent=2))
        log.info("Saved %d trial results → %s", len(results), out)


# ===========================================================================
# SECTION 6: Entry Point
# ===========================================================================

def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna HPO for QLearningRegressionTest",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-trials",    type=int,   default=QLEARNING_CONFIG.n_trials,
                   help="Number of optuna trials")
    p.add_argument("--n-jobs",      type=int,   default=QLEARNING_CONFIG.n_jobs,
                   help="Parallel trials (each trial runs the exe)")
    p.add_argument("--study-name",  type=str,   default=QLEARNING_CONFIG.study_name)
    p.add_argument("--storage",     type=str,   default=None,
                   help="Optuna storage URL. Auto-generated from config if omitted.")
    p.add_argument("--results",     type=str,   default=None,
                   help="Path to save JSON results. Auto-generated from config if omitted.")
    p.add_argument("--num-threads", type=int,   default=10,
                   help="--num-threads passed to the exe")
    p.add_argument("--num-rounds",  type=int,   default=100_000_000,
                   help="--num-rounds passed to the exe")
    p.add_argument("--verbose",     action="store_true")
    return p.parse_args()


def main():
    args = parse_cli()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Override fixed args first so the slug reflects any CLI overrides
    QLEARNING_CONFIG.fixed_args.args["--num-threads"] = args.num_threads
    QLEARNING_CONFIG.fixed_args.args["--num-rounds"]  = args.num_rounds

    # Auto-generate storage URL and results path from the config slug.
    # Paths are anchored to the project root so they're consistent regardless
    # of which directory you invoke the script from.
    slug = QLEARNING_CONFIG.fixed_args.to_slug()
    db_path   = _PROJECT_ROOT / f"hpo_{slug}.db"
    json_path = _PROJECT_ROOT / f"hpo_{slug}.json"
    QLEARNING_CONFIG.storage      = args.storage  or f"sqlite:///{db_path}"
    QLEARNING_CONFIG.results_path = args.results  or str(json_path)

    QLEARNING_CONFIG.n_trials    = args.n_trials
    QLEARNING_CONFIG.n_jobs      = args.n_jobs

    # Append a short hash of the param space to the study name so that any
    # change in the search space (choices, bounds, types) automatically creates
    # a new study instead of clashing with an existing one in the same DB.
    space_hash = _param_space_hash(QLEARNING_CONFIG.param_space)
    QLEARNING_CONFIG.study_name  = f"{args.study_name}_{space_hash}"
    log.info("Study name: %s", QLEARNING_CONFIG.study_name)

    study_runner = HPOStudy(QLEARNING_CONFIG)
    study_runner.run()


if __name__ == "__main__":
    main()
