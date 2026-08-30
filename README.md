# BlackjaclCpp

A C++20 blackjack simulation and research project with:

- A reusable blackjack engine
- Classic blackjack and Double Down Madness table variants
- Table-driven and learned playing strategies
- Card-count-aware betting and regression tooling
- Three research apps — `AlternatingOptimization`, `CompareCountStrategies`,
  and the blackjack-first `QuantizationEffect` evaluator — using the shared
  `Game` traits/app-dispatch shape (see `apps/GameTraits.h`)
- Standalone simulation regression targets

## Repository Layout

```text
.
├── apps/                         # Training, strategy comparison, and count-quantization apps
├── basic_strategy_tables/
│   ├── blackjack/                # One JSON table per classic-blackjack rule combo
│   └── double_down_madness/      # Base table + per-version overrides
├── checkpoints/
│   ├── alternating-checkpoints/  # AlternatingOptimization runs and resumable state (blackjack)
│   ├── double-down-madness-alternating-checkpoints/  # same, for DDM
│   ├── CompareCountStrategies/   # Compare app logs + graph artifacts (blackjack)
│   ├── DoubleDownMadnessCompareCountStrategies/       # same, for DDM
│   └── QuantizationEffect/       # quantized-count tables, metrics, and plots
├── hpo/                          # Python hyperparameter tuning helpers
├── include/                      # Third-party headers
├── regression/                   # Regression tests: sim output vs. reference JSON tables
├── src/
│   ├── Game/                     # Blackjack rules, table flow, players, betting
│   ├── RL/                       # Strategy interfaces and learning implementations
│   ├── Shoe/                     # Cards, decks, shoes, debug shoe
│   └── Utils/                    # Parallel simulation helpers and run logging
└── CMakeLists.txt
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Production code is compiled once into the `blackjack_objs` object library and reused by apps and regression tests.

## Test

```bash
cmake --build build --target tests
```

Regression targets include:

- `BasicStrategyRegressionTest`, `QLearningRegressionTest`,
  `DoubleDownMadnessEdgeRegressionTest`, `DoubleDownMadnessQLearningRegressionTest`,
  `CountQuantizationTest`
  — standalone regression executables (no GoogleTest) that compare simulated
  output against reference JSON tables, in `regression/`

## Quick Start

```bash
# Alternating optimization (RL policy <-> count regression), classic blackjack,
# default rules, default count objective/constraint (classical OLS, sum-zero
# with W1's bias fixed for W2+):
./build/bin/AlternatingOptimization --game blackjack

# Same, but Double Down Madness paytable version 2:
./build/bin/AlternatingOptimization --game ddm --version 2

# Resume an interrupted/previous run by its checkpoint folder name:
./build/bin/AlternatingOptimization --game blackjack --load-checkpoint <folder-name>

# Compare basic strategy / Illustrious 18 / full deviations under Hi-Lo (default):
./build/bin/CompareCountStrategies --game blackjack

# Compare a specific policy+count learned by AlternatingOptimization (P1 with
# its own W1) instead of training/using Hi-Lo:
./build/bin/CompareCountStrategies --game blackjack \
    --deviations-checkpoint <folder-name> --deviations-agent P1

# Measure W3 with the policy from which it was learned (P2) across the default
# quantization grid. Rules and performance defaults come from meta.json:
./build/bin/QuantizationEffect --game blackjack \
    --checkpoint <folder-or-path> --count W3 --policy P2 --seed 12345

# Full flag reference:
./build/bin/AlternatingOptimization --game <blackjack|ddm> --help
./build/bin/CompareCountStrategies --game <blackjack|ddm> --help
./build/bin/QuantizationEffect --game blackjack --help
```

`<folder-name>` is whatever `AlternatingOptimization` printed after `Folder:`
when it started (or `--checkpoint-name` if you set one) — both apps look for
it under `checkpoints/alternating-checkpoints/` for blackjack and
`checkpoints/double-down-madness-alternating-checkpoints/` for DDM.

## Apps

The app surface has three executables, each implemented in one `.cpp` file.
`AlternatingOptimization` and `CompareCountStrategies` are instantiated for
both games. `QuantizationEffect` follows the same `template <typename Game>`
and `--game` dispatch shape, but currently enables only its blackjack
instantiation so Double Down Madness support can be added without redesigning
the app.
Earlier, standalone precursors — `FindDeviations` (Q-learning checkpoints),
`FindOptimalCount` (OLS count checkpoints), and `MeasureEdge` (fixed-strategy
evaluation) — have been removed; their functionality is fully covered by
`AlternatingOptimization`'s `W0 -> P0 -> W1 -> P1 -> ...` loop and by
`CompareCountStrategies` reading its checkpoints directly.

- `AlternatingOptimization`
  Selects classic blackjack with `--game blackjack` (the default) or Double
  Down Madness with `--game ddm`, then alternates between policy and count learning:
  `W0 -> P0 -> W1 -> P1 -> ...`
  with resumable checkpoints, EV-vs-count graphs, cumulative overlays, histogram artifacts,
  conditional `E[X^2 | count]` graphs, round-return variance, and Kelly-multiplier
  sweep graphs. DDM supports `--version 1|2|3`; blackjack-only rule flags are
  ignored with warnings in DDM mode.

  Use `--graph-rounds <N>` to choose the number of rounds used to form EV/count,
  histogram, and conditional-second-moment graphs independently of
  `--eval-rounds`. If omitted, graph rounds default to the evaluation rounds.

  Count fitting selects an objective independently from its constraints. Use
  `--count-classical-ols` or `--count-quadratic-kelly`, then optionally select
  `--count-unconstrained`, `--count-sum-zero`,
  `--count-sum-zero-fixed-b1`, or `--count-sum-zero-fixed-p0-edge`. The default
  remains classical OLS with sum-zero weights and the `W1` bias fixed for `W2+`.
  For example:

  ```bash
  ./build/bin/AlternatingOptimization --game blackjack \
      --count-classical-ols --count-sum-zero
  ./build/bin/AlternatingOptimization --game ddm \
      --count-quadratic-kelly --count-sum-zero-fixed-b1
  ```

  Quadratic-Kelly direct-wager mode streams
  `A = sum(X^2 cc^T)` and `b = sum(Xc)` from flat rounds and solves `Aw=b`, where
  `c` is the pre-round normalized removed-card vector and `X` is complete unit-wager
  round profit. The feature vector includes a constant entry, so the fit learns
  its bias together with the rank weights and evaluates `max(0,w^Tc)` continuously.
  Learned rank tags are normalized so the average 10/J/Q/K value is `-1`, with
  inverse factor scaling preserving the fitted signal.

- `CompareCountStrategies`
  Uses the same `--game <blackjack|ddm>` selection. Blackjack compares basic
  strategy, Illustrious 18, and full deviations; DDM compares its known
  version-specific policy with full deviations (Illustrious 18 is Hi-Lo/blackjack-specific,
  so it's skipped for DDM). Full deviations come from one of three sources:
  - `--deviations-checkpoint <dir> --deviations-agent Pk` loads a policy
    straight from an `AlternatingOptimization` checkpoint folder for the
    same game — `Pk_agent.json` for the policy and, unless the count was
    explicitly overridden on the command line, the matching `Wk.json`
    (or `W0`'s implicit all-zero count) for the count config.
  - `--strategy-checkpoint <dir>` reloads a policy that `CompareCountStrategies`
    itself trained fresh in a *previous* run, together with whatever count
    it was trained against (Hi-Lo by default, or any explicit `--count`).
    Every fresh-training run auto-saves this into its own output folder as
    `full_deviations_strategy.json` + `full_deviations_meta.json`, so pointing
    `--strategy-checkpoint` at that folder (by name under the checkpoint
    root, or by absolute path) skips retraining next time. Mutually
    exclusive with `--deviations-checkpoint`.
  - Neither flag: trains full deviations fresh using the RL flags, and
    auto-saves it as above.

  It writes a run log, EV-vs-count graph/data, a count histogram, spread-return variance,
  conditional `E[X^2 | count]` curves, and an overlaid Kelly-multiplier sweep for
  all policies. Irrelevant game-specific CLI flags are ignored with warnings.

- `QuantizationEffect`
  Loads a learned alternating pair `Wk + P(k-1)` (latest complete pair by
  default), reconstructs the learned tags from `raw_solution` at full
  precision, and evaluates the exact count plus user-selected quantization
  steps. Balanced counts use minimum-squared-error constrained rounding, keep
  `sum(weights)=0`, and retain one shared 10/J/Q/K tag. The default grid is
  `0,0.01,0.05,0.1,0.5,1.0`; zero is simulated once as the exact reference.
  Rules, spread rounds, threads, and Kelly measurement count default from the
  source `meta.json`. Kelly multipliers default to `0.75..1.25` in `0.05`
  steps, and `--seed` provides repeatable common random seeds across quantum
  levels. Results are written as a run log, JSON, CSV, and SVG under
  `checkpoints/QuantizationEffect/` unless `--output-dir` is supplied.

Conditional second-moment curves use only flat simulations with an initial wager
of exactly 1, so `X` is the normalized net profit from one complete round. The
statistics are accumulated in the same count bins as the EV graph and are not
measured during spread simulations.

Both apps accept `--kelly-fraction-min`, `--kelly-fraction-max`, and
`--kelly-fraction-step`. By default, each curve rounds its flat-simulation estimate
`1/E[X^2]` to the nearest fraction step, spans `+/-0.25` around that center, clamps
the lower endpoint at zero, and uses increments of `0.05`. Explicit minimum and
maximum flags override their respective dynamic endpoints.
For `--count-quadratic-kelly`, the learned signal is already a wager fraction, so
its multiplier sweep is centered at `1.0` instead of `1/E[X^2]`.
Each fraction runs 10 independent experiments by default, configurable with
`--kelly-measurements`. Kelly graphs show the mean growth with error bars equal to
the sample standard deviation across those experiments.
Because `KellyBetting` uses `bet / bankroll = multiplier * estimatedEV`, the
small-edge prediction for the optimal multiplier is `1 / E[X^2]`, where `X` is
the net unit-bet return for one complete blackjack round.

DDM permits repeated doubling, so an initially small bankroll fraction can
become a catastrophic single-hand exposure after multiplication by `2^d`.
Quadratic Kelly is only a second-order approximation and does not currently
enforce `1+fX>0`. A ruined worker has log bankroll `-infinity`, which makes that
entire parallel Kelly experiment report zero growth. Treat DDM Kelly curves with
large standard deviations as ruin mixtures, not ordinary Monte Carlo noise.

## Checkpoints And Run Artifacts

- `checkpoints/alternating-checkpoints/<folder>/`
  `meta.json`, `state.json`, `P*.json`, `P*_agent.json`, `P*_strategy.json`, `W*.json`,
  `W*_data.json`, `W*_second_moment_graph.json/svg`, cumulative
  `W*_second_moment_graph_overlay.json/svg`, `W*_kelly_graph.json/svg`, and
  cumulative `W*_kelly_graph_overlay.json/svg`

- `checkpoints/double-down-madness-alternating-checkpoints/<folder>/`
  Uses the same resumable policy, count, EV, second-moment, histogram, and Kelly
  artifact layout for Double Down Madness.

- `checkpoints/DoubleDownMadnessCompareCountStrategies/<run-name>/`
  DDM policy-comparison logs and EV/count, histogram, second-moment, and Kelly
  graph artifacts.

- `checkpoints/CompareCountStrategies/<run-name>/`
  `run.log`, `ev_count_graph.json/svg`, `count_histograms.json/svg`,
  `kelly_fraction_graph.json/svg`

- `checkpoints/QuantizationEffect/<run-name>/`
  `run.log`, `results.json`, `results.csv`, and `quantization_effect.svg`

All apps log terminal output into their checkpoint/run folder through `RunLogger`.

## Current Research Direction

The project's research path is joint count/policy improvement with alternating
optimization (`AlternatingOptimization`). `CompareCountStrategies` exists to
validate and inspect the resulting checkpoints against reference strategies.

## Notes For Another Agent

- `Player` is the bridge between raw table state and strategy state keys. Count discretization, betting context, regression sampling, EV-count bins, and streaming return moments all live there.
- `BlackjackTable::round()` captures net round return after all splits, doubles, and settlements. Variance tracking stores only count, sum, and sum of squares, and these totals merge across threads without retaining outcome histories.
- `DoubleDownMadnessTable` models the one-card player start, repeated doubling,
  continued play after hitting an initial Ace, a one-card limit when doubling
  that Ace, immediate version-specific blackjack payouts, a hidden dealer hole
  card, and a dealer-22 push.
- `CompareCountStrategies` is an evaluation app; it should not leave a live `QLearningStrategy` exploring during measurement.

## Architecture Notes

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current runtime flow, class relationships, and how counting, betting, regression, graphing, and learning interact.
