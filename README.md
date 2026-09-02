# BlackjaclCpp

A C++20 blackjack simulation and research project with:

- A reusable blackjack engine
- Classic blackjack and Double Down Madness table variants
- Table-driven and learned playing strategies
- Card-count-aware betting and regression tooling
- `AlternatingOptimization` plus one authoritative fixed count/policy evaluator,
  `EvaluateCountPolicy`; Python workflows orchestrate policy comparisons and
  count quantization through that evaluator
- Standalone simulation regression targets

## Repository Layout

```text
.
├── apps/                         # Training and fixed count/policy evaluation apps
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
├── scripts/                      # Quantization and policy-comparison workflows
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
  `CountQuantizationTest`, `KellyExposureTest`, `CountPolicyEvaluationTest`
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

# Evaluate one named count and one policy:
./build/bin/EvaluateCountPolicy --game blackjack \
    --count-name halves --policy basic --factor 0.005 --bias -0.005

# Evaluate a learned W3 + P2 pair. Rules and simulation defaults come from the folder:
./build/bin/EvaluateCountPolicy --game blackjack \
    --checkpoint <folder-or-path> --count W3 --policy P2 --seed 12345

# Cap the initial wager plus all split/double wagers at 25% of the bankroll
# measured immediately before the round's initial wager:
./build/bin/EvaluateCountPolicy --game blackjack \
    --checkpoint <folder-or-path> --count W3 --policy P2 \
    --max-total-wager-fraction 0.25

# Compare basic / Illustrious 18 / full deviations through that same evaluator:
./scripts/compare_count_strategies.py --game blackjack --count hilo

# Measure W3 with the policy from which it was learned (P2) across the default
# quantization grid. Rules and performance defaults come from meta.json:
./scripts/quantization_effect.py --game blackjack \
    --checkpoint <folder-or-path> --count W3 --policy P2 --seed 12345

# Full flag reference:
./build/bin/AlternatingOptimization --game <blackjack|ddm> --help
./build/bin/EvaluateCountPolicy --game blackjack --help
./scripts/compare_count_strategies.py --help
./scripts/quantization_effect.py --help
```

`<folder-name>` is whatever `AlternatingOptimization` printed after `Folder:`
when it started (or `--checkpoint-name` if you set one). The evaluator also
accepts an explicit path, including experiment folders under `paper_materials`.

## Apps

The primary app surface consists of `AlternatingOptimization` and
`EvaluateCountPolicy`. The latter has the same `--game` dispatch shape as the
other research apps, but intentionally enables blackjack only for now so a
Spanish 21 implementation can be connected later without changing its CLI.
The old C++ comparison/quantization executables remain temporarily for command
compatibility; new work should use the scripts, which call only
`EvaluateCountPolicy` for simulation.
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

- `EvaluateCountPolicy`
  Takes one count and one matching fixed policy, then measures a 1:10 spread
  edge and a Kelly-multiplier curve. Counts may be named, supplied as 13 rank
  tags/JSON, or loaded as `Wk` from an alternating checkpoint. Policies may be
  basic strategy, standard Hi-Lo Illustrious 18, a policy JSON, an alternating
  `Pk`, or fresh full deviations trained under the selected count.

  The betting transform is explicit: `f(c)=bias+factor*c`. `c` can be a true
  count or an unnormalized running count. `--initial-count` and
  `--initial-count-per-deck` support systems with a nonzero initial running
  count; for example Spanish 21's `-4*N` start uses
  `--count-normalization running --initial-count-per-deck -4`.

  Every Kelly multiplier aggregates wager diagnostics across every measurement.
  `gross_exposure` is the sum of all wagers placed in the complete round divided
  by bankroll before the initial bet. Thus an initial wager followed by a split
  and one double contributes `3x`; four hands all doubled contribute `8x`.
  `absolute_return` is the realized `|round_profit / bankroll| = |fX|`. JSON
  contains histograms, tail probabilities, exact `log(1+r)`, its quadratic
  approximation `r-r^2/2`, and their error. Multiplier 1.0 is always evaluated
  and receives dedicated JSON/CSV/SVG artifacts. Exposure SVGs use a logarithmic
  probability axis, percentage ticks, grids, labeled axes, and side-by-side
  gross-exposure/`|fX|` bars so the no-bet bin does not hide rare tails.

  Kelly evaluation accepts `--max-total-wager-fraction <0..1>` (alias
  `--max-total-wager`). The default is `1.0`. The cap is based on bankroll
  immediately before the initial wager and applies to cumulative gross wagers:
  the initial wager plus every split and double still counts after an earlier
  hand settles. It does not cap positive payouts, so a 5% wager cap can still
  produce a 7.5% `|fX|` on a 3:2 natural blackjack.

- `scripts/compare_count_strategies.py`
  Runs basic strategy, Illustrious 18 when the count is true-counted Hi-Lo, and
  fresh or pre-calculated full deviations as separate calls to
  `EvaluateCountPolicy`, then combines their spread, Kelly-at-1, and wager
  diagnostics.

- `scripts/quantization_effect.py`
  Loads a learned alternating pair `Wk + P(k-1)` (latest complete pair by
  default), reconstructs the learned tags from `raw_solution` at full
  precision, and evaluates the exact count plus user-selected quantization
  steps. Balanced counts use minimum-squared-error constrained rounding, keep
  `sum(weights)=0`, and retain one shared 10/J/Q/K tag. The default grid is
  `0,0.01,0.05,0.1,0.5,1.0`; zero is simulated once as the exact reference.
  Rules, spread rounds, threads, and Kelly measurement count default from the
  source `meta.json`. Kelly multipliers default to `0.75..1.25` in `0.05`
  steps, and `--seed` provides repeatable common random seeds across quantum
  levels. Each level is evaluated by `EvaluateCountPolicy`; the combined report
  uses multiplier-1 Kelly results rather than the empirical optimum, so it does
  not hide degradation through fraction retuning. Results are written under
  `checkpoints/QuantizationEffect/` unless `--output-dir` is supplied.

Conditional second-moment curves use only flat simulations with an initial wager
of exactly 1, so `X` is the normalized net profit from one complete round. The
statistics are accumulated in the same count bins as the EV graph and are not
measured during spread simulations.

`AlternatingOptimization` accepts `--kelly-fraction-min`,
`--kelly-fraction-max`, and `--kelly-fraction-step`. Its classical-OLS curve is
centered on the flat-simulation estimate `1/E[X^2]`; a quadratic-Kelly count is
already a wager fraction, so its sweep is centered at `1.0`.
`EvaluateCountPolicy` instead defaults to the explicit `0.75..1.25` multiplier
range requested for fixed-count comparisons and accepts the shorter
`--kelly-min`, `--kelly-max`, and `--kelly-step` flags (the longer aliases also
work). It always inserts multiplier `1.0` as a direct, unmitigated reference.
Each fraction runs 10 independent experiments by default, configurable with
`--kelly-measurements`. Kelly graphs show the mean growth with error bars equal to
the sample standard deviation across those experiments.
Because `KellyBetting` uses `bet / bankroll = multiplier * estimatedEV`, the
small-edge prediction for the optimal multiplier is `1 / E[X^2]`, where `X` is
the net unit-bet return for one complete blackjack round.

DDM permits repeated doubling, so an initially small bankroll fraction can
become a catastrophic single-hand exposure after multiplication by `2^d`.
The cumulative-wager cap applies to DDM doubling as well. Quadratic Kelly is
only a second-order approximation; without a restrictive cap, a ruined worker
has log bankroll `-infinity`, which makes that
entire parallel Kelly experiment report zero growth. Treat DDM Kelly curves with
large standard deviations as ruin mixtures, not ordinary Monte Carlo noise.

## Checkpoints And Run Artifacts

- `checkpoints/alternating-checkpoints/<folder>/`
  `meta.json`, `state.json`, `P*.json`, `P*_agent.json`, `P*_strategy.json`, `W*.json`,
  `W*_data.json`, `W*_second_moment_graph.json/svg`, cumulative
  `W*_second_moment_graph_overlay.json/svg`, `W*_kelly_graph.json/svg`, and
  cumulative `W*_kelly_graph_overlay.json/svg`. Each `Wk` also writes
  `Wk_kelly_exposure_at_1.json/csv/svg`.

- `checkpoints/double-down-madness-alternating-checkpoints/<folder>/`
  Uses the same resumable policy, count, EV, second-moment, histogram, and Kelly
  artifact layout for Double Down Madness.

- `checkpoints/DoubleDownMadnessCompareCountStrategies/<run-name>/`
  DDM policy-comparison logs and EV/count, histogram, second-moment, and Kelly
  graph artifacts.

- `checkpoints/CompareCountStrategies/<run-name>/`
  One subfolder per policy, a combined `results.json/csv`, and each policy's
  `kelly_exposure_at_1.json/csv/svg`.

- `checkpoints/EvaluateCountPolicy/<run-name>/`
  `run.log`, `results.json/csv`, `kelly_growth.json/svg`, and
  `kelly_exposure_at_1.json/csv/svg`.

- `checkpoints/QuantizationEffect/<run-name>/`
  One subfolder per quantum, the exact quantized count JSON files, and a
  combined `results.json/csv`. Every child contains its Kelly exposure artifacts.

All apps log terminal output into their checkpoint/run folder through `RunLogger`.

## Current Research Direction

The project's research path is joint count/policy improvement with alternating
optimization (`AlternatingOptimization`). The comparison and quantization
scripts validate those checkpoints through the shared fixed evaluator.

## Notes For Another Agent

- `Player` is the bridge between raw table state and strategy state keys. Count discretization, betting context, regression sampling, EV-count bins, and streaming return moments all live there.
- `BlackjackTable::round()` captures net round return and cumulative gross wager
  after all splits, doubles, surrenders, and settlements. Streaming exposure
  histograms merge across threads and Kelly measurements without retaining
  individual rounds.
- `DoubleDownMadnessTable` models the one-card player start, repeated doubling,
  continued play after hitting an initial Ace, a one-card limit when doubling
  that Ace, immediate version-specific blackjack payouts, a hidden dealer hole
  card, and a dealer-22 push.
- `EvaluateCountPolicy` converts a trained `QLearningStrategy` to a fixed greedy
  `BasicStrategy` before measurement; exploration is never live during evaluation.

## Architecture Notes

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current runtime flow, class relationships, and how counting, betting, regression, graphing, and learning interact.
