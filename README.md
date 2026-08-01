# BlackjaclCpp

A C++20 blackjack simulation and research project with:

- A reusable blackjack engine
- Classic blackjack and Double Down Madness table variants
- Table-driven and learned playing strategies
- Card-count-aware betting and regression tooling
- Standalone research apps for RL, OLS, alternating optimization, and count comparison
- GoogleTest-based unit, regression, and benchmark targets

## Repository Layout

```text
.
├── apps/                         # Standalone executables for training and analysis
├── basic_strategy_tables/        # JSON basic-strategy tables and edge references
├── checkpoints/
│   ├── alternating-checkpoints/  # AlternatingOptimization runs and resumable state
│   ├── double-down-madness-alternating-checkpoints/
│   ├── checkpoints_QLearning/    # FindDeviations Q-learning checkpoints
│   ├── checkpoints_ols/          # FindOptimalCount OLS checkpoints
│   ├── CompareCountStrategies/   # Compare app logs + graph artifacts
│   └── MeasureEdge/              # MeasureEdge run logs
├── hpo/                          # Python hyperparameter tuning helpers
├── include/                      # Third-party headers
├── src/
│   ├── Game/                     # Blackjack rules, table flow, players, betting
│   ├── RL/                       # Strategy interfaces and learning implementations
│   ├── Shoe/                     # Cards, decks, shoes, debug shoe
│   └── Utils/                    # Parallel simulation helpers and run logging
├── unittest/                     # Unit, regression, and benchmark tests
└── CMakeLists.txt
```

## Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

Production code is compiled once into the `blackjack_objs` object library and reused by apps and tests.

## Test

```bash
ctest --test-dir build
```

Main test targets include:

- `ShoeTest`
- `HandTest`
- `BlackjackTableTest`
- `BasicStrategyTest`
- `DebugShoeTest`
- `BasicStrategyRegressionTest`
- `QLearningRegressionTest`
- `DoubleDownMadnessEdgeRegressionTest`
- `DoubleDownMadnessQLearningRegressionTest`
- `PerformanceBenchmark`
- `BlackjackBenchmark`

## Apps

- `FindDeviations`
  Trains or resumes Q-learning policy checkpoints. Supports fixed-round training and diff-based stopping by comparing adjacent Q-table snapshots.

- `FindOptimalCount`
  Runs OLS regression on removed-card features to derive a counting system and saves resumable `XtX/Xty` checkpoints.

- `MeasureEdge`
  Evaluates a fixed playing strategy with flat betting, spread betting, or Kelly betting. Learned Q checkpoints are converted to fixed greedy `BasicStrategy` tables before evaluation.

- `AlternatingOptimization`
  Selects classic blackjack with `--game blackjack` (the default) or Double
  Down Madness with `--game ddm`, then alternates between policy and count learning:
  `W0 -> P0 -> W1 -> P1 -> ...`
  with resumable checkpoints, EV-vs-count graphs, cumulative overlays, histogram artifacts,
  conditional `E[X^2 | count]` graphs, round-return variance, and Kelly-multiplier
  sweep graphs. DDM supports `--version 1|2|3`; blackjack-only rule flags are
  ignored with warnings in DDM mode.

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
  strategy, Illustrious 18, and full deviations. DDM compares its known
  version-specific policy with full deviations.
  It writes a run log, EV-vs-count graph/data, a count histogram, spread-return variance,
  conditional `E[X^2 | count]` curves, and an overlaid Kelly-multiplier sweep for
  all policies. Irrelevant game-specific CLI flags are ignored with warnings.

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

- `checkpoints/checkpoints_QLearning/<folder>/`
  `meta.json`, per-agent Q tables, and strategy tables from `FindDeviations`

- `checkpoints/checkpoints_ols/<folder>/`
  `meta.json`, `data.json` (`XtX`, `Xty`, rounds) from `FindOptimalCount`

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

- `checkpoints/MeasureEdge/<run-name>/`
  `run.log`

Most apps now log terminal output into their checkpoint/run folder through `RunLogger`.

## Current Research Direction

The project currently has two major research paths:

- Count discovery with OLS (`FindOptimalCount`)
- Joint count/policy improvement with alternating optimization (`AlternatingOptimization`)

The compare and measurement apps exist mainly to validate and inspect those learned artifacts.

## Notes For Another Agent

- `Player` is the bridge between raw table state and strategy state keys. Count discretization, betting context, regression sampling, EV-count bins, and streaming return moments all live there.
- `BlackjackTable::round()` captures net round return after all splits, doubles, and settlements. Variance tracking stores only count, sum, and sum of squares, and these totals merge across threads without retaining outcome histories.
- `DoubleDownMadnessTable` models the one-card player start, repeated doubling,
  continued play after hitting an initial Ace, a one-card limit when doubling
  that Ace, immediate version-specific blackjack payouts, a hidden dealer hole
  card, and a dealer-22 push.
- `MeasureEdge` and `CompareCountStrategies` are evaluation apps; they should not leave a live `QLearningStrategy` exploring during measurement.

## Architecture Notes

See [ARCHITECTURE.md](ARCHITECTURE.md) for the current runtime flow, class relationships, and how counting, betting, regression, graphing, and learning interact.
