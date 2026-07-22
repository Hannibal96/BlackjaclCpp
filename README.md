# BlackjaclCpp

A C++20 blackjack simulation and research project with:

- A reusable blackjack engine
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
  Alternates between policy learning and count learning:
  `W0 -> P0 -> W1 -> P1 -> ...`
  with resumable checkpoints, EV-vs-count graphs, cumulative overlays, and histogram artifacts.

- `CompareCountStrategies`
  Compares one count system under:
  basic strategy, Illustrious 18, and full deviations.
  It writes a run log, EV-vs-count graph/data, and a count histogram beside the run folder.

## Checkpoints And Run Artifacts

- `checkpoints/checkpoints_QLearning/<folder>/`
  `meta.json`, per-agent Q tables, and strategy tables from `FindDeviations`

- `checkpoints/checkpoints_ols/<folder>/`
  `meta.json`, `data.json` (`XtX`, `Xty`, rounds) from `FindOptimalCount`

- `checkpoints/alternating-checkpoints/<folder>/`
  `meta.json`, `state.json`, `P*.json`, `P*_agent.json`, `P*_strategy.json`, `W*.json`, `W*_data.json`

- `checkpoints/CompareCountStrategies/<run-name>/`
  `run.log`, `ev_count_graph.json/svg`, `count_histograms.json/svg`

- `checkpoints/MeasureEdge/<run-name>/`
  `run.log`

Most apps now log terminal output into their checkpoint/run folder through `RunLogger`.

## Current Research Direction

The project currently has two major research paths:

- Count discovery with OLS (`FindOptimalCount`)
- Joint count/policy improvement with alternating optimization (`AlternatingOptimization`)

The compare and measurement apps exist mainly to validate and inspect those learned artifacts.

## Notes For Another Agent

- `Player` is the bridge between raw table state and strategy state keys. Count discretization, count clamping, betting context, regression sampling, and EV-count graph collection all live there.
- `BlackjackTable::round()` is where per-round tracking hooks are captured for both regression and EV-count graphing.
- `MeasureEdge` and `CompareCountStrategies` are evaluation apps; they should not leave a live `QLearningStrategy` exploring during measurement.
- The compare app currently contains temporary Kelly per-run debug printing in its run log and console output.

## Architecture Notes

See [architecture.md](architecture.md) for the current runtime flow, class relationships, and how counting, betting, regression, graphing, and learning interact.
