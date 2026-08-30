# BlackjaclCpp Architecture

## Overview

The codebase is organized around a blackjack simulation engine with four layers:

1. `src/Shoe` produces cards and tracks exposed-card history.
2. `src/Game` applies blackjack rules, manages players, bets, hands, and round flow.
3. `src/RL` chooses actions from transformed game state and optionally learns from rewards.
4. `src/Utils` runs parallel simulations and mirrors terminal output into per-run log files.

The apps in `apps/` are orchestration binaries built on top of those shared layers.

## High-Level Flow

```text
Shoe / DebugShoe
    ↓
BlackjackTable::round()
    ↓
collectBets()
dealInitialCards()
playersPlay()
dealerPlays()
evaluate()
    ↓
Player bankroll updates
Strategy updates (when learning)
Regression / EV-count graph tracking (when enabled)
```

## Core Runtime Pieces

### `src/Shoe`

- `Deck` builds a standard 52-card deck.
- `Card` converts rank/suit into blackjack values and printable text.
- `Shoe` combines one or more decks, shuffles, deals cards, tracks `removedCards`, and supports covered hole cards.
- `DebugShoe` provides deterministic dealing for reproducible tests and debugging.

`removedCards` is the shared data source for:

- true count computation
- strategy state discretization
- betting decisions
- OLS regression features
- EV-vs-count graph collection

### `src/Game`

- `Rules` stores generic game settings.
- `BlackjackRules` extends it with blackjack-specific options such as surrender, peek, split rules, and min/max bet.
- `Hand` models blackjack hand value, softness, blackjack detection, and splitting.
- `Slot` groups one seat's active hands after splits.
- `Player` owns bankroll state, the playing strategy, counting configuration, optional betting strategy, regression accumulators, and EV-count histogram bins.
- `Table` is the abstract round coordinator.
- `BlackjackTable` is the concrete blackjack implementation.

`BlackjackTable` is the runtime heart of the project.

### `src/RL`

- `Strategy` is the core playing-decision interface.
- `RandomStrategy` samples a legal action uniformly.
- `BasicStrategy` loads or materializes a fixed JSON strategy table and can print itself.
- `QLearningStrategy` stores the Q-table, exploration settings, alpha schedules, and checkpoint serialization.
- `State` is the raw visible game state.
- `StateKey` is the compact strategy lookup key.
- `DecayingParameter` models alpha and exploration schedules.

## Player As The Adapter

`Player` is the key adapter between the table engine and the strategy/betting layers.

It is responsible for:

- converting raw `State` into `StateKey`
- computing the true count from `removedCards`
- discretizing and clamping the count into `[minCount, maxCount]`
- building `BettingContext`
- tracking bankroll and cached log-bankroll
- accumulating OLS regression matrices
- accumulating EV-count graph bins

This means most count-related behavior is centralized in `Player`, not duplicated across strategies.

## Round Lifecycle Details

`Table` defines the skeleton and `BlackjackTable` supplies the blackjack-specific rules:

```text
round
├── reset shoe if penetration reached
├── capture pre-round tracking state
├── collectBets
├── dealInitialCards
├── playersPlay
├── dealerPlays
├── evaluate payouts
└── record round-level regression / graph data
```

Important current details:

- Covered dealer hole cards are uncovered later so no-peek and peek counting behavior stays correct.
- Dealer-blackjack rounds can change bankroll without performing a learning update.
- Split hands are represented inside `Slot`, not by cloning players.
- Regression and EV-count graph features are sampled from the pre-round shoe state and paired with the round outcome.

## Counting, Discretization, And Betting

Card counting is not embedded inside `BasicStrategy` or `QLearningStrategy`. Instead:

- `Shoe` tracks exposed cards
- `Player` converts them to a true count
- `Player` discretizes that count for strategy lookup
- `Player` computes `expectedValue = bias + factor * trueCount`
- `BettingStrategy` consumes that expected value and bankroll

Current betting implementations:

- `SpreadBetting`
- `KellyBetting`

This separation lets the same play policy be evaluated under multiple betting models.

## Learning Modes

### Q-Learning

`QLearningStrategy` stores values keyed by:

```text
(count, handType, playerSum, dealerCard, action)
```

Key properties:

- learning-rate and exploration schedules are decayed per state
- checkpoints serialize the full Q table and metadata
- strategies support `operator+=` and `operator*=` so parallel threads can be merged by averaging
- Q-table snapshots can be compared via average absolute entry difference for diff-based stopping

### OLS Count Fitting

The count-learning half of `AlternatingOptimization` accumulates:

- `XtX` (14 x 14)
- `Xty` (14)
- sampled round count

The 14th parameter is the bias term. These normal-equation statistics are
streamed (never retaining round histories) and saved per-step in `W*_data.json`,
resumable and later converted into a count system.

## EV-Count Graphing

The project now supports dedicated EV-vs-count measurement runs.

The flow is:

1. run a flat-bet simulation with a fixed play strategy and count system
2. collect per-count-bin:
   - number of samples
   - sum of rewards
   - sum of squared rewards
3. derive:
   - empirical mean reward
   - standard deviation
   - 95% confidence bounds
4. save JSON data and SVG visualizations

This is used heavily by:

- `AlternatingOptimization`
- `CompareCountStrategies`

## Apps Layer

The app surface is three binaries, each implemented in one file:
`apps/AlternatingOptimization.cpp`, `apps/CompareCountStrategies.cpp`, and
`apps/QuantizationEffect.cpp`. Each follows the `template <typename Game>`
engine and shared-dispatch shape. The first two instantiate both games;
`QuantizationEffect` currently enables blackjack and retains a clear DDM
dispatch boundary for later support. Game selection
is dispatched by `--game` through the header-only `apps/GameAppDispatcher.h`
(shared by all three apps, since the routing logic is identical). Everything that
differs between the two games — `Rules` construction, the `Case` parameter
struct, CLI rule flags, meta.json fields, checkpoint-root naming, basic-strategy
loading, and whether Illustrious 18 applies — lives in the shared
`apps/GameTraits.h`; the rest of the app logic is written once. Earlier
standalone precursors —
`FindDeviations` (Q-learning checkpoints under `checkpoints/checkpoints_QLearning/`),
`FindOptimalCount` (OLS checkpoints under `checkpoints/checkpoints_ols/`), and
`MeasureEdge` (fixed-strategy evaluation under `checkpoints/MeasureEdge/`) —
have been removed; `AlternatingOptimization`'s loop and
`CompareCountStrategies` reading its checkpoints directly now cover that
ground.

### `AlternatingOptimization`

- selects classic blackjack or Double Down Madness with `--game`
- ignores explicitly supplied rules that do not apply to the selected game and
  prints a warning
- alternates:
  `W0 -> P0 -> W1 -> P1 -> ...`
- `P_k` is learned by RL with a fixed count
- `W_k` is learned with a fixed policy using either classical OLS or the
  quadratic-Kelly objective
- selects the objective independently with `--count-classical-ols` or
  `--count-quadratic-kelly`
- applies the same constraint flags to either objective:
  `--count-unconstrained`, `--count-sum-zero`,
  `--count-sum-zero-fixed-b1`, or `--count-sum-zero-fixed-p0-edge`
- stores the objective and constraint separately in checkpoint metadata
- supports resume via `state.json`
- saves:
  - `P*.json`
  - `P*_agent.json`
  - `P*_strategy.json`
  - `W*.json`
  - `W*_data.json`
  - `W*_graph.json/svg`
  - `W*_graph_overlay.*`
- stores blackjack runs under `checkpoints/alternating-checkpoints/`
- stores DDM runs under `checkpoints/double-down-madness-alternating-checkpoints/`

### `CompareCountStrategies`

- selects classic blackjack or Double Down Madness with `--game`
- blackjack compares basic strategy, Illustrious 18, and full deviations
- DDM compares its known version-specific strategy and full deviations
  (Illustrious 18 is Hi-Lo/blackjack-specific, so it's skipped for DDM)
- full deviations come from one of three sources:
  - `--deviations-checkpoint <dir> --deviations-agent Pk`: that game's own
    `AlternatingOptimization` checkpoint folder (`Pk_agent.json` plus,
    unless the count was explicitly overridden, the matching `Wk.json`/`W0`
    count config)
  - `--strategy-checkpoint <dir>`: a `full_deviations_strategy.json` +
    `full_deviations_meta.json` pair saved by a previous
    `CompareCountStrategies` fresh-training run (mutually exclusive with
    `--deviations-checkpoint`)
  - neither flag: trained fresh, then auto-saved as the pair above into this
    run's own output folder
- writes:
  - `run.log`
  - `ev_count_graph.json/svg`
  - `count_histograms.json/svg`
  under the game-specific comparison checkpoint root
- writes aggregate Kelly growth curves and error bars; individual Kelly trial
  debug printing is disabled

### `QuantizationEffect`

- loads a learned count `Wk` and its generating policy `P(k-1)` from an
  alternating checkpoint, defaulting to the latest complete pair
- reconstructs full-precision learned tags from `raw_solution` and
  `normalization_scale`
- rounds to a user-supplied quantum grid while preserving a source zero-sum
  constraint and shared 10/J/Q/K tags
- reruns quantum zero once as an exact reference
- evaluates 1:10 spread edge and a configurable Kelly multiplier sweep
- can use deterministic common seeds across quantum levels
- writes `run.log`, `results.json`, `results.csv`, and
  `quantization_effect.svg` under `checkpoints/QuantizationEffect/`

## Checkpoint And Logging Convention

The project now keeps run output beside the relevant checkpoint family instead of in a separate log tree.

Examples:

- `checkpoints/alternating-checkpoints/<folder>/` (blackjack)
- `checkpoints/double-down-madness-alternating-checkpoints/<folder>/` (DDM)
- `checkpoints/CompareCountStrategies/<run-name>/` (blackjack)
- `checkpoints/DoubleDownMadnessCompareCountStrategies/<run-name>/` (DDM)
- `checkpoints/QuantizationEffect/<run-name>/` (blackjack-first)

`RunLogger` mirrors `stdout` and `stderr` into `run.log` inside those folders.

## Extension Points

The cleanest extension points are:

- add a new play strategy by implementing `Strategy`
- add a new betting model by implementing `BettingStrategy`
- add a new counting system through `CountingMethods` or explicit weights
- add new analysis binaries in `apps/` without changing the engine core

## Important Files

- `apps/CMakeLists.txt`
- `src/Game/BlackjackTable.cpp`
- `src/Game/Player.cpp`
- `src/Game/BettingStrategy.h`
- `src/RL/BasicStrategy.cpp`
- `src/RL/QLearningStrategy.cpp`
- `src/Utils/RunLogger.cpp`
