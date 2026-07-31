# BlackjaclCpp — Code Architecture Reference

This file is the compact reference companion to [architecture.md](architecture.md).

## Directory Layout

```text
BlackjaclCpp/
├── src/
│   ├── Shoe/          # Card / deck / shoe primitives
│   ├── Game/          # Game mechanics, table flow, betting, player state
│   ├── RL/            # Strategies and reinforcement learning
│   └── Utils/         # Parallel simulation + run logging
├── apps/              # Standalone training / evaluation executables
├── unittest/          # Unit, regression, and benchmark tests
├── basic_strategy_tables/
├── checkpoints/
│   ├── alternating-checkpoints/
│   ├── double-down-madness-alternating-checkpoints/
│   ├── checkpoints_QLearning/
│   ├── checkpoints_ols/
│   ├── CompareCountStrategies/
│   └── MeasureEdge/
└── CMakeLists.txt
```

## Main App Targets

| App | Purpose | Output Root |
|---|---|---|
| `FindDeviations` | Train/resume Q-learning playing deviations | `checkpoints/checkpoints_QLearning/` |
| `FindOptimalCount` | Fit OLS count weights | `checkpoints/checkpoints_ols/` |
| `MeasureEdge` | Evaluate fixed strategy + betting model | `checkpoints/MeasureEdge/` |
| `AlternatingOptimization` | Unified blackjack/DDM RL and OLS pipeline selected by `--game` | Game-specific alternating checkpoint root |
| `CompareCountStrategies` | Unified blackjack/DDM strategy comparison selected by `--game` | Game-specific comparison checkpoint root |

## Class / Type Hierarchy

### `src/Shoe`

| Type | Kind | Inherits |
|---|---|---|
| `Card` | `struct` | — |
| `Deck` | `class` | — |
| `Shoe` | `class` | — |
| `DebugShoe` | `class` | `Shoe` |

### `src/Game`

| Type | Kind | Inherits |
|---|---|---|
| `Rules` | `struct` | — |
| `BlackjackRules` | `struct` | `Rules` |
| `DoubleDownMadnessRules` | `struct` | `Rules` |
| `Hand` | `class` | — |
| `Slot` | `class` | — |
| `Player` | `class` | — |
| `Table` | `abstract class` | — |
| `BlackjackTable` | `class` | `Table` |
| `DoubleDownMadnessTable` | `class` | `Table` |

### `src/RL`

| Type | Kind | Inherits |
|---|---|---|
| `Strategy` | `abstract class` | — |
| `RandomStrategy` | `class` | `Strategy` |
| `BasicStrategy` | `class` | `Strategy` |
| `QLearningStrategy` | `class` | `Strategy` |
| `DecayingParameter` | `abstract class` | — |
| `EpsilonDecayingParameter` | `class` | `DecayingParameter` |
| `LinearDecayingParameter` | `class` | `DecayingParameter` |

### `src/Game` Betting

| Type | Kind | Inherits |
|---|---|---|
| `BettingStrategy` | `abstract class` | — |
| `SpreadBetting` | `class` | `BettingStrategy` |
| `KellyBetting` | `class` | `BettingStrategy` |

## Key State Shapes

```text
StateKey = (count, handType, playerSum, dealerCard)
QTableKey = (StateKey, action)
```

`Player::stateToKey()` is the central place where:

- true count is computed
- count is discretized / clamped
- hand type is derived
- raw table state becomes a strategy lookup key

## Round Skeleton

```text
BlackjackTable::round()
├── reset shoe if penetration reached
├── capture pre-round state for tracking
├── collectBets()
├── dealInitialCards()
├── playersPlay()
├── dealerPlays()
├── evaluate()
└── record round outcome for moments, regression, and graph bins
```

`DoubleDownMadnessTable::round()` follows the same outer tracking skeleton but
uses a one-card player start, a covered dealer hole card, repeated doubling,
continued decisions after hitting an initial Ace, a one-card limit after
doubling that Ace, immediate two-card blackjack settlement, and a push when the
dealer busts with exactly 22.

DDM Q-values are normalized to the wager entering each state. A hit carries
`V(next)` forward, while a double that continues carries `2 * V(next)`. This is
required because the strategy state intentionally excludes bankroll and wager
size, while DDM permits repeated doubling.

## Learning / Analysis Data

### Q-learning checkpoints

- `meta.json`
- `<agent>_agent.json`
- `<agent>_strategy.json`

### OLS checkpoints

- `meta.json`
- `data.json` containing:
  - `XtX`
  - `Xty`
  - sampled rounds

### Alternating optimization checkpoints

- `meta.json`
- `state.json`
- `P*.json`
- `P*_agent.json`
- `P*_strategy.json`
- `W*.json`
- `W*_data.json`
- `W*_graph.json/svg`
- `W*_graph_overlay.*`
- `W*_second_moment_graph.json/svg`
- `W*_second_moment_graph_overlay.*` containing cumulative W1 through Wk curves
- `W*_kelly_graph.json/svg`
- `W*_kelly_graph_overlay.*` containing cumulative W1 through Wk curves

The Double Down Madness optimizer writes the same artifact set beneath
`checkpoints/double-down-madness-alternating-checkpoints/`; its metadata records
the paytable version in addition to decks, H17/S17, and penetration.

### Compare artifacts

- `run.log`
- `ev_count_graph.json/svg`
- `count_histograms.json/svg`
- `second_moment_count_graph.json/svg` overlaying all compared policies
- `kelly_fraction_graph.json/svg` overlaying all compared policies

The DDM comparison app writes the same artifact families for known basic
strategy and full deviations. A loaded alternating policy `Pk` is paired with
`Wk` by default so both policies use the same count and betting model.

### Return moments and Kelly sweeps

`Player` optionally tracks the global round-return moments `n`, `sum(X)`, and
`sum(X^2)`. Thread-local totals are added without scaling, allowing the apps to
report empirical edge, sample `std(X)`, and `E[X^2]` without retaining round
histories. `BlackjackTable` reuses scratch bankroll storage and skips card-feature
construction when only moments are enabled.

The EV-vs-count flat simulation also bins `sum(X^2)` by count. Its dedicated
graphs report `E[X^2 | count]` for a fixed unit initial wager. Alternating
optimization writes a per-weight curve and cumulative W1-through-Wk overlays;
CompareCountStrategies overlays the compared policies. Spread simulations do
not contribute to these conditional curves.

`KellyBetting` uses `bet / bankroll = k * estimatedEV`. Under the small-edge
quadratic approximation, the optimal multiplier is therefore `k = 1/E[X^2]`.
The apps compare that prediction with empirical growth over a configurable
fraction grid.

## Important Architectural Notes

- Count logic lives in `Player`, not in the strategy classes.
- Learned Q policies are converted to greedy `BasicStrategy` tables before evaluation apps use them.
- Regression and EV-count graph collection use pre-round shoe state with round outcome as the target.
- Round-return variance includes the complete round result, including splits, doubles, surrender, blackjack payout, and the selected wager size.
- Parallel simulations merge players and strategies through averaging operators.
- `RunLogger` mirrors terminal output into the same run folder used by checkpoints or graph artifacts.

## Files To Read First

- [src/Game/BlackjackTable.cpp](/home/neria/Desktop/BlackjaclCpp/src/Game/BlackjackTable.cpp)
- [src/Game/DoubleDownMadnessTable.cpp](/home/neria/Desktop/BlackjaclCpp/src/Game/DoubleDownMadnessTable.cpp)
- [src/Game/Player.cpp](/home/neria/Desktop/BlackjaclCpp/src/Game/Player.cpp)
- [src/Game/BettingStrategy.h](/home/neria/Desktop/BlackjaclCpp/src/Game/BettingStrategy.h)
- [src/RL/BasicStrategy.h](/home/neria/Desktop/BlackjaclCpp/src/RL/BasicStrategy.h)
- [src/RL/QLearningStrategy.h](/home/neria/Desktop/BlackjaclCpp/src/RL/QLearningStrategy.h)
- [apps/AlternatingOptimization.cpp](/home/neria/Desktop/BlackjaclCpp/apps/AlternatingOptimization.cpp)
- [apps/BlackjackAlternatingOptimization.cpp](/home/neria/Desktop/BlackjaclCpp/apps/BlackjackAlternatingOptimization.cpp)
- [apps/DoubleDownMadnessAlternatingOptimization.cpp](/home/neria/Desktop/BlackjaclCpp/apps/DoubleDownMadnessAlternatingOptimization.cpp)
- [apps/CompareCountStrategies.cpp](/home/neria/Desktop/BlackjaclCpp/apps/CompareCountStrategies.cpp)
- [apps/BlackjackCompareCountStrategies.cpp](/home/neria/Desktop/BlackjaclCpp/apps/BlackjackCompareCountStrategies.cpp)
- [apps/DoubleDownMadnessCompareCountStrategies.cpp](/home/neria/Desktop/BlackjaclCpp/apps/DoubleDownMadnessCompareCountStrategies.cpp)
