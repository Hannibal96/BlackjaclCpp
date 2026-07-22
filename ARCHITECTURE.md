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
| `AlternatingOptimization` | Alternate between RL policy learning and OLS count learning | `checkpoints/alternating-checkpoints/` |
| `CompareCountStrategies` | Compare basic / I18 / full deviations for one count | `checkpoints/CompareCountStrategies/` |

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
| `Hand` | `class` | — |
| `Slot` | `class` | — |
| `Player` | `class` | — |
| `Table` | `abstract class` | — |
| `BlackjackTable` | `class` | `Table` |

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
└── recordRound() for regression / graph bins
```

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

### Compare artifacts

- `run.log`
- `ev_count_graph.json/svg`
- `count_histograms.json/svg`

## Important Architectural Notes

- Count logic lives in `Player`, not in the strategy classes.
- Learned Q policies are converted to greedy `BasicStrategy` tables before evaluation apps use them.
- Regression and EV-count graph collection use pre-round shoe state with round outcome as the target.
- Parallel simulations merge players and strategies through averaging operators.
- `RunLogger` mirrors terminal output into the same run folder used by checkpoints or graph artifacts.

## Files To Read First

- [src/Game/BlackjackTable.cpp](/home/neria/Desktop/BlackjaclCpp/src/Game/BlackjackTable.cpp)
- [src/Game/Player.cpp](/home/neria/Desktop/BlackjaclCpp/src/Game/Player.cpp)
- [src/Game/BettingStrategy.h](/home/neria/Desktop/BlackjaclCpp/src/Game/BettingStrategy.h)
- [src/RL/BasicStrategy.h](/home/neria/Desktop/BlackjaclCpp/src/RL/BasicStrategy.h)
- [src/RL/QLearningStrategy.h](/home/neria/Desktop/BlackjaclCpp/src/RL/QLearningStrategy.h)
- [apps/AlternatingOptimization.cpp](/home/neria/Desktop/BlackjaclCpp/apps/AlternatingOptimization.cpp)
- [apps/CompareCountStrategies.cpp](/home/neria/Desktop/BlackjaclCpp/apps/CompareCountStrategies.cpp)
