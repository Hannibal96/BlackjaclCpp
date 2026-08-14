# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BlackjaclCpp is a C++ blackjack simulator with reinforcement learning capabilities. The project simulates blackjack games with configurable rules and allows training AI agents using Q-learning to develop optimal playing strategies.

## Build Commands

### Initial Setup
```bash
# Configure with CMake (creates build directory)
cmake -B build/debug -DCMAKE_BUILD_TYPE=Debug
cmake -B build/release -DCMAKE_BUILD_TYPE=Release

# Build all targets
cmake --build build/debug
cmake --build build/release
```

### Running Tests
```bash
# Build all regression tests
cmake --build build/debug --target tests
cmake --build build/release --target tests

# Run a regression executable
./build/debug/bin/BasicStrategyRegressionTest
./build/debug/bin/QLearningRegressionTest
./build/debug/bin/DoubleDownMadnessEdgeRegressionTest
./build/debug/bin/DoubleDownMadnessQLearningRegressionTest
```

### Development Workflow
```bash
# Rebuild after changes
cmake --build build/debug

# Run regression tests (compares against reference JSON tables)
./build/debug/bin/BasicStrategyRegressionTest
./build/debug/bin/QLearningRegressionTest
```

## Code Architecture

### Component Organization

The codebase follows a layered architecture with clear separation of concerns:

**src/Shoe/** - Card deck management
- `Deck`: Represents a single 52-card deck
- `Shoe`: Multi-deck shoe with penetration settings and shuffling

**src/Game/** - Core game mechanics
- `Hand`: Manages cards, calculates values (soft/hard), handles splits/blackjack detection
- `Player`: Contains strategy and bankroll management
- `Slot`: Manages multiple hands for a single player (after splits)
- `Table` (abstract): Template method pattern for game flow
- `BlackjackTable`: Concrete implementation with full blackjack rules
- `Rules` / `BlackjackRules`: Configurable game rules (dealer behavior, splits, surrender, etc.)

**src/RL/** - Reinforcement learning strategies
- `Strategy` (abstract): Base interface for decision-making agents
- `RandomStrategy`: Baseline random player
- `BasicStrategy`: Lookup-table based optimal play (can be loaded from JSON)
- `QLearningStrategy`: Q-learning agent with per-state learning parameters
- `DecayingParameter`: Abstract decay strategies (epsilon/alpha decay)
- `State`: Encapsulates game state (player hand, dealer card, allowed actions, count)
- `Action`: Enum for player actions (HIT, STAND, DOUBLE_DOWN, SPLIT, SURRENDER)

**src/Utils/** - Utility functions

### Key Design Patterns

1. **Strategy Pattern**: All player strategies inherit from `Strategy` base class
   - Enables swapping AI algorithms without changing game logic
   - `getAction(State)` is the core interface
   - Learning strategies override `updateTable()` for training

2. **Template Method Pattern**: `Table` defines game flow skeleton
   - `round()` orchestrates: collectBets → dealInitialCards → playersPlay → dealerPlays → evaluate
   - Subclasses (`BlackjackTable`) implement specific game logic

3. **Polymorphic Cloning**: Strategies use `clone()` for deep copying
   - Essential for parallel simulations
   - Returns `unique_ptr<Strategy>` to maintain polymorphism

4. **Per-State Learning Parameters**: `QLearningStrategy` uses per-state alpha/epsilon
   - Each state has independent learning rate and exploration decay
   - Enables faster convergence for frequently-visited states

### State Representation

The Q-learning agent uses a simplified state key:
- `StateKey = (count, handType, playerSum, dealerCard)`
- `handType`: HARD, SOFT, or PAIR
- For pairs, uses single card value (not total) to distinguish AA from 22

### Object Library Pattern

The project compiles all production code once as `blackjack_objs` (object library), then links it into the apps and regression executables. This avoids recompiling source files for each target.

## Testing Strategy

`regression/` contains standalone executables (no GoogleTest) that compare
simulation results and generated strategy tables against reference data:

  - `BasicStrategyRegressionTest`: Validates optimal basic strategy tables
  - `QLearningRegressionTest`: Checks Q-learning convergence
  - `DoubleDownMadnessEdgeRegressionTest`, `DoubleDownMadnessQLearningRegressionTest`: same, for DDM

## Important Implementation Details

### Hand Value Calculation
- Aces are dynamically valued (11 or 1) to maximize hand value without busting
- `getHandType()` returns HARD, SOFT, or PAIR based on ace usage and card composition
- Pairs are only recognized with exactly 2 cards of matching rank (ten-value cards DO pair: K+Q is a pair)

### Splitting Logic
- `Slot` manages multiple hands after splits
- Each split hand tracks `isSplit` flag
- Rules control: max splits, resplit aces, hit split aces, double after split

### Strategy Fallbacks
- `BasicStrategy` uses `ActionWithFallback` structure
- Primary action + fallback for when primary isn't allowed
- Example: DOUBLE_DOWN falls back to HIT if doubling not available

### Q-Learning Specifics
- Uses epsilon-greedy exploration with per-state epsilon decay
- Gamma (discount factor) is typically 1.0 (no discounting)
- SPLIT action special case: expected value doubles (two hands played)
- Can convert Q-table to BasicStrategy via `toBasicStrategy()`

### Basic Strategy Tables
- Stored as JSON files under `basic_strategy_tables/`, one subfolder per game
  so neither game's tables sit in the shared root: `basic_strategy_tables/blackjack/`
  and `basic_strategy_tables/double_down_madness/`
- Blackjack filename encodes rules: `decks=N_ss17=BOOL_das=BOOL_surr=X_peek=BOOL.json`
- DDM has a base `basic_strategy.json` plus a `version_<N>.json` override merged on top
- Used for regression testing and as reference strategies

## Debug Mode and Reproducibility

The project supports fully deterministic simulations for debugging and cross-validation with Python.

### Quick Debug Mode Usage

```cpp
// Option 1: Use DebugShoe (deterministic LCG)
DebugShoe shoe(1, 75.0, 42);  // seed=42

// Option 2: Seed regular components
Shoe shoe(6, 75.0, 42);  // MT19937 with seed
auto strategy = std::make_unique<RandomStrategy>(42);
auto qlearning = std::make_unique<QLearningStrategy>(
    std::make_unique<EpsilonDecayingParameter>(0.1, 0.01, 0.9999),
    std::make_unique<LinearDecayingParameter>(0.5, 0.1, 1000),
    1.0,  // gamma
    42    // seed
);
```

### Running Reproducibility Test

The ReproducibilityTest is a **standalone executable** that demonstrates deterministic Q-learning training using DebugShoe and explicit seeds. It runs single-threaded for full reproducibility.

```bash
# Build and run the reproducibility test
cmake --build build/debug
./build/debug/bin/ReproducibilityTest

# Run multiple times - results should be identical each time
./build/debug/bin/ReproducibilityTest > run1.txt
./build/debug/bin/ReproducibilityTest > run2.txt
diff run1.txt run2.txt  # Should show no differences
```

### Python Cross-Validation

The `DebugShoe` uses a Linear Congruential Generator that can be exactly replicated in Python:

```python
# See debug_shoe_python_example.py for full implementation
from debug_shoe_python_example import DebugShoe

shoe = DebugShoe(num_decks=1, penetration=75.0, seed=42)
cards = [shoe.deal_card() for _ in range(20)]
```

See [DEBUG_MODE.md](DEBUG_MODE.md) for comprehensive guide on debug mode and Python alignment.

## Current Alternating-Optimization Research

The app surface is exactly two executables, each exactly one file:
`apps/AlternatingOptimization.cpp` and `apps/CompareCountStrategies.cpp`.
Each holds a `template <typename Game>` engine, its two instantiations
(`Game = BlackjackGame` / `DoubleDownMadnessGame`), and `main()`. Nothing
else includes those engines, so there's no header/source split for them.
Game-specific behavior (Rules construction, CLI rule flags, meta.json
fields, checkpoint-root naming, basic-strategy loading, whether Illustrious
18 applies) lives in the shared `apps/GameTraits.h`. `apps/GameAppDispatcher.h`
is a header-only helper (also shared, since it's identical for both apps)
that does the `--game`/`--table-type` argv routing and filters out
irrelevant game-specific flags before calling into the right instantiation.
The older standalone `FindDeviations`, `FindOptimalCount`, and `MeasureEdge`
apps are gone — superseded by `AlternatingOptimization`'s loop and by
`CompareCountStrategies` reading its checkpoints directly.

`AlternatingOptimization` is the unified CLI for classic blackjack and Double
Down Madness:

```bash
./build/bin/AlternatingOptimization --game blackjack [game/options]
./build/bin/AlternatingOptimization --game ddm --version 1 [game/options]
```

It alternates `W0 -> P0 -> W1 -> P1 -> ...`, saves resumable policy/count
artifacts, and generates EV/count, histogram, conditional second-moment, and
Kelly-multiplier graphs. Count objective and constraints are independent:

- Objectives: `--count-classical-ols`, `--count-quadratic-kelly`.
- Constraints: `--count-unconstrained`, `--count-sum-zero`,
  `--count-sum-zero-fixed-b1`, `--count-sum-zero-fixed-p0-edge`.
- Default: classical OLS plus `--count-sum-zero-fixed-b1`.
- Legacy `--count-quadratic-kelly-zero-bias` alias remains accepted (both games).

`CompareCountStrategies --deviations-checkpoint <dir> --deviations-agent Pk`
reads full deviations straight from an `AlternatingOptimization` checkpoint
folder for the same game (`Pk_agent.json` plus the matching `Wk.json`/`W0`
count) — this is now the only supported checkpoint format for both games.

`CompareCountStrategies` also owns its own, separate checkpoint format for
whatever count it trained full deviations against fresh (Hi-Lo by default,
or any `--count`/`--count-weights`) — every fresh-training run auto-saves
`full_deviations_strategy.json` + `full_deviations_meta.json` (the greedy
`BasicStrategy` plus its count/game config) into that run's own output
folder. `--strategy-checkpoint <dir>` reloads one instead of retraining
(`<dir>` is a folder name under `checkpoints/CompareCountStrategies/` /
`checkpoints/DoubleDownMadnessCompareCountStrategies/`, or an absolute path
to any prior run's output folder). Mutually exclusive with
`--deviations-checkpoint`.

Both objectives use streamed normal-equation statistics. OLS has
`A=E[cc^T]`; quadratic Kelly has `A=E[X^2cc^T]`; both use `d=E[Xc]`. Sum-zero
and fixed-bias modes solve KKT systems directly from `A,d`, without retaining
round histories. New metadata stores `count_regression_objective` and
`count_regression_constraint` separately while retaining legacy checkpoint
loading.

Known limitation: DDM allows repeated doubling, so exposure becomes `2^d` times
the initial Kelly wager. The quadratic approximation does not enforce
`1+fX>0`; rare ruin produces `-infinity` log bankroll and zero growth for the
whole parallel experiment. Large DDM Kelly standard deviations are usually a
mixture of near-one growth and ruined zero-growth trials. Do not interpret them
as ordinary estimator noise.

The user prefers regression tests over low-value unit tests for simulation
accuracy and never wants more than one 10-thread research simulation running at
once because concurrent runs overheat the machine.

## Branch Strategy

- `master`: Main branch for stable code
- `Q-learning`: Current development branch for Q-learning features
- Create PRs targeting `master` when features are complete
