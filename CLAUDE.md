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
./build/debug/bin/Spanish21EdgeRegressionTest
./build/debug/bin/Spanish21QLearningRegressionTest
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
- `StateKey = (count, handType, playerSum, dealerCard, cardCount)`
- `handType`: HARD, SOFT, PAIR, ZOMBIE, BLACKJACK, or AFTER_DOUBLE
- For pairs, uses single card value (not total) to distinguish AA from 22
- `cardCount` (added for Spanish 21): the hand's card count, capped at 6. Blackjack
  and DDM players never set `Player::trackHandCardCount`, so this field is always
  the constant `2` for them — their behavior and JSON tables are unaffected. Spanish
  21 sets it `true` since its 5/6/7-card 21 bonus payouts make card count strategically
  relevant. `AFTER_DOUBLE` (also Spanish-21-specific) is a `HandType` a hand takes on
  once doubled, overriding HARD/SOFT/PAIR regardless of total — the redouble/rescue
  decision node.
- `BasicStrategy`'s lookup table has a matching extra nesting level:
  `count -> handType -> playerSum -> dealerCard -> cardCount -> action`. Every
  existing `basic_strategy_tables/blackjack/*.json` and `double_down_madness/*.json`
  file was migrated to wrap its leaf actions as `{"2": "<action>"}` to match.

### Object Library Pattern

The project compiles all production code once as `blackjack_objs` (object library), then links it into the apps and regression executables. This avoids recompiling source files for each target.

## Testing Strategy

`regression/` contains standalone executables (no GoogleTest) that compare
simulation results and generated strategy tables against reference data:

  - `BasicStrategyRegressionTest`: Validates optimal basic strategy tables
  - `QLearningRegressionTest`: Checks Q-learning convergence
  - `DoubleDownMadnessEdgeRegressionTest`, `DoubleDownMadnessQLearningRegressionTest`: same, for DDM
  - `Spanish21EdgeRegressionTest`, `Spanish21QLearningRegressionTest`: same, for Spanish 21

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
  so neither game's tables sit in the shared root: `basic_strategy_tables/blackjack/`,
  `basic_strategy_tables/double_down_madness/`, and `basic_strategy_tables/spanish21/`
- Blackjack filename encodes rules: `decks=N_ss17=BOOL_das=BOOL_surr=X_peek=BOOL.json`
- DDM has a base `basic_strategy.json` plus a `version_<N>.json` override merged on top
- Spanish 21 has `s17.json`/`h17.json` (base, no redouble) and
  `h17_redouble_first.json`/`h17_redouble_after.json` (H17 + redouble, merged on top —
  WoO's only published pre-double AND after-double charts for this combination).
  Double-down rescue (DDR) is a standard WoO rule (defaults on — see
  `SpanishRules`), not a separate variant, but WoO never published a
  dedicated after-double chart for "DDR without redouble" — only for
  H17+redouble — so `h17_redouble_after.json` is reused as the best available
  AFTER_DOUBLE reference whenever redouble or DDR makes that decision node
  live (`loadSpanishBasicStrategy()`). S17 + redouble has no *pre*-double
  chart at all, so it falls back to the plain S17 base table there. All
  four were hand-transcribed from wizardofodds.com's strategy chart images and
  deliberately drop the images' 6-7-8/Super-Bonus-reachability footnote markers
  (documented v1 simplification — costs a small, bounded amount of EV, reflected in
  `Spanish21EdgeRegressionTest`'s wider tolerance).
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
Each holds a `template <typename Game>` engine, its three instantiations
(`Game = BlackjackGame` / `DoubleDownMadnessGame` / `Spanish21Game`), and
`main()`. Nothing else includes those engines, so there's no header/source
split for them. Game-specific behavior (Rules construction, CLI rule flags,
meta.json fields, checkpoint-root naming, basic-strategy loading, whether
Illustrious 18 applies) lives in the shared `apps/GameTraits.h`.
`apps/GameAppDispatcher.h` is a header-only helper (also shared, since it's
identical across apps) that does the `--game`/`--table-type` argv routing and
filters out irrelevant game-specific flags before calling into the right
instantiation. The older standalone `FindDeviations`, `FindOptimalCount`, and
`MeasureEdge` apps are gone — superseded by `AlternatingOptimization`'s loop
and by `CompareCountStrategies` reading its checkpoints directly.

`AlternatingOptimization` is the unified CLI for classic blackjack, Double
Down Madness, and Spanish 21:

```bash
./build/bin/AlternatingOptimization --game blackjack [game/options]
./build/bin/AlternatingOptimization --game ddm --version 1 [game/options]
./build/bin/AlternatingOptimization --game spanish21 --ss17 true [game/options]
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

### Spanish 21

Modeled after wizardofodds.com/games/spanish-21/: 48-card shoe (no rank-TEN
cards; J/Q/K remain and still count as 10 — `SpanishShoe`/`Deck(spanish=true)`),
player blackjack always beats a dealer blackjack (pays 3:2, not a push), player
21 of any card count always wins (no push, even vs. a dealer 21), 5/6/7+-card 21
pays 3:2/2:1/3:1, and the 6-7-8/7-7-7 suited bonus pays 3:2 mixed / 2:1 suited /
3:1 spades. `SpanishTable`/`SpanishRules` follow `BlackjackTable`'s structure
(uses `Slot`, unlike DDM). Double is allowed on any card count, not just the
first two, via the `StateKey` `cardCount` field described above.

`SpanishTable::round()` skips the dealer's play entirely once every alive
hand already has an unbeatable outcome (blackjack or a plain 21) — unlike
`BlackjackTable`, this holds even against a possible dealer blackjack (player
blackjack always beats it here), so there's no push ambiguity left to probe
for and no need for `BlackjackTable`'s peek-style `CHECK_BLACKJACK` dealer
action. Found via `DebugPlayerBehavior`: the dealer was drawing full hands
even when a player had already clinched a 21, burning shoe cards a real
table would never deal and skewing the count seen by later rounds.

Redoubling (`SpanishRules::maxRedoubles`, an `int` — 0 disables it, N allows
up to N redoubles beyond the first ordinary double, e.g. the default "on"
value maxRedoubles=1 caps a hand at 2 total doublings / 4x the original bet)
is fully implemented (bet compounding, `AFTER_DOUBLE` action gating,
Q-learning normalization via `nextValueMultiplier`) and verified: simulated
edge lands within the same tolerance band as S17/H17 against WoO's published
H17+redouble figure (0.42%, verified at both maxRedoubles=1 and 2), asserted
in `Spanish21EdgeRegressionTest` alongside the other two configurations. An
earlier ~13%-simulated-edge finding
turned out to be a data bug, not an engine bug: the transcription script
merged the WoO "already doubled" chart's HARD and SOFT sections into one dict
keyed only by numeric total, and since hard/soft totals share the same range
(13-21), the soft entries silently clobbered the hard ones. Fixed by giving
doubled soft hands their own `HandType::AFTER_DOUBLE_SOFT` (mirrors the
existing HARD/SOFT split) so the two WoO chart sections stay separate
throughout. `maxRedoubles` still defaults to `0` — it's the less commonly
offered table variant, not a known-issue flag.

The user prefers regression tests over low-value unit tests for simulation
accuracy and never wants more than one 10-thread research simulation running at
once because concurrent runs overheat the machine.

## Branch Strategy

- `master`: Main branch for stable code
- `Q-learning`: Current development branch for Q-learning features
- Create PRs targeting `master` when features are complete
