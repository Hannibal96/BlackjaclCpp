# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Configure (one-time)
cmake -B build/debug  -DCMAKE_BUILD_TYPE=Debug
cmake -B build/release -DCMAKE_BUILD_TYPE=Release

# Build everything
cmake --build build/debug
cmake --build build/release

# Run all tests
ctest --test-dir build/debug

# Run a single test binary directly
./build/debug/bin/QLearningRegressionTest --num-threads 1 --num-rounds 1000000
./build/debug/bin/QLearningDebugRegressionTest --seed 42 --num-rounds 1000000
./build/debug/bin/ShoeTest
```

All production code is compiled once as the **`blackjack_objs` object library** (no `main`). Every test executable links against it. When adding a new `.cpp` source file, add it to `blackjack_objs` in [CMakeLists.txt](CMakeLists.txt) and register the test in [unittest/regression/CMakeLists.txt](unittest/regression/CMakeLists.txt).

Test executables land in `build/<type>/bin/`. Unit tests use Google Test (fetched automatically). Regression tests are standalone binaries with their own `main` and CLI argument parsing.

## Architecture

### Layer stack

```
Utils/GlobalRNG          — cross-reproducible LCG RNG (matches Python SeededRNG)
Shoe / DebugShoe         — multi-deck card source; DebugShoe is deterministic
Game/Hand, Slot          — cards, bets, split tracking per hand / betting slot
Game/Player              — money + Strategy (polymorphic)
Game/Table (abstract)    — template-method game loop
Game/BlackjackTable      — concrete rules, round orchestration
RL/Strategy (abstract)   — getAction / updateTable / clone / averaging operators
RL/QLearningStrategy     — Q-table, per-state α/ε decay, toBasicStrategy()
```

### Game loop (`BlackjackTable`)

`round()` calls in order: `collectBets → dealInitialCards → playersPlay → dealerPlays → evaluate`.

`playersPlay()` issues Q-updates immediately for HIT (reward=0, real nextState) and for busts/surrenders (reward=±bet, empty nextState). Non-bust STAND and DOUBLE_DOWN hands are stored in `aliveHandIndices` and updated later in `evaluate()` with the final reward.

This means the nextState passed to `updateTable` for terminal hands always has **empty allowedActions**, so `getMaxQValue` returns 0 for those.

### Q-learning key types

```cpp
StateKey  = tuple<int count, HandType, int playerSum, int dealerCard>
QTableKey = pair<StateKey, Action>
```

`HandType` is `HARD / SOFT / PAIR / ZOMBIE`. PAIR keys use the single card value (not the sum). `stateToKey` calls `hand.getHandType(pairAllowed=true)`.

Per-state alpha and epsilon have their own `DecayingParameter` maps. Both decay on every Q-update call (not just on exploration steps).

SPLIT updates: `maxNextQ *= 2.0` before the Bellman update.

### Multi-threaded averaging

`runParallelSimulation` (in `Utils.cpp`) clones one player per thread, runs async, then merges:

1. Clone thread-0 result into `resultPlayers` (bypasses `operator+=`, so `qTableCount` for thread 0 is not incremented).
2. `operator+=` threads 1…N-1 (increments `qTableCount` per entry).
3. `operator*=` divides each entry by its `qTableCount` (number of threads that contributed via `+=`).

This means thread-0 data is always included but never counted — for well-explored states (all threads visit them) the effective divisor is N-1, not N.

### Cross-reproducible RNG

`GlobalRNG` (header-only, `src/Utils/GlobalRNG.h`) and Python's `SeededRNG` use the identical LCG:

```
state = (state * 1664525 + 1013904223) mod 2^32
nextInt(n)  → return state % n, then advance
nextFloat() → return state / 2^32, then advance
```

`DebugShoe` can take a `GlobalRNG*`; when non-null, every `dealCard()` draws the index from that shared RNG instead of its own internal state, giving a single unified randomness stream shared with the Q-learning agent.

### Basic strategy JSON tables

Stored in `basic_strategy_tables/`. Keyed by `count → handType → playerSum → dealerCard → action string`. Loaded by `BasicStrategy`. `QLearningStrategy::toBasicStrategy()` converts a learned Q-table into the same structure.
