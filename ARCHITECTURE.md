# BlackjaclCpp — Code Architecture Reference

## Directory Layout

```
BlackjaclCpp/
├── src/
│   ├── Shoe/          # Card / deck / shoe primitives
│   ├── Game/          # Game mechanics and table logic
│   ├── RL/            # Strategies and reinforcement learning
│   └── Utils/         # Shared utilities
├── unittest/
│   ├── regression/    # Regression tests vs. reference JSON tables
│   └── *.cpp          # Unit / integration / benchmark tests
├── basic_strategy_tables/   # Reference JSON strategy tables
├── hpo/               # Python hyperparameter optimisation scripts
├── CMakeLists.txt
└── unittest/CMakeLists.txt
```

---

## Class / Type Hierarchy

### src/Shoe/

| Type | Kind | Inherits | File |
|------|------|----------|------|
| `Rank` | `enum class` | — | [Deck.h](src/Shoe/Deck.h) |
| `Suit` | `enum class` | — | [Deck.h](src/Shoe/Deck.h) |
| `Card` | `struct` | — | [Deck.h](src/Shoe/Deck.h) |
| `Deck` | `class` | — | [Deck.h](src/Shoe/Deck.h) |
| `Shoe` | `class` | — | [Shoe.h](src/Shoe/Shoe.h) |
| `DebugShoe` | `class` | `: public Shoe` | [Shoe.h](src/Shoe/Shoe.h) |

`DebugShoe` uses a Linear Congruential Generator for fully deterministic card dealing (reproducible across C++ and Python).

### src/Game/

| Type | Kind | Inherits | File |
|------|------|----------|------|
| `Rules` | `struct` | — | [Rules.h](src/Game/Rules.h) |
| `BlackjackRules` | `struct` | `: public Rules` | [BlackjackRules.h](src/Game/BlackjackRules.h) |
| `DoubleDownOn` | `enum class` | — | [BlackjackRules.h](src/Game/BlackjackRules.h) |
| `Surrender` | `enum class` | — | [BlackjackRules.h](src/Game/BlackjackRules.h) |
| `HandType` | `enum class` | — | [Hand.h](src/Game/Hand.h) |
| `Hand` | `class` | — | [Hand.h](src/Game/Hand.h) |
| `Player` | `class` | — | [Player.h](src/Game/Player.h) |
| `Slot` | `class` | — | [Slot.h](src/Game/Slot.h) |
| `Table` | `abstract class` | — | [Table.h](src/Game/Table.h) |
| `BlackjackTable` | `class` | `: public Table` | [BlackjackTable.h](src/Game/BlackjackTable.h) |

**`Table` (Template Method pattern)** — defines the round skeleton:
```
round() → collectBets → dealInitialCards → playersPlay → dealerPlays → evaluate
```
All five steps are pure virtual; `BlackjackTable` is the only concrete implementation.

**`Slot`** — holds one or more `Hand` objects for a single player seat (created by splitting).

### src/RL/

| Type | Kind | Inherits | File |
|------|------|----------|------|
| `Action` | `enum class` | — | [Action.h](src/RL/Action.h) |
| `State` | `struct` | — | [State.h](src/RL/State.h) |
| `Strategy` | `abstract class` | — | [Strategy.h](src/RL/Strategy.h) |
| `RandomStrategy` | `class` | `: public Strategy` | [RandomStrategy.h](src/RL/RandomStrategy.h) |
| `BasicStrategy` | `class` | `: public Strategy` | [BasicStrategy.h](src/RL/BasicStrategy.h) |
| `QLearningStrategy` | `class` | `: public Strategy` | [QLearningStrategy.h](src/RL/QLearningStrategy.h) |
| `ActionWithFallback` | `struct` | — | [BasicStrategy.h](src/RL/BasicStrategy.h) |
| `DecayingParameter` | `abstract class` | — | [DecayingParameter.h](src/RL/DecayingParameter.h) |
| `EpsilonDecayingParameter` | `class` | `: public DecayingParameter` | [DecayingParameter.h](src/RL/DecayingParameter.h) |
| `LinearDecayingParameter` | `class` | `: public DecayingParameter` | [DecayingParameter.h](src/RL/DecayingParameter.h) |
| `ExplorationMode` | `enum class` | — | [QLearningStrategy.h](src/RL/QLearningStrategy.h) |

**`Strategy` (Strategy pattern)** — core interface:
- `getAction(State) → Action` — decision
- `updateTable(state, action, reward, nextState)` — learning (no-op by default)
- `clone() → unique_ptr<Strategy>` — deep copy for parallel simulations
- `operator+=` / `operator*=` — merge / scale Q-tables

**`QLearningStrategy` key types:**
```cpp
StateKey  = tuple<int, HandType, unsigned int, unsigned int>
            // (count, hand_type, player_sum, dealer_card)
QTableKey = pair<StateKey, Action>
```
Per-state alpha and epsilon are stored independently for faster convergence on frequently-visited states.

---

## Inheritance Trees

```
Shoe
└── DebugShoe

Rules
└── BlackjackRules

Table  (abstract)
└── BlackjackTable

Strategy  (abstract)
├── RandomStrategy
├── BasicStrategy
└── QLearningStrategy

DecayingParameter  (abstract)
├── EpsilonDecayingParameter
└── LinearDecayingParameter
```

---

## Test Executables

| Binary | Source | Type |
|--------|--------|------|
| `ShoeTest` | unittest/ShoeTest.cpp | Unit |
| `HandTest` | unittest/HandTest.cpp | Unit |
| `BlackjackTableTest` | unittest/BlackjackTableTest.cpp | Integration |
| `BasicStrategyTest` | unittest/BasicStrategyTest.cpp | Strategy |
| `DebugShoeTest` | unittest/DebugShoeTest.cpp | Determinism |
| `BasicStrategyRegressionTest` | unittest/regression/BasicStrategyRegressionTest.cpp | Regression |
| `QLearningRegressionTest` | unittest/regression/QLearningRegressionTest.cpp | Regression |
| `PerformanceBenchmark` | unittest/PerformanceBenchmark.cpp | Benchmark |
| `BlackjackBenchmark` | unittest/BlackjackBenchmark.cpp | Benchmark |

Regression tests compare generated strategy tables against the JSON files in `basic_strategy_tables/`.  
JSON filename convention: `decks=N_ss17=BOOL_das=BOOL_surr=TYPE_peek=BOOL.json`

---

## Build Targets

All production source is compiled once as `blackjack_objs` (CMake object library) and linked into every test binary — avoids redundant compilation.

```bash
cmake -B build/debug  -DCMAKE_BUILD_TYPE=Debug
cmake -B build/release -DCMAKE_BUILD_TYPE=Release
cmake --build build/debug
ctest --test-dir build/debug
```

---

## Key Data-Flow (one round)

```
Shoe  ──dealCard()──►  BlackjackTable
                              │
                    ┌─────────▼──────────┐
                    │   Table::round()   │
                    │  collectBets       │
                    │  dealInitialCards  │
                    │  playersPlay  ─────┼──► Player::getAction(State)
                    │                   │       └─► Strategy::getAction(State)
                    │  dealerPlays      │
                    │  evaluate         │
                    └─────────┬──────────┘
                              │ reward
                              ▼
                  Strategy::updateTable()   ◄── Q-learning training
```
