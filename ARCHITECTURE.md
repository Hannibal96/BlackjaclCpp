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
├── apps/              # AlternatingOptimization + CompareCountStrategies (all games)
├── regression/        # Regression tests: sim output vs. reference JSON tables
├── basic_strategy_tables/
│   ├── blackjack/
│   ├── double_down_madness/
│   └── spanish21/
├── checkpoints/
│   ├── alternating-checkpoints/
│   ├── double-down-madness-alternating-checkpoints/
│   ├── spanish21-alternating-checkpoints/
│   ├── CompareCountStrategies/
│   ├── DoubleDownMadnessCompareCountStrategies/
│   └── Spanish21CompareCountStrategies/
└── CMakeLists.txt
```

## Main App Targets

There are exactly two app binaries, each a `template <typename Game>` engine
instantiated over `BlackjackGame`/`DoubleDownMadnessGame`/`Spanish21Game`
(`apps/GameTraits.h`):

| App | Purpose | Output Root |
|---|---|---|
| `AlternatingOptimization` | RL and count-regression pipeline selected by `--game` | Game-specific alternating checkpoint root |
| `CompareCountStrategies` | Strategy comparison selected by `--game` | Game-specific comparison checkpoint root |

The older standalone `FindDeviations`, `FindOptimalCount`, and `MeasureEdge`
apps have been removed; `AlternatingOptimization` covers their training
workflows and `CompareCountStrategies` reads its checkpoints directly via
`--deviations-checkpoint`.

## Class / Type Hierarchy

### `src/Shoe`

| Type | Kind | Inherits |
|---|---|---|
| `Card` | `struct` | — |
| `Deck` | `class` | — |
| `Shoe` | `class` | — |
| `DebugShoe` | `class` | `Shoe` |
| `SpanishShoe` | `class` | `Shoe` |

### `src/Game`

| Type | Kind | Inherits |
|---|---|---|
| `Rules` | `struct` | — |
| `BlackjackRules` | `struct` | `Rules` |
| `DoubleDownMadnessRules` | `struct` | `Rules` |
| `SpanishRules` | `struct` | `Rules` |
| `Hand` | `class` | — |
| `Slot` | `class` | — |
| `Player` | `class` | — |
| `Table` | `abstract class` | — |
| `BlackjackTable` | `class` | `Table` |
| `DoubleDownMadnessTable` | `class` | `Table` |
| `SpanishTable` | `class` | `Table` |

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
StateKey = (count, handType, playerSum, dealerCard, cardCount)
QTableKey = (StateKey, action)
```

`cardCount` (the hand's card count, capped at 6) is a Spanish-21-only
dimension: Blackjack and DDM players never set `Player::trackHandCardCount`,
so it's always the constant `2` for them — their tables and behavior are
unaffected. `HandType` includes `AFTER_DOUBLE`/`AFTER_DOUBLE_SOFT` (Spanish
21's redouble/rescue decision node, split by hard vs. soft since WoO's chart
treats them differently for the same numeric total).

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

`SpanishTable::round()` follows the same skeleton, structurally closest to
`BlackjackTable` (uses `Slot` for splits, unlike DDM), built on a 48-card
no-tens `SpanishShoe`. Its `evaluate()` differs from Blackjack's: player
blackjack always beats a dealer blackjack (pays 3:2, not a push), player 21 of
any card count always wins (no push, even vs. a dealer 21), and card-count/
suited-composition bonus tiers (5/6/7+-card 21, 6-7-8/7-7-7) replace the plain
1:1 payout when they apply. Doubling is legal on any card count (not just the
first two) as long as the hand hasn't already doubled; once doubled, the hand
enters the `AFTER_DOUBLE[_SOFT]` decision node (stand, redouble up to
`maxRedoubles`, or rescue/surrender).

DDM Q-values are normalized to the wager entering each state. A hit carries
`V(next)` forward, while a double that continues carries `2 * V(next)`. This is
required because the strategy state intentionally excludes bankroll and wager
size, while DDM permits repeated doubling. `SpanishTable` reuses the same
`nextValueMultiplier` mechanism for redoubling.

## Learning / Analysis Data

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
the paytable version in addition to decks, H17/S17, and penetration. Spanish 21
writes the same set beneath `checkpoints/spanish21-alternating-checkpoints/`,
recording `redouble`/`ddr` in addition to decks and H17/S17.

### Compare artifacts

- `run.log`
- `ev_count_graph.json/svg`
- `count_histograms.json/svg`
- `second_moment_count_graph.json/svg` overlaying all compared policies
- `kelly_fraction_graph.json/svg` overlaying all compared policies
- `full_deviations_strategy.json` + `full_deviations_meta.json`, written
  whenever full deviations were trained fresh this run (skipped when loaded
  from either `--deviations-checkpoint` or `--strategy-checkpoint`)

The DDM and Spanish 21 comparison apps write the same artifact families for
known basic strategy and full deviations, beneath
`DoubleDownMadnessCompareCountStrategies/` and `Spanish21CompareCountStrategies/`
respectively. A loaded alternating policy `Pk` is paired with `Wk` by default
so both policies use the same count and betting model; a loaded
`--strategy-checkpoint` is paired with its own saved count the same way.

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

Alternating optimization separates the count objective from its constraints.
Classical OLS streams `A=E[cc^T]`, `d=E[Xc]`; quadratic Kelly streams
`A=E[X^2cc^T]`, `d=E[Xc]`. Both minimize the quadratic form
`0.5 theta^T A theta-d^T theta`, with `theta=[w,b]`, so they share the same
streaming-compatible constraint solvers:

- Unconstrained: `A theta=d`.
- Sum-zero: solve `[A q; q^T 0][theta;lambda]=[d;0]`, where
  `q=[1,...,1,0]` excludes the bias.
- Sum-zero with fixed bias `b0`: solve
  `[Aww 1; 1^T 0][w;lambda]=[dw-Awb*b0;0]`.
- Fixed `b1`: `W1` estimates a free bias and `W2+` reuse it.
- Fixed `P0` edge: `W1+` use the measured flat edge of `P0`.

The objective flags are `--count-classical-ols` and
`--count-quadratic-kelly`. Constraint flags are `--count-unconstrained`,
`--count-sum-zero`, `--count-sum-zero-fixed-b1`, and
`--count-sum-zero-fixed-p0-edge`. Checkpoints store
`count_regression_objective` and `count_regression_constraint` separately and
still load legacy combined-mode metadata.

For quadratic Kelly, pre-round normalized removed-card features `c` and
unit-wager round profit `X` produce the second-order approximation
`log(1+theta^Tc X) ~= theta^Tc X - 0.5(theta^Tc X)^2`. Learned rank weights
are scaled so the average 10/J/Q/K tag is `-1`, with the betting factor scaled
inversely to preserve `w^Tc` exactly, while policy lookup remains discretized. Bet sizing uses the continuous
signal and clamps negative fractions to zero. Because the signal is already a
quadratic-Kelly fraction, its empirical multiplier sweep is centered at `1.0`.

DDM's repeated doubling can multiply the initial wager by `2^d`. The current
quadratic approximation has no admissibility constraint ensuring `1+fX>0`, so
rare long double chains can ruin a worker. Parallel log aggregation propagates
that worker's `-infinity`, yielding a zero-growth experiment and a characteristic
near-Bernoulli standard deviation. This remains an open research issue.

## Important Architectural Notes

- Count logic lives in `Player`, not in the strategy classes.
- Learned Q policies are converted to greedy `BasicStrategy` tables before evaluation apps use them.
- Regression and EV-count graph collection use pre-round shoe state with round outcome as the target.
- Quadratic-Kelly matrices are tagged separately in `W*_data.json`; their EV/count graphs omit the OLS regression line because the fitted line represents wager fraction, not expectancy.
- Round-return variance includes the complete round result, including splits, doubles, surrender, blackjack payout, and the selected wager size.
- Parallel simulations merge players and strategies through averaging operators.
- `RunLogger` mirrors terminal output into the same run folder used by checkpoints or graph artifacts.

## Files To Read First

- [src/Game/BlackjackTable.cpp](src/Game/BlackjackTable.cpp)
- [src/Game/DoubleDownMadnessTable.cpp](src/Game/DoubleDownMadnessTable.cpp)
- [src/Game/SpanishTable.cpp](src/Game/SpanishTable.cpp)
- [src/Game/Player.cpp](src/Game/Player.cpp)
- [src/Game/BettingStrategy.h](src/Game/BettingStrategy.h)
- [src/RL/BasicStrategy.h](src/RL/BasicStrategy.h)
- [src/RL/QLearningStrategy.h](src/RL/QLearningStrategy.h)
- [apps/GameTraits.h](apps/GameTraits.h) — everything that differs between blackjack, DDM, and Spanish 21
- [apps/AlternatingOptimization.cpp](apps/AlternatingOptimization.cpp) — engine + all three instantiations + main()
- [apps/CompareCountStrategies.cpp](apps/CompareCountStrategies.cpp) — engine + all three instantiations + main()
