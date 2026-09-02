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
├── apps/              # Training and authoritative fixed count/policy evaluator
├── scripts/           # Comparison and quantization orchestration
├── regression/        # Regression tests: sim output vs. reference JSON tables
├── basic_strategy_tables/
│   ├── blackjack/
│   └── double_down_madness/
├── checkpoints/
│   ├── alternating-checkpoints/
│   ├── double-down-madness-alternating-checkpoints/
│   ├── CompareCountStrategies/
│   ├── DoubleDownMadnessCompareCountStrategies/
│   └── QuantizationEffect/
└── CMakeLists.txt
```

## Main App Targets

The primary binaries use the shared `template <typename Game>` shape and game
traits (`apps/GameTraits.h`):

| App | Purpose | Output Root |
|---|---|---|
| `AlternatingOptimization` | Blackjack/DDM RL and count-regression pipeline selected by `--game` | Game-specific alternating checkpoint root |
| `EvaluateCountPolicy` | One fixed count + one fixed policy; blackjack enabled first | `checkpoints/EvaluateCountPolicy/` |

`scripts/compare_count_strategies.py` and `scripts/quantization_effect.py`
orchestrate repeated `EvaluateCountPolicy` calls. This keeps spread/Kelly and
wager-distribution semantics in one implementation.

The older standalone `FindDeviations`, `FindOptimalCount`, and `MeasureEdge`
apps have been removed; `AlternatingOptimization` covers their training
workflows and `EvaluateCountPolicy` reads its `Wk`/`Pk` artifacts directly.

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

- the selected true count or running count is computed, including constant and
  per-deck initial-count offsets
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
└── record round outcome, gross wagers, regression, and graph bins
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

### Policy-comparison artifacts

- `run.log`
- one child folder per fixed policy
- combined `results.json/csv`
- each child's `kelly_growth.json/svg` and
  `kelly_exposure_at_1.json/csv/svg`

The new shared evaluator is blackjack-only at present. Its `--game` dispatch
boundary is ready for additional games once their table implementation is
available on the relevant branch.

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

During every Kelly measurement, `BlackjackTable` also streams two distributions:
gross exposure `total wagers / pre-round bankroll`, and the exact realized
absolute return `|round profit / pre-round bankroll| = |fX|`. Total wagers add
the initial bet plus every split and double wager; surrender does not erase
money that was originally at risk. The same accumulator compares exact
`log(1+r)` with `r-r^2/2`, and thread/measurement merges preserve absolute
histogram counts. Spread simulations deliberately do not collect these metrics.

Kelly players also carry a configurable maximum total-wager fraction. Tables
cap the initial wager and remove split/double actions when cumulative gross
wagers would exceed that fraction of the round-start bankroll; settled or
released wagers still count. The default is `1.0`, exposed by
`--max-total-wager-fraction`. Exposure SVGs plot the 0.1%-bankroll histograms
side by side against a base-10 logarithmic probability axis with percentage
ticks, labeled axes, and grid lines.

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
- [src/Game/Player.cpp](src/Game/Player.cpp)
- [src/Game/BettingStrategy.h](src/Game/BettingStrategy.h)
- [src/RL/BasicStrategy.h](src/RL/BasicStrategy.h)
- [src/RL/QLearningStrategy.h](src/RL/QLearningStrategy.h)
- [apps/GameTraits.h](apps/GameTraits.h) — everything that differs between blackjack and DDM
- [apps/AlternatingOptimization.cpp](apps/AlternatingOptimization.cpp) — engine + both instantiations + main()
- [src/Utils/CountPolicyEvaluation.h](src/Utils/CountPolicyEvaluation.h) — shared spread/Kelly evaluator
- [apps/EvaluateCountPolicy.cpp](apps/EvaluateCountPolicy.cpp) — count/policy loading, full-deviation training, CLI, artifacts
- [scripts/compare_count_strategies.py](scripts/compare_count_strategies.py)
- [scripts/quantization_effect.py](scripts/quantization_effect.py)
