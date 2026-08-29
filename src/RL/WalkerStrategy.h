#pragma once

#include "BasicStrategy.h"
#include "Game/SpanishRules.h"
#include <memory>
#include <ostream>

// Katarina Walker's published Spanish 21 card-counting method (see "The Pro's
// Guide to Spanish 21 and Australian Pontoon"): true-counted Hi-Lo, with the
// running count starting at -4 per deck to correct for the 48-card shoe's
// missing TEN rank (CountingMethods::walker()). This file provides her
// published index deviations, applied on top of the WoO-transcribed base
// Spanish 21 strategy (loadSpanishBasicStrategy()) -- the counting-deviations
// analogue of Illustrious 18 for classic blackjack, but Spanish-21-specific.
//
// Sourced from the confirmed deviation table (rules confirmed against
// wizardofodds.com/games/spanish-21/ and Walker's own published method
// summary): all thresholds use the "true count >= index" convention (same as
// this project's standard Hi-Lo Illustrious 18 table in CompareCountStrategies.cpp);
// when two thresholds for the same hand both hold, the higher (later, in
// ascending order) one wins.

// Builds a BasicStrategy expanded across [minCount, maxCount] from the base
// Spanish 21 strategy for `rules`, with no deviations applied -- i.e. the same
// action at every count. Used for the "no deviations" comparison rows (which
// still differ in bet sizing/count context depending on which CountingSystem
// drives the count key).
std::unique_ptr<BasicStrategy> expandSpanishBasicStrategyAcrossCounts(
    const BasicStrategy& base, int minCount, int maxCount);

// Builds a BasicStrategy expanded across [minCount, maxCount] with Walker's
// published index deviations applied on top of the base Spanish 21 strategy.
std::unique_ptr<BasicStrategy> loadWalkerBasicStrategy(
    const SpanishRules& rules, int minCount, int maxCount);

// Applies Walker's deviations in place to an already count-expanded strategy
// (exposed separately so callers that already have an expanded base strategy,
// e.g. a redouble/DDR-specific one, can reuse it without reloading).
void applyWalkerDeviations(BasicStrategy& strategy, int minCount, int maxCount);

// Backfills any (count, handType, playerSum, dealerCard, cardCount) cell
// `learned` is missing (in [minCount, maxCount]) from `fallback`'s entry for
// that same cell. Needed because QLearningStrategy::toBasicStrategy() only
// writes states actually visited during training -- Walker's wide count
// range x 5 card-counts is sparse enough that rare cells can go unvisited
// even at large round counts, and BasicStrategy::getAction() (unlike
// getActionFromTable()) throws on a genuinely missing HARD/SOFT entry rather
// than defaulting. `fallback` should be fully populated across the same
// range (e.g. via expandSpanishBasicStrategyAcrossCounts()) so the result has
// no gaps.
std::unique_ptr<BasicStrategy> backfillMissingEntries(
    const BasicStrategy& learned, const BasicStrategy& fallback, int minCount, int maxCount);

// Human-readable summary of Walker's index table, for --help / run banners.
void printWalkerDeviationsList(std::ostream& os);

// Walker's own table spans thresholds from -11 to +3; this range gives every
// index room to bind plus headroom for realistic play, and is used as the
// default count range for all Walker-related comparisons/debugging.
inline constexpr int kWalkerMinCount = -15;
inline constexpr int kWalkerMaxCount = 10;
