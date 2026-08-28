#pragma once

#include "BasicStrategy.h"
#include "Game/SpanishRules.h"
#include <memory>

// Loads the WoO-transcribed Spanish 21 reference strategy matching the given rules.
// Base tables (s17.json / h17.json) cover the no-redouble case. When redoubling is
// enabled under H17, WoO publishes a distinct pair of charts — "first double"
// (replaces the base table) and "after double" (the AFTER_DOUBLE decision) — which
// are loaded and merged together. S17 + redouble has no published WoO chart, so it
// falls back to the S17 base table (mechanically correct; not chart-verified for the
// redouble decisions themselves — see basic_strategy_tables/spanish21/README).
std::unique_ptr<BasicStrategy> loadSpanishBasicStrategy(const SpanishRules& rules);
