#include "SpanishStrategy.h"

#include <stdexcept>

std::unique_ptr<BasicStrategy> loadSpanishBasicStrategy(const SpanishRules& rules) {
    auto strategy = std::make_unique<BasicStrategy>();

    // Base (pre-double) decisions. WoO only publishes a distinct "not yet doubled"
    // chart for H17 + redouble (it differs from the plain H17 base table since
    // knowing redouble is available makes more totals worth doubling); every other
    // combination uses the plain S17/H17 base table.
    if (rules.maxRedoubles > 0 && !rules.standSoft17) {
        if (!strategy->loadFromFile("basic_strategy_tables/spanish21/h17_redouble_first.json")) {
            throw std::runtime_error("Failed to load Spanish 21 H17+redouble basic strategy");
        }
    } else {
        const std::string base = rules.standSoft17
            ? "basic_strategy_tables/spanish21/s17.json"
            : "basic_strategy_tables/spanish21/h17.json";
        if (!strategy->loadFromFile(base)) {
            throw std::runtime_error("Failed to load Spanish 21 basic strategy: " + base);
        }
    }

    // AFTER_DOUBLE (redouble/rescue) decisions. WoO only publishes one such chart
    // (H17 + redouble), but the decision itself is not meaningfully sensitive to the
    // dealer's soft-17 rule, so it's reused as the best available reference whenever
    // there's a real AFTER_DOUBLE decision to make (redoubling or rescue enabled).
    // BasicStrategy::getAction() falls back to STAND for any AFTER_DOUBLE state
    // missing from the table, so this merge is optional but improves accuracy.
    if (rules.maxRedoubles > 0 || rules.allowDoubleDownRescue) {
        if (!strategy->mergeFromFile("basic_strategy_tables/spanish21/h17_redouble_after.json")) {
            throw std::runtime_error("Failed to load Spanish 21 after-double strategy");
        }
    }

    return strategy;
}
