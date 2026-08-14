#include "DoubleDownMadnessStrategy.h"

#include <stdexcept>
#include <string>

std::unique_ptr<BasicStrategy> loadDoubleDownMadnessBasicStrategy(
        DoubleDownMadnessVersion version) {
    auto strategy = std::make_unique<BasicStrategy>();
    if (!strategy->loadFromFile(
            "basic_strategy_tables/double_down_madness/basic_strategy.json")) {
        throw std::runtime_error(
            "Failed to load Double Down Madness basic strategy");
    }

    const int versionNumber = static_cast<int>(version);
    const std::string overridePath =
        "basic_strategy_tables/double_down_madness/version_" +
        std::to_string(versionNumber) + ".json";
    if (!strategy->mergeFromFile(overridePath)) {
        throw std::runtime_error(
            "Failed to load Double Down Madness version override");
    }
    return strategy;
}
