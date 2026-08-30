#pragma once

#include <array>

struct CountQuantizationResult {
    std::array<double, 13> weights{};
    bool sourceWasBalanced = false;
    bool balancePreserved = false;
};

// Quantize rank tags to integer multiples of quantum. A zero quantum is an
// exact pass-through. Balanced source counts are rounded under an exact
// integer-sum constraint, and 10/J/Q/K remain one shared tag.
CountQuantizationResult quantizeCountWeights(
    const std::array<double, 13>& weights,
    double quantum,
    double balanceTolerance = 1e-9);
