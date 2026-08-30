#include "CountQuantization.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

constexpr std::array<int, 9> kNonTenValueIndices = {0, 1, 2, 3, 4, 5, 6, 7, 12};

double squaredError(const std::array<long long, 13>& lattice,
                    const std::array<double, 13>& scaled) {
    double error = 0.0;
    for (size_t i = 0; i < lattice.size(); ++i) {
        const double difference = static_cast<double>(lattice[i]) - scaled[i];
        error += difference * difference;
    }
    return error;
}

std::array<long long, 9> closestIntegersWithSum(
        const std::array<double, 9>& values,
        long long targetSum) {
    std::array<long long, 9> result{};
    long long currentSum = 0;
    for (size_t i = 0; i < values.size(); ++i) {
        result[i] = static_cast<long long>(std::floor(values[i]));
        currentSum += result[i];
    }

    // Separable squared error is discrete-convex. Repeatedly taking the
    // cheapest one-step move therefore gives the exact fixed-sum solution.
    while (currentSum < targetSum) {
        size_t best = 0;
        double bestCost = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < values.size(); ++i) {
            const double x = static_cast<double>(result[i]);
            const double cost = (x + 1.0 - values[i]) * (x + 1.0 - values[i])
                              - (x - values[i]) * (x - values[i]);
            if (cost < bestCost) {
                bestCost = cost;
                best = i;
            }
        }
        ++result[best];
        ++currentSum;
    }
    while (currentSum > targetSum) {
        size_t best = 0;
        double bestCost = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < values.size(); ++i) {
            const double x = static_cast<double>(result[i]);
            const double cost = (x - 1.0 - values[i]) * (x - 1.0 - values[i])
                              - (x - values[i]) * (x - values[i]);
            if (cost < bestCost) {
                bestCost = cost;
                best = i;
            }
        }
        --result[best];
        --currentSum;
    }
    return result;
}

bool lexicographicallyLess(const std::array<long long, 13>& lhs,
                           const std::array<long long, 13>& rhs) {
    return std::lexicographical_compare(lhs.begin(), lhs.end(), rhs.begin(), rhs.end());
}

} // namespace

CountQuantizationResult quantizeCountWeights(
        const std::array<double, 13>& weights,
        double quantum,
        double balanceTolerance) {
    if (!std::isfinite(quantum) || quantum < 0.0)
        throw std::invalid_argument("Quantization quantum must be finite and non-negative");
    if (!std::isfinite(balanceTolerance) || balanceTolerance < 0.0)
        throw std::invalid_argument("Balance tolerance must be finite and non-negative");
    for (double weight : weights) {
        if (!std::isfinite(weight))
            throw std::invalid_argument("Count weights must be finite");
    }

    CountQuantizationResult result;
    result.weights = weights;
    double sourceSum = 0.0;
    double sourceMagnitude = 0.0;
    for (double weight : weights) {
        sourceSum += weight;
        sourceMagnitude += std::abs(weight);
    }
    result.sourceWasBalanced =
        std::abs(sourceSum) <= balanceTolerance * std::max(1.0, sourceMagnitude);

    // Quantum zero is deliberately bit-for-bit identical to the source count.
    if (quantum == 0.0) {
        result.balancePreserved = result.sourceWasBalanced;
        return result;
    }

    std::array<double, 13> scaled{};
    constexpr double kSafeIntegerMagnitude =
        static_cast<double>(std::numeric_limits<long long>::max()) / 16.0;
    for (size_t i = 0; i < weights.size(); ++i) {
        scaled[i] = weights[i] / quantum;
        if (!std::isfinite(scaled[i]) || std::abs(scaled[i]) > kSafeIntegerMagnitude) {
            throw std::invalid_argument(
                "Quantization quantum is too small for the supplied count weights");
        }
    }

    if (!result.sourceWasBalanced) {
        for (size_t i = 0; i < result.weights.size(); ++i) {
            const long long tag = static_cast<long long>(std::llround(scaled[i]));
            result.weights[i] = tag == 0 ? 0.0 : quantum * static_cast<double>(tag);
        }
        return result;
    }

    std::array<double, 9> nonTenScaled{};
    for (size_t i = 0; i < kNonTenValueIndices.size(); ++i)
        nonTenScaled[i] = scaled[kNonTenValueIndices[i]];

    const double tenValueMean =
        (scaled[8] + scaled[9] + scaled[10] + scaled[11]) / 4.0;
    const long long center = static_cast<long long>(std::llround(tenValueMean));

    std::array<long long, 13> best{};
    double bestError = std::numeric_limits<double>::infinity();
    bool found = false;

    // With a balanced source, moving the shared ten-value tag more than three
    // lattice units from its unconstrained optimum cannot beat the nearby
    // feasible solution. The wider seven-point window also makes ties stable.
    for (long long tenTag = center - 3; tenTag <= center + 3; ++tenTag) {
        std::array<long long, 13> candidate{};
        candidate[8] = candidate[9] = candidate[10] = candidate[11] = tenTag;
        const auto nonTenTags = closestIntegersWithSum(nonTenScaled, -4LL * tenTag);
        for (size_t i = 0; i < kNonTenValueIndices.size(); ++i)
            candidate[kNonTenValueIndices[i]] = nonTenTags[i];

        const double error = squaredError(candidate, scaled);
        if (!found || error < bestError - 1e-12 ||
            (std::abs(error - bestError) <= 1e-12 &&
             lexicographicallyLess(candidate, best))) {
            best = candidate;
            bestError = error;
            found = true;
        }
    }

    long long latticeSum = 0;
    for (size_t i = 0; i < result.weights.size(); ++i) {
        latticeSum += best[i];
        result.weights[i] = best[i] == 0
            ? 0.0
            : quantum * static_cast<double>(best[i]);
    }
    result.balancePreserved = latticeSum == 0;
    return result;
}
