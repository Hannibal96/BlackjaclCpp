#include "Game/CountingMethods.h"
#include "RegressionTestUtils.h"
#include "Utils/CountPolicyEvaluation.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

template <size_t Size>
uint64_t sumHistogram(const std::array<uint64_t, Size>& histogram) {
    uint64_t total = 0;
    for (uint64_t count : histogram) total += count;
    return total;
}

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    const Case gameCase{
        1,
        false,
        true,
        4,
        DoubleDownOn::ANY,
        false,
        false,
        true,
        Surrender::NO_SURRENDER,
        1.5f
    };
    BasicStrategy policy;
    if (!policy.loadFromJson(ToStringTableName(gameCase))) {
        std::cerr << "Could not load test basic strategy\n";
        return 1;
    }

    CountPolicyConfig count;
    count.system = *CountingMethods::fromName("hilo");
    count.minCount = -5;
    count.maxCount = 5;

    BlackjackRules rules(
        gameCase.blackJackPay,
        gameCase.standSoft17,
        gameCase.deckSize,
        75.0,
        gameCase.peek,
        gameCase.splitAfterSplit,
        gameCase.doubleAfterSplit,
        gameCase.reSplitAces,
        gameCase.hitSplitAces,
        gameCase.surrender,
        gameCase.doubleOn);

    CountPolicyEvaluationOptions options;
    options.spreadRounds = 2'000;
    options.kellyRoundsPerMeasurement = 1'000;
    options.kellyMeasurements = 2;
    options.numThreads = 2;
    options.kellyFractions = {1.0};
    options.maximumTotalWagerFraction = 0.01;
    options.seed = 12345;

    const CountPolicyEvaluationResult result =
        evaluateCountPolicy(rules, policy, count, options, "test");
    check(result.spread.samples == options.spreadRounds, "spread round count");
    check(result.kelly.points.size() == 1, "Kelly point count");
    const KellyExposureStatistics& exposure = result.kelly.points.front().exposure;
    const uint64_t expectedKellyRounds =
        options.kellyRoundsPerMeasurement * options.kellyMeasurements;
    check(exposure.rounds == expectedKellyRounds, "exposure round count");
    check(exposure.validBankrollRounds == expectedKellyRounds, "valid bankroll count");
    check(sumHistogram(exposure.grossExposureHistogram) == expectedKellyRounds,
          "gross exposure histogram count");
    check(sumHistogram(exposure.absoluteReturnHistogram) == expectedKellyRounds,
          "absolute return histogram count");
    check(sumHistogram(exposure.wagerMultipleHistogram) == expectedKellyRounds,
          "wager multiple histogram count");
    check(exposure.grossExposureMaximum <=
              options.maximumTotalWagerFraction + 1e-10,
          "cumulative gross wager cap");
    check(kellyPointAtFraction(result.kelly, 1.0) != nullptr,
          "Kelly multiplier 1 lookup");

    std::cout << "CountPolicyEvaluationTest passed\n";
    return 0;
}
