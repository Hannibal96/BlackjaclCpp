#include "Game/Player.h"
#include "Game/CountingMethods.h"
#include "RL/BasicStrategy.h"
#include "Utils/SimulationAnalysis.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

bool close(double lhs, double rhs, double tolerance = 1e-12) {
    return std::abs(lhs - rhs) <= tolerance;
}

void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

} // namespace

int main() {
    KellyExposureStatistics first;
    first.record(100.0, 1.0, 3.0, -2.0);
    check(first.rounds == 1, "round count");
    check(first.validBankrollRounds == 1, "valid bankroll round count");
    check(close(first.grossExposureSum, 0.03), "gross exposure sum");
    check(close(first.grossExposureMaximum, 0.03), "gross exposure maximum");
    check(close(first.absoluteReturnSum, 0.02), "absolute return sum");
    check(close(first.signedReturnSum, -0.02), "signed return sum");
    check(first.wagerMultipleHistogram[3] == 1, "wager multiple");
    check(first.grossExposureTailCounts[2] == 1, "gross exposure 1% tail");
    check(first.absoluteReturnTailCounts[3] == 1, "absolute return 2% tail");
    check(first.invalidLogRounds == 0, "valid logarithm");
    check(close(first.exactLogIncrementSum, std::log1p(-0.02)), "exact log");
    check(close(first.quadraticLogIncrementSum, -0.02 - 0.5 * 0.02 * 0.02),
          "quadratic log approximation");

    KellyExposureStatistics second;
    second.record(100.0, 0.0, 0.0, 0.0);
    second.record(0.0, 0.0, 0.0, 0.0);
    first += second;
    check(first.rounds == 3, "merged rounds");
    check(first.validBankrollRounds == 2, "merged valid bankroll rounds");
    check(first.invalidBankrollRounds == 1, "invalid bankroll rounds");
    check(first.zeroWagerRounds == 1, "zero wager rounds");
    check(first.wagerMultipleHistogram[0] == 1, "zero wager multiple");

    auto strategy = std::make_unique<BasicStrategy>();
    Player player(100.0, std::move(strategy));
    CountingSystem running;
    running.weights[0] = 1.0;
    running.normalization = CountNormalization::RUNNING_COUNT;
    running.initialCountPerDeck = -4.0;
    player.setNumDecks(6);
    player.setCountSystem(running);
    std::array<int, 13> removed{};
    removed[0] = 2;
    check(close(player.countValue(removed), -22.0), "running count with deck offset");

    running.normalization = CountNormalization::TRUE_COUNT;
    running.initialCount = 0.0;
    running.initialCountPerDeck = 0.0;
    player.setNumDecks(1);
    player.setCountSystem(running);
    removed = {};
    removed[0] = 2;
    const double remainingDecks = 50.0 / 52.0;
    check(close(player.countValue(removed), 2.0 / remainingDecks), "true count");

    const auto namedKo = CountingMethods::fromName("ko");
    check(namedKo.has_value(), "named KO exists");
    check(namedKo->normalization == CountNormalization::RUNNING_COUNT,
          "named KO uses a running count");
    check(close(namedKo->initialCount, 0.0) &&
              close(namedKo->initialCountPerDeck, 0.0),
          "named KO uses the zero-IRC convention");

    player.setEnforceBankrollActionLimits(true);
    player.setMaximumTotalWagerFraction(0.25);
    check(player.canAffordAdditionalWager(5.0, 10.0, 20.0, 100.0),
          "cumulative wager exactly at cap");
    check(!player.canAffordAdditionalWager(5.01, 10.0, 20.0, 100.0),
          "cumulative wager above cap");
    check(!player.canAffordAdditionalWager(91.0, 10.0, 0.0, 100.0),
          "concurrent bankroll limit remains active");

    bool rejectedInvalidCap = false;
    try {
        player.setMaximumTotalWagerFraction(1.01);
    } catch (const std::invalid_argument&) {
        rejectedInvalidCap = true;
    }
    check(rejectedInvalidCap, "invalid total wager cap rejected");

    const std::string exposureSvg =
        kellyExposureStatisticsToSvg("Exposure test", first);
    check(exposureSvg.find("Probability per bin (log scale)") != std::string::npos,
          "logarithmic y-axis name");
    check(exposureSvg.find("Fraction of round-start bankroll") != std::string::npos,
          "x-axis name");
    check(exposureSvg.find("stroke=\"#d6dbe1\"") != std::string::npos,
          "horizontal grid");
    check(exposureSvg.find("Gross exposure") != std::string::npos,
          "exposure legend");

    std::cout << "KellyExposureTest passed\n";
    return 0;
}
