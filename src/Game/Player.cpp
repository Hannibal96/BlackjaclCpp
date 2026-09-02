#include "Player.h"
#include <limits>
#include <stdexcept>

namespace {

size_t histogramIndex(double value) {
    if (!std::isfinite(value) || value >= KellyExposureStatistics::kHistogramMaximum)
        return KellyExposureStatistics::kHistogramBins - 1;
    if (value <= 0.0) return 0;
    return std::min(
        static_cast<size_t>(value / KellyExposureStatistics::kHistogramBinWidth),
        KellyExposureStatistics::kRegularHistogramBins - 1);
}

} // namespace

void KellyExposureStatistics::record(double bankrollBeforeRound,
                                     double initialWager,
                                     double totalWager,
                                     double roundProfit) {
    ++rounds;
    if (!std::isfinite(bankrollBeforeRound) || bankrollBeforeRound <= 0.0 ||
        !std::isfinite(initialWager) || !std::isfinite(totalWager) ||
        !std::isfinite(roundProfit) || initialWager < 0.0 || totalWager < 0.0) {
        ++invalidBankrollRounds;
        return;
    }

    ++validBankrollRounds;
    if (totalWager == 0.0) ++zeroWagerRounds;

    const double grossExposure = totalWager / bankrollBeforeRound;
    const double signedReturn = roundProfit / bankrollBeforeRound;
    const double absoluteReturn = std::abs(signedReturn);
    grossExposureSum += grossExposure;
    grossExposureMaximum = std::max(grossExposureMaximum, grossExposure);
    absoluteReturnSum += absoluteReturn;
    absoluteReturnMaximum = std::max(absoluteReturnMaximum, absoluteReturn);
    signedReturnSum += signedReturn;
    signedReturnSquaredSum += signedReturn * signedReturn;
    ++grossExposureHistogram[histogramIndex(grossExposure)];
    ++absoluteReturnHistogram[histogramIndex(absoluteReturn)];

    size_t wagerMultipleIndex = 0;
    if (initialWager > 0.0) {
        const double multiple = totalWager / initialWager;
        const long long rounded = std::llround(multiple);
        wagerMultipleIndex = rounded <= 0
            ? 0
            : std::min(static_cast<size_t>(rounded), kWagerMultipleBins - 1);
    }
    ++wagerMultipleHistogram[wagerMultipleIndex];

    for (size_t i = 0; i < kTailThresholds.size(); ++i) {
        if (grossExposure >= kTailThresholds[i]) ++grossExposureTailCounts[i];
        if (absoluteReturn >= kTailThresholds[i]) ++absoluteReturnTailCounts[i];
    }

    if (signedReturn <= -1.0) {
        ++invalidLogRounds;
        return;
    }
    const double exact = std::log1p(signedReturn);
    const double quadratic = signedReturn - 0.5 * signedReturn * signedReturn;
    const double error = exact - quadratic;
    exactLogIncrementSum += exact;
    quadraticLogIncrementSum += quadratic;
    taylorErrorSum += error;
    absoluteTaylorErrorSum += std::abs(error);
    absoluteTaylorErrorMaximum = std::max(absoluteTaylorErrorMaximum, std::abs(error));
}

KellyExposureStatistics& KellyExposureStatistics::operator+=(
        const KellyExposureStatistics& other) {
    rounds += other.rounds;
    validBankrollRounds += other.validBankrollRounds;
    invalidBankrollRounds += other.invalidBankrollRounds;
    zeroWagerRounds += other.zeroWagerRounds;
    invalidLogRounds += other.invalidLogRounds;
    grossExposureSum += other.grossExposureSum;
    grossExposureMaximum = std::max(grossExposureMaximum, other.grossExposureMaximum);
    absoluteReturnSum += other.absoluteReturnSum;
    absoluteReturnMaximum = std::max(absoluteReturnMaximum, other.absoluteReturnMaximum);
    signedReturnSum += other.signedReturnSum;
    signedReturnSquaredSum += other.signedReturnSquaredSum;
    exactLogIncrementSum += other.exactLogIncrementSum;
    quadraticLogIncrementSum += other.quadraticLogIncrementSum;
    taylorErrorSum += other.taylorErrorSum;
    absoluteTaylorErrorSum += other.absoluteTaylorErrorSum;
    absoluteTaylorErrorMaximum =
        std::max(absoluteTaylorErrorMaximum, other.absoluteTaylorErrorMaximum);
    for (size_t i = 0; i < kHistogramBins; ++i) {
        grossExposureHistogram[i] += other.grossExposureHistogram[i];
        absoluteReturnHistogram[i] += other.absoluteReturnHistogram[i];
    }
    for (size_t i = 0; i < kWagerMultipleBins; ++i)
        wagerMultipleHistogram[i] += other.wagerMultipleHistogram[i];
    for (size_t i = 0; i < kTailThresholds.size(); ++i) {
        grossExposureTailCounts[i] += other.grossExposureTailCounts[i];
        absoluteReturnTailCounts[i] += other.absoluteReturnTailCounts[i];
    }
    return *this;
}

// Constructor
Player::Player(double initialMoney, std::unique_ptr<Strategy> strat, std::string name)
    : money(initialMoney), strategy(std::move(strat)), name(name) {
    if (!strategy) {
        throw std::invalid_argument("Player must have a strategy");
    }
}

// Convert a raw State to a StateKey
StateKey Player::stateToKey(const State& state) const {
    const double selectedCount = countValue(state.removedCards);

    // Discretize: round to nearest multiple of countResolution, clamp to [minCount, maxCount]
    int discretizedCount = std::clamp(
        static_cast<int>(std::round(selectedCount / countResolution) * countResolution),
        minCount, maxCount);

    // Determine hand type — treat as PAIR only if split is actually allowed
    bool splitAllowed = std::find(state.allowedActions.begin(), state.allowedActions.end(),
                                  Action::SPLIT) != state.allowedActions.end();
    HandType handType = state.playerHand.getHandType(splitAllowed);

    unsigned int dealerCard = static_cast<unsigned int>(state.dealerCard.getValue());
    unsigned int playerSum;
    if (handType == HandType::PAIR) {
        playerSum = static_cast<unsigned int>(state.playerHand[0].getValue());
    } else {
        playerSum = static_cast<unsigned int>(state.playerHand.getValue());
    }

    return std::make_tuple(discretizedCount, handType, playerSum, dealerCard);
}

// Get action based on game state (delegates to strategy via StateKey)
Action Player::getAction(const State& state) {
    return strategy->getAction(stateToKey(state), state.allowedActions);
}

Action Player::getAction(const State& state,
                         const std::vector<Action>& allowedActions) {
    return strategy->getAction(stateToKey(state), allowedActions);
}

// Get bet amount — computes BettingContext and delegates to bettingStrategy
double Player::getBet(const std::array<int, 13>& removedCards) {
    if (!bettingStrategy) return 1.0;

    BettingContext ctx;
    ctx.bankroll = money;
    ctx.trueCount = countValue(removedCards);
    if (!continuousBettingCount && countResolution > 0.0)
        ctx.trueCount = std::round(ctx.trueCount / countResolution) * countResolution;
    ctx.expectedValue = countBias + countFactor * ctx.trueCount;

    return bettingStrategy->getBet(ctx);
}

double Player::countValue(const std::array<int, 13>& removedCards) const {
    double runningCount = initialCount + initialCountPerDeck * static_cast<double>(numDecks);
    int totalRemoved = 0;
    for (int i = 0; i < 13; ++i) {
        runningCount += countWeights[i] * static_cast<double>(removedCards[i]);
        totalRemoved += removedCards[i];
    }
    if (countNormalization == CountNormalization::RUNNING_COUNT)
        return runningCount;
    const double remainingDecks =
        static_cast<double>(numDecks * 52 - totalRemoved) / 52.0;
    return remainingDecks > 0.0 ? runningCount / remainingDecks : 0.0;
}

double Player::bettingSignal(const std::array<int, 13>& removedCards) const {
    double count = countValue(removedCards);
    if (!continuousBettingCount && countResolution > 0.0)
        count = std::round(count / countResolution) * countResolution;
    return countBias + countFactor * count;
}

double Player::getLogMoney() const {
    if (logMoneyDirty) {
        logMoney = (money > 0.0)
            ? std::log(money)
            : -std::numeric_limits<double>::infinity();
        logMoneyDirty = false;
    }
    return logMoney;
}

void Player::setBettingStrategy(std::unique_ptr<BettingStrategy> bs) {
    bettingStrategy = std::move(bs);
}

bool Player::canAffordAdditionalWager(double additionalWager,
                                     double committedWager) const {
    if (!enforceBankrollActionLimits) return true;

    constexpr double tolerance = 1e-12;
    return additionalWager <= money - committedWager + tolerance;
}

void Player::setMaximumTotalWagerFraction(double fraction) {
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0) {
        throw std::invalid_argument(
            "Maximum total wager fraction must be finite and between 0 and 1");
    }
    maximumTotalWagerFraction = fraction;
}

bool Player::canAffordAdditionalWager(double additionalWager,
                                     double committedWager,
                                     double cumulativeWager,
                                     double bankrollBeforeRound) const {
    if (!enforceBankrollActionLimits) return true;
    if (!std::isfinite(additionalWager) || !std::isfinite(committedWager) ||
        !std::isfinite(cumulativeWager) || !std::isfinite(bankrollBeforeRound) ||
        additionalWager < 0.0 || committedWager < 0.0 || cumulativeWager < 0.0 ||
        bankrollBeforeRound < 0.0) {
        return false;
    }

    const double tolerance = 1e-12 * std::max(1.0, bankrollBeforeRound);
    const double maximumTotalWager =
        bankrollBeforeRound * maximumTotalWagerFraction;
    return additionalWager <= money - committedWager + tolerance &&
           cumulativeWager + additionalWager <= maximumTotalWager + tolerance;
}

// Update the player's money with SARS parameters for learning strategies
void Player::updateMoney(double reward, const State& state, Action action,
                         const State& nextState,
                         double learningRewardDivisor,
                         double nextValueMultiplier) {
    if (learningRewardDivisor <= 0.0)
        throw std::invalid_argument("Learning reward divisor must be positive");

    money += reward;
    logMoneyDirty = true;
    strategy->updateTable(
        stateToKey(state), action, reward / learningRewardDivisor,
        stateToKey(nextState), nextState.allowedActions, nextValueMultiplier);
}

void Player::resetPlayer(double m) {
    money = m;
    logMoneyDirty = true;
}

// Set strategy
void Player::setStrategy(std::unique_ptr<Strategy> strat) {
    if (!strat) {
        throw std::invalid_argument("Player must have a strategy");
    }
    strategy = std::move(strat);
}

void Player::setCountWeights(const std::array<double, 13>& weights) {
    countWeights = weights;
}

// Set the count range — the discretized count is clamped to [min, max]
void Player::setCountRange(int min, int max) {
    minCount = min;
    maxCount = max;
}

// Set the total number of decks in the shoe (needed for true-count normalization)
void Player::setNumDecks(int decks) {
    numDecks = decks;
}

void Player::recordRoundOutcome(double reward) {
    if (!roundStatsEnabled) return;
    ++roundStatsCount;
    roundRewardSum += reward;
    roundRewardSumSq += reward * reward;
}

void Player::recordKellyExposure(double bankrollBeforeRound,
                                 double initialWager,
                                 double totalWager,
                                 double roundProfit) {
    if (!kellyExposureStatsEnabled) return;
    kellyExposureStats.record(
        bankrollBeforeRound, initialWager, totalWager, roundProfit);
}

void Player::recordRound(const std::array<double, 13>& x,
                         double y,
                         std::optional<double> observedCount) {
    if (countGraphEnabled) {
        double trueCount = observedCount.value_or(0.0);
        if (!observedCount) {
            for (int i = 0; i < 13; ++i)
                trueCount += countWeights[i] * x[i];
        }
        int binIndex = (countGraphResolution > 0.0)
            ? static_cast<int>(std::llround(trueCount / countGraphResolution))
            : static_cast<int>(std::llround(trueCount / 0.25));
        auto& bin = countGraphBins[binIndex];
        ++bin.n;
        bin.sumReward += y;
        bin.sumRewardSq += y * y;
    }

    if (!regressionEnabled) return;
    ++regressionSampleCounter;
    if (regressionSampleCounter % regressionSampleEvery != 0) return;

    std::array<double, 14> xb;
    for (int i = 0; i < 13; ++i) xb[i] = x[i];
    xb[13] = 1.0;  // bias term
    const double covarianceWeight =
        regressionObjective == RegressionObjective::QUADRATIC_KELLY ? y * y : 1.0;

    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 14; ++j)
            XtX[i][j] += covarianceWeight * xb[i] * xb[j];
        Xty[i] += xb[i] * y;
    }
    ++regressionRounds;
}

// Averaging operators for combining players from parallel simulations
Player& Player::operator+=(const Player& other) {
    const double thisLog = getLogMoney();
    const double otherLog = other.getLogMoney();
    money += other.money;
    logMoney = thisLog + otherLog;
    logMoneyDirty = false;
    if (strategy && other.strategy) {
        *strategy += *other.strategy;
    }
    if (regressionEnabled && other.regressionEnabled &&
        regressionObjective != other.regressionObjective) {
        throw std::logic_error("Cannot combine players with different regression objectives");
    }
    if (regressionEnabled || other.regressionEnabled) {
        if (!regressionEnabled) regressionObjective = other.regressionObjective;
        regressionEnabled = true;
        for (int i = 0; i < 14; ++i) {
            for (int j = 0; j < 14; ++j)
                XtX[i][j] += other.XtX[i][j];
            Xty[i] += other.Xty[i];
        }
        regressionRounds += other.regressionRounds;
    }
    if (countGraphEnabled || other.countGraphEnabled) {
        countGraphEnabled = countGraphEnabled || other.countGraphEnabled;
        if (other.countGraphEnabled) countGraphResolution = other.countGraphResolution;
        for (const auto& [binIndex, stats] : other.countGraphBins) {
            auto& dst = countGraphBins[binIndex];
            dst.n += stats.n;
            dst.sumReward += stats.sumReward;
            dst.sumRewardSq += stats.sumRewardSq;
        }
    }
    if (roundStatsEnabled || other.roundStatsEnabled) {
        roundStatsEnabled = roundStatsEnabled || other.roundStatsEnabled;
        roundStatsCount += other.roundStatsCount;
        roundRewardSum += other.roundRewardSum;
        roundRewardSumSq += other.roundRewardSumSq;
    }
    if (kellyExposureStatsEnabled || other.kellyExposureStatsEnabled) {
        kellyExposureStatsEnabled = true;
        kellyExposureStats += other.kellyExposureStats;
    }
    return *this;
}

Player& Player::operator*=(double factor) {
    const double currentLog = getLogMoney();
    money *= factor;
    logMoney = currentLog * factor;
    logMoneyDirty = false;
    if (strategy) {
        *strategy *= factor;
    }
    // Do NOT scale regression/count-graph/round-stat totals: they accumulate
    // absolute observations rather than per-thread averages.
    return *this;
}

// Clone the player
Player* Player::clone() const {
    if (!strategy) {
        throw std::logic_error("Cannot clone player without strategy");
    }
    auto* p = new Player(money, strategy->clone(), name);
    p->logMoney = logMoney;
    p->logMoneyDirty = logMoneyDirty;
    p->countWeights = countWeights;
    p->countResolution = countResolution;
    p->minCount = minCount;
    p->maxCount = maxCount;
    p->numDecks = numDecks;
    p->regressionEnabled = regressionEnabled;
    p->regressionObjective = regressionObjective;
    p->XtX = XtX;
    p->Xty = Xty;
    p->regressionRounds = regressionRounds;
    p->regressionSampleEvery = regressionSampleEvery;
    p->regressionSampleCounter = regressionSampleCounter;
    p->countGraphEnabled = countGraphEnabled;
    p->countGraphResolution = countGraphResolution;
    p->countGraphBins = countGraphBins;
    p->roundStatsEnabled = roundStatsEnabled;
    p->roundStatsCount = roundStatsCount;
    p->roundRewardSum = roundRewardSum;
    p->roundRewardSumSq = roundRewardSumSq;
    p->kellyExposureStatsEnabled = kellyExposureStatsEnabled;
    p->kellyExposureStats = kellyExposureStats;
    if (bettingStrategy) p->bettingStrategy = std::unique_ptr<BettingStrategy>(bettingStrategy->clone());
    p->enforceBankrollActionLimits = enforceBankrollActionLimits;
    p->maximumTotalWagerFraction = maximumTotalWagerFraction;
    p->countFactor = countFactor;
    p->countBias   = countBias;
    p->continuousBettingCount = continuousBettingCount;
    p->countNormalization = countNormalization;
    p->initialCount = initialCount;
    p->initialCountPerDeck = initialCountPerDeck;
    return p;
}
