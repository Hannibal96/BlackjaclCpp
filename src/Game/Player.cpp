#include "Player.h"
#include <limits>
#include <stdexcept>

// Constructor
Player::Player(double initialMoney, std::unique_ptr<Strategy> strat, std::string name)
    : money(initialMoney), strategy(std::move(strat)), name(name) {
    if (!strategy) {
        throw std::invalid_argument("Player must have a strategy");
    }
}

// Convert a raw State to a StateKey
StateKey Player::stateToKey(const State& state) const {
    // Compute raw count as dot product of weights and removed cards
    double rawCount = 0.0;
    int totalRemoved = 0;
    for (int i = 0; i < 13; ++i) {
        rawCount += countWeights[i] * static_cast<double>(state.removedCards[i]);
        totalRemoved += state.removedCards[i];
    }

    // Normalize by remaining decks to get the true count
    double remainingDecks = static_cast<double>(numDecks * 52 - totalRemoved) / 52.0;
    double trueCount = (remainingDecks > 0.0) ? (rawCount / remainingDecks) : 0.0;

    // Discretize: round to nearest multiple of countResolution, clamp to [minCount, maxCount]
    int discretizedCount = std::clamp(
        static_cast<int>(std::round(trueCount / countResolution) * countResolution),
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

// Get bet amount — computes BettingContext and delegates to bettingStrategy
double Player::getBet(const std::array<int, 13>& removedCards) {
    if (!bettingStrategy) return 1.0;

    int totalRemoved = 0;
    for (int c : removedCards) totalRemoved += c;
    double remainingDecks = static_cast<double>(numDecks * 52 - totalRemoved) / 52.0;

    BettingContext ctx;
    ctx.bankroll = money;

    if (remainingDecks > 0.0) {
        double rawCount = 0.0;
        for (int i = 0; i < 13; ++i)
            rawCount += countWeights[i] * static_cast<double>(removedCards[i]);
        double tc = rawCount / remainingDecks;
        // Discretize to the same resolution used for strategy state keys
        if (countResolution > 0.0)
            tc = std::round(tc / countResolution) * countResolution;
        ctx.trueCount = tc;
    } else {
        ctx.trueCount = 0.0;
    }
    ctx.expectedValue = countBias + countFactor * ctx.trueCount;

    return bettingStrategy->getBet(ctx);
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

void Player::recordRound(const std::array<double, 13>& x, double y) {
    if (countGraphEnabled) {
        double trueCount = 0.0;
        for (int i = 0; i < 13; ++i)
            trueCount += countWeights[i] * x[i];
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

    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 14; ++j)
            XtX[i][j] += xb[i] * xb[j];
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
    if (regressionEnabled) {
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
    if (bettingStrategy) p->bettingStrategy = std::unique_ptr<BettingStrategy>(bettingStrategy->clone());
    p->countFactor = countFactor;
    p->countBias   = countBias;
    return p;
}
