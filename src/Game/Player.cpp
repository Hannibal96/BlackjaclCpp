#include "Player.h"
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

// Get bet amount
double Player::getBet(const std::array<int, 13>& /*removedCards*/) {
    return 1.0;
}

// Update the player's money with SARS parameters for learning strategies
void Player::updateMoney(double reward, const State& state, Action action, const State& nextState) {
    money += reward;
    strategy->updateTable(stateToKey(state), action, reward,
                          stateToKey(nextState), nextState.allowedActions);
}

void Player::resetPlayer(double m) {
    money = m;
}

// Set strategy
void Player::setStrategy(std::unique_ptr<Strategy> strat) {
    if (!strat) {
        throw std::invalid_argument("Player must have a strategy");
    }
    strategy = std::move(strat);
}

// Set card counting weights and resolution
void Player::setCountWeights(const std::array<double, 13>& weights, double resolution) {
    countWeights = weights;
    countResolution = resolution;
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

void Player::recordRound(const std::array<double, 13>& x, double y) {
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
    money += other.money;
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
    return *this;
}

Player& Player::operator*=(double factor) {
    money *= factor;
    if (strategy) {
        *strategy *= factor;
    }
    // Do NOT scale XtX, Xty, or regressionRounds — they accumulate absolute totals,
    // not per-thread averages. After runParallelSimulation the matrices already hold
    // the correct sum across all threads.
    return *this;
}

// Clone the player
Player* Player::clone() const {
    if (!strategy) {
        throw std::logic_error("Cannot clone player without strategy");
    }
    auto* p = new Player(money, strategy->clone(), name);
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
    return p;
}
