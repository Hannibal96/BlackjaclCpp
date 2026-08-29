#pragma once
#include "RL/State.h"
#include "RL/Action.h"
#include "RL/Strategy.h"
#include "RL/StateKey.h"
#include "BettingStrategy.h"
#include "CountingMethods.h"
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <algorithm>
#include <cmath>

enum class RegressionObjective {
    EXPECTED_VALUE_OLS,
    QUADRATIC_KELLY
};

// Player class - can be used directly or inherited for advanced behavior
class Player {
protected:
    double money;
    mutable double logMoney = 0.0;
    mutable bool logMoneyDirty = true;
    std::string name;
    std::unique_ptr<Strategy> strategy;

    // Card counting fields
    std::array<double, 13> countWeights{};  // Weights per rank (index 0=TWO ... 12=ACE); default all zeros
    double countResolution = 1.0;           // Discretization step for the true count
    int minCount = 0;                       // Lower clamp bound for the discretized count
    int maxCount = 0;                       // Upper clamp bound for the discretized count
    int numDecks = 1;                       // Total decks in the shoe (needed to compute remaining decks)
    double countOffsetPerDeck = 0.0;        // Running count starting offset per deck (see CountingSystem)

    // When true, StateKey's cardCount field carries the real (capped at 6) number
    // of cards in the player's hand. Games that don't need this distinction (all
    // but Spanish 21) leave this false, so the field is always the constant 2.
    bool trackHandCardCount = false;

    // Betting strategy — nullptr means fixed unit bet of 1.0
    std::unique_ptr<BettingStrategy> bettingStrategy;

    // When enabled, actions requiring another wager are removed if the
    // bankroll is already committed to the current round. Kelly simulations
    // enable this; training and fixed/spread edge simulations leave it off.
    bool enforceBankrollActionLimits = false;

    // Betting signal = countBias + countFactor * trueCount. Usually this is EV;
    // quadratic-Kelly counts fit the wager fraction directly.
    // For OLS-derived systems: countFactor=1.0, countBias=w[13] (weights already in EV units).
    // For traditional systems (Hi-Lo etc.): countFactor≈0.005, countBias≈-(house edge).
    double countFactor = 1.0;
    double countBias   = 0.0;
    bool continuousBettingCount = false;

    // Streaming regression accumulation. OLS stores X^T X; quadratic Kelly
    // stores sum(y^2 X^T X). Both store X^T y in Xty.
    // X row: [removedCards[i]/remainingDecks (13-dim), 1.0 bias]  (14-dim total)
    // y: net outcome of the round summed over all hands
    bool regressionEnabled = false;
    RegressionObjective regressionObjective = RegressionObjective::EXPECTED_VALUE_OLS;
    std::array<std::array<double, 14>, 14> XtX{};  // accumulated X^T X  (14×14, last = bias)
    std::array<double, 14> Xty{};                   // accumulated X^T y  (14×1)
    uint64_t regressionRounds = 0;
    uint64_t regressionSampleEvery = 1;             // collect every N-th round
    uint64_t regressionSampleCounter = 0;

    struct CountGraphBinStats {
        uint64_t n = 0;
        double sumReward = 0.0;
        double sumRewardSq = 0.0;
    };
    bool countGraphEnabled = false;
    double countGraphResolution = 0.25;
    std::map<int, CountGraphBinStats> countGraphBins;

    // Optional global round-return moments. These remain absolute totals when
    // thread-local players are combined, unlike bankroll/Q-table averages.
    bool roundStatsEnabled = false;
    uint64_t roundStatsCount = 0;
    double roundRewardSum = 0.0;
    double roundRewardSumSq = 0.0;

public:
    // Constructor
    Player(double initialMoney, std::unique_ptr<Strategy> strat, std::string name = "Uzan");

    // Virtual destructor for proper cleanup
    virtual ~Player() = default;

    // Convert a raw State to a StateKey by computing and discretizing the count
    StateKey stateToKey(const State& state) const;

    // Compute the discretized true count for a removed-cards snapshot (same
    // formula stateToKey uses internally) — exposed for verbose debug tracing.
    int computeTrueCount(const std::array<int, 13>& removedCards) const;

    // Get action based on game state (delegates to strategy via StateKey)
    virtual Action getAction(const State& state);
    Action getAction(const State& state, const std::vector<Action>& allowedActions);

    // Get bet amount — receives current shoe removed-cards for count-aware sizing.
    // Currently always returns 1.0; override or extend for count-based betting.
    virtual double getBet(const std::array<int, 13>& removedCards);

    // Update the player's money with SARS parameters for learning strategies
    void updateMoney(double reward, const State& state, Action action,
                     const State& nextState,
                     double learningRewardDivisor = 1.0,
                     double nextValueMultiplier = 1.0);

    // Update money only — no Q-table update. Use when the player made no decision
    // (e.g. dealer blackjack resolved before player acts in a peek game).
    void addReward(double reward) { money += reward; logMoneyDirty = true; }

    void resetPlayer(double money = 0);

    // Get current money
    double getMoney() const { return money; }
    double getLogMoney() const;

    std::string getName() const { return name; }

    // Set strategy
    void setStrategy(std::unique_ptr<Strategy> strat);

    // Get strategy (const access)
    const Strategy* getStrategy() const { return strategy.get(); }

    // Get mutable strategy access (for averaging operations)
    Strategy* getMutableStrategy() { return strategy.get(); }

    // --- Betting strategy ---
    void setBettingStrategy(std::unique_ptr<BettingStrategy> bs);
    bool hasBettingStrategy() const { return bettingStrategy != nullptr; }
    void setEnforceBankrollActionLimits(bool enforce) {
        enforceBankrollActionLimits = enforce;
    }
    bool shouldEnforceBankrollActionLimits() const {
        return enforceBankrollActionLimits;
    }
    bool canAffordAdditionalWager(double additionalWager,
                                  double committedWager) const;

    // Set E[game] model parameters individually or from a CountingSystem.
    void setCountFactor(double f) { countFactor = f; }
    void setCountBias(double b)   { countBias   = b; }
    void setCountSystem(const CountingSystem& cs) {
        setCountWeights(cs.weights);
        setCountFactor(cs.factor);
        setCountBias(cs.bias);
        continuousBettingCount = cs.continuousBettingCount;
        countOffsetPerDeck = cs.rawCountOffsetPerDeck;
    }

    // Set card counting weights (rank tags only; resolution is separate)
    void setCountWeights(const std::array<double, 13>& weights);

    // Set true-count discretization step (used both for strategy lookup and bet sizing)
    void setCountResolution(double resolution) { countResolution = resolution; }

    // Set the count range — the discretized count is clamped to [min, max]
    void setCountRange(int min, int max);

    // Set the total number of decks in the shoe (needed for true-count normalization)
    void setNumDecks(int decks);

    // Set whether StateKey's cardCount field should carry the hand's real card
    // count (Spanish 21) instead of the constant 2 (every other game).
    void setTrackHandCardCount(bool track) { trackHandCardCount = track; }
    bool getTrackHandCardCount() const { return trackHandCardCount; }

    // --- Regression ---
    void enableRegression(
        RegressionObjective objective = RegressionObjective::EXPECTED_VALUE_OLS) {
        regressionEnabled = true;
        regressionObjective = objective;
    }
    bool isRegressionEnabled() const { return regressionEnabled; }
    RegressionObjective getRegressionObjective() const { return regressionObjective; }
    void enableCountGraph(double resolution) {
        countGraphEnabled = true;
        countGraphResolution = (resolution > 0.0 ? resolution : 0.25);
    }
    bool isCountGraphEnabled() const { return countGraphEnabled; }
    double getCountGraphResolution() const { return countGraphResolution; }

    void enableRoundStats() { roundStatsEnabled = true; }
    bool isRoundStatsEnabled() const { return roundStatsEnabled; }
    void recordRoundOutcome(double reward);
    uint64_t getRoundStatsCount() const { return roundStatsCount; }
    double getRoundRewardSum() const { return roundRewardSum; }
    double getRoundRewardSumSq() const { return roundRewardSumSq; }

    // Accumulate one round: x = pre-round normalized removed-cards, y = round net outcome
    void recordRound(const std::array<double, 13>& x, double y);

    const std::array<std::array<double, 14>, 14>& getXtX() const { return XtX; }
    const std::array<double, 14>& getXty() const { return Xty; }
    const std::map<int, CountGraphBinStats>& getCountGraphBins() const { return countGraphBins; }
    void setRegressionSampleEvery(uint64_t n) { regressionSampleEvery = (n < 1 ? 1 : n); }
    uint64_t getRegressionRounds() const { return regressionRounds; }

    // Averaging operators for combining players from parallel simulations
    Player& operator+=(const Player& other);
    Player& operator*=(double factor);

    // Clone the player
    Player* clone() const;
};
