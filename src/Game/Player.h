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

    // Betting strategy — nullptr means fixed unit bet of 1.0
    std::unique_ptr<BettingStrategy> bettingStrategy;

    // E[game] model parameters: E[game] = countBias + countFactor * trueCount
    // For OLS-derived systems: countFactor=1.0, countBias=w[13] (weights already in EV units).
    // For traditional systems (Hi-Lo etc.): countFactor≈0.005, countBias≈-(house edge).
    double countFactor = 1.0;
    double countBias   = 0.0;

    // OLS regression accumulation:  w = (X^T X)^-1 X^T y
    // X row: [removedCards[i]/remainingDecks (13-dim), 1.0 bias]  (14-dim total)
    // y: net outcome of the round summed over all hands
    bool regressionEnabled = false;
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

    // Get action based on game state (delegates to strategy via StateKey)
    virtual Action getAction(const State& state);

    // Get bet amount — receives current shoe removed-cards for count-aware sizing.
    // Currently always returns 1.0; override or extend for count-based betting.
    virtual double getBet(const std::array<int, 13>& removedCards);

    // Update the player's money with SARS parameters for learning strategies
    void updateMoney(double reward, const State& state, Action action, const State& nextState);

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

    // Set E[game] model parameters individually or from a CountingSystem.
    void setCountFactor(double f) { countFactor = f; }
    void setCountBias(double b)   { countBias   = b; }
    void setCountSystem(const CountingSystem& cs) {
        setCountWeights(cs.weights);
        setCountFactor(cs.factor);
        setCountBias(cs.bias);
    }

    // Set card counting weights (rank tags only; resolution is separate)
    void setCountWeights(const std::array<double, 13>& weights);

    // Set true-count discretization step (used both for strategy lookup and bet sizing)
    void setCountResolution(double resolution) { countResolution = resolution; }

    // Set the count range — the discretized count is clamped to [min, max]
    void setCountRange(int min, int max);

    // Set the total number of decks in the shoe (needed for true-count normalization)
    void setNumDecks(int decks);

    // --- Regression ---
    void enableRegression() { regressionEnabled = true; }
    bool isRegressionEnabled() const { return regressionEnabled; }
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
