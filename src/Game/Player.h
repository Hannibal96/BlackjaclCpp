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
#include <optional>
#include <vector>
#include <algorithm>
#include <cmath>

struct KellyExposureStatistics {
    static constexpr double kHistogramBinWidth = 0.001;
    static constexpr double kHistogramMaximum = 1.0;
    static constexpr size_t kRegularHistogramBins = 1000;
    static constexpr size_t kHistogramBins = kRegularHistogramBins + 1; // last is >= 1
    static constexpr size_t kWagerMultipleBins = 33; // last is >= 32
    static constexpr std::array<double, 9> kTailThresholds = {
        0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0
    };

    uint64_t rounds = 0;
    uint64_t validBankrollRounds = 0;
    uint64_t invalidBankrollRounds = 0;
    uint64_t zeroWagerRounds = 0;
    uint64_t invalidLogRounds = 0;
    double grossExposureSum = 0.0;
    double grossExposureMaximum = 0.0;
    double absoluteReturnSum = 0.0;
    double absoluteReturnMaximum = 0.0;
    double signedReturnSum = 0.0;
    double signedReturnSquaredSum = 0.0;
    double exactLogIncrementSum = 0.0;
    double quadraticLogIncrementSum = 0.0;
    double taylorErrorSum = 0.0;
    double absoluteTaylorErrorSum = 0.0;
    double absoluteTaylorErrorMaximum = 0.0;
    std::array<uint64_t, kHistogramBins> grossExposureHistogram{};
    std::array<uint64_t, kHistogramBins> absoluteReturnHistogram{};
    std::array<uint64_t, kWagerMultipleBins> wagerMultipleHistogram{};
    std::array<uint64_t, kTailThresholds.size()> grossExposureTailCounts{};
    std::array<uint64_t, kTailThresholds.size()> absoluteReturnTailCounts{};

    void record(double bankrollBeforeRound,
                double initialWager,
                double totalWager,
                double roundProfit);
    KellyExposureStatistics& operator+=(const KellyExposureStatistics& other);
};

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

    // Betting strategy — nullptr means fixed unit bet of 1.0
    std::unique_ptr<BettingStrategy> bettingStrategy;

    // When enabled, actions requiring another wager are removed if the
    // bankroll is already committed to the current round. Kelly simulations
    // enable this; training and fixed/spread edge simulations leave it off.
    bool enforceBankrollActionLimits = false;
    // Maximum cumulative gross wager in one round, expressed as a fraction of
    // bankroll immediately before the initial wager. This is deliberately
    // cumulative: wagers remain part of the limit after a hand busts or settles.
    double maximumTotalWagerFraction = 1.0;

    // Betting signal = countBias + countFactor * selectedCount. Usually this is EV;
    // quadratic-Kelly counts fit the wager fraction directly.
    // For OLS-derived systems: countFactor=1.0, countBias=w[13] (weights already in EV units).
    // For traditional systems (Hi-Lo etc.): countFactor≈0.005, countBias≈-(house edge).
    double countFactor = 1.0;
    double countBias   = 0.0;
    bool continuousBettingCount = false;
    CountNormalization countNormalization = CountNormalization::TRUE_COUNT;
    double initialCount = 0.0;
    double initialCountPerDeck = 0.0;

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

    bool kellyExposureStatsEnabled = false;
    KellyExposureStatistics kellyExposureStats;

public:
    // Constructor
    Player(double initialMoney, std::unique_ptr<Strategy> strat, std::string name = "Uzan");

    // Virtual destructor for proper cleanup
    virtual ~Player() = default;

    // Convert a raw State to a StateKey by computing and discretizing the count
    StateKey stateToKey(const State& state) const;

    // Get action based on game state (delegates to strategy via StateKey)
    virtual Action getAction(const State& state);
    Action getAction(const State& state, const std::vector<Action>& allowedActions);

    // Get bet amount — receives current shoe removed-cards for count-aware sizing.
    // Currently always returns 1.0; override or extend for count-based betting.
    virtual double getBet(const std::array<int, 13>& removedCards);

    double countValue(const std::array<int, 13>& removedCards) const;
    double bettingSignal(const std::array<int, 13>& removedCards) const;

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
    void setMaximumTotalWagerFraction(double fraction);
    double getMaximumTotalWagerFraction() const {
        return maximumTotalWagerFraction;
    }
    bool canAffordAdditionalWager(double additionalWager,
                                  double committedWager) const;
    bool canAffordAdditionalWager(double additionalWager,
                                  double committedWager,
                                  double cumulativeWager,
                                  double bankrollBeforeRound) const;

    // Set E[game] model parameters individually or from a CountingSystem.
    void setCountFactor(double f) { countFactor = f; }
    void setCountBias(double b)   { countBias   = b; }
    void setCountSystem(const CountingSystem& cs) {
        setCountWeights(cs.weights);
        setCountFactor(cs.factor);
        setCountBias(cs.bias);
        continuousBettingCount = cs.continuousBettingCount;
        countNormalization = cs.normalization;
        initialCount = cs.initialCount;
        initialCountPerDeck = cs.initialCountPerDeck;
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

    void enableKellyExposureStats() { kellyExposureStatsEnabled = true; }
    bool isKellyExposureStatsEnabled() const { return kellyExposureStatsEnabled; }
    void recordKellyExposure(double bankrollBeforeRound,
                             double initialWager,
                             double totalWager,
                             double roundProfit);
    const KellyExposureStatistics& getKellyExposureStats() const {
        return kellyExposureStats;
    }

    // Accumulate one round: x = pre-round normalized removed-cards, y = round net outcome
    void recordRound(const std::array<double, 13>& x,
                     double y,
                     std::optional<double> observedCount = std::nullopt);

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
