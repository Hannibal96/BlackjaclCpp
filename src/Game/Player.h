#pragma once
#include "RL/State.h"
#include "RL/Action.h"
#include "RL/Strategy.h"
#include "RL/StateKey.h"
#include <array>
#include <memory>
#include <algorithm>
#include <cmath>

// Player class - can be used directly or inherited for advanced behavior
class Player {
protected:
    double money;
    std::string name;
    std::unique_ptr<Strategy> strategy;

    // Card counting fields
    std::array<double, 13> countWeights{};  // Weights per rank (index 0=TWO ... 12=ACE); default all zeros
    double countResolution = 1.0;           // Discretization step for the true count
    int minCount = 0;                       // Lower clamp bound for the discretized count
    int maxCount = 0;                       // Upper clamp bound for the discretized count
    int numDecks = 1;                       // Total decks in the shoe (needed to compute remaining decks)

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
    void resetPlayer(double money = 0);

    // Get current money
    double getMoney() const { return money; }

    std::string getName() const { return name; }

    // Set strategy
    void setStrategy(std::unique_ptr<Strategy> strat);

    // Get strategy (const access)
    const Strategy* getStrategy() const { return strategy.get(); }

    // Get mutable strategy access (for averaging operations)
    Strategy* getMutableStrategy() { return strategy.get(); }

    // Set card counting weights and resolution
    void setCountWeights(const std::array<double, 13>& weights, double resolution = 1.0);

    // Set the count range — the discretized count is clamped to [min, max]
    void setCountRange(int min, int max);

    // Set the total number of decks in the shoe (needed for true-count normalization)
    void setNumDecks(int decks);

    // Averaging operators for combining players from parallel simulations
    Player& operator+=(const Player& other);
    Player& operator*=(double factor);

    // Clone the player
    Player* clone() const;
};
