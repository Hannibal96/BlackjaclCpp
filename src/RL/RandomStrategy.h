#pragma once
#include "Strategy.h"
#include <random>

// Random strategy that randomly selects from allowed actions
class RandomStrategy : public Strategy {
private:
    mutable std::mt19937 rng;

public:
    // Constructor
    RandomStrategy();

    // Constructor with seed for reproducibility
    explicit RandomStrategy(unsigned int seed);

    // Get random action from allowed actions (ignores key)
    Action getAction(const StateKey& key, const std::vector<Action>& allowedActions) override;

    // Clone the strategy
    std::unique_ptr<Strategy> clone() const override;
};

