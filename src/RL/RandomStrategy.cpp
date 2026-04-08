#include "RandomStrategy.h"
#include <random>
#include <stdexcept>

// Constructor with random seed
RandomStrategy::RandomStrategy()
    : rng(std::random_device{}()) {
}

// Constructor with specific seed for reproducibility
RandomStrategy::RandomStrategy(unsigned int seed)
    : rng(seed) {
}

// Get random action from allowed actions (ignores key)
Action RandomStrategy::getAction(const StateKey& /*key*/, const std::vector<Action>& allowedActions) {
    if (allowedActions.empty()) {
        throw std::logic_error("No allowed actions available");
    }

    std::uniform_int_distribution<size_t> dist(0, allowedActions.size() - 1);
    return allowedActions[dist(rng)];
}

std::unique_ptr<Strategy> RandomStrategy::clone() const {
    return std::make_unique<RandomStrategy>();
}
