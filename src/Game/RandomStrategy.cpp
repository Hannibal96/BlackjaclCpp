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

// Get random action from allowed actions
Action RandomStrategy::getAction(const State& state) {
    if (state.allowedActions.empty()) {
        throw std::logic_error("No allowed actions available");
    }
    
    // Generate random index
    std::uniform_int_distribution<size_t> dist(0, state.allowedActions.size() - 1);
    size_t randomIndex = dist(rng);
    
    return state.allowedActions[randomIndex];
}

std::unique_ptr<Strategy> RandomStrategy::clone() const {
    return std::make_unique<RandomStrategy>();
}

