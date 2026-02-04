#pragma once
#include "State.h"
#include "Action.h"
#include <memory>

// Strategy class for decision-making
// This will be implemented in derived classes
class Strategy {
public:
    virtual ~Strategy() = default;
    
    // Pure virtual method to get action based on state
    virtual Action getAction(const State& state) = 0;
    
    // Pure virtual method to clone the strategy
    virtual std::unique_ptr<Strategy> clone() const = 0;
    
    // Virtual method to update strategy table (for learning strategies)
    // Default implementation does nothing for non-learning strategies
    virtual void updateTable(const State& state, Action action, double reward, const State& nextState) {
        // Empty implementation for strategies that don't learn
    }
    
    // Virtual operators for averaging strategies across parallel simulations
    // += operator: accumulate another strategy's values into this one
    virtual Strategy& operator+=(const Strategy& other) {
        // Default implementation does nothing
        return *this;
    }
    
    // * operator: scale strategy values by a factor (for averaging)
    virtual Strategy& operator*=(double factor) {
        // Default implementation does nothing
        return *this;
    }
};

