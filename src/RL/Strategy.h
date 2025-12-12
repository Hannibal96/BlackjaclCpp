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
};

