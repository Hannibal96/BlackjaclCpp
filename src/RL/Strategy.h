#pragma once
#include "StateKey.h"
#include "Action.h"
#include <memory>
#include <vector>

// Strategy class for decision-making.
// Operates on StateKey (produced by Player) and allowedActions — never on raw State.
class Strategy {
public:
    virtual ~Strategy() = default;

    // Pure virtual method to get action given a resolved state key and allowed actions
    virtual Action getAction(const StateKey& key, const std::vector<Action>& allowedActions) = 0;

    // Pure virtual method to clone the strategy
    virtual std::unique_ptr<Strategy> clone() const = 0;

    // Virtual method to update strategy table (for learning strategies).
    // nextAllowedActions is needed to compute max Q-value for the next state.
    virtual void updateTable(const StateKey& currentKey, Action action, double reward,
                             const StateKey& nextKey, const std::vector<Action>& nextAllowedActions) {
        // Empty implementation for strategies that don't learn
    }

    // Virtual operators for averaging strategies across parallel simulations
    virtual Strategy& operator+=(const Strategy& other) {
        return *this;
    }

    virtual Strategy& operator*=(double factor) {
        return *this;
    }
};
