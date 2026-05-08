#pragma once
#include "Strategy.h"
#include "Game/Hand.h"
#include <map>
#include <string>
#include <memory>
#include <utility>
#include <nlohmann/json.hpp>

// Struct to hold an action and its fallback
struct ActionWithFallback {
    Action primary;
    Action fallback;
    
    ActionWithFallback() : primary(Action::HIT), fallback(Action::HIT) {}
    ActionWithFallback(Action p, Action f) : primary(p), fallback(f) {}
    ActionWithFallback(Action p) : primary(p), fallback(p) {}
};

// BasicStrategy class that uses a lookup table from JSON
class BasicStrategy : public Strategy {
private:
    // Nested map structure: count -> hand_type -> player_sum -> dealer_card -> action
    // Using shared_ptr so clones can share the same lookup table (it's read-only during gameplay)
    std::shared_ptr<std::map<int, std::map<HandType, std::map<int, std::map<int, ActionWithFallback>>>>> lookupTable;
    
    // Helper function to convert string to HandType
    HandType stringToHandType(const std::string& handTypeStr) const;

public:
    // Constructor
    BasicStrategy();
    
    // Destructor
    ~BasicStrategy() override = default;
    
    // Load strategy from JSON file
    bool loadFromJson(const std::string& filepath);

    // Load strategy from a project-relative file path.
    bool loadFromFile(const std::string& filepath);
    
    // Get action based on state key and allowed actions (implements pure virtual from Strategy)
    Action getAction(const StateKey& key, const std::vector<Action>& allowedActions) override;

    void printTo(std::ostream& os) const override;
    
    // Clone the strategy
    std::unique_ptr<Strategy> clone() const override;
    
    // Check if the lookup table is loaded
    bool isLoaded() const { return lookupTable && !lookupTable->empty(); }
    
    // Set action in the lookup table for a specific state
    void setAction(int count, HandType handType, unsigned int playerSum, unsigned int dealerHand, ActionWithFallback action);
    
    // Get action directly from lookup table by keys (returns HIT/HIT if not found)
    ActionWithFallback getActionFromTable(int count, HandType handType, int playerSum, int dealerCard) const;

    // Return the minimum and maximum count keys currently present in the table.
    std::pair<int, int> getCountRange() const;

    // Serialize the lookup table to a JSON object (same format as basic_strategy_tables/*.json)
    nlohmann::json toJson() const;

    // Save the lookup table to a JSON file (path is relative to PROJECT_ROOT)
    bool saveToJson(const std::string& filepath) const;
    
    // Averaging operators (no-op for BasicStrategy since it's read-only)
    Strategy& operator+=(const Strategy& other) override {
        // BasicStrategy is read-only, no accumulation needed
        return *this;
    }
    
    Strategy& operator*=(double factor) override {
        // BasicStrategy is read-only, no scaling needed
        return *this;
    }
    
    // Friend function for printing the strategy table
    friend std::ostream& operator<<(std::ostream& os, const BasicStrategy& strategy);
};
