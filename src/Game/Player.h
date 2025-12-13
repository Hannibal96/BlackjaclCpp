#pragma once
#include "RL/State.h"
#include "RL/Action.h"
#include "RL/Strategy.h"
#include <memory>

// Player class - can be used directly or inherited for advanced behavior
class Player {
protected:
    double money;
    std::string name;
    std::unique_ptr<Strategy> strategy;
    
public:
    // Constructor
    Player(double initialMoney, std::unique_ptr<Strategy> strat, std::string name = "Uzan");
    
    // Virtual destructor for proper cleanup
    virtual ~Player() = default;
    
    // Get action based on game state (delegates to strategy)
    virtual Action getAction(const State& state);
    
    // Get bet amount (returns 1, can be overridden for advanced betting strategies)
    virtual double getBet();
    
    // Update the player's money with SARS parameters for learning strategies
    void updateMoney(double reward, const State& state, Action action, const State& nextState);
    void resetPlayer(double moeny = 0);
    
    // Get current money
    double getMoney() const { return money; }

    std::string getName() const { return name; }
    
    // Set strategy
    void setStrategy(std::unique_ptr<Strategy> strat);
    
    // Get strategy (const access)
    const Strategy* getStrategy() const { return strategy.get(); }
    
    // Get mutable strategy access (for averaging operations)
    Strategy* getMutableStrategy() { return strategy.get(); }
    
    // Averaging operators for combining players from parallel simulations
    Player& operator+=(const Player& other);
    Player& operator*=(double factor);
    
    // Clone the player
    Player* clone() const;
};

