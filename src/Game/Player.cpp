#include "Player.h"
#include <stdexcept>

// Constructor
Player::Player(double initialMoney, std::unique_ptr<Strategy> strat, std::string name)
    : money(initialMoney), strategy(std::move(strat)), name(name) {
    if (!strategy) {
        throw std::invalid_argument("Player must have a strategy");
    }
}

// Get action based on game state (delegates to strategy)
Action Player::getAction(const State& state) {
    return strategy->getAction(state);
}

// Get bet amount (returns 1, can be overridden for advanced betting strategies)
double Player::getBet() {
    return 1.0;
}

// Update the player's money with SARS parameters for learning strategies
void Player::updateMoney(double reward, const State& state, Action action, const State& nextState) {
    money += reward;
    // Update strategy table (for learning strategies like Q-learning)
    strategy->updateTable(state, action, reward, nextState);
}

void Player::resetPlayer(double moeny){
    money = 0;
}

// Set strategy
void Player::setStrategy(std::unique_ptr<Strategy> strat) {
    if (!strat) {
        throw std::invalid_argument("Player must have a strategy");
    }
    strategy = std::move(strat);
}

// Averaging operators for combining players from parallel simulations
Player& Player::operator+=(const Player& other) {
    // Sum the money
    money += other.money;
    
    // Apply += to strategy
    if (strategy && other.strategy) {
        *strategy += *other.strategy;
    }
    
    return *this;
}

Player& Player::operator*=(double factor) {
    // Scale the money
    money *= factor;
    
    // Apply *= to strategy
    if (strategy) {
        *strategy *= factor;
    }
    
    return *this;
}

// Clone the player
Player* Player::clone() const {
    if (!strategy) {
        throw std::logic_error("Cannot clone player without strategy");
    }
    return new Player(money, strategy->clone());
}

