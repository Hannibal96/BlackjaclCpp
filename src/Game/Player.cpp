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

// Update the player's money
void Player::updateMoney(double amount) {
    money += amount;
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

// Clone the player
Player* Player::clone() const {
    if (!strategy) {
        throw std::logic_error("Cannot clone player without strategy");
    }
    return new Player(money, strategy->clone());
}

