#pragma once
#include "Game/Hand.h"
#include "Action.h"
#include "Shoe/Deck.h"
#include <vector>

// State struct representing the current game state
struct State {
    Hand playerHand;
    Card dealerCard;
    std::vector<Action> allowedActions;
    
    // Constructor with all relevant fields
    State(const Hand& hand, const Card& dealer, const std::vector<Action>& actions)
        : playerHand(hand), dealerCard(dealer), allowedActions(actions) {}
};

