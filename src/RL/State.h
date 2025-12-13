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
    int count;
    
    // Constructor with all relevant fields (count defaults to 0)
    State(const Hand& hand, const Card& dealer, const std::vector<Action>& actions, int count = 0)
        : playerHand(hand), dealerCard(dealer), allowedActions(actions), count(count) {}
};

