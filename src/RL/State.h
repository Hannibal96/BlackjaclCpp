#pragma once
#include "Game/Hand.h"
#include "Action.h"
#include "Shoe/Deck.h"
#include <vector>
#include <array>

// State struct representing the current game state as seen by the Player.
// Strategy methods never receive this directly — Player converts it to a StateKey first.
struct State {
    Hand playerHand;
    Card dealerCard;
    std::vector<Action> allowedActions;
    std::array<int, 13> removedCards{};  // Dealt card counts per rank (index 0=TWO ... 12=ACE)

    State(const Hand& hand, const Card& dealer, const std::vector<Action>& actions,
          const std::array<int, 13>& removed = {})
        : playerHand(hand), dealerCard(dealer), allowedActions(actions), removedCards(removed) {}
};
