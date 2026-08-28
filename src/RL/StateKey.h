#pragma once
#include "Game/Hand.h"
#include <tuple>

// StateKey: (discretizedCount, handType, playerSum, dealerCard, cardCount)
// Produced by Player::stateToKey(), consumed by Strategy methods.
// cardCount is the number of cards in the player's hand, capped at 6. Games
// that don't care about it (Blackjack, Double Down Madness) always report
// the constant 2 here (see Player::trackHandCardCount) so their behavior,
// JSON tables, and regression tests are unaffected. Spanish 21 reports the
// real (capped) count, since bonus payouts on 21 depend on how many cards
// were used to reach it.
using StateKey = std::tuple<int, HandType, unsigned int, unsigned int, unsigned int>;
