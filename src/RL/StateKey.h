#pragma once
#include "Game/Hand.h"
#include <tuple>

// StateKey: (discretizedCount, handType, playerSum, dealerCard)
// Produced by Player::stateToKey(), consumed by Strategy methods.
using StateKey = std::tuple<int, HandType, unsigned int, unsigned int>;
