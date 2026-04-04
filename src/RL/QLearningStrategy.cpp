#include "QLearningStrategy.h"

__attribute__((noinline)) double QLearningStrategy::getQValueDebug(const HandType handType, int playerSum, int dealerHand, Action action) const {
    StateKey key = std::make_tuple(0, handType, playerSum, dealerHand);
    auto it = qTable.find(std::make_pair(key, action));
    return (it != qTable.end()) ? it->second : 0.0;
}
