#include "Slot.h"
#include <stdexcept>

// Constructor with bet amount
Slot::Slot(double betAmount) : splitCount(0) {
    hands.emplace_back(betAmount);
}

// Add a hand to the slot (used after a split)
void Slot::addHand(const Hand& hand) {
    hands.push_back(hand);
    splitCount++;
}

// Access hand by index
Hand& Slot::operator[](size_t index) {
    if (index >= hands.size()) {
        throw std::out_of_range("Hand index out of range");
    }
    return hands[index];
}

// Access hand by index (const version)
const Hand& Slot::operator[](size_t index) const {
    if (index >= hands.size()) {
        throw std::out_of_range("Hand index out of range");
    }
    return hands[index];
}

// Output operator for Slot
std::ostream& operator<<(std::ostream& os, const Slot& slot) {
    for (auto &hand: slot.hands) {
        os << hand << std::endl;
    }
    return os;
}
