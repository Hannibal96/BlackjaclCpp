#pragma once
#include <vector>
#include <ostream>
#include "Hand.h"

// Slot class representing a betting position that can contain multiple hands from splits
class Slot {
private:
    std::vector<Hand> hands;
    int splitCount;  // Number of splits that have occurred
    
public:
    // Constructor
    explicit Slot(double betAmount, size_t maxHands = 4);
    
    // Add a hand to the slot (used after a split)
    void addHand(const Hand& hand);
    
    // Access hand by index
    Hand& operator[](size_t index);
    const Hand& operator[](size_t index) const;
    
    // Get all hands in the slot
    std::vector<Hand>& getHands() { return hands; }
    const std::vector<Hand>& getHands() const { return hands; }
    
    friend std::ostream& operator<<(std::ostream& os, const Slot& slot);
};
