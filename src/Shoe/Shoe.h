#pragma once
#include "Deck.h"
#include <vector>
#include <random>

// Shoe class for managing multiple decks of cards
class Shoe {
protected:
    std::vector<Card> cards;
    size_t currentIndex;
    int numDecks;
    double penetration;  // Percentage (0.0 to 1.0)
    size_t penetrationThreshold;  // Card index where penetration is reached
    bool endShoe;
    std::mt19937 rng;  // Persistent RNG for shuffling (seeded once)
    
    void initializeFromDecks();
    void calculatePenetrationThreshold();
    
public:
    // Constructor: numDecks - number of decks to use, penetration - percentage of cards to deal (0-100)
    Shoe(int numDecks, double penetration);
    
    // Shuffle the shoe
    virtual void shuffle();
    
    // Deal a card from the shoe
    virtual Card dealCard();
    
    // Check if the penetration threshold has been reached
    bool isEndShoe() const { return endShoe; }
    
    // Get number of cards remaining in the shoe
    size_t cardsRemaining() const { return cards.size() - currentIndex; }
    
    // Get total number of cards in the shoe
    size_t totalCards() const { return cards.size(); }
    
    // Reset the shoe (reshuffle all cards)
    virtual void reset();
};

class DebugShoe : public Shoe {
protected:
    uint32_t state;
    uint32_t initialSeed;

public:
    DebugShoe(int numDecks, double penetration, int seed = 42);
    
    void shuffle() override;
    Card dealCard() override;
    void reset() override;
};

