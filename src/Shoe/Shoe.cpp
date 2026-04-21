#include "Shoe.h"
#include <algorithm>
#include <random>
#include <stdexcept>
#include <chrono>
#include <thread>

// Shoe methods
Shoe::Shoe(int numDecks, double penetration) 
    : currentIndex(0), numDecks(numDecks), penetration(penetration / 100.0), endShoe(false),
      rng(static_cast<unsigned int>(
          std::chrono::high_resolution_clock::now().time_since_epoch().count() + 
          std::hash<std::thread::id>{}(std::this_thread::get_id()))) {  // Fast seed using time + thread ID
    
    if (numDecks <= 0) {
        throw std::invalid_argument("Number of decks must be positive");
    }
    
    if (penetration < 0.0 || penetration > 100.0) {
        throw std::invalid_argument("Penetration must be between 0 and 100");
    }
    
    initializeFromDecks();
    calculatePenetrationThreshold();
    shuffle();
}

void Shoe::initializeFromDecks() {
    cards.clear();
    
    // Create num_decks Deck instances and combine all their cards
    for (int i = 0; i < numDecks; ++i) {
        Deck deck;
        const std::vector<Card>& deckCards = deck.getCards();
        cards.insert(cards.end(), deckCards.begin(), deckCards.end());
    }
}

void Shoe::calculatePenetrationThreshold() {
    penetrationThreshold = static_cast<size_t>(cards.size() * penetration);
}

void Shoe::shuffle() {
    // Use the persistent RNG member instead of creating new random_device each time
    std::shuffle(cards.begin(), cards.end(), rng);
    currentIndex = 0;
    endShoe = false;
    removedCards.fill(0);
}

Card Shoe::dealCard(bool covered) {
    if (currentIndex >= cards.size()) {
        reset();
    }

    Card card = cards[currentIndex++];
    if (!covered)
        removedCards[static_cast<int>(card.rank) - 2]++;

    if (currentIndex >= penetrationThreshold)
        endShoe = true;

    return card;
}

void Shoe::uncoverCard(const Card& card) {
    removedCards[static_cast<int>(card.rank) - 2]++;
}

void Shoe::reset() {
    currentIndex = 0;
    endShoe = false;
    shuffle();
}


// DebugShoe methods
DebugShoe::DebugShoe(int numDecks, double penetration, int seed)
    : Shoe(numDecks, penetration), state(static_cast<uint32_t>(seed)), initialSeed(static_cast<uint32_t>(seed)) {
    reset();
}

void DebugShoe::shuffle() {
    // DebugShoe does not shuffle
}

void DebugShoe::reset() {
    initializeFromDecks();
    state = initialSeed;
    currentIndex = 0;
    endShoe = false;
    calculatePenetrationThreshold();
}

Card DebugShoe::dealCard(bool covered) {
    if (cards.empty()) {
        reset();
    }

    size_t n = cards.size();
    size_t index = state % n;

    Card card = cards[index];
    cards.erase(cards.begin() + index);

    state = state * 1664525 + 1013904223;

    if (!covered)
        removedCards[static_cast<int>(card.rank) - 2]++;

    size_t total = numDecks * 52;
    if (cards.size() <= total - penetrationThreshold)
        endShoe = true;

    return card;
}
