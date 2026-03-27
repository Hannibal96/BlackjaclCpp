#include "Shoe.h"
#include "../Utils/GlobalRNG.h"
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
}

Card Shoe::dealCard() {
    if (currentIndex >= cards.size()) {
        reset();
        // throw std::runtime_error("No more cards in the shoe");
    }
    
    Card card = cards[currentIndex++];
    
    // Check if we've reached the penetration threshold
    if (currentIndex >= penetrationThreshold) {
        endShoe = true;
    }
    
    return card;
}

void Shoe::reset() {
    currentIndex = 0;
    endShoe = false;
    shuffle();
}


// DebugShoe methods

// Internal RNG constructor (backward compatible)
DebugShoe::DebugShoe(int numDecks, double penetration, int seed)
    : Shoe(numDecks, penetration), state(static_cast<uint32_t>(seed)),
      initialSeed(static_cast<uint32_t>(seed)), externalRng(nullptr) {
    reset();
}

// External shared RNG constructor — cross-reproducible with Python SeededRNG
DebugShoe::DebugShoe(int numDecks, double penetration, GlobalRNG* rng)
    : Shoe(numDecks, penetration), state(0), initialSeed(0), externalRng(rng) {
    initializeFromDecks();
    calculatePenetrationThreshold();
    // No shuffle, rng is not reset — matches Python DebugShoe.__init__
}

void DebugShoe::shuffle() {
    // DebugShoe does not shuffle
}

void DebugShoe::reset() {
    initializeFromDecks();
    calculatePenetrationThreshold();
    currentIndex = 0;
    endShoe = false;
    if (externalRng == nullptr) {
        state = initialSeed;  // Only reset internal state; external rng keeps advancing
    }
    // External rng intentionally not reset — matches Python remake()
}

Card DebugShoe::dealCard() {
    if (cards.empty()) {
        reset();
    }

    size_t n = cards.size();
    size_t index;

    if (externalRng) {
        index = static_cast<size_t>(externalRng->nextInt(static_cast<int>(n)));
    } else {
        index = state % n;
        state = state * 1664525u + 1013904223u;
    }

    Card card = cards[index];
    cards.erase(cards.begin() + index);

    size_t totalCards = numDecks * 52;
    if (cards.size() <= totalCards - penetrationThreshold) {
        endShoe = true;
    }

    return card;
}
