#pragma once

// Base Rules struct for all game variants
struct Rules {
    double blackjackPayout;  // Blackjack payout multiplier (e.g., 1.5 for 3:2, 1.2 for 6:5)
    bool standSoft17;        // True if dealer stands on soft 17, false if dealer hits
    int numDecks;            // Number of decks in the shoe
    double penetration;      // Penetration percentage (0-100)
    
    Rules(double bjPayout, bool standS17, int decks, double pen)
        : blackjackPayout(bjPayout), standSoft17(standS17), numDecks(decks), penetration(pen) {}
};

