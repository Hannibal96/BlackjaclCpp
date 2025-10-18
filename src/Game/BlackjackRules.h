#pragma once
#include "Rules.h"

// Allowed cards for double down
enum class DoubleDownOn {
    ANY,              // Can double on any total
    NINE_TEN_ELEVEN,  // Can only double on 9, 10, or 11
    TEN_ELEVEN        // Can only double on 10 or 11
};

// Surrender options
enum class Surrender {
    NO_SURRENDER,     // Cannot surrender
    SURRENDER_ANY,    // Can surrender on any hand
    SURRENDER_NO_ACE  // Can surrender only if no ace in hand
};

// Blackjack-specific rules struct
struct BlackjackRules : public Rules {
    bool peekBlackjack;         // True = dealer peeks for blackjack (American), False = no peek (European)
    unsigned int maxSplits;     // Maximum number of splits allowed (e.g., 3 means 4 hands total)
    bool doubleAfterSplit;      // Can player double down after split
    bool resplitAces;           // Can player resplit aces
    bool hitSplitAces;          // Can player hit split aces
    Surrender surrender;        // Surrender options
    DoubleDownOn doubleDownOn;  // What totals allow double down
    
    // Default constructor with classic blackjack rules (European style)
    BlackjackRules()
        : Rules(1.5, true, 6, 75.0),  // 3:2 payout, stand soft 17, 6 decks, 75% penetration
          peekBlackjack(false),        // European style - dealer peeks for blackjack
          maxSplits(4),
          doubleAfterSplit(true),
          resplitAces(false),
          hitSplitAces(false),
          surrender(Surrender::NO_SURRENDER),
          doubleDownOn(DoubleDownOn::ANY)
    {}
    
    // Custom constructor for rule variations
    BlackjackRules(double bjPayout, bool standS17, int decks, double pen, bool peek,
                   unsigned int maxSplit, bool doubleAfterSpl, bool resplitAce, bool hitSplitAce,
                   Surrender surr, DoubleDownOn ddOn)
        : Rules(bjPayout, standS17, decks, pen),
          peekBlackjack(peek),
          maxSplits(maxSplit),
          doubleAfterSplit(doubleAfterSpl),
          resplitAces(resplitAce),
          hitSplitAces(hitSplitAce),
          surrender(surr),
          doubleDownOn(ddOn)
    {}
};

