#pragma once

#include "Rules.h"
#include <limits>

// Spanish 21 rules struct. Splitting/DAS/surrender apply much like Blackjack,
// plus Spanish-21-specific fields for the post-double (redouble/rescue)
// mechanics and bonus paytable. blackjackPayout (inherited) defaults to 1.5 —
// natural blackjack always pays 3:2 and always beats a dealer blackjack, so
// there's no push variant to model.
struct SpanishRules : public Rules {
    unsigned int maxSplits;     // Maximum number of splits allowed
    bool doubleAfterSplit;      // Can player double down after split
    bool resplitAces;           // Can player resplit aces
    bool hitSplitAces;          // Can player hit/double split aces (WoO: "usually" allowed)
    bool surrenderAllowed;      // Late surrender, on the first two cards only (not ace-conditional)
    bool peekBlackjack;         // Dealer peeks for blackjack before players act (American-style)

    // Number of times a hand may be REdoubled after its first double (0 = no
    // redoubling; the ordinary single double is always available regardless of
    // this field). E.g. maxRedoubles=1 means: double, then optionally redouble
    // once more, for a maximum of 2 total doublings (4x the original bet).
    // This is the sole on/off switch for redoubling -- there's no separate
    // bool, since 0 already means "off". Verified against WoO's published
    // H17+redouble edge (0.42%) in Spanish21EdgeRegressionTest at both
    // maxRedoubles=1 and 2 -- simulated edge lands within the same tolerance
    // band as the non-redouble cases (~0.17-0.26pp gap, consistent with the
    // documented dropped-footnote-nuance simplification). Defaults to 0
    // because redoubling is still the less commonly offered table variant,
    // not because of any known issue.
    int maxRedoubles;
    bool allowDoubleDownRescue; // Surrender after doubling, forfeiting only the original bet

    bool paySuitedBonus;        // Pay the 6-7-8 / 7-7-7 suited bonus (mixed 3:2, suited 2:1, spades 3:1)

    double minBet = 0.0;
    double maxBet = std::numeric_limits<double>::max();

    // Default constructor: 8 decks, dealer hits soft 17, no redouble, DDR on --
    // the most commonly cited Spanish 21 configuration. WoO lists double-down
    // rescue under "The Rules" (standard, always-on), unlike redoubling which
    // is a separate "Variable Rule" with its own published edge figures -- so
    // WoO's published S17/H17/H17+redouble edges already assume DDR is in
    // play, and this defaults on to match. Its only published AFTER_DOUBLE
    // strategy chart is for H17+redouble specifically though
    // (loadSpanishBasicStrategy() reuses it as the best available reference
    // whenever AFTER_DOUBLE has a real decision to make), not a dedicated
    // chart for this exact maxRedoubles=0/DDR-on combination.
    SpanishRules()
        : Rules(1.5, false, 8, 75.0),
          maxSplits(4),
          doubleAfterSplit(true),
          resplitAces(true),
          hitSplitAces(true),
          surrenderAllowed(true),
          peekBlackjack(true),
          maxRedoubles(0),
          allowDoubleDownRescue(true),
          paySuitedBonus(true)
    {}

    SpanishRules(bool standS17, int decks, double pen, unsigned int maxSplit,
                bool doubleAfterSpl, bool resplitAce, bool hitSplitAce,
                bool surrender, bool peek, int maxRedouble,
                bool ddr, bool suitedBonus)
        : Rules(1.5, standS17, decks, pen),
          maxSplits(maxSplit),
          doubleAfterSplit(doubleAfterSpl),
          resplitAces(resplitAce),
          hitSplitAces(hitSplitAce),
          surrenderAllowed(surrender),
          peekBlackjack(peek),
          maxRedoubles(maxRedouble),
          allowDoubleDownRescue(ddr),
          paySuitedBonus(suitedBonus)
    {}
};
