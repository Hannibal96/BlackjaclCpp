#pragma once

// Shared formatting helpers for Table verbose mode (see Table::verbose):
// a compact human-readable trace of each hand decision and each hand's final
// outcome, used identically by BlackjackTable, DoubleDownMadnessTable, and
// SpanishTable so the three tables' debug output stays consistent.

#include "Hand.h"
#include "Player.h"
#include "RL/Action.h"
#include "../Shoe/Deck.h"
#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>

namespace VerboseTrace {

inline std::string formatHand(const Hand& hand) {
    std::ostringstream oss;
    for (size_t i = 0; i < hand.cardCount(); ++i) {
        if (i) oss << " ";
        oss << hand[i];
    }
    oss << " (";
    switch (hand.getHandType()) {
        case HandType::BLACKJACK:         oss << "BLACKJACK"; break;
        case HandType::AFTER_DOUBLE:      oss << "AFTER_DOUBLE " << hand.getValue(); break;
        case HandType::AFTER_DOUBLE_SOFT: oss << "AFTER_DOUBLE_SOFT " << hand.getValue(); break;
        // Single-card value, not hand total, so A,A (PAIR 11) and T,T (PAIR 10)
        // display distinctly -- matches StateKey's own pair convention (see
        // Player::stateToKey) instead of Hand::getValue()'s soft-adjusted total,
        // which would otherwise show both as "PAIR 12" (A+A softened to 12).
        case HandType::PAIR:              oss << "PAIR " << hand[0].getValue(); break;
        case HandType::SOFT:              oss << "SOFT " << hand.getValue(); break;
        case HandType::HARD:              oss << "HARD " << hand.getValue(); break;
        case HandType::ZOMBIE:            oss << "1 card"; break;
    }
    oss << ")";
    return oss.str();
}

inline void printDecision(std::ostream& out, uint64_t roundNumber, const Player& player,
                           const Hand& hand, const Card& dealerUpCard, int trueCount,
                           Action action) {
    out << "[round " << roundNumber << "] " << player.getName()
        << " count=" << (trueCount > 0 ? "+" : "") << trueCount
        << " | " << formatHand(hand) << " vs dealer " << dealerUpCard
        << " -> " << action << "\n";
}

inline void printOutcome(std::ostream& out, uint64_t roundNumber, const Player& player,
                          const Hand& hand, const Hand& dealerHand, double reward) {
    const char* outcome = reward > 0.0 ? "WIN" : (reward < 0.0 ? "LOSE" : "PUSH");
    out << "[round " << roundNumber << "] " << player.getName()
        << " final " << formatHand(hand) << " vs dealer " << formatHand(dealerHand)
        << " => " << outcome << " " << (reward >= 0 ? "+" : "") << reward
        << " | balance=" << player.getMoney() << "\n";
}

} // namespace VerboseTrace
