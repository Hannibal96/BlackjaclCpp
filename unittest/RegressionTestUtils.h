#pragma once

#include "Game/BlackjackRules.h"
#include "RL/Action.h"
#include <string>
#include <sstream>
#include <tuple>

// Common test case structure for regression tests
struct Case {
    int deckSize;
    bool standSoft17;
    bool doubleAfterSplit;
    int splitAfterSplit;
    DoubleDownOn doubleOn;
    bool reSplitAces;
    bool hitSplitAces;
    bool peek;
    Surrender surrender;
    float blackJackPay;
};

// Common result structure
struct RegressionResult {
    double empiric_edge;
    double time_sec;
};

// Convert Case to tuple for comparison
inline auto as_tuple(const Case& s) {
    return std::tie(s.deckSize, s.standSoft17, s.doubleAfterSplit, s.splitAfterSplit, 
                    s.doubleOn, s.reSplitAces, s.hitSplitAces, s.peek, s.surrender, s.blackJackPay);
}

// Calculate distance between two cases (number of differing fields)
inline int caseDistance(const Case& x, const Case& y) {
    auto tx = as_tuple(x);
    auto ty = as_tuple(y);

    std::size_t n = 0;

    std::apply([&](auto&&... xs) {
        std::apply([&](auto&&... ys) {
            ((n += (xs != ys ? 1 : 0)), ...);
        }, ty);
    }, tx);

    return n;
}

// Convert Case to table name string (for JSON file lookup)
inline std::string ToStringTableName(const Case& c) {
    std::ostringstream os;
    os << "decks=" << (c.deckSize > 4 ? 4 : c.deckSize)
       << "_ss17=" << (c.standSoft17 ? "True" : "False")
       << "_das=" << (c.doubleAfterSplit ? "True" : "False")
       << "_surr=" << (c.surrender == Surrender::NO_SURRENDER ? "no" : c.surrender == Surrender::SURRENDER_ANY ? "yes" : "2-10")
       << "_peek=" << (c.peek ? "True" : "False");
    return os.str();
}

// Convert Case to full string representation
inline std::string ToString(const Case& c) {
    std::ostringstream os;
    os << "decks=" << c.deckSize
       << "_ss17=" << (c.standSoft17 ? "True" : "False")
       << "_das=" << (c.doubleAfterSplit ? "True" : "False")
       << "_sas=" << c.splitAfterSplit
       << "_don=" << (c.doubleOn == DoubleDownOn::ANY ? "Any" : c.doubleOn == DoubleDownOn::NINE_TEN_ELEVEN ? "9-11" : "10-11")
       << "_reSplitAces=" << (c.reSplitAces ? "True" : "False")
       << "_hitSplitAces=" << (c.hitSplitAces ? "True" : "False")
       << "_surr=" << (c.surrender == Surrender::NO_SURRENDER ? "no" : c.surrender == Surrender::SURRENDER_ANY ? "yes" : "2-10")
       << "_peek=" << (c.peek ? "True" : "False") 
       << "_bj-pays=" << c.blackJackPay;
    return os.str();
}

// Convert action string from JSON to Action enum
inline Action stringToAction(const std::string& actionStr) {
    if (actionStr == "H") return Action::HIT;
    if (actionStr == "S") return Action::STAND;
    if (actionStr == "D" || actionStr == "Dh" || actionStr == "Ds") return Action::DOUBLE_DOWN;
    if (actionStr == "P") return Action::SPLIT;
    if (actionStr == "X" || actionStr == "Xh" || actionStr == "Xs") return Action::SURRENDER;
    return Action::HIT;  // Default
}
