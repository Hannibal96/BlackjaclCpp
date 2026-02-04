#pragma once

#include "Game/BlackjackRules.h"
#include "RL/Action.h"
#include <string>
#include <sstream>
#include <tuple>
#include <vector>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <type_traits>

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

// Helper function to convert string to DoubleDownOn enum
inline DoubleDownOn stringToDoubleOn(const std::string& str) {
    if (str == "ANY") return DoubleDownOn::ANY;
    if (str == "9-11") return DoubleDownOn::NINE_TEN_ELEVEN;
    if (str == "10-11") return DoubleDownOn::TEN_ELEVEN;
    throw std::invalid_argument("Invalid double-on value: " + str);
}

// Helper function to convert string to Surrender enum
inline Surrender stringToSurrender(const std::string& str) {
    if (str == "no") return Surrender::NO_SURRENDER;
    if (str == "yes") return Surrender::SURRENDER_ANY;
    if (str == "2-10") return Surrender::SURRENDER_NO_ACE;
    throw std::invalid_argument("Invalid surrender value: " + str);
}

// Helper function to parse comma-separated list from string like "[1,2,3]" or "1,2,3"
template<typename T>
inline std::vector<T> parseList(const std::string& str) {
    std::vector<T> result;
    std::string cleaned = str;

    // Remove brackets if present
    if (!cleaned.empty() && cleaned.front() == '[') cleaned.erase(0, 1);
    if (!cleaned.empty() && cleaned.back() == ']') cleaned.pop_back();

    std::stringstream ss(cleaned);
    std::string item;

    while (std::getline(ss, item, ',')) {
        // Trim whitespace
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);

        if constexpr (std::is_same_v<T, int>) {
            result.push_back(std::stoi(item));
        } else if constexpr (std::is_same_v<T, float>) {
            result.push_back(std::stof(item));
        } else if constexpr (std::is_same_v<T, bool>) {
            if (item == "true" || item == "1" || item == "True") result.push_back(true);
            else if (item == "false" || item == "0" || item == "False") result.push_back(false);
            else throw std::invalid_argument("Invalid boolean value: " + item);
        } else if constexpr (std::is_same_v<T, std::string>) {
            result.push_back(item);
        }
    }

    return result;
}

// Generate test cases based on configuration vectors
inline std::vector<Case> generateTestCases(
    const std::vector<int>& deckSizes,
    const std::vector<bool>& standSoft17,
    const std::vector<bool>& doubleAfterSplit,
    const std::vector<int>& splitAfterSplit,
    const std::vector<std::string>& doubleOn,
    const std::vector<bool>& reSplitAces,
    const std::vector<bool>& hitSplitAces,
    const std::vector<bool>& peek,
    const std::vector<std::string>& surrender,
    const std::vector<float>& blackjackPay,
    unsigned int seed = 1234)
{
    std::vector<Case> v;
    for (int d : deckSizes)
        for (bool ss17 : standSoft17)
            for (bool das : doubleAfterSplit)
                for (int sas : splitAfterSplit)
                    for (const auto& don_str : doubleOn)
                        for (bool rsa : reSplitAces)
                            for (bool hsa : hitSplitAces)
                                for (bool pk : peek)
                                    for (const auto& surr_str : surrender)
                                        for (float bj : blackjackPay) {
                                            DoubleDownOn don = stringToDoubleOn(don_str);
                                            Surrender surr = stringToSurrender(surr_str);
                                            Case new_case{d, ss17, das, sas, don, rsa, hsa, pk, surr, bj};
                                            v.push_back(new_case);
                                        }
    std::mt19937 rng(seed);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}
