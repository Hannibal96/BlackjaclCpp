#pragma once
#include <vector>
#include <utility>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <sstream>

// Pre-computed values passed from Player to BettingStrategy each round.
struct BettingContext {
    double trueCount;     // dot(countWeights, removedCards) / remainingDecks
    double expectedValue; // EV estimate, or a directly fitted wager fraction for quadratic Kelly
    double bankroll;      // player's current money
};

class BettingStrategy {
public:
    virtual ~BettingStrategy() = default;
    virtual double getBet(const BettingContext& ctx) const = 0;
    virtual BettingStrategy* clone() const = 0;
};

// Step-function bet sizing keyed on true count.
// Format: sorted (threshold, betAmount) pairs.
// Semantics: return the bet for the highest threshold that the count strictly exceeds.
//   "1:10"        → count > 1: bet 10; count <= 1: return 0 (table clamps to minBet)
//   "1:1,2:4,3:8" → count > 3: bet 8; count > 2: bet 4; count > 1: bet 1; else 0
// The table is responsible for enforcing min/max bet limits.
class SpreadBetting : public BettingStrategy {
    std::vector<std::pair<double, double>> steps_; // sorted ascending by threshold

public:
    explicit SpreadBetting(std::vector<std::pair<double, double>> steps)
        : steps_(std::move(steps))
    {
        std::sort(steps_.begin(), steps_.end());
        if (steps_.empty())
            throw std::invalid_argument("SpreadBetting: spread map cannot be empty");
    }

    // Parse compact format: "1:1,2:4,3:8,4:12"  (threshold:bet,...)
    static SpreadBetting fromString(const std::string& s) {
        std::vector<std::pair<double, double>> steps;
        std::istringstream stream(s);
        std::string token;
        while (std::getline(stream, token, ',')) {
            size_t colon = token.find(':');
            if (colon == std::string::npos)
                throw std::invalid_argument("SpreadBetting: expected 'threshold:bet', got '" + token + "'");
            double threshold = std::stod(token.substr(0, colon));
            double bet       = std::stod(token.substr(colon + 1));
            steps.emplace_back(threshold, bet);
        }
        return SpreadBetting(std::move(steps));
    }

    double getBet(const BettingContext& ctx) const override {
        // Iterate from highest threshold down; return the bet for the first threshold exceeded.
        for (auto it = steps_.rbegin(); it != steps_.rend(); ++it)
            if (ctx.trueCount >= it->first)
                return it->second;
        return 0.0; // count below all thresholds; table clamps to minBet
    }

    BettingStrategy* clone() const override {
        return new SpreadBetting(steps_);
    }
};

// Kelly criterion: bet = bankroll * kellyFraction * max(0, E[game]).
// E[game] is the expected net outcome per unit wagered.
// The table is responsible for enforcing min/max bet limits.
class KellyBetting : public BettingStrategy {
    double kellyFraction_; // scale applied to the Kelly fraction (default 1.0)

public:
    explicit KellyBetting(double kellyFraction = 1.0)
        : kellyFraction_(kellyFraction) {}

    double getBet(const BettingContext& ctx) const override {
        if (ctx.bankroll <= 0.0) return 0.0;
        double f = std::max(0.0, ctx.expectedValue) * kellyFraction_;
        return ctx.bankroll * f;
    }

    BettingStrategy* clone() const override {
        return new KellyBetting(kellyFraction_);
    }
};
