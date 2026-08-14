#pragma once

#include "Rules.h"
#include <limits>
#include <stdexcept>

enum class DoubleDownMadnessVersion {
    VERSION_1 = 1,
    VERSION_2 = 2,
    VERSION_3 = 3
};

struct DoubleDownMadnessRules : public Rules {
    DoubleDownMadnessVersion version = DoubleDownMadnessVersion::VERSION_1;
    double minBet = 0.0;
    double maxBet = std::numeric_limits<double>::max();

    DoubleDownMadnessRules()
        : Rules(1.5, false, 6, 75.0) {}

    DoubleDownMadnessRules(DoubleDownMadnessVersion gameVersion,
                           bool standS17,
                           int decks,
                           double penetration)
        : Rules(1.5, standS17, decks, penetration),
          version(gameVersion) {
        if (version != DoubleDownMadnessVersion::VERSION_1 &&
            version != DoubleDownMadnessVersion::VERSION_2 &&
            version != DoubleDownMadnessVersion::VERSION_3) {
            throw std::invalid_argument("Double Down Madness version must be 1, 2, or 3");
        }
    }

    double blackjackPayoutFor(bool suited) const {
        switch (version) {
            case DoubleDownMadnessVersion::VERSION_1:
                return suited ? 2.0 : 1.5;
            case DoubleDownMadnessVersion::VERSION_2:
                return 1.5;
            case DoubleDownMadnessVersion::VERSION_3:
                return suited ? 3.0 : 1.0;
        }
        throw std::logic_error("Unknown Double Down Madness version");
    }
};
