#include "WalkerStrategy.h"
#include "SpanishStrategy.h"
#include <vector>

std::unique_ptr<BasicStrategy> expandSpanishBasicStrategyAcrossCounts(
        const BasicStrategy& base, int minCount, int maxCount) {
    auto strategy = std::make_unique<BasicStrategy>();
    for (int count = minCount; count <= maxCount; ++count) {
        for (int dealer = 2; dealer <= 11; ++dealer) {
            for (int cardCount = 2; cardCount <= 6; ++cardCount) {
                for (int hard = 4; hard <= 21; ++hard)
                    strategy->setAction(count, HandType::HARD, hard, dealer,
                        base.getActionFromTable(0, HandType::HARD, hard, dealer, cardCount),
                        cardCount);
                for (int soft = 12; soft <= 21; ++soft)
                    strategy->setAction(count, HandType::SOFT, soft, dealer,
                        base.getActionFromTable(0, HandType::SOFT, soft, dealer, cardCount),
                        cardCount);
                for (int total = 4; total <= 21; ++total)
                    strategy->setAction(count, HandType::AFTER_DOUBLE, total, dealer,
                        base.getActionFromTable(0, HandType::AFTER_DOUBLE, total, dealer, cardCount),
                        cardCount);
                for (int total = 12; total <= 21; ++total)
                    strategy->setAction(count, HandType::AFTER_DOUBLE_SOFT, total, dealer,
                        base.getActionFromTable(0, HandType::AFTER_DOUBLE_SOFT, total, dealer, cardCount),
                        cardCount);
            }
            // A pair is always exactly 2 cards -- no cardCount dimension to sweep.
            for (int pairValue = 2; pairValue <= 11; ++pairValue)
                strategy->setAction(count, HandType::PAIR, pairValue, dealer,
                    base.getActionFromTable(0, HandType::PAIR, pairValue, dealer, 2), 2);
        }
    }
    return strategy;
}

namespace {

struct WalkerThreshold {
    int index;  // true count >= index triggers `action`
    ActionWithFallback action;
};

struct WalkerDeviation {
    HandType handType;
    int playerSum;   // or single-card value for PAIR
    int dealerCard;
    ActionWithFallback floorAction;              // action below every threshold
    std::vector<WalkerThreshold> thresholds;      // ascending index order
    const char* label;                            // for printWalkerDeviationsList()
};

// Katarina Walker's published Spanish 21 index deviations. Applied uniformly
// across every cardCount (2-6): Walker's table is card-count-agnostic (a
// human counter tracks total-vs-dealer, not how many cards it took to get
// there), so a deviation fires the same way regardless of hand composition.
// "Forfeit" = double-down rescue (Action::SURRENDER at the AFTER_DOUBLE node).
const std::vector<WalkerDeviation>& walkerDeviations() {
    static const std::vector<WalkerDeviation> table = {
        {HandType::HARD, 16, 10, ActionWithFallback(Action::HIT),
            {{-2, ActionWithFallback(Action::SURRENDER, Action::HIT)},
             { 3, ActionWithFallback(Action::STAND)}},
            "16 vs X: Surrender -2, Stand +3"},
        {HandType::HARD, 17, 11, ActionWithFallback(Action::HIT),
            {{-6, ActionWithFallback(Action::SURRENDER, Action::HIT)},
             {-3, ActionWithFallback(Action::STAND)}},
            "17 vs A: Surrender -6, Stand -3"},
        {HandType::AFTER_DOUBLE, 17, 11, ActionWithFallback(Action::STAND),
            {{2, ActionWithFallback(Action::SURRENDER, Action::STAND)}},
            "doubled 17 vs A: Forfeit +2"},
        {HandType::SOFT, 17, 4, ActionWithFallback(Action::HIT),
            {{-5, ActionWithFallback(Action::DOUBLE_DOWN, Action::HIT)}},
            "A,6 vs 4: Double -5"},
        {HandType::SOFT, 18, 4, ActionWithFallback(Action::HIT),
            {{-5, ActionWithFallback(Action::DOUBLE_DOWN, Action::HIT)}},
            "A,7 vs 4: Double -5"},
        {HandType::PAIR, 8, 9, ActionWithFallback(Action::HIT),
            {{-11, ActionWithFallback(Action::SPLIT, Action::HIT)}},
            "8,8 vs 9: Split -11"},
        {HandType::HARD, 9, 6, ActionWithFallback(Action::HIT),
            {{-3, ActionWithFallback(Action::DOUBLE_DOWN, Action::HIT)}},
            "9 vs 6: Double -3"},
        {HandType::HARD, 11, 9, ActionWithFallback(Action::HIT),
            {{-5, ActionWithFallback(Action::DOUBLE_DOWN, Action::HIT)}},
            "11 vs 9: Double -5"},
    };
    return table;
}

} // namespace

std::unique_ptr<BasicStrategy> backfillMissingEntries(
        const BasicStrategy& learned, const BasicStrategy& fallback, int minCount, int maxCount) {
    auto strategy = std::make_unique<BasicStrategy>();
    auto copyCell = [&](int count, HandType handType, int sum, int dealer, int cardCount) {
        const BasicStrategy& source = learned.hasEntry(count, handType, sum, dealer, cardCount)
            ? learned : fallback;
        strategy->setAction(count, handType, sum, dealer,
                            source.getActionFromTable(count, handType, sum, dealer, cardCount), cardCount);
    };
    for (int count = minCount; count <= maxCount; ++count) {
        for (int dealer = 2; dealer <= 11; ++dealer) {
            for (int cardCount = 2; cardCount <= 6; ++cardCount) {
                for (int hard = 4; hard <= 21; ++hard)
                    copyCell(count, HandType::HARD, hard, dealer, cardCount);
                for (int soft = 12; soft <= 21; ++soft)
                    copyCell(count, HandType::SOFT, soft, dealer, cardCount);
                for (int total = 4; total <= 21; ++total)
                    copyCell(count, HandType::AFTER_DOUBLE, total, dealer, cardCount);
                for (int total = 12; total <= 21; ++total)
                    copyCell(count, HandType::AFTER_DOUBLE_SOFT, total, dealer, cardCount);
            }
            for (int pairValue = 2; pairValue <= 11; ++pairValue)
                copyCell(count, HandType::PAIR, pairValue, dealer, 2);
        }
    }
    return strategy;
}

void applyWalkerDeviations(BasicStrategy& strategy, int minCount, int maxCount) {
    for (const auto& dev : walkerDeviations()) {
        for (int count = minCount; count <= maxCount; ++count) {
            ActionWithFallback chosen = dev.floorAction;
            for (const auto& t : dev.thresholds) {
                if (count >= t.index) chosen = t.action;
            }
            for (int cardCount = 2; cardCount <= 6; ++cardCount) {
                strategy.setAction(count, dev.handType, dev.playerSum, dev.dealerCard,
                                   chosen, cardCount);
            }
        }
    }
}

std::unique_ptr<BasicStrategy> loadWalkerBasicStrategy(
        const SpanishRules& rules, int minCount, int maxCount) {
    auto base = loadSpanishBasicStrategy(rules);
    auto strategy = expandSpanishBasicStrategyAcrossCounts(*base, minCount, maxCount);
    applyWalkerDeviations(*strategy, minCount, maxCount);
    return strategy;
}

void printWalkerDeviationsList(std::ostream& os) {
    os << "Katarina Walker's Spanish 21 index deviations (true-counted Hi-Lo, "
          "running count starts at -4/deck):\n";
    for (const auto& dev : walkerDeviations())
        os << "  " << dev.label << "\n";
}
