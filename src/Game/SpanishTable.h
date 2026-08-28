#pragma once
#include "Table.h"
#include "SpanishRules.h"
#include "Hand.h"
#include "Slot.h"
#include <map>
#include <vector>
#include <utility>
#include <ostream>

// Concrete Spanish 21 Table implementation.
class SpanishTable : public Table {
private:
    // Rule fields from SpanishRules
    double blackjackPayout;
    bool standSoft17;
    bool peekBlackjack;
    unsigned int maxSplits;
    bool doubleAfterSplit;
    bool resplitAces;
    bool hitSplitAces;
    bool surrenderAllowed;
    int maxRedoubles;  // 0 = redoubling off; see SpanishRules::maxRedoubles
    bool allowDoubleDownRescue;
    bool paySuitedBonus;
    double minBet;
    double maxBet;
    uint64_t round_number;

    // Game state
    Hand dealerHand;
    std::map<Player*, std::vector<Slot>> playerSlots;
    std::map<Player*, double> committedWagers;
    std::vector<double> moneyBeforeScratch;

    // Helper methods
    void clearHands();
    bool shouldDealerHit() const;
    std::vector<Action> getAllowedActions(const Hand& hand, size_t handsInSlot) const;
    void releaseCommittedWager(Player* player, double wager);

    // Payout helpers
    double suitedSequenceMultiplier(const Hand& hand) const;  // 6-7-8 / 7-7-7 bonus tier, 0 if none
    double winMultiplier(const Hand& hand) const;              // multiplier for a hand that wins outright

protected:
    void collectBets() override;
    bool dealInitialCards() override;
    std::vector<std::tuple<Player*, Hand*, State, Action>> playersPlay() override;
    void dealerPlays(DealerAction dealer_action) override;
    void evaluate(const std::vector<std::tuple<Player*, Hand*, State, Action>>& aliveHands) override;

public:
    SpanishTable(const SpanishRules& gameRules, std::vector<Player*> gamePlayers);

    void round() override;

    const Hand& getDealerHand() const { return dealerHand; }

    friend std::ostream& operator<<(std::ostream& os, const SpanishTable& table);
};
