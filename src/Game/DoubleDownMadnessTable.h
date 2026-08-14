#pragma once

#include "DoubleDownMadnessRules.h"
#include "Hand.h"
#include "Table.h"
#include <map>
#include <ostream>
#include <tuple>
#include <vector>

class DoubleDownMadnessTable : public Table {
private:
    bool standSoft17;
    double minBet;
    double maxBet;
    double suitedBlackjackPayout;
    double unsuitedBlackjackPayout;
    uint64_t roundNumber = 0;

    Hand dealerHand;
    std::map<Player*, Hand> playerHands;
    std::map<Player*, double> committedWagers;
    std::vector<double> moneyBeforeScratch;

    void clearHands();
    bool shouldDealerHit() const;
    std::vector<Action> getAllowedActions(const Hand& hand) const;
    bool isSuitedBlackjack(const Hand& hand) const;
    double blackjackPayout(const Hand& hand) const;

protected:
    void collectBets() override;
    bool dealInitialCards() override;
    std::vector<std::tuple<Player*, Hand*, State, Action>> playersPlay() override;
    void dealerPlays(DealerAction dealerAction) override;
    void evaluate(
        const std::vector<std::tuple<Player*, Hand*, State, Action>>& aliveHands) override;

public:
    DoubleDownMadnessTable(const DoubleDownMadnessRules& rules,
                           std::vector<Player*> players);

    void round() override;

    const Hand& getDealerHand() const { return dealerHand; }

    friend std::ostream& operator<<(std::ostream& os,
                                    const DoubleDownMadnessTable& table);
};
