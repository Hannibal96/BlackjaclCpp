#pragma once
#include "Table.h"
#include "BlackjackRules.h"
#include "Hand.h"
#include <map>
#include <vector>
#include <utility>
#include <ostream>


// Concrete Blackjack Table implementation
class BlackjackTable : public Table {
private:

    // Rule fields from BlackjackRules
    double blackjackPayout;
    bool standSoft17;
    bool peekBlackjack;
    unsigned int maxSplits;
    bool doubleAfterSplit;
    bool resplitAces;
    bool hitSplitAces;
    Surrender surrender;
    DoubleDownOn doubleDownOn;
    uint64_t round_number;
    
    // Game state
    Hand dealerHand;
    std::map<Player*, std::vector<Hand>> playerHands;   // TODO: replace this with slots for enabling to track the number of splits
    
    // Helper methods
    void clearHands();
    bool shouldDealerHit() const;
    std::vector<Action> getAllowedActions(const Hand& hand) const;
    
protected:
    // Implement abstract methods from Table
    void collectBets() override;
    bool dealInitialCards() override;  // Returns true if round is over (dealer BJ or all players BJ)
    std::vector<std::pair<Player*, Hand*>> playersPlay() override;  // Returns alive hands (not bust, not surrendered)
    void dealerPlays(DealerAction dealer_action) override;
    void evaluate(const std::vector<std::pair<Player*, Hand*>>& aliveHands) override;
    
public:
    // Constructor - creates shoe based on rules
    BlackjackTable(const BlackjackRules& gameRules, std::vector<Player*> gamePlayers);
    
    // Implement round method
    void round() override;
    
    // Get dealer hand (for testing/display)
    const Hand& getDealerHand() const { return dealerHand; }

    friend std::ostream& operator<<(std::ostream& os, const BlackjackTable& table);
};
