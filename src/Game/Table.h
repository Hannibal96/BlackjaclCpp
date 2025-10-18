#pragma once
#include "Rules.h"
#include "Player.h"
#include "Hand.h"
#include "../Shoe/Shoe.h"
#include <vector>
#include <memory>
#include <utility>

// Abstract Table class for managing game flow
class Table {
protected:
    enum class DealerAction {
        SKIP,
        CHECK_BLACKJACK,
        PLAY
    };  

    std::vector<Player*> players;
    std::unique_ptr<Shoe> shoe;
    
    // Protected helper methods for subclasses
    virtual void collectBets() = 0;
    virtual bool dealInitialCards() = 0;  // Returns true if round is over (dealer BJ or all players BJ)
    virtual std::vector<std::pair<Player*, Hand*>> playersPlay() = 0;  // Returns alive hands (not bust, not surrendered)
    virtual void dealerPlays(DealerAction dealer_action) = 0;
    virtual void evaluate(const std::vector<std::pair<Player*, Hand*>>& aliveHands) = 0;
    
public:
    // Constructor - receives rules, creates shoe, does NOT store rules
    Table(const Rules& gameRules, std::vector<Player*> gamePlayers);
    
    // Virtual destructor
    virtual ~Table() = default;
    
    // Main game round - manages complete round flow
    virtual void round() = 0;
    
    // Get players
    const std::vector<Player*>& getPlayers() const { return players; }
    
    // Get shoe
    Shoe* getShoe() const { return shoe.get(); }
};

