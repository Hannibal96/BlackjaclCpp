#pragma once
#include "Rules.h"
#include "Player.h"
#include "Hand.h"
#include "../Shoe/Shoe.h"
#include "RL/State.h"
#include "RL/Action.h"
#include <vector>
#include <memory>
#include <utility>
#include <tuple>
#include <ostream>

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

    // When true, concrete round() implementations print a per-decision and
    // per-outcome trace to *verboseOut. Debug-only: no effect on simulation
    // results, intended for single-threaded runs (DebugPlayerBehavior app).
    bool verbose = false;
    std::ostream* verboseOut = nullptr;

    // Protected helper methods for subclasses
    virtual void collectBets() = 0;
    virtual bool dealInitialCards() = 0;  // Returns true if round is over (dealer BJ or all players BJ)
    virtual std::vector<std::tuple<Player*, Hand*, State, Action>> playersPlay() = 0;  // Returns alive hands with state and action
    virtual void dealerPlays(DealerAction dealer_action) = 0;
    virtual void evaluate(const std::vector<std::tuple<Player*, Hand*, State, Action>>& aliveHands) = 0;
    
public:
    // Constructor - receives rules, creates shoe, does NOT store rules.
    // spanishDeck=true builds the shoe from 48-card Spanish decks (no rank-TEN
    // cards) instead of standard 52-card decks; used by SpanishTable.
    Table(const Rules& gameRules, std::vector<Player*> gamePlayers, bool spanishDeck = false);
    
    // Virtual destructor
    virtual ~Table() = default;
    
    // Main game round - manages complete round flow
    virtual void round() = 0;
    
    // Get players
    const std::vector<Player*>& getPlayers() const { return players; }
    
    // Get shoe
    Shoe* getShoe() const { return shoe.get(); }

    // Enable/disable the per-decision/per-outcome debug trace (see `verbose` above).
    void setVerbose(bool v, std::ostream* out = nullptr) {
        verbose = v;
        verboseOut = v ? out : nullptr;
    }
};

