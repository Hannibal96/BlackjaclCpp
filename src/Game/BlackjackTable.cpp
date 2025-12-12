#include "BlackjackTable.h"
#include "BlackjackRules.h"
#include "RL/State.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>

// Constructor
BlackjackTable::BlackjackTable(const BlackjackRules& gameRules, std::vector<Player*> gamePlayers)
    : Table(gameRules, std::move(gamePlayers)),
      blackjackPayout(gameRules.blackjackPayout),
      standSoft17(gameRules.standSoft17),
      peekBlackjack(gameRules.peekBlackjack),
      maxSplits(gameRules.maxSplits),
      doubleAfterSplit(gameRules.doubleAfterSplit),
      resplitAces(gameRules.resplitAces),
      hitSplitAces(gameRules.hitSplitAces),
      surrender(gameRules.surrender),
      doubleDownOn(gameRules.doubleDownOn),
      round_number(0ULL)
{
    // Initialize player slots map
    for (auto* player : players) {
        playerSlots[player] = std::vector<Slot>();
    }
}

// Main round flow
void BlackjackTable::round() {
    round_number++;
    // Check if shoe needs reshuffle
    if (shoe->isEndShoe()) {
        shoe->reset();
    }
    
    clearHands();
    collectBets();
    
    // Deal initial cards - returns true if round is over
    bool delaer_bj = dealInitialCards();
    
    std::vector<std::pair<Player*, Hand*>> aliveHands;
    
    // Only continue with player actions if there is no dealer bj
    if (!delaer_bj) {
        aliveHands = playersPlay();
    } else {
        for (auto* player : players) {
            for (auto& slot : playerSlots[player]) {
                for (auto& hand : slot.getHands()) {
                    aliveHands.push_back({player, &hand});
                }
            }
        }
    }

    bool all_players_bj = std::all_of(aliveHands.begin(), aliveHands.end(), [](auto& p) { return p.second->isBlackjack(); });
    bool dealer_has_chance_bj = dealerHand[0].getValue() == 10 || dealerHand[0].getValue() == 11;

    DealerAction dealer_action;
    if(aliveHands.empty()){
        dealer_action = DealerAction::SKIP;
    } else if(all_players_bj) {
        if(dealer_has_chance_bj && !peekBlackjack){
            dealer_action = DealerAction::CHECK_BLACKJACK;
        } else {
            dealer_action = DealerAction::SKIP;
        }
    } else{ 
        dealer_action = DealerAction::PLAY;
    }

    dealerPlays(dealer_action);
    
    // Evaluate alive hands and process payments
    evaluate(aliveHands);
}

// Clear all hands from previous round
void BlackjackTable::clearHands() {
    dealerHand.clear();
    for (auto& pair : playerSlots) {
        pair.second.clear();
    }
}

// Collect bets from all players
void BlackjackTable::collectBets() {
    for (auto* player : players) {
        double bet = player->getBet();
        playerSlots[player].emplace_back(bet);
    }
}

// Deal initial two cards to each player and dealer
// Returns true if round is over (dealer blackjack), keeping the possibility to double down on blackjack for later
bool BlackjackTable::dealInitialCards() {
    // First card to each slot
    for (auto* player : players) {
        for (auto& slot : playerSlots[player]) {
            Card card = shoe->dealCard();
            slot[0].addCard(card);
        }
    }
    
    dealerHand.addCard(shoe->dealCard());
    
    // Second card to each slot
    for (auto* player : players) {
        for (auto& slot : playerSlots[player]) {
            Card card = shoe->dealCard();
            slot[0].addCard(card);
        }
    }
    
    // American vs European blackjack
    if (peekBlackjack) {
        dealerHand.addCard(shoe->dealCard());
    }

    // TODO: Implement Insurance logic
    return dealerHand.isBlackjack();
}

// Let each player play their hands
// Returns alive hands (not bust, not surrendered)
std::vector<std::pair<Player*, Hand*>> BlackjackTable::playersPlay() {
    std::vector<std::pair<Player*, std::pair<size_t, size_t>>> aliveHandIndices;
    Card dealerUpCard = dealerHand[0];
    
    // Iterate over each slot for each player
    for (auto* player : players) {
        std::vector<Slot>& slots = playerSlots[player];
        for (size_t slotIdx = 0; slotIdx < slots.size(); ++slotIdx) {
            Slot& slot = slots[slotIdx];
            // Iterate over each hand in the slot
            for (size_t handIdx = 0; handIdx < slot.getHands().size(); ++handIdx) {
                while(true) {
                    // Refresh hand reference each iteration to avoid invalidation
                    Hand& hand = slot[handIdx];
                    // Calculate allowed actions based on rules
                    std::vector<Action> allowedActions = getAllowedActions(hand, slot.getHands().size());
                    
                    State state(hand, dealerUpCard, allowedActions);
                    Action action = player->getAction(state);
                    
                    if (action == Action::HIT) {
                        hand.addCard(shoe->dealCard());
                        if (hand.isBust()) {
                            player->updateMoney(-hand.getBet());
                            break;  
                        }
                    }
                    else if (action == Action::STAND) {
                        aliveHandIndices.push_back({player, {slotIdx, handIdx}});
                        break;
                    }
                    else if (action == Action::DOUBLE_DOWN) {
                        hand.multiplyBet(2.0f);
                        hand.addCard(shoe->dealCard());
                        if (hand.isBust()) {
                            player->updateMoney(-hand.getBet());
                        }
                        aliveHandIndices.push_back({player, {slotIdx, handIdx}});
                        break;
                    }
                    else if (action == Action::SPLIT) {
                        // Split the hand - new hand is added to the slot
                        Hand newHand = hand.split();
                        slot.addHand(newHand);
                        // Continue playing the current hand
                    }
                    // TODO: move surrender to pre action
                    else if (action == Action::SURRENDER) {
                        double originalBet = hand.getBet();
                        hand.multiplyBet(0.5f);
                        double lossAmount = hand.getBet();
                        player->updateMoney(-lossAmount);
                        break;  
                    }
                    else {
                        throw std::invalid_argument("Invalid action");
                    }
                }
            }
        }
    }
    
    // Convert indices to pointers now that all vector modifications are complete
    std::vector<std::pair<Player*, Hand*>> aliveHands;
    for (const auto& [player, indices] : aliveHandIndices) {
        size_t slotIdx = indices.first;
        size_t handIdx = indices.second;
        aliveHands.push_back({player, &playerSlots[player][slotIdx][handIdx]});
    }
    
    return aliveHands;
}

// Dealer plays according to rules
void BlackjackTable::dealerPlays(DealerAction dealer_action) {
    if(dealer_action == DealerAction::SKIP)
        return;
    while (shouldDealerHit()) {
        dealerHand.addCard(shoe->dealCard());
        if(dealer_action == DealerAction::CHECK_BLACKJACK){
            break;
        }
    }
}

// Check if dealer should hit based on rules
bool BlackjackTable::shouldDealerHit() const {
    int dealerValue = dealerHand.getValue();
    if (dealerValue < 17) {
        return true;
    }
    // Check if dealer should hit on soft 17
    if (dealerValue == 17 && !standSoft17 && dealerHand.isSoft()) {
        return true;
    }
    return false;
}

// Calculate allowed actions based on hand and table rules
std::vector<Action> BlackjackTable::getAllowedActions(const Hand& hand, size_t handsInSlot) const {
    if (handsInSlot > maxSplits){
        throw std::logic_error("Cannot have more hands than maxSplit");
    }
    std::vector<Action> actions;

    if(hand.cardCount() == 1) {
        actions.push_back(Action::HIT);
        return actions;
    }   

    actions.push_back(Action::HIT);
    actions.push_back(Action::STAND);
    
    if(hand.cardCount() > 2) {        
        return actions;
    }

    bool allow_double = doubleDownOn == DoubleDownOn::ANY || 
        ((doubleDownOn == DoubleDownOn::NINE_TEN_ELEVEN) && hand.isHard() && (9 <= hand.getValue() && hand.getValue() <= 11)) || 
        ((doubleDownOn == DoubleDownOn::TEN_ELEVEN) && hand.isHard() && (10 <= hand.getValue() && hand.getValue() <= 11)) ;

    if(hand.getIsSplit()){
        if((!hitSplitAces) && (hand[0].getValue() == 11)){
            actions = std::vector<Action>{Action::STAND};
        } 
        // Check if more splits are allowed based on maxSplits
        if(handsInSlot < maxSplits && hand.isPair()){
            if(hand[0].getValue() != 11){
                actions.push_back(Action::SPLIT);
            } else if(resplitAces){
                actions.push_back(Action::SPLIT);
            }
        } 
        if(doubleAfterSplit && allow_double){
            if(hand[0].getValue() != 11){
                actions.push_back(Action::DOUBLE_DOWN);
            } else if(hitSplitAces){
                actions.push_back(Action::DOUBLE_DOWN);
            }
        }
        return actions;
    } 

    if(allow_double){
        actions.push_back(Action::DOUBLE_DOWN);
    }
    if(hand.isPair()){
        actions.push_back(Action::SPLIT);
    }

    bool allow_surrender = surrender == Surrender::SURRENDER_ANY || 
                        (surrender == Surrender::SURRENDER_NO_ACE && (dealerHand[0].getValue() != 11));
    
    if(allow_surrender){
        actions.push_back(Action::SURRENDER);
    }

    return actions;
}

// Evaluate alive hands and process payments
void BlackjackTable::evaluate(const std::vector<std::pair<Player*, Hand*>>& aliveHands) {
    int dealerValue = dealerHand.getValue();
    bool dealerBusted = dealerHand.isBust();
    bool dealerBlackjack = dealerHand.isBlackjack();
    
    // Only evaluate hands that are still alive (not bust, not surrendered)
    for (const auto& [player, hand] : aliveHands) {
        int playerValue = hand->getValue();
        bool playerBlackjack = hand->isBlackjack();
        double bet = hand->getBet();
        
        if (playerBlackjack && dealerBlackjack) {
            continue;
        }
        
        if (playerBlackjack) {
            player->updateMoney(bet * blackjackPayout);
            continue;
        }
        
        if (dealerBlackjack) {
            player->updateMoney(-bet);
            continue;
        }
        
        if (dealerBusted) {
            player->updateMoney(bet);
            continue;
        }
        
        if (playerValue > dealerValue) {
            player->updateMoney(bet);
        }
        else if (playerValue < dealerValue) {
            player->updateMoney(-bet);
        }
        // else: playerValue == dealerValue - PUSH (no money exchanged)
    }
}


std::ostream& operator<<(std::ostream& os, const BlackjackTable& table){
    os << "*========* Round number: " << table.round_number << " *========* " << std::endl;
    os << "Dealer: " << table.dealerHand << std::endl;

    for(const auto& [player, slots] : table.playerSlots){
        os << "Player Name: " << player->getName() << ", Money: " << player->getMoney() << std::endl;
        for(size_t i = 0; i < slots.size(); ++i){
            os << "Slot " << (i + 1) << ": " << std::endl << slots[i] << std::endl;
        }
    }
    return os;
}