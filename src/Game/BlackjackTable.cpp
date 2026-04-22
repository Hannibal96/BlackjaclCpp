#include "BlackjackTable.h"
#include "BlackjackRules.h"
#include "RL/State.h"
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <tuple>

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
    if (shoe->isEndShoe()) shoe->reset();

    // Capture pre-round state for regression (must happen after a potential reset
    // so the shoe state is fresh/accurate, and before any cards are dealt).
    bool anyRegression = false;
    for (auto* p : players) if (p->isRegressionEnabled()) { anyRegression = true; break; }

    std::array<int, 13> removedBefore{};
    double remainingDecksBefore = 0.0;
    std::vector<double> moneyBefore;
    if (anyRegression) {
        removedBefore = shoe->getRemovedCards();
        remainingDecksBefore = shoe->cardsRemaining() / 52.0;
        moneyBefore.resize(players.size());
        for (size_t i = 0; i < players.size(); ++i)
            moneyBefore[i] = players[i]->getMoney();
    }

    clearHands();
    collectBets();
    
    // Deal initial cards - returns true if round is over
    bool delaer_bj = dealInitialCards();
    
    std::vector<std::tuple<Player*, Hand*, State, Action>> aliveHands;
    
    // Only continue with player actions if there is no dealer bj
    if (!delaer_bj) {
        aliveHands = playersPlay();
    } else {
        // Dealer has blackjack (peek already resolved before players act).
        // Players never made a decision, so we update money only — no Q-table update.
        // Player BJ vs dealer BJ is a push; all other hands lose their bet.
        for (auto* player : players) {
            for (auto& slot : playerSlots[player]) {
                for (auto& hand : slot.getHands()) {
                    double reward = hand.isBlackjack() ? 0.0 : -hand.getBet();
                    player->addReward(reward);
                }
            }
        }
        // aliveHands stays empty → dealerPlays will DealerAction::SKIP
    }

    bool all_players_bj = std::all_of(aliveHands.begin(), aliveHands.end(), [](auto& h) { return std::get<1>(h)->isBlackjack(); });
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

    // Update regression accumulators with this round's outcome
    if (anyRegression) {
        std::array<double, 13> x;
        for (int i = 0; i < 13; ++i)
            x[i] = (remainingDecksBefore > 0.0)
                    ? static_cast<double>(removedBefore[i]) / remainingDecksBefore
                    : 0.0;
        for (size_t i = 0; i < players.size(); ++i)
            players[i]->recordRound(x, players[i]->getMoney() - moneyBefore[i]);
    }
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
        double bet = player->getBet(shoe->getRemovedCards());
        playerSlots[player].emplace_back(bet, maxSplits);
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
    
    // American style: deal the dealer's hole card face-down so card counters
    // cannot observe it. uncoverCard() is called at the start of dealerPlays().
    if (peekBlackjack) {
        dealerHand.addCard(shoe->dealCard(/*covered=*/true));
    }

    // TODO: Implement Insurance logic
    return dealerHand.isBlackjack();
}

// Let each player play their hands
// Returns alive hands (not bust, not surrendered) with their state and action
std::vector<std::tuple<Player*, Hand*, State, Action>> BlackjackTable::playersPlay() {
    std::vector<std::tuple<Player*, std::pair<size_t, size_t>, State, Action>> aliveHandIndices;
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
                    
                    State state(hand, dealerUpCard, allowedActions, shoe->getRemovedCards());
                    Action action = player->getAction(state);

                    if (action == Action::HIT) {
                        hand.addCard(shoe->dealCard());
                        if (hand.isBust()) {
                            // Create terminal state (bust) with empty allowed actions
                            State nextState(hand, dealerUpCard, {}, shoe->getRemovedCards());
                            double reward = -hand.getBet();
                            player->updateMoney(reward, state, action, nextState);
                            break;
                        } else {
                            // Non-terminal state - player continues with updated hand
                            std::vector<Action> nextAllowedActions = getAllowedActions(hand, slot.getHands().size());
                            State nextState(hand, dealerUpCard, nextAllowedActions, shoe->getRemovedCards());
                            player->updateMoney(0.0, state, action, nextState);
                        }
                    }
                    else if (action == Action::STAND) {
                        aliveHandIndices.push_back({player, {slotIdx, handIdx}, state, action});
                        break;
                    }
                    else if (action == Action::DOUBLE_DOWN) {
                        hand.multiplyBet(2.0f);
                        hand.addCard(shoe->dealCard());
                        if (hand.isBust()) {
                            // Create terminal state (bust) with empty allowed actions
                            State nextState(hand, dealerUpCard, {}, shoe->getRemovedCards());
                            double reward = -hand.getBet();
                            player->updateMoney(reward, state, action, nextState);
                        } else {
                            // Non-bust double down - still alive
                            aliveHandIndices.push_back({player, {slotIdx, handIdx}, state, action});
                        }
                        break;
                    }
                    else if (action == Action::SPLIT) {
                        // Split the hand - new hand is added to the slot
                        Hand newHand = hand.split();
                        slot.addHand(newHand);
                        // Update with reward 0 - split creates two hands with equal expected value
                        std::vector<Action> nextAllowedActions = getAllowedActions(hand, slot.getHands().size());
                        State nextState(hand, dealerUpCard, nextAllowedActions, shoe->getRemovedCards());
                        player->updateMoney(0.0, state, action, nextState);
                        // Continue playing the current hand
                    }
                    // TODO: move surrender to pre action
                    else if (action == Action::SURRENDER) {
                        double originalBet = hand.getBet();
                        hand.multiplyBet(0.5f);
                        double lossAmount = hand.getBet();
                        // Create terminal state (surrendered) with empty allowed actions
                        State nextState(hand, dealerUpCard, {}, shoe->getRemovedCards());
                        double reward = -lossAmount;
                        player->updateMoney(reward, state, action, nextState);
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
    std::vector<std::tuple<Player*, Hand*, State, Action>> aliveHands;
    for (const auto& [player, indices, state, action] : aliveHandIndices) {
        size_t slotIdx = indices.first;
        size_t handIdx = indices.second;
        aliveHands.push_back({player, &playerSlots[player][slotIdx][handIdx], state, action});
    }
    
    return aliveHands;
}

// Dealer plays according to rules
void BlackjackTable::dealerPlays(DealerAction dealer_action) {
    // In peek games the hole card was dealt covered. Reveal it now — regardless of
    // dealer_action — so the running count is accurate for all subsequent rounds.
    if (peekBlackjack)
        shoe->uncoverCard(dealerHand[1]);

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
void BlackjackTable::evaluate(const std::vector<std::tuple<Player*, Hand*, State, Action>>& aliveHands) {
    int dealerValue = dealerHand.getValue();
    bool dealerBusted = dealerHand.isBust();
    bool dealerBlackjack = dealerHand.isBlackjack();
    Card dealerUpCard = dealerHand[0];
    
    // Only evaluate hands that are still alive (not bust, not surrendered)
    for (const auto& [player, hand, state, action] : aliveHands) {
        int playerValue = hand->getValue();
        bool playerBlackjack = hand->isBlackjack();
        double bet = hand->getBet();
        
        // Create terminal next state after dealer plays
        State nextState(*hand, dealerUpCard, {}, shoe->getRemovedCards());
        
        if (playerBlackjack && dealerBlackjack) {
            // Push - no reward
            player->updateMoney(0.0, state, action, nextState);
            continue;
        }
        
        if (playerBlackjack) {
            double reward = bet * blackjackPayout;
            player->updateMoney(reward, state, action, nextState);
            continue;
        }
        
        if (dealerBlackjack) {
            double reward = -bet;
            player->updateMoney(reward, state, action, nextState);
            continue;
        }
        
        if (dealerBusted) {
            double reward = bet;
            player->updateMoney(reward, state, action, nextState);
            continue;
        }
        
        if (playerValue > dealerValue) {
            double reward = bet;
            player->updateMoney(reward, state, action, nextState);
        }
        else if (playerValue < dealerValue) {
            double reward = -bet;
            player->updateMoney(reward, state, action, nextState);
        }
        else {
            // Push - no reward
            player->updateMoney(0.0, state, action, nextState);
        }
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