#include "SpanishTable.h"
#include "RL/State.h"
#include "VerboseTrace.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <tuple>

namespace {

// Bet-size normalization for Q-learning, needed once a hand's wager can compound
// across multiple decision points (redoubling) — same mechanism DoubleDownMadnessTable
// uses. A no-op in the common case (no redouble/rescue) since bet size is then constant
// across a hand's decisions.
double learningWager(const State& state) {
    return state.playerHand.getBet() > 0.0 ? state.playerHand.getBet() : 1.0;
}

} // namespace

SpanishTable::SpanishTable(const SpanishRules& gameRules, std::vector<Player*> gamePlayers)
    : Table(gameRules, std::move(gamePlayers), /*spanishDeck=*/true),
      blackjackPayout(gameRules.blackjackPayout),
      standSoft17(gameRules.standSoft17),
      peekBlackjack(gameRules.peekBlackjack),
      maxSplits(gameRules.maxSplits),
      doubleAfterSplit(gameRules.doubleAfterSplit),
      resplitAces(gameRules.resplitAces),
      hitSplitAces(gameRules.hitSplitAces),
      surrenderAllowed(gameRules.surrenderAllowed),
      maxRedoubles(gameRules.maxRedoubles),
      allowDoubleDownRescue(gameRules.allowDoubleDownRescue),
      paySuitedBonus(gameRules.paySuitedBonus),
      minBet(gameRules.minBet),
      maxBet(gameRules.maxBet),
      round_number(0ULL)
{
    moneyBeforeScratch.resize(players.size());
    for (auto* player : players) {
        playerSlots[player] = std::vector<Slot>();
        committedWagers[player] = 0.0;
    }
}

void SpanishTable::round() {
    round_number++;
    if (shoe->isEndShoe()) shoe->reset();

    bool anyRoundTracking = false;
    bool anyFeatureTracking = false;
    for (auto* p : players) {
        if (p->isRegressionEnabled() || p->isCountGraphEnabled()) {
            anyRoundTracking = true;
            anyFeatureTracking = true;
        }
        if (p->isRoundStatsEnabled()) anyRoundTracking = true;
    }

    std::array<int, 13> removedBefore{};
    double remainingDecksBefore = 0.0;
    if (anyRoundTracking) {
        if (anyFeatureTracking) {
            removedBefore = shoe->getRemovedCards();
            remainingDecksBefore = shoe->cardsRemaining() / 52.0;
        }
        for (size_t i = 0; i < players.size(); ++i)
            moneyBeforeScratch[i] = players[i]->getMoney();
    }

    clearHands();
    collectBets();

    bool dealerBJ = dealInitialCards();

    std::vector<std::tuple<Player*, Hand*, State, Action>> aliveHands;
    if (dealerBJ) {
        // Player blackjack still beats a dealer blackjack in Spanish 21 (pays out,
        // not a push); every other hand loses. Players never made a decision.
        for (auto* player : players) {
            for (auto& slot : playerSlots[player]) {
                for (auto& hand : slot.getHands()) {
                    double reward = hand.isBlackjack() ? hand.getBet() * blackjackPayout
                                                        : -hand.getBet();
                    player->addReward(reward);
                    if (verbose) {
                        VerboseTrace::printOutcome(*verboseOut, round_number, *player, hand,
                                                    dealerHand, reward);
                    }
                }
            }
        }
    } else {
        aliveHands = playersPlay();
    }

    // A hand that's already blackjack or a plain 21 always wins regardless of
    // the dealer's total (see evaluate()) -- even against a dealer blackjack,
    // unlike classic blackjack. So once every alive hand is one of those, the
    // dealer never needs to draw beyond the initial deal; there's no push
    // ambiguity left to resolve, so (unlike BlackjackTable) there's no need
    // for a peek-style CHECK_BLACKJACK probe here either.
    bool allHandsAlreadyWin = std::all_of(aliveHands.begin(), aliveHands.end(),
        [](auto& h) {
            const Hand* hand = std::get<1>(h);
            return hand->isBlackjack() || hand->getValue() == 21;
        });

    DealerAction dealer_action =
        (aliveHands.empty() || allHandsAlreadyWin) ? DealerAction::SKIP : DealerAction::PLAY;

    dealerPlays(dealer_action);
    evaluate(aliveHands);

    if (anyRoundTracking) {
        std::array<double, 13> x{};
        if (anyFeatureTracking) {
            for (int i = 0; i < 13; ++i) {
                x[i] = remainingDecksBefore > 0.0
                    ? static_cast<double>(removedBefore[i]) / remainingDecksBefore
                    : 0.0;
            }
        }
        for (size_t i = 0; i < players.size(); ++i) {
            const double reward = players[i]->getMoney() - moneyBeforeScratch[i];
            players[i]->recordRoundOutcome(reward);
            if (players[i]->isRegressionEnabled() || players[i]->isCountGraphEnabled())
                players[i]->recordRound(x, reward);
        }
    }
}

void SpanishTable::clearHands() {
    dealerHand.clear();
    for (auto& [player, slots] : playerSlots) {
        slots.clear();
        committedWagers[player] = 0.0;
    }
}

void SpanishTable::collectBets() {
    for (auto* player : players) {
        double bet = std::clamp(player->getBet(shoe->getRemovedCards()), minBet, maxBet);
        playerSlots[player].emplace_back(bet, maxSplits);
        if (player->shouldEnforceBankrollActionLimits())
            committedWagers[player] = bet;
    }
}

bool SpanishTable::dealInitialCards() {
    for (auto* player : players)
        for (auto& slot : playerSlots[player])
            slot[0].addCard(shoe->dealCard());

    dealerHand.addCard(shoe->dealCard());

    for (auto* player : players)
        for (auto& slot : playerSlots[player])
            slot[0].addCard(shoe->dealCard());

    // American style: hole card dealt face-down so card counters can't observe it.
    // uncoverCard() is called at the start of dealerPlays().
    if (peekBlackjack) {
        dealerHand.addCard(shoe->dealCard(/*covered=*/true));
    }

    return dealerHand.isBlackjack();
}

std::vector<Action> SpanishTable::getAllowedActions(const Hand& hand, size_t handsInSlot) const {
    if (handsInSlot > maxSplits) {
        throw std::logic_error("Cannot have more hands than maxSplits");
    }

    // A freshly split-off hand starts with one card and must draw its second
    // before facing any real decision (matches BlackjackTable's convention).
    if (hand.cardCount() == 1) {
        return {Action::HIT};
    }

    // AFTER_DOUBLE decision node: stand, redouble (if any redoubles remain),
    // or rescue (surrender, forfeiting only the original bet). getDoubleCount()
    // is 1 right after the first (ordinary) double, i.e. 0 redoubles spent yet
    // -- so getDoubleCount() <= maxRedoubles is "at least one redouble left".
    if (hand.getIsDoubled()) {
        std::vector<Action> actions{Action::STAND};
        if (hand.getDoubleCount() <= maxRedoubles) {
            actions.push_back(Action::DOUBLE_DOWN);
        }
        if (allowDoubleDownRescue) {
            actions.push_back(Action::SURRENDER);
        }
        return actions;
    }

    std::vector<Action> actions{Action::HIT, Action::STAND};

    if (hand.getIsSplit()) {
        if (!hitSplitAces && hand[0].getValue() == 11) {
            actions = std::vector<Action>{Action::STAND};
        }
        if (handsInSlot < maxSplits && hand.isPair()) {
            if (hand[0].getValue() != 11) {
                actions.push_back(Action::SPLIT);
            } else if (resplitAces) {
                actions.push_back(Action::SPLIT);
            }
        }
        if (doubleAfterSplit) {
            if (hand[0].getValue() != 11) {
                actions.push_back(Action::DOUBLE_DOWN);
            } else if (hitSplitAces) {
                actions.push_back(Action::DOUBLE_DOWN);
            }
        }
        return actions;
    }

    // Spanish 21: double allowed on any number of cards (not just the first two),
    // as long as the hand hasn't been doubled yet.
    actions.push_back(Action::DOUBLE_DOWN);

    if (hand.isPair()) {
        actions.push_back(Action::SPLIT);
    }

    // Late surrender is only available as the very first decision.
    if (surrenderAllowed && hand.cardCount() == 2) {
        actions.push_back(Action::SURRENDER);
    }

    return actions;
}

std::vector<std::tuple<Player*, Hand*, State, Action>> SpanishTable::playersPlay() {
    std::vector<std::tuple<Player*, std::pair<size_t, size_t>, State, Action>> aliveHandIndices;
    Card dealerUpCard = dealerHand[0];

    for (auto* player : players) {
        std::vector<Slot>& slots = playerSlots[player];
        for (size_t slotIdx = 0; slotIdx < slots.size(); ++slotIdx) {
            Slot& slot = slots[slotIdx];
            for (size_t handIdx = 0; handIdx < slot.getHands().size(); ++handIdx) {
                while (true) {
                    Hand& hand = slot[handIdx];
                    std::vector<Action> allowedActions = getAllowedActions(hand, slot.getHands().size());

                    std::vector<Action> affordableActions = allowedActions;
                    if (player->shouldEnforceBankrollActionLimits() &&
                        !player->canAffordAdditionalWager(hand.getBet(), committedWagers.at(player))) {
                        affordableActions.erase(
                            std::remove(affordableActions.begin(), affordableActions.end(), Action::DOUBLE_DOWN),
                            affordableActions.end());
                        affordableActions.erase(
                            std::remove(affordableActions.begin(), affordableActions.end(), Action::SPLIT),
                            affordableActions.end());
                    }

                    State state(hand, dealerUpCard, allowedActions, shoe->getRemovedCards());
                    Action action = player->getAction(state, affordableActions);

                    if (verbose) {
                        int trueCount = player->computeTrueCount(state.removedCards);
                        VerboseTrace::printDecision(*verboseOut, round_number, *player, hand,
                                                     dealerUpCard, trueCount, action);
                    }

                    if (action == Action::HIT) {
                        hand.addCard(shoe->dealCard());
                        if (hand.isBust()) {
                            State nextState(hand, dealerUpCard, {}, shoe->getRemovedCards());
                            const double reward = -hand.getBet();
                            player->updateMoney(reward, state, action, nextState, learningWager(state));
                            releaseCommittedWager(player, hand.getBet());
                            if (verbose) {
                                VerboseTrace::printOutcome(*verboseOut, round_number, *player,
                                                            hand, dealerHand, reward);
                            }
                            break;
                        }
                        std::vector<Action> nextAllowedActions = getAllowedActions(hand, slot.getHands().size());
                        State nextState(hand, dealerUpCard, nextAllowedActions, shoe->getRemovedCards());
                        player->updateMoney(0.0, state, action, nextState, learningWager(state));
                    }
                    else if (action == Action::STAND) {
                        aliveHandIndices.push_back({player, {slotIdx, handIdx}, state, action});
                        break;
                    }
                    else if (action == Action::DOUBLE_DOWN) {
                        const double additionalWager = hand.getBet();
                        hand.multiplyBet(2.0f);
                        if (player->shouldEnforceBankrollActionLimits())
                            committedWagers[player] += additionalWager;
                        hand.addCard(shoe->dealCard());
                        hand.incrementDoubleCount();

                        if (hand.isBust()) {
                            State nextState(hand, dealerUpCard, {}, shoe->getRemovedCards());
                            const double reward = -hand.getBet();
                            player->updateMoney(reward, state, action, nextState, learningWager(state));
                            releaseCommittedWager(player, hand.getBet());
                            if (verbose) {
                                VerboseTrace::printOutcome(*verboseOut, round_number, *player,
                                                            hand, dealerHand, reward);
                            }
                            break;
                        }
                        // The hand continues into the AFTER_DOUBLE decision node (stand,
                        // possibly redouble/rescue) — same loop, not a break.
                        std::vector<Action> nextAllowedActions = getAllowedActions(hand, slot.getHands().size());
                        State nextState(hand, dealerUpCard, nextAllowedActions, shoe->getRemovedCards());
                        player->updateMoney(0.0, state, action, nextState,
                                            learningWager(state), /*nextValueMultiplier=*/2.0);
                    }
                    else if (action == Action::SPLIT) {
                        Hand newHand = hand.split();
                        slot.addHand(newHand);
                        if (player->shouldEnforceBankrollActionLimits())
                            committedWagers[player] += newHand.getBet();
                        std::vector<Action> nextAllowedActions = getAllowedActions(hand, slot.getHands().size());
                        State nextState(hand, dealerUpCard, nextAllowedActions, shoe->getRemovedCards());
                        player->updateMoney(0.0, state, action, nextState, learningWager(state));
                        // Continue playing the current hand.
                    }
                    else if (action == Action::SURRENDER) {
                        // Plain late surrender (first two cards): lose half the bet.
                        // Double-down rescue (after doubling): lose only the ORIGINAL
                        // (pre-double) bet — the current bet is 2^doubleCount times it.
                        const double lossAmount = hand.getIsDoubled()
                            ? hand.getBet() / std::pow(2.0, hand.getDoubleCount())
                            : hand.getBet() * 0.5;
                        State nextState(hand, dealerUpCard, {}, shoe->getRemovedCards());
                        const double reward = -lossAmount;
                        player->updateMoney(reward, state, action, nextState, learningWager(state));
                        releaseCommittedWager(player, hand.getBet());
                        if (verbose) {
                            VerboseTrace::printOutcome(*verboseOut, round_number, *player,
                                                        hand, dealerHand, reward);
                        }
                        break;
                    }
                    else {
                        throw std::invalid_argument("Invalid action");
                    }
                }
            }
        }
    }

    std::vector<std::tuple<Player*, Hand*, State, Action>> aliveHands;
    for (const auto& [player, indices, state, action] : aliveHandIndices) {
        size_t slotIdx = indices.first;
        size_t handIdx = indices.second;
        aliveHands.push_back({player, &playerSlots[player][slotIdx][handIdx], state, action});
    }

    return aliveHands;
}

void SpanishTable::dealerPlays(DealerAction dealer_action) {
    if (peekBlackjack)
        shoe->uncoverCard(dealerHand[1]);

    if (dealer_action == DealerAction::SKIP)
        return;

    while (shouldDealerHit()) {
        dealerHand.addCard(shoe->dealCard());
        if (dealer_action == DealerAction::CHECK_BLACKJACK) {
            break;
        }
    }
}

bool SpanishTable::shouldDealerHit() const {
    int dealerValue = dealerHand.getValue();
    if (dealerValue < 17) return true;
    return dealerValue == 17 && !standSoft17 && dealerHand.isSoft();
}

void SpanishTable::releaseCommittedWager(Player* player, double wager) {
    if (!player->shouldEnforceBankrollActionLimits()) return;
    double& committed = committedWagers.at(player);
    committed = std::max(0.0, committed - wager);
}

double SpanishTable::suitedSequenceMultiplier(const Hand& hand) const {
    if (hand.cardCount() != 3) return 0.0;

    std::array<int, 3> vals = {hand[0].getValue(), hand[1].getValue(), hand[2].getValue()};
    std::sort(vals.begin(), vals.end());
    const bool is678 = (vals[0] == 6 && vals[1] == 7 && vals[2] == 8);
    const bool is777 = (vals[0] == 7 && vals[1] == 7 && vals[2] == 7);
    if (!is678 && !is777) return 0.0;

    const bool allSameSuit = hand[0].suit == hand[1].suit && hand[1].suit == hand[2].suit;
    if (!allSameSuit) return 1.5;                       // mixed suits
    if (hand[0].suit == Suit::SPADES) return 3.0;        // suited spades
    return 2.0;                                          // suited, not spades
}

double SpanishTable::winMultiplier(const Hand& hand) const {
    // Neither bonus pays after doubling.
    if (hand.getIsDoubled()) return 1.0;

    if (paySuitedBonus) {
        const double suitedTier = suitedSequenceMultiplier(hand);
        if (suitedTier > 0.0) return suitedTier;
    }

    const size_t n = hand.cardCount();
    if (n >= 7) return 3.0;
    if (n == 6) return 2.0;
    if (n == 5) return 1.5;
    return 1.0;
}

void SpanishTable::evaluate(const std::vector<std::tuple<Player*, Hand*, State, Action>>& aliveHands) {
    const int dealerValue = dealerHand.getValue();
    const bool dealerBusted = dealerHand.isBust();
    const bool dealerBlackjack = dealerHand.isBlackjack();
    const Card dealerUpCard = dealerHand[0];

    for (const auto& [player, hand, state, action] : aliveHands) {
        const int playerValue = hand->getValue();
        const bool playerBlackjack = hand->isBlackjack();
        const double bet = hand->getBet();

        State nextState(*hand, dealerUpCard, {}, shoe->getRemovedCards());

        double reward;
        if (playerBlackjack) {
            // Player blackjack always beats the dealer, including a dealer blackjack.
            reward = bet * blackjackPayout;
        } else if (playerValue == 21) {
            // Player 21 always wins (no push, even vs. a dealer 21); winMultiplier()
            // applies the card-count/suited bonus tiers when they apply.
            reward = bet * winMultiplier(*hand);
        } else if (dealerBlackjack) {
            reward = -bet;
        } else if (dealerBusted) {
            reward = bet;
        } else if (playerValue > dealerValue) {
            reward = bet;
        } else if (playerValue < dealerValue) {
            reward = -bet;
        } else {
            reward = 0.0;
        }

        player->updateMoney(reward, state, action, nextState, learningWager(state));
        releaseCommittedWager(player, bet);
        if (verbose) {
            VerboseTrace::printOutcome(*verboseOut, round_number, *player, *hand, dealerHand, reward);
        }
    }
}

std::ostream& operator<<(std::ostream& os, const SpanishTable& table) {
    os << "*========* Spanish 21 round " << table.round_number << " *========*\n";
    os << "Dealer: " << table.dealerHand << "\n";
    for (const auto& [player, slots] : table.playerSlots) {
        os << "Player Name: " << player->getName() << ", Money: " << player->getMoney() << "\n";
        for (size_t i = 0; i < slots.size(); ++i) {
            os << "Slot " << (i + 1) << ": \n" << slots[i] << "\n";
        }
    }
    return os;
}
