#include "DoubleDownMadnessTable.h"

#include <algorithm>
#include <stdexcept>

namespace {

double learningWager(const State& state) {
    // A Kelly evaluator may intentionally wager zero. Its fixed strategy ignores
    // updates, so use a neutral divisor while preserving the actual zero payout.
    return state.playerHand.getBet() > 0.0 ? state.playerHand.getBet() : 1.0;
}

} // namespace

DoubleDownMadnessTable::DoubleDownMadnessTable(
        const DoubleDownMadnessRules& rules,
        std::vector<Player*> gamePlayers)
    : Table(rules, std::move(gamePlayers)),
      standSoft17(rules.standSoft17),
      minBet(rules.minBet),
      maxBet(rules.maxBet),
      suitedBlackjackPayout(rules.blackjackPayoutFor(true)),
      unsuitedBlackjackPayout(rules.blackjackPayoutFor(false)) {
    moneyBeforeScratch.resize(players.size());
    for (auto* player : players) {
        playerHands.emplace(player, Hand{});
        committedWagers.emplace(player, 0.0);
    }
}

void DoubleDownMadnessTable::round() {
    ++roundNumber;
    if (shoe->isEndShoe()) shoe->reset();

    bool anyRoundTracking = false;
    bool anyFeatureTracking = false;
    for (auto* player : players) {
        if (player->isRegressionEnabled() || player->isCountGraphEnabled()) {
            anyRoundTracking = true;
            anyFeatureTracking = true;
        }
        if (player->isRoundStatsEnabled()) anyRoundTracking = true;
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

    const bool dealerBlackjack = dealInitialCards();
    std::vector<std::tuple<Player*, Hand*, State, Action>> aliveHands;
    if (dealerBlackjack) {
        for (auto* player : players)
            player->addReward(-playerHands.at(player).getBet());
    } else {
        aliveHands = playersPlay();
    }

    dealerPlays(aliveHands.empty() ? DealerAction::SKIP : DealerAction::PLAY);
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

void DoubleDownMadnessTable::clearHands() {
    dealerHand.clear();
    for (auto& [player, hand] : playerHands) {
        hand.clear();
        committedWagers[player] = 0.0;
    }
}

void DoubleDownMadnessTable::collectBets() {
    for (auto* player : players) {
        const double bet =
            std::clamp(player->getBet(shoe->getRemovedCards()), minBet, maxBet);
        playerHands.at(player).setBet(bet);
        if (player->shouldEnforceBankrollActionLimits())
            committedWagers[player] = bet;
    }
}

bool DoubleDownMadnessTable::dealInitialCards() {
    // The physical order is dealer hole card, one player card per spot, dealer up-card.
    // Keep the up-card at dealerHand[0] so State follows the classic table convention.
    const Card holeCard = shoe->dealCard(/*covered=*/true);
    for (auto* player : players)
        playerHands.at(player).addCard(shoe->dealCard());
    dealerHand.addCard(shoe->dealCard());
    dealerHand.addCard(holeCard);

    return dealerHand.isBlackjack();
}

std::vector<Action> DoubleDownMadnessTable::getAllowedActions(
        const Hand& hand) const {
    if (hand.isBust() || hand.getValue() >= 21) return {};

    return
        hand.cardCount() == 1 && hand[0].rank == Rank::ACE
            ? std::vector<Action>{Action::HIT, Action::DOUBLE_DOWN}
            : std::vector<Action>{Action::HIT, Action::STAND, Action::DOUBLE_DOWN};
}

std::vector<std::tuple<Player*, Hand*, State, Action>>
DoubleDownMadnessTable::playersPlay() {
    std::vector<std::tuple<Player*, Hand*, State, Action>> aliveHands;
    const Card dealerUpCard = dealerHand[0];

    for (auto* player : players) {
        Hand& hand = playerHands.at(player);
        while (true) {
            const std::vector<Action> allowedActions = getAllowedActions(hand);
            if (allowedActions.empty())
                throw std::logic_error("Double Down Madness hand has no legal action");

            std::vector<Action> affordableActions = allowedActions;
            if (player->shouldEnforceBankrollActionLimits() &&
                !player->canAffordAdditionalWager(
                    hand.getBet(), committedWagers.at(player))) {
                affordableActions.erase(
                    std::remove(affordableActions.begin(), affordableActions.end(),
                                Action::DOUBLE_DOWN),
                    affordableActions.end());
            }

            State state(hand, dealerUpCard, allowedActions, shoe->getRemovedCards());
            const Action action = player->getAction(state, affordableActions);
            if (std::find(affordableActions.begin(), affordableActions.end(), action) ==
                affordableActions.end()) {
                throw std::logic_error("Strategy selected an illegal Double Down Madness action");
            }

            if (action == Action::STAND) {
                aliveHands.emplace_back(player, &hand, state, action);
                break;
            }

            const bool doubleOnInitialAce =
                action == Action::DOUBLE_DOWN &&
                hand.cardCount() == 1 &&
                hand[0].rank == Rank::ACE;
            if (action == Action::DOUBLE_DOWN) {
                if (player->shouldEnforceBankrollActionLimits())
                    committedWagers[player] += hand.getBet();
                hand.multiplyBet(2.0f);
            }
            hand.addCard(shoe->dealCard());

            State terminalState(hand, dealerUpCard, {}, shoe->getRemovedCards());
            if (hand.isBust()) {
                player->updateMoney(
                    -hand.getBet(), state, action, terminalState,
                    learningWager(state));
                if (player->shouldEnforceBankrollActionLimits())
                    committedWagers[player] = 0.0;
                break;
            }
            if (hand.isBlackjack()) {
                player->updateMoney(
                    hand.getBet() * blackjackPayout(hand), state, action, terminalState,
                    learningWager(state));
                if (player->shouldEnforceBankrollActionLimits())
                    committedWagers[player] = 0.0;
                break;
            }
            // Hitting an initial Ace may continue (A,A becomes soft 12), but
            // doubling an initial Ace is a one-card draw with no further action.
            if (doubleOnInitialAce || hand.getValue() == 21) {
                aliveHands.emplace_back(player, &hand, state, action);
                break;
            }

            const std::vector<Action> nextActions = getAllowedActions(hand);
            State nextState(hand, dealerUpCard, nextActions, shoe->getRemovedCards());
            const double nextValueMultiplier =
                action == Action::DOUBLE_DOWN ? 2.0 : 1.0;
            player->updateMoney(
                0.0, state, action, nextState,
                learningWager(state), nextValueMultiplier);
        }
    }

    return aliveHands;
}

void DoubleDownMadnessTable::dealerPlays(DealerAction dealerAction) {
    // The dealer hole card remains invisible to counting and policy state until now.
    shoe->uncoverCard(dealerHand[1]);
    if (dealerAction == DealerAction::SKIP) return;

    while (shouldDealerHit())
        dealerHand.addCard(shoe->dealCard());
}

bool DoubleDownMadnessTable::shouldDealerHit() const {
    const int value = dealerHand.getValue();
    if (value < 17) return true;
    return value == 17 && dealerHand.isSoft() && !standSoft17;
}

bool DoubleDownMadnessTable::isSuitedBlackjack(const Hand& hand) const {
    return hand.isBlackjack() && hand[0].suit == hand[1].suit;
}

double DoubleDownMadnessTable::blackjackPayout(const Hand& hand) const {
    return isSuitedBlackjack(hand)
        ? suitedBlackjackPayout
        : unsuitedBlackjackPayout;
}

void DoubleDownMadnessTable::evaluate(
        const std::vector<std::tuple<Player*, Hand*, State, Action>>& aliveHands) {
    const int dealerValue = dealerHand.getValue();
    const bool dealerBusted = dealerHand.isBust();
    const Card dealerUpCard = dealerHand[0];

    for (const auto& [player, hand, state, action] : aliveHands) {
        State nextState(*hand, dealerUpCard, {}, shoe->getRemovedCards());
        double reward = 0.0;
        if (dealerBusted && dealerValue == 22) {
            reward = 0.0;
        } else if (dealerBusted || hand->getValue() > dealerValue) {
            reward = hand->getBet();
        } else if (hand->getValue() < dealerValue) {
            reward = -hand->getBet();
        }
        player->updateMoney(
            reward, state, action, nextState, learningWager(state));
        if (player->shouldEnforceBankrollActionLimits())
            committedWagers[player] = 0.0;
    }
}

std::ostream& operator<<(std::ostream& os, const DoubleDownMadnessTable& table) {
    os << "*========* Double Down Madness round " << table.roundNumber
       << " *========*\n";
    os << "Dealer: " << table.dealerHand << "\n";
    for (const auto& [player, hand] : table.playerHands)
        os << "Player " << player->getName() << ", money=" << player->getMoney()
           << "\n" << hand << "\n";
    return os;
}
