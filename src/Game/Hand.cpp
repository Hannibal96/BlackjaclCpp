#include "Hand.h"
#include <stdexcept>

// Default constructor
Hand::Hand() : bet(0), isSplit(false) {
}

// Constructor with bet amount
Hand::Hand(double betAmount) : bet(betAmount), isSplit(false) {
}

// Add a card to the hand
void Hand::addCard(const Card& card) {
    cards.push_back(card);
}

// Access card by index (const version)
const Card& Hand::operator[](size_t index) const {
    if (index >= cards.size()) {
        throw std::out_of_range("Hand index out of range: " + std::to_string(index));
    }
    return cards[index];
}

// Access card by index (non-const version)
Card& Hand::operator[](size_t index) {
    if (index >= cards.size()) {
        throw std::out_of_range("Hand index out of range: " + std::to_string(index));
    }
    return cards[index];
}

// Multiply the bet by a factor
void Hand::multiplyBet(float multiplier) {
    bet = bet * multiplier;
}

// Clear the hand
void Hand::clear() {
    cards.clear();
    bet = 0;
    isSplit = false;
}

// Calculate hand value (handles soft/hard totals)
int Hand::getValue() const {
    int total = 0;
    int aces = 0;
    
    // Sum up all card values and count aces
    for (const auto& card : cards) {
        int value = card.getValue();
        total += value;
        if (value == 11) {  // Ace
            aces++;
        }
    }
    
    // Adjust for aces if bust (convert from soft to hard)
    while (total > 21 && aces > 0) {
        total -= 10;  // Convert an ace from 11 to 1
        aces--;
    }
    
    return total;
}

// Check if hand is a pair (for split logic)
bool Hand::isPair() const {
    if (cards.size() != 2) {
        return false;
    }
    
    // Check if both cards have the same value
    return cards[0].getValue() == cards[1].getValue();
}

// Check if hand is blackjack (21 with 2 cards)
bool Hand::isBlackjack() const {
    return cards.size() == 2 && getValue() == 21 && !isSplit;
}

// Check if hand is bust (over 21)
bool Hand::isBust() const {
    return getValue() > 21;
}

// Check if hand is soft (has an Ace counted as 11)
bool Hand::isSoft() const {
    int total = 0;
	bool hasAce = false;
    
    // Sum up all card values and check for aces
    for (const auto& card : cards) {
        int value = card.getValue();
        total += value;
        if (value == 11) {  // Ace
            total -= 10;
			hasAce = true;
        }
    }
    
    // Hand is soft if it has an Ace and the total is <= 21 (meaning the Ace is counted as 11)
    return hasAce && total <= 11;
}

bool Hand::isHard() const{
    return !isSoft();
}

// Split the hand into two hands
Hand Hand::split() {
    // Must have exactly 2 cards to split
    if (cards.size() != 2) {
        throw std::logic_error("Cannot split hand with " + std::to_string(cards.size()) + " cards");
    }
    
    // Save the original bet and cards
    double originalBet = bet;
    Card firstCard = cards[0];
    Card secondCard = cards[1];
    
    // Create new hand with second card
    Hand newHand(originalBet);
    newHand.addCard(secondCard);
    newHand.setIsSplit(true);
    
    // Modify current hand to keep only first card
    clear();
    bet = originalBet;
    addCard(firstCard);
    isSplit = true;
    
    return newHand;
}


HandType Hand::getHandType(bool pairAllowed) const {
    if(cardCount() == 1){
        return HandType::ZOMBIE;
    } else if (isBlackjack()) {
        return HandType::BLACKJACK;
    } else if (isPair() and pairAllowed) {
        return HandType::PAIR;
    } else if (isSoft()) {
        return HandType::SOFT;
    } else if (isHard()) {
        return HandType::HARD;
    }

    throw std::logic_error("Unable to determine hand type for hand with " + std::to_string(cardCount()) + " cards");
}

std::ostream& operator<<(std::ostream& os, const Hand& hand) {
    os << "Bet: " << hand.bet << (hand.isSplit ? " After Split" : " Not splited") << std::endl;
    for(Card card : hand.cards){
        os << card << " ";
    }
    return os;
}

