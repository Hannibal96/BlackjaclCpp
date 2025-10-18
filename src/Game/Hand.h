#pragma once
#include <vector>
#include "../Shoe/Deck.h"

// Hand class representing a player's hand in blackjack
class Hand {
private:
    std::vector<Card> cards;
    double bet;
    bool isSplit;
    
public:
    // Constructors
    Hand();
    explicit Hand(double betAmount);
    
    // Add a card to the hand
    void addCard(const Card& card);
    
    // Get all cards in the hand
    const std::vector<Card>& getCards() const { return cards; }
    
    // Access card by index (with bounds checking)
    const Card& operator[](size_t index) const;
    Card& operator[](size_t index);
    
    // Get the bet amount
    double getBet() const { return bet; }
    
    // Set the bet amount
    void setBet(double betAmount) { bet = betAmount; }
    
    // Multiply the bet by a factor (default 2 for double down)
    void multiplyBet(float multiplier = 2.0f);
    
    // Check if this hand is from a split
    bool getIsSplit() const { return isSplit; }
    
    // Set the split flag
    void setIsSplit(bool split) { isSplit = split; }
    
    // Get the number of cards in the hand
    size_t cardCount() const { return cards.size(); }
    
    // Clear the hand
    void clear();
    
    // Calculate hand value (handles soft/hard totals)
    int getValue() const;
    
    // Check if hand is a pair (for split logic)
    bool isPair() const;
    
    // Check if hand is blackjack (21 with 2 cards)
    bool isBlackjack() const;
    
    // Check if hand is bust (over 21)
    bool isBust() const;
    
    // Check if hand is soft (has an Ace counted as 11)
    bool isSoft() const;
    bool isHard() const;
    
    // Split the hand into two hands
    // Modifies current hand to keep only first card, returns new hand with second card
    // Both hands marked as split with same bet
    Hand split();

    friend std::ostream& operator<<(std::ostream& os, const Hand& hand);
};

