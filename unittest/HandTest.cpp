#include <gtest/gtest.h>
#include "../src/Game/Hand.h"

// Test fixture for Hand tests
class HandTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// Test: Default constructor
TEST_F(HandTest, DefaultConstructor) {
    Hand hand;
    EXPECT_EQ(hand.getBet(), 0);
    EXPECT_EQ(hand.cardCount(), 0);
    EXPECT_FALSE(hand.getIsSplit());
}

// Test: Constructor with bet amount
TEST_F(HandTest, ConstructorWithBet) {
    Hand hand(100);
    EXPECT_EQ(hand.getBet(), 100);
    EXPECT_EQ(hand.cardCount(), 0);
    EXPECT_FALSE(hand.getIsSplit());
}

// Test: Add cards to hand
TEST_F(HandTest, AddCard) {
    Hand hand;
    Card card1(Rank::KING, Suit::HEARTS);
    Card card2(Rank::SEVEN, Suit::SPADES);
    
    hand.addCard(card1);
    EXPECT_EQ(hand.cardCount(), 1);
    
    hand.addCard(card2);
    EXPECT_EQ(hand.cardCount(), 2);
    EXPECT_EQ(hand.getCards().size(), 2);
}

// Test: Hard hand values (no Aces or Ace as 1)
TEST_F(HandTest, HardHandValues) {
    Hand hand;
    
    // King + 7 = 17
    hand.addCard(Card(Rank::KING, Suit::HEARTS));
    hand.addCard(Card(Rank::SEVEN, Suit::SPADES));
    EXPECT_EQ(hand.getValue(), 17);
}

TEST_F(HandTest, HardHandWithMultipleCards) {
    Hand hand;
    
    // 5 + 3 + 2 + 8 = 18
    hand.addCard(Card(Rank::FIVE, Suit::HEARTS));
    hand.addCard(Card(Rank::THREE, Suit::SPADES));
    hand.addCard(Card(Rank::TWO, Suit::DIAMONDS));
    hand.addCard(Card(Rank::EIGHT, Suit::CLUBS));
    EXPECT_EQ(hand.getValue(), 18);
}

// Test: Soft hand values (Ace as 11)
TEST_F(HandTest, SoftHandAceAs11) {
    Hand hand;
    
    // Ace + 6 = 17 (soft)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::SIX, Suit::SPADES));
    EXPECT_EQ(hand.getValue(), 17);
}

TEST_F(HandTest, SoftHandAceWithTen) {
    Hand hand;
    
    // Ace + King = 21 (blackjack)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::KING, Suit::SPADES));
    EXPECT_EQ(hand.getValue(), 21);
}

// Test: Ace converts from 11 to 1 when busting
TEST_F(HandTest, AceConvertsTo1WhenBusting) {
    Hand hand;
    
    // Ace (11) + King (10) + 5 = would be 26, but Ace converts to 1 = 16
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::KING, Suit::SPADES));
    hand.addCard(Card(Rank::FIVE, Suit::DIAMONDS));
    EXPECT_EQ(hand.getValue(), 16);
}

TEST_F(HandTest, AceConvertsTo1WithHighCard) {
    Hand hand;
    
    // Ace (11) + 9 + 5 = would be 25, but Ace converts to 1 = 15
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::NINE, Suit::SPADES));
    hand.addCard(Card(Rank::FIVE, Suit::DIAMONDS));
    EXPECT_EQ(hand.getValue(), 15);
}

// Test: Multiple Aces
TEST_F(HandTest, MultipleAces) {
    Hand hand;
    
    // Ace + Ace = 12 (one as 11, one as 1)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::ACE, Suit::SPADES));
    EXPECT_EQ(hand.getValue(), 12);
}

TEST_F(HandTest, ThreeAces) {
    Hand hand;
    
    // Ace + Ace + Ace = 13 (one as 11, two as 1)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::ACE, Suit::SPADES));
    hand.addCard(Card(Rank::ACE, Suit::DIAMONDS));
    EXPECT_EQ(hand.getValue(), 13);
}

TEST_F(HandTest, MultipleAcesWithOtherCards) {
    Hand hand;
    
    // Ace + Ace + 9 = 21 (one Ace as 11, one as 1)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::ACE, Suit::SPADES));
    hand.addCard(Card(Rank::NINE, Suit::DIAMONDS));
    EXPECT_EQ(hand.getValue(), 21);
}

TEST_F(HandTest, FourAces) {
    Hand hand;
    
    // Four Aces = 14 (one as 11, three as 1)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::ACE, Suit::SPADES));
    hand.addCard(Card(Rank::ACE, Suit::DIAMONDS));
    hand.addCard(Card(Rank::ACE, Suit::CLUBS));
    EXPECT_EQ(hand.getValue(), 14);
}

// Test: isPair method
TEST_F(HandTest, IsPairWithPair) {
    Hand hand;
    
    // Two 8s
    hand.addCard(Card(Rank::EIGHT, Suit::HEARTS));
    hand.addCard(Card(Rank::EIGHT, Suit::SPADES));
    EXPECT_TRUE(hand.isPair());
}

TEST_F(HandTest, IsPairWithFaceCards) {
    Hand hand;
    
    // King + King
    hand.addCard(Card(Rank::KING, Suit::HEARTS));
    hand.addCard(Card(Rank::KING, Suit::SPADES));
    EXPECT_TRUE(hand.isPair());
}

TEST_F(HandTest, IsPairWithTenValueCards) {
    Hand hand;
    
    // King + Queen (both value 10, but should still be pair based on value)
    hand.addCard(Card(Rank::KING, Suit::HEARTS));
    hand.addCard(Card(Rank::QUEEN, Suit::SPADES));
    EXPECT_TRUE(hand.isPair()); // Both have value 10
}

TEST_F(HandTest, IsNotPairWithDifferentValues) {
    Hand hand;
    
    // 8 + 9
    hand.addCard(Card(Rank::EIGHT, Suit::HEARTS));
    hand.addCard(Card(Rank::NINE, Suit::SPADES));
    EXPECT_FALSE(hand.isPair());
}

TEST_F(HandTest, IsNotPairWithMoreThanTwoCards) {
    Hand hand;
    
    // Three 8s
    hand.addCard(Card(Rank::EIGHT, Suit::HEARTS));
    hand.addCard(Card(Rank::EIGHT, Suit::SPADES));
    hand.addCard(Card(Rank::EIGHT, Suit::DIAMONDS));
    EXPECT_FALSE(hand.isPair());
}

TEST_F(HandTest, IsNotPairWithOneCard) {
    Hand hand;
    
    hand.addCard(Card(Rank::EIGHT, Suit::HEARTS));
    EXPECT_FALSE(hand.isPair());
}

// Test: isBlackjack method
TEST_F(HandTest, IsBlackjackWithAceKing) {
    Hand hand;
    
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::KING, Suit::SPADES));
    EXPECT_TRUE(hand.isBlackjack());
}

TEST_F(HandTest, IsBlackjackWithAceQueen) {
    Hand hand;
    
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::QUEEN, Suit::SPADES));
    EXPECT_TRUE(hand.isBlackjack());
}

TEST_F(HandTest, IsBlackjackWithAceTen) {
    Hand hand;
    
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::TEN, Suit::SPADES));
    EXPECT_TRUE(hand.isBlackjack());
}

TEST_F(HandTest, IsNotBlackjackWith21ThreeCards) {
    Hand hand;
    
    // 7 + 7 + 7 = 21, but not blackjack
    hand.addCard(Card(Rank::SEVEN, Suit::HEARTS));
    hand.addCard(Card(Rank::SEVEN, Suit::SPADES));
    hand.addCard(Card(Rank::SEVEN, Suit::DIAMONDS));
    EXPECT_FALSE(hand.isBlackjack());
}

TEST_F(HandTest, IsNotBlackjackWithNon21) {
    Hand hand;
    
    hand.addCard(Card(Rank::KING, Suit::HEARTS));
    hand.addCard(Card(Rank::NINE, Suit::SPADES));
    EXPECT_FALSE(hand.isBlackjack());
}

// Test: isBust method
TEST_F(HandTest, IsBustOver21) {
    Hand hand;
    
    // King + Queen + 5 = 25
    hand.addCard(Card(Rank::KING, Suit::HEARTS));
    hand.addCard(Card(Rank::QUEEN, Suit::SPADES));
    hand.addCard(Card(Rank::FIVE, Suit::DIAMONDS));
    EXPECT_TRUE(hand.isBust());
}

TEST_F(HandTest, IsNotBustAt21) {
    Hand hand;
    
    // King + Jack + Ace = 21
    hand.addCard(Card(Rank::KING, Suit::HEARTS));
    hand.addCard(Card(Rank::JACK, Suit::SPADES));
    hand.addCard(Card(Rank::ACE, Suit::DIAMONDS));
    EXPECT_FALSE(hand.isBust());
}

TEST_F(HandTest, IsNotBustUnder21) {
    Hand hand;
    
    // 9 + 8 = 17
    hand.addCard(Card(Rank::NINE, Suit::HEARTS));
    hand.addCard(Card(Rank::EIGHT, Suit::SPADES));
    EXPECT_FALSE(hand.isBust());
}

TEST_F(HandTest, IsNotBustWithAceConversion) {
    Hand hand;
    
    // Ace + King + 9 = 20 (Ace converts to 1)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::KING, Suit::SPADES));
    hand.addCard(Card(Rank::NINE, Suit::DIAMONDS));
    EXPECT_FALSE(hand.isBust());
}

// Test: Bet management
TEST_F(HandTest, SetAndGetBet) {
    Hand hand;
    
    hand.setBet(50);
    EXPECT_EQ(hand.getBet(), 50);
    
    hand.setBet(200);
    EXPECT_EQ(hand.getBet(), 200);
}

TEST_F(HandTest, MultiplyBetDefault) {
    Hand hand(100);
    
    hand.multiplyBet(); // Default multiplier is 2.0
    EXPECT_EQ(hand.getBet(), 200);
}

TEST_F(HandTest, MultiplyBetCustomMultiplier) {
    Hand hand(100);
    
    hand.multiplyBet(1.5f);
    EXPECT_EQ(hand.getBet(), 150);
}

TEST_F(HandTest, MultiplyBetTriple) {
    Hand hand(50);
    
    hand.multiplyBet(3.0f);
    EXPECT_EQ(hand.getBet(), 150);
}

TEST_F(HandTest, MultiplyBetZero) {
    Hand hand(100);
    
    hand.multiplyBet(0.0f);
    EXPECT_EQ(hand.getBet(), 0);
}

// Test: Split flag management
TEST_F(HandTest, SetAndGetIsSplit) {
    Hand hand;
    
    EXPECT_FALSE(hand.getIsSplit());
    
    hand.setIsSplit(true);
    EXPECT_TRUE(hand.getIsSplit());
    
    hand.setIsSplit(false);
    EXPECT_FALSE(hand.getIsSplit());
}

// Test: Clear method
TEST_F(HandTest, ClearHand) {
    Hand hand(100);
    
    hand.addCard(Card(Rank::KING, Suit::HEARTS));
    hand.addCard(Card(Rank::SEVEN, Suit::SPADES));
    hand.setIsSplit(true);
    
    EXPECT_EQ(hand.cardCount(), 2);
    EXPECT_EQ(hand.getBet(), 100);
    EXPECT_TRUE(hand.getIsSplit());
    
    hand.clear();
    
    EXPECT_EQ(hand.cardCount(), 0);
    EXPECT_EQ(hand.getBet(), 0);
    EXPECT_FALSE(hand.getIsSplit());
}

// Test: Edge cases
TEST_F(HandTest, EmptyHandValue) {
    Hand hand;
    EXPECT_EQ(hand.getValue(), 0);
}

TEST_F(HandTest, SingleCardValue) {
    Hand hand;
    hand.addCard(Card(Rank::NINE, Suit::HEARTS));
    EXPECT_EQ(hand.getValue(), 9);
    EXPECT_FALSE(hand.isSoft());
    EXPECT_FALSE(hand.isBust());
}

TEST_F(HandTest, SingleAceValue) {
    Hand hand;
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    EXPECT_EQ(hand.getValue(), 11);
}

TEST_F(HandTest, MaximumValue) {
    Hand hand;
    
    // Four 5s + Ace = 21 (5+5+5+5+1)
    hand.addCard(Card(Rank::FIVE, Suit::HEARTS));
    hand.addCard(Card(Rank::FIVE, Suit::SPADES));
    hand.addCard(Card(Rank::FIVE, Suit::DIAMONDS));
    hand.addCard(Card(Rank::FIVE, Suit::CLUBS));
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    EXPECT_EQ(hand.getValue(), 21);
}

// Test: isSoft method - Testing soft/hard hand transitions
TEST_F(HandTest, IsSoftWithAceLowCard) {
    Hand hand;
    
    // Ace + 5 = soft 16
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::FIVE, Suit::SPADES));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 16);
}

TEST_F(HandTest, IsSoftWithAceSix) {
    Hand hand;
    
    // Ace + 6 = soft 17
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::SIX, Suit::SPADES));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 17);
}

TEST_F(HandTest, IsSoftBlackjack) {
    Hand hand;
    
    // Ace + King = soft 21 (blackjack)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::KING, Suit::SPADES));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 21);
}

TEST_F(HandTest, IsNotSoftHardHand) {
    Hand hand;
    
    // King + 7 = hard 17 (no Ace)
    hand.addCard(Card(Rank::KING, Suit::HEARTS));
    hand.addCard(Card(Rank::SEVEN, Suit::SPADES));
    EXPECT_FALSE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 17);
}

TEST_F(HandTest, IsNotSoftAceConvertedToOne) {
    Hand hand;
    
    // Ace + King + 5 = hard 16 (Ace counted as 1)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::KING, Suit::SPADES));
    hand.addCard(Card(Rank::FIVE, Suit::DIAMONDS));
    EXPECT_FALSE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 16);
}

TEST_F(HandTest, SoftToHardTransition) {
    Hand hand;
    
    // Start soft: Ace + 5 = soft 16
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::FIVE, Suit::SPADES));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 16);
    
    // Add a card that keeps it soft: Ace + 5 + 2 = soft 18
    hand.addCard(Card(Rank::TWO, Suit::DIAMONDS));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 18);
    
    // Add a card that converts to hard: Ace + 5 + 2 + 6 = hard 14 (Ace as 1)
    hand.addCard(Card(Rank::SIX, Suit::CLUBS));
    EXPECT_FALSE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 14);
}

TEST_F(HandTest, SoftToHardTransitionWithTen) {
    Hand hand;
    
    // Start soft: Ace + 6 = soft 17
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::SIX, Suit::SPADES));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 17);
    
    // Add 10 to force Ace to 1: Ace + 6 + 10 = hard 17 (Ace as 1)
    hand.addCard(Card(Rank::TEN, Suit::DIAMONDS));
    EXPECT_FALSE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 17);
}

TEST_F(HandTest, SoftToHardWithMultipleCards) {
    Hand hand;
    
    // Ace + 3 = soft 14
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::THREE, Suit::SPADES));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 14);
    
    // Ace + 3 + 2 = soft 16
    hand.addCard(Card(Rank::TWO, Suit::DIAMONDS));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 16);
    
    // Ace + 3 + 2 + 3 = soft 19
    hand.addCard(Card(Rank::THREE, Suit::CLUBS));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 19);
    
    // Ace + 3 + 2 + 3 + 8 = hard 17 (Ace as 1)
    hand.addCard(Card(Rank::EIGHT, Suit::HEARTS));
    EXPECT_FALSE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 17);
}

TEST_F(HandTest, IsSoftWithMultipleAces) {
    Hand hand;
    
    // Ace + Ace = soft 12 (one Ace as 11, one as 1)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::ACE, Suit::SPADES));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 12);
}

TEST_F(HandTest, MultipleAcesSoftToHard) {
    Hand hand;
    
    // Ace + Ace = soft 12
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::ACE, Suit::SPADES));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 12);
    
    // Ace + Ace + 9 = 21 still soft (one Ace as 11)
    hand.addCard(Card(Rank::NINE, Suit::DIAMONDS));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 21);
}

TEST_F(HandTest, MultipleAcesBecomesHard) {
    Hand hand;
    
    // Ace + Ace = soft 12
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::ACE, Suit::SPADES));
    EXPECT_TRUE(hand.isSoft());
    
    // Ace + Ace + King = hard 12 (both Aces as 1)
    hand.addCard(Card(Rank::KING, Suit::DIAMONDS));
    EXPECT_FALSE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 12);
}

TEST_F(HandTest, ThreeAcesSoftToHard) {
    Hand hand;
    
    // Ace + Ace + Ace = soft 13 (one as 11, two as 1)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::ACE, Suit::SPADES));
    hand.addCard(Card(Rank::ACE, Suit::DIAMONDS));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 13);
    
    // Ace + Ace + Ace + 9 = 21 still soft
    hand.addCard(Card(Rank::NINE, Suit::CLUBS));
    EXPECT_FALSE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 12);
}

TEST_F(HandTest, SingleAceIsSoft) {
    Hand hand;
    
    // Single Ace = soft 11
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 11);
}

TEST_F(HandTest, EmptyHandIsNotSoft) {
    Hand hand;
    EXPECT_FALSE(hand.isSoft());
}

TEST_F(HandTest, NoAceIsNotSoft) {
    Hand hand;
    
    // 10 + 7 = 17 (no Ace)
    hand.addCard(Card(Rank::TEN, Suit::HEARTS));
    hand.addCard(Card(Rank::SEVEN, Suit::SPADES));
    EXPECT_FALSE(hand.isSoft());
}

TEST_F(HandTest, BustedHandWithAceIsNotSoft) {
    Hand hand;
    
    // Ace + King + Queen = 21 (Ace as 1)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::KING, Suit::SPADES));
    hand.addCard(Card(Rank::QUEEN, Suit::DIAMONDS));
    EXPECT_FALSE(hand.isSoft()); // Ace counted as 1, so not soft
    EXPECT_EQ(hand.getValue(), 21);
}

TEST_F(HandTest, Soft17ExactCase) {
    Hand hand;
    
    // Ace + 6 = soft 17 (dealer rule test case)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::SIX, Suit::SPADES));
    EXPECT_TRUE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 17);
}

TEST_F(HandTest, Hard17IsNotSoft) {
    Hand hand;
    
    // 10 + 7 = hard 17
    hand.addCard(Card(Rank::TEN, Suit::HEARTS));
    hand.addCard(Card(Rank::SEVEN, Suit::SPADES));
    EXPECT_FALSE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 17);
}

TEST_F(HandTest, AceConvertedDueToMultipleCardsIsNotSoft) {
    Hand hand;
    
    // Ace + 2 + 3 + 4 + 5 = hard 15 (Ace as 1)
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::TWO, Suit::SPADES));
    hand.addCard(Card(Rank::THREE, Suit::DIAMONDS));
    hand.addCard(Card(Rank::FOUR, Suit::CLUBS));
    hand.addCard(Card(Rank::FIVE, Suit::HEARTS));
    EXPECT_FALSE(hand.isSoft());
    EXPECT_EQ(hand.getValue(), 15);
}

// Test: split method - Testing hand splitting functionality
TEST_F(HandTest, SplitBasicPair) {
    Hand hand(100);
    
    // Create a pair of 8s
    hand.addCard(Card(Rank::EIGHT, Suit::HEARTS));
    hand.addCard(Card(Rank::EIGHT, Suit::SPADES));
    
    EXPECT_FALSE(hand.getIsSplit());
    EXPECT_EQ(hand.cardCount(), 2);
    
    // Split the hand
    Hand newHand = hand.split();
    
    // Check original hand
    EXPECT_TRUE(hand.getIsSplit());
    EXPECT_EQ(hand.cardCount(), 1);
    EXPECT_EQ(hand.getBet(), 100);
    EXPECT_EQ(hand.getValue(), 8);
    EXPECT_EQ(hand.getCards()[0].rank, Rank::EIGHT);
    EXPECT_EQ(hand.getCards()[0].suit, Suit::HEARTS);
    
    // Check new hand
    EXPECT_TRUE(newHand.getIsSplit());
    EXPECT_EQ(newHand.cardCount(), 1);
    EXPECT_EQ(newHand.getBet(), 100);
    EXPECT_EQ(newHand.getValue(), 8);
    EXPECT_EQ(newHand.getCards()[0].getValue(), 8);
    EXPECT_EQ(newHand.getCards()[0].rank, Rank::EIGHT);
    EXPECT_EQ(newHand.getCards()[0].suit, Suit::SPADES);
}

TEST_F(HandTest, SplitAces) {
    Hand hand(50);
    
    // Create a pair of Aces
    hand.addCard(Card(Rank::ACE, Suit::HEARTS));
    hand.addCard(Card(Rank::ACE, Suit::DIAMONDS));
    
    // Split the hand
    Hand newHand = hand.split();
    
    // Check both hands have one Ace each
    EXPECT_EQ(hand.cardCount(), 1);
    EXPECT_EQ(hand.getValue(), 11);
    EXPECT_EQ(hand.getCards()[0].rank, Rank::ACE);
    EXPECT_EQ(hand.getCards()[0].suit, Suit::HEARTS);

    EXPECT_EQ(newHand.cardCount(), 1);
    EXPECT_EQ(newHand.getValue(), 11);
    EXPECT_EQ(newHand.getCards()[0].rank, Rank::ACE);
    EXPECT_EQ(newHand.getCards()[0].suit, Suit::DIAMONDS);
    
    // Both should be marked as split
    EXPECT_TRUE(hand.getIsSplit());
    EXPECT_TRUE(newHand.getIsSplit());
    
    // Both should have same bet
    EXPECT_EQ(hand.getBet(), 50);
    EXPECT_EQ(newHand.getBet(), 50);
}

TEST_F(HandTest, SplitFaceCards) {
    Hand hand(200);
    
    // Create a pair of face cards (King and Queen both value 10)
    hand.addCard(Card(Rank::KING, Suit::HEARTS));
    hand.addCard(Card(Rank::QUEEN, Suit::SPADES));
    
    // Split the hand
    Hand newHand = hand.split();
    
    // Check original hand has King
    EXPECT_EQ(hand.cardCount(), 1);
    EXPECT_EQ(hand.getValue(), 10);
    EXPECT_EQ(hand.getCards()[0].getValue(), 10);
    
    // Check new hand has Queen
    EXPECT_EQ(newHand.cardCount(), 1);
    EXPECT_EQ(newHand.getValue(), 10);
    EXPECT_EQ(newHand.getCards()[0].getValue(), 10);
}

TEST_F(HandTest, SplitPreservesBet) {
    Hand hand(500);
    
    hand.addCard(Card(Rank::SEVEN, Suit::HEARTS));
    hand.addCard(Card(Rank::SEVEN, Suit::CLUBS));
    
    Hand newHand = hand.split();
    
    // Both hands should have the original bet amount
    EXPECT_EQ(hand.getBet(), 500);
    EXPECT_EQ(newHand.getBet(), 500);
}

TEST_F(HandTest, SplitThrowsExceptionWithOneCard) {
    Hand hand(100);
    hand.addCard(Card(Rank::EIGHT, Suit::HEARTS));
    
    // Should throw exception - can't split with only 1 card
    EXPECT_THROW(hand.split(), std::logic_error);
}

TEST_F(HandTest, SplitThrowsExceptionWithThreeCards) {
    Hand hand(100);
    hand.addCard(Card(Rank::EIGHT, Suit::HEARTS));
    hand.addCard(Card(Rank::EIGHT, Suit::SPADES));
    hand.addCard(Card(Rank::THREE, Suit::DIAMONDS));
    
    // Should throw exception - can't split with 3 cards
    EXPECT_THROW(hand.split(), std::logic_error);
}

TEST_F(HandTest, SplitThrowsExceptionWithEmptyHand) {
    Hand hand(100);
    
    // Should throw exception - can't split empty hand
    EXPECT_THROW(hand.split(), std::logic_error);
}

TEST_F(HandTest, SplitAllowsSubsequentCardAddition) {
    Hand hand(100);
    
    hand.addCard(Card(Rank::NINE, Suit::HEARTS));
    hand.addCard(Card(Rank::NINE, Suit::SPADES));
    
    Hand newHand = hand.split();
    
    // Add cards to both split hands
    hand.addCard(Card(Rank::FIVE, Suit::DIAMONDS));
    newHand.addCard(Card(Rank::ACE, Suit::CLUBS));
    
    // Check values
    EXPECT_EQ(hand.cardCount(), 2);
    EXPECT_EQ(hand.getValue(), 14);  // 9 + 5
    
    EXPECT_EQ(newHand.cardCount(), 2);
    EXPECT_EQ(newHand.getValue(), 20);  // 9 + 11 (Ace)
}

TEST_F(HandTest, SplitWithDifferentBetAmounts) {
    Hand hand1(50);
    hand1.addCard(Card(Rank::SIX, Suit::HEARTS));
    hand1.addCard(Card(Rank::SIX, Suit::SPADES));
    Hand newHand1 = hand1.split();
    
    Hand hand2(300);
    hand2.addCard(Card(Rank::TEN, Suit::HEARTS));
    hand2.addCard(Card(Rank::TEN, Suit::CLUBS));
    Hand newHand2 = hand2.split();
    
    // Verify bets are preserved correctly
    EXPECT_EQ(hand1.getBet(), 50);
    EXPECT_EQ(newHand1.getBet(), 50);
    EXPECT_EQ(hand2.getBet(), 300);
    EXPECT_EQ(newHand2.getBet(), 300);
}

TEST_F(HandTest, SplitMaintainsSplitFlag) {
    Hand hand(100);
    
    hand.addCard(Card(Rank::FOUR, Suit::HEARTS));
    hand.addCard(Card(Rank::FOUR, Suit::SPADES));
    
    // Initially not split
    EXPECT_FALSE(hand.getIsSplit());
    
    Hand newHand = hand.split();
    
    // After split, both should be marked
    EXPECT_TRUE(hand.getIsSplit());
    EXPECT_TRUE(newHand.getIsSplit());
    
    // Add cards shouldn't change split flag
    hand.addCard(Card(Rank::SEVEN, Suit::DIAMONDS));
    newHand.addCard(Card(Rank::KING, Suit::CLUBS));
    
    EXPECT_TRUE(hand.getIsSplit());
    EXPECT_TRUE(newHand.getIsSplit());
}

TEST_F(HandTest, SplitWithLowCards) {
    Hand hand(75);
    
    hand.addCard(Card(Rank::TWO, Suit::HEARTS));
    hand.addCard(Card(Rank::TWO, Suit::DIAMONDS));
    
    Hand newHand = hand.split();
    
    // Each hand should have one 2
    EXPECT_EQ(hand.getValue(), 2);
    EXPECT_EQ(newHand.getValue(), 2);
    EXPECT_EQ(hand.cardCount(), 1);
    EXPECT_EQ(newHand.cardCount(), 1);
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

