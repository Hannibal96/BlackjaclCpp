#include <gtest/gtest.h>
#include "../src/Shoe/Shoe.h"
#include <vector>

class DebugShoeTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(DebugShoeTest, DeterministicBehavior) {
    int numDecks = 1;
    double penetration = 75.0;
    int seed = 42;
    
    DebugShoe shoe1(numDecks, penetration, seed);
    DebugShoe shoe2(numDecks, penetration, seed);
    
    for (int i = 0; i < 10; ++i) {
        Card c1 = shoe1.dealCard();
        Card c2 = shoe2.dealCard();
        
        EXPECT_EQ(c1.rank, c2.rank);
        EXPECT_EQ(c1.suit, c2.suit);
    }
}

TEST_F(DebugShoeTest, ResetBehavior) {
    int numDecks = 1;
    double penetration = 75.0;
    int seed = 42;
    
    DebugShoe shoe(numDecks, penetration, seed);
    
    std::vector<Card> dealtCards;
    for (int i = 0; i < 10; ++i) {
        dealtCards.push_back(shoe.dealCard());
    }
    
    shoe.reset();
    
    for (int i = 0; i < 10; ++i) {
        Card c = shoe.dealCard();
        EXPECT_EQ(c.rank, dealtCards[i].rank);
        EXPECT_EQ(c.suit, dealtCards[i].suit);
    }
}

TEST_F(DebugShoeTest, CardRemoval) {
    int numDecks = 1;
    double penetration = 75.0;
    int seed = 42;
    
    DebugShoe shoe(numDecks, penetration, seed);
    size_t initialCount = shoe.totalCards();
    
    shoe.dealCard();
    // In DebugShoe, cards are removed.
    // cardsRemaining() = cards.size() - currentIndex (0) = cards.size()
    // So it should decrease by 1.
    EXPECT_EQ(shoe.cardsRemaining(), initialCount - 1);
    
    shoe.dealCard();
    EXPECT_EQ(shoe.cardsRemaining(), initialCount - 2);
}
