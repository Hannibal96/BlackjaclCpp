#include <gtest/gtest.h>
#include "../src/Shoe/Shoe.h"
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <string>

// Test fixture for Shoe tests
class ShoeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// Test: Constructor with valid parameters
TEST_F(ShoeTest, ConstructorValidParameters) {
    EXPECT_NO_THROW({
        Shoe shoe(1, 75.0);
    });
    
    EXPECT_NO_THROW({
        Shoe shoe(6, 50.0);
    });
    
    EXPECT_NO_THROW({
        Shoe shoe(8, 100.0);
    });
}

// Test: Constructor with invalid number of decks
TEST_F(ShoeTest, ConstructorInvalidNumDecks) {
    EXPECT_THROW({
        Shoe shoe(0, 75.0);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        Shoe shoe(-1, 75.0);
    }, std::invalid_argument);
}

// Test: Constructor with invalid penetration
TEST_F(ShoeTest, ConstructorInvalidPenetration) {
    EXPECT_THROW({
        Shoe shoe(6, -10.0);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        Shoe shoe(6, 150.0);
    }, std::invalid_argument);
}

// Test: Total cards in shoe equals num_decks * 52
TEST_F(ShoeTest, TotalCardsCorrect) {
    Shoe shoe1(1, 75.0);
    EXPECT_EQ(shoe1.totalCards(), 52);
    
    Shoe shoe2(6, 75.0);
    EXPECT_EQ(shoe2.totalCards(), 312);  // 6 * 52
    
    Shoe shoe3(8, 75.0);
    EXPECT_EQ(shoe3.totalCards(), 416);  // 8 * 52
}

// Test: Cards remaining decreases after dealing
TEST_F(ShoeTest, CardsRemainingDecreasesAfterDeal) {
    Shoe shoe(1, 75.0);
    size_t initialRemaining = shoe.cardsRemaining();
    
    shoe.dealCard();
    EXPECT_EQ(shoe.cardsRemaining(), initialRemaining - 1);
    
    shoe.dealCard();
    EXPECT_EQ(shoe.cardsRemaining(), initialRemaining - 2);
}

// Test: Deal card returns valid cards
TEST_F(ShoeTest, DealCardReturnsValidCards) {
    Shoe shoe(1, 75.0);
    
    for (int i = 0; i < 10; ++i) {
        Card card = shoe.dealCard();
        int value = card.getValue();
        
        // Card values should be between 2 and 11 (Ace = 11)
        EXPECT_GE(value, 2);
        EXPECT_LE(value, 11);
    }
}

// Test: Penetration threshold triggers end_shoe flag
TEST_F(ShoeTest, PenetrationThresholdTriggersEndShoe) {
    Shoe shoe(1, 75.0);
    
    // Initially, end_shoe should be false
    EXPECT_FALSE(shoe.isEndShoe());
    
    // Deal 75% of the cards (39 cards for a single deck)
    size_t cardsToDeal = static_cast<size_t>(52 * 0.75);
    
    for (size_t i = 0; i < cardsToDeal; ++i) {
        shoe.dealCard();
    }
    
    // After dealing 75% of cards, end_shoe should be true
    EXPECT_TRUE(shoe.isEndShoe());
}

// Test: Different penetration levels
TEST_F(ShoeTest, DifferentPenetrationLevels) {
    // Test 50% penetration
    Shoe shoe50(2, 50.0);
    size_t halfCards = static_cast<size_t>(104 * 0.5);
    
    for (size_t i = 0; i < halfCards; ++i) {
        shoe50.dealCard();
    }
    EXPECT_TRUE(shoe50.isEndShoe());
    
    // Test 100% penetration
    Shoe shoe100(1, 100.0);
    for (size_t i = 0; i < 52; ++i) {
        shoe100.dealCard();
    }
    EXPECT_TRUE(shoe100.isEndShoe());
}

// Test: Reset functionality
TEST_F(ShoeTest, ResetRestoresShoe) {
    Shoe shoe(2, 75.0);
    
    // Deal some cards
    for (int i = 0; i < 50; ++i) {
        shoe.dealCard();
    }
    
    size_t remainingBeforeReset = shoe.cardsRemaining();
    EXPECT_LT(remainingBeforeReset, 104);
    
    // Reset the shoe
    shoe.reset();
    
    // After reset, all cards should be available again
    EXPECT_EQ(shoe.cardsRemaining(), 104);
    EXPECT_FALSE(shoe.isEndShoe());
}

// Test: Shuffle randomizes cards
TEST_F(ShoeTest, ShuffleRandomizesCards) {
    Shoe shoe1(1, 100.0);
    Shoe shoe2(1, 100.0);
    
    // Deal first 10 cards from each shoe
    std::vector<std::string> cards1;
    std::vector<std::string> cards2;
    
    for (int i = 0; i < 10; ++i) {
        cards1.push_back(shoe1.dealCard().toString());
        cards2.push_back(shoe2.dealCard().toString());
    }
    
    // The shuffled shoes should likely have different card orders
    // (There's a very small chance they could be the same, but it's extremely unlikely)
    bool different = false;
    for (size_t i = 0; i < cards1.size(); ++i) {
        if (cards1[i] != cards2[i]) {
            different = true;
            break;
        }
    }
    
    EXPECT_TRUE(different);
}

// Test: All 52 unique cards exist in a single deck
TEST_F(ShoeTest, SingleDeckHas52UniqueCards) {
    Shoe shoe(1, 100.0);
    
    std::set<std::string> cardStrings;
    
    // Deal all 52 cards
    for (int i = 0; i < 52; ++i) {
        Card card = shoe.dealCard();
        cardStrings.insert(card.toString());
    }
    
    // Should have exactly 52 unique cards
    EXPECT_EQ(cardStrings.size(), 52);
}

// Test: Multiple decks have correct card distribution
TEST_F(ShoeTest, MultipleDecksHaveCorrectDistribution) {
    int numDecks = 6;
    Shoe shoe(numDecks, 100.0);
    
    std::map<std::string, int> cardCounts;
    
    // Deal all cards
    size_t totalCards = shoe.totalCards();
    for (size_t i = 0; i < totalCards; ++i) {
        Card card = shoe.dealCard();
        cardCounts[card.toString()]++;
    }
    
    // Each unique card should appear exactly numDecks times
    EXPECT_EQ(cardCounts.size(), 52);  // 52 unique cards
    
    for (const auto& pair : cardCounts) {
        EXPECT_EQ(pair.second, numDecks);
    }
}

// Test: Cannot deal more cards than available
TEST_F(ShoeTest, ThrowsWhenDealingBeyondLimit) {
    Shoe shoe(1, 100.0);
    
    // Deal all 52 cards
    for (int i = 0; i < 52; ++i) {
        EXPECT_NO_THROW(shoe.dealCard());
    }
    
    // Trying to deal one more should throw
    EXPECT_THROW(shoe.dealCard(), std::runtime_error);
}

// Test: Card values are correct
TEST_F(ShoeTest, CardValuesAreCorrect) {
    Shoe shoe(6, 100.0);
    
    std::map<int, int> valueCounts;
    
    // Deal all cards and count values
    size_t totalCards = shoe.totalCards();
    for (size_t i = 0; i < totalCards; ++i) {
        Card card = shoe.dealCard();
        int value = card.getValue();
        valueCounts[value]++;
    }
    
    // In 6 decks:
    // Cards 2-9: 24 of each (6 decks * 4 suits)
    // Cards with value 10 (10, J, Q, K): 96 total (6 decks * 4 suits * 4 ranks)
    // Aces (value 11): 24 (6 decks * 4 suits)
    
    EXPECT_EQ(valueCounts[2], 24);
    EXPECT_EQ(valueCounts[3], 24);
    EXPECT_EQ(valueCounts[10], 96);  // 10, J, Q, K all have value 10
    EXPECT_EQ(valueCounts[11], 24);  // Aces
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

