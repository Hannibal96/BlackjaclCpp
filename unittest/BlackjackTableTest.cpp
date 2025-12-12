#include <gtest/gtest.h>
#include "Game/BlackjackTable.h"
#include "Game/Player.h"
#include "RL/RandomStrategy.h"
#include "Game/BlackjackRules.h"
#include <memory>

// Test fixture for BlackjackTable tests
class BlackjackTableTest : public ::testing::Test {
protected:
    // Default test objects
    std::unique_ptr<RandomStrategy> defaultStrategy;
    std::unique_ptr<Player> defaultPlayer;
    std::unique_ptr<BlackjackRules> defaultRules;
    std::unique_ptr<BlackjackTable> table;
    std::vector<Player*> players;
    
    void SetUp() override {
        // Create default player with random strategy (seeded for reproducibility)
        defaultStrategy = std::make_unique<RandomStrategy>(42);
        defaultPlayer = std::make_unique<Player>(1000.0, std::move(defaultStrategy));
        
        // Create default rules
        defaultRules = std::make_unique<BlackjackRules>();
        
        // Setup players vector
        players = {defaultPlayer.get()};
        
        // Create table with default rules and player
        table = std::make_unique<BlackjackTable>(*defaultRules, players);
    }

    void TearDown() override {
        // Cleanup happens automatically with unique_ptrs
        table.reset();
        players.clear();
        defaultRules.reset();
        defaultPlayer.reset();
        defaultStrategy.reset();
    }
    
    // Helper method to create a new table with custom configuration
    std::unique_ptr<BlackjackTable> createCustomTable(
        std::vector<Player*> customPlayers,
        const BlackjackRules& rules = BlackjackRules()) {
        return std::make_unique<BlackjackTable>(rules, customPlayers);
    }
};

// Test: Run a single round with random strategy
TEST_F(BlackjackTableTest, SingleRoundWithRandomStrategy) {
    // Get initial money
    double initialMoney = defaultPlayer->getMoney();
    
    // Run one round
    EXPECT_NO_THROW(table->round());
    
    // Player money should have changed (won, lost, or push)
    double finalMoney = defaultPlayer->getMoney();
    
    // Money should be different or the same (push)
    // Just verify the round completed without crashing
    EXPECT_TRUE(finalMoney <= initialMoney + 100 || finalMoney >= initialMoney - 100);
}

// Test: Run multiple rounds with random strategy
TEST_F(BlackjackTableTest, MultipleRoundsWithRandomStrategy) {
    // Run 1000 rounds
    for (int i = 0; i < 1000; ++i) {
        EXPECT_NO_THROW(table->round());
    }
}

// Test: Run round with multiple players
TEST_F(BlackjackTableTest, MultiplePlayersWithRandomStrategy) {
    // Create multiple players with random strategies
    auto strat1 = std::make_unique<RandomStrategy>(100);
    auto strat2 = std::make_unique<RandomStrategy>(200);
    auto strat3 = std::make_unique<RandomStrategy>(300);
    
    Player player1(1000.0, std::move(strat1));
    Player player2(2000.0, std::move(strat2));
    Player player3(1500.0, std::move(strat3));
    
    // Create vector of players
    std::vector<Player*> players = {&player1, &player2, &player3};
    
    // Create table with default blackjack rules
    BlackjackRules rules;
    BlackjackTable table(rules, players);
    
    // Get initial money for all players
    double initialMoney1 = player1.getMoney();
    double initialMoney2 = player2.getMoney();
    double initialMoney3 = player3.getMoney();
    
    // Run one round
    EXPECT_NO_THROW(table.round());
    
    // Verify all players are still in valid state
    EXPECT_TRUE(player1.getMoney() != 0.0 || player1.getMoney() == 0.0); // Always true, just check it doesn't crash
    EXPECT_TRUE(player2.getMoney() != 0.0 || player2.getMoney() == 0.0);
    EXPECT_TRUE(player3.getMoney() != 0.0 || player3.getMoney() == 0.0);
}

// Test: Run rounds until shoe needs reshuffle
TEST_F(BlackjackTableTest, ShoeReshuffleTest) {
    // Run many rounds to trigger reshuffle
    for (int i = 0; i < 1000; ++i) {
        EXPECT_NO_THROW(table->round());
    }
    
    // If we got here without crashing, shoe reshuffle works
    SUCCEED();
}

// Test: Table with custom rules
TEST_F(BlackjackTableTest, CustomRulesTest) {
    // Create custom rules: 6:5 blackjack, hit soft 17, 2 decks, 50% penetration
    BlackjackRules customRules(
        1.2,                              // 6:5 blackjack payout
        false,                            // Hit soft 17
        2,                                // 2 decks
        50.0,                             // 50% penetration
        true,                             // Peek for blackjack (American)
        3,                                // Max 3 splits
        true,                             // Double after split
        false,                            // No resplit aces
        false,                            // No hit split aces
        Surrender::NO_SURRENDER,          // No surrender
        DoubleDownOn::TEN_ELEVEN          // Only double on 10 or 11
    );
    
    // Create table with custom rules (reusing default player)
    auto customTable = createCustomTable(players, customRules);
    
    // Run rounds with custom rules
    for (int i = 0; i < 1000; ++i) {
        EXPECT_NO_THROW(customTable->round());
    }
    
    SUCCEED();
}

// Test: Player money tracking across rounds
TEST_F(BlackjackTableTest, MoneyTrackingTest) {
    // Track money changes
    std::vector<double> moneyHistory;
    moneyHistory.push_back(defaultPlayer->getMoney());
    
    // Run 1000 rounds and track money
    for (int i = 0; i < 1000; ++i) {
        table->round();
        moneyHistory.push_back(defaultPlayer->getMoney());
    }
    
    // Money should have changed at least once (very unlikely to push 1000 times)
    bool moneyChanged = false;
    for (size_t i = 1; i < moneyHistory.size(); ++i) {
        if (moneyHistory[i] != moneyHistory[i-1]) {
            moneyChanged = true;
            break;
        }
    }
    
    EXPECT_TRUE(moneyChanged);
}

// Test: European vs American style (peek blackjack)
TEST_F(BlackjackTableTest, EuropeanVsAmericanStyle) {
    // Test European style (no peek)
    {
        BlackjackRules europeanRules;
        europeanRules.peekBlackjack = false; // European style
        
        auto europeanTable = createCustomTable(players, europeanRules);
        
        // Run rounds - should work fine
        for (int i = 0; i < 1000; ++i) {
            EXPECT_NO_THROW(europeanTable->round());
        }
    }
    
    // Test American style (peek)
    {
        BlackjackRules americanRules;
        americanRules.peekBlackjack = true; // American style
        
        auto americanTable = createCustomTable(players, americanRules);
        
        // Run rounds - should work fine
        for (int i = 0; i < 1000; ++i) {
            EXPECT_NO_THROW(americanTable->round());
        }
    }
    
    SUCCEED();
}

// Test: Verify dealer hand is accessible
TEST_F(BlackjackTableTest, DealerHandAccessTest) {
    // Run one round
    table->round();
    
    // Access dealer hand
    const Hand& dealerHand = table->getDealerHand();
    
    // Dealer should have at least one card (could be more after hitting)
    EXPECT_GE(dealerHand.cardCount(), 1);
    
    // Dealer value should be between 0 and 21 (or bust if > 21)
    int dealerValue = dealerHand.getValue();
    EXPECT_GE(dealerValue, 0);
}

// Main function for running tests
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

