#include <gtest/gtest.h>
#include "RL/BasicStrategy.h"
#include <sstream>

// Test fixture for BasicStrategy tests
class BasicStrategyTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup code if needed
    }

    void TearDown() override {
        // Cleanup code if needed
    }
};

// Test: Print empty strategy
TEST_F(BasicStrategyTest, PrintEmptyStrategy) {
    BasicStrategy strategy;
    std::ostringstream oss;
    oss << strategy;
    
    std::string output = oss.str();
    EXPECT_TRUE(output.find("empty") != std::string::npos || 
                output.find("not loaded") != std::string::npos);
}

// Test: Print strategy with some actions
TEST_F(BasicStrategyTest, PrintStrategyWithActions) {
    BasicStrategy strategy;
    
    // Set some sample actions for count=0, HARD hand type
    strategy.setAction(0, HandType::HARD, 12, 2, ActionWithFallback(Action::HIT));
    strategy.setAction(0, HandType::HARD, 12, 3, ActionWithFallback(Action::HIT));
    strategy.setAction(0, HandType::HARD, 12, 4, ActionWithFallback(Action::STAND));
    strategy.setAction(0, HandType::HARD, 13, 2, ActionWithFallback(Action::STAND));
    strategy.setAction(0, HandType::HARD, 13, 3, ActionWithFallback(Action::STAND));
    strategy.setAction(0, HandType::HARD, 13, 4, ActionWithFallback(Action::STAND));
    
    // Set some SOFT hand actions
    strategy.setAction(0, HandType::SOFT, 18, 2, ActionWithFallback(Action::STAND));
    strategy.setAction(0, HandType::SOFT, 18, 3, ActionWithFallback(Action::DOUBLE_DOWN, Action::STAND));
    strategy.setAction(0, HandType::SOFT, 17, 2, ActionWithFallback(Action::HIT));
    
    // Set some PAIR actions
    strategy.setAction(0, HandType::PAIR, 8, 2, ActionWithFallback(Action::SPLIT));
    strategy.setAction(0, HandType::PAIR, 8, 3, ActionWithFallback(Action::SPLIT));
    strategy.setAction(0, HandType::PAIR, 6, 2, ActionWithFallback(Action::HIT));
    
    std::ostringstream oss;
    oss << strategy;
    
    std::string output = oss.str();
    
    // Verify output contains expected elements
    EXPECT_TRUE(output.find("Count: 0") != std::string::npos);
    EXPECT_TRUE(output.find("Hard") != std::string::npos);
    EXPECT_TRUE(output.find("Soft") != std::string::npos);
    EXPECT_TRUE(output.find("Pair") != std::string::npos);
    
    // Verify dealer card headers appear
    EXPECT_TRUE(output.find("2") != std::string::npos);
    EXPECT_TRUE(output.find("3") != std::string::npos);
    EXPECT_TRUE(output.find("4") != std::string::npos);
    
    // Verify player sums appear
    EXPECT_TRUE(output.find("12") != std::string::npos);
    EXPECT_TRUE(output.find("13") != std::string::npos);
    EXPECT_TRUE(output.find("17") != std::string::npos);
    EXPECT_TRUE(output.find("18") != std::string::npos);
    
    // Verify actions appear (H, S, Ds, P)
    EXPECT_TRUE(output.find("H") != std::string::npos);
    EXPECT_TRUE(output.find("S") != std::string::npos);
    EXPECT_TRUE(output.find("Ds") != std::string::npos);
    EXPECT_TRUE(output.find("P") != std::string::npos);
}

// Test: Print strategy with surrender actions
TEST_F(BasicStrategyTest, PrintStrategyWithSurrenderActions) {
    BasicStrategy strategy;
    
    strategy.setAction(0, HandType::HARD, 16, 9, ActionWithFallback(Action::SURRENDER, Action::HIT));
    strategy.setAction(0, HandType::HARD, 16, 10, ActionWithFallback(Action::SURRENDER, Action::STAND));
    strategy.setAction(0, HandType::HARD, 15, 10, ActionWithFallback(Action::SURRENDER, Action::HIT));
    
    std::ostringstream oss;
    oss << strategy;
    
    std::string output = oss.str();
    
    // Verify surrender actions appear
    EXPECT_TRUE(output.find("Xh") != std::string::npos || output.find("Xs") != std::string::npos);
}

// Test: Load from JSON and print
TEST_F(BasicStrategyTest, LoadAndPrintFromJson) {
    BasicStrategy strategy;
    
    // Try to load a strategy file
    bool loaded = strategy.loadFromJson("decks=1_ss17=False_das=False_surr=no_peek=True");
    
    if (loaded) {
        std::ostringstream oss;
        oss << strategy;
        
        std::string output = oss.str();
        
        // Verify it's not empty
        EXPECT_FALSE(output.empty());
        EXPECT_TRUE(output.find("Count") != std::string::npos);
        EXPECT_TRUE(output.find("Hard") != std::string::npos);
        
        // Print to console for manual inspection (optional)
        // std::cout << output << std::endl;
        std::cout << strategy << std::endl;
    } else {
        // If file doesn't exist, just verify the strategy is not loaded
        EXPECT_FALSE(strategy.isLoaded());
    }
}
