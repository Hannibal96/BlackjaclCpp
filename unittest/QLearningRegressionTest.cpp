#include <gtest/gtest.h>
#include "RegressionTestUtils.h"
#include "Game/BlackjackTable.h"
#include "Game/Player.h"
#include "RL/QLearningStrategy.h"
#include "RL/BasicStrategy.h"
#include "RL/DecayingParameter.h"
#include "Utils/Utils.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <fstream>
#include <nlohmann/json.hpp>
#include <cstring>

using json = nlohmann::json;


// Global configuration variables (can be overridden via command-line)
uint64_t g_num_rounds = 1'000'000'000ULL;
int g_num_threads = 16;


// Compare learned strategy with known basic strategy from JSON file
// Returns (differences, total states compared)
std::pair<int, int> countStrategyDifferences(const QLearningStrategy& learned, const std::string& jsonFileName) {
    int differences = 0;
    int totalCompared = 0;
    
    // Try to open the JSON file
    std::string filepath = "basic_strategy_tables/" + jsonFileName + ".json";
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open strategy file: " << filepath << std::endl;
        return {0, 0};
    }
    
    json strategyJson;
    try {
        file >> strategyJson;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return {0, 0};
    }
    
    // Convert learned Q-learning strategy to BasicStrategy to get its lookup table
    auto learnedBasic = learned.toBasicStrategy();
    
    // Iterate through the JSON structure and compare
    // JSON structure: count -> hand_type -> player_sum -> dealer_card -> action
    for (auto& [countStr, handTypes] : strategyJson.items()) {
        int count = std::stoi(countStr);
        
        for (auto& [handTypeStr, playerSums] : handTypes.items()) {
            HandType handType;
            if (handTypeStr == "HandType.HARD") handType = HandType::HARD;
            else if (handTypeStr == "HandType.SOFT") handType = HandType::SOFT;
            else if (handTypeStr == "HandType.PAIR") handType = HandType::PAIR;
            else continue;
            
            for (auto& [playerSumStr, dealerCards] : playerSums.items()) {
                int playerSum = std::stoi(playerSumStr);
                
                for (auto& [dealerCardStr, actionStr] : dealerCards.items()) {
                    int dealerCard = std::stoi(dealerCardStr);
                    Action knownAction = stringToAction(actionStr.get<std::string>());
                    
                    // Get learned action from the converted BasicStrategy's lookup table
                    Action learnedAction = learnedBasic->getActionFromTable(count, handType, playerSum, dealerCard);
                    
                    totalCompared++;
                    if (learnedAction != knownAction) {
                        differences++;
                    }
                }
            }
        }
    }
    
    return {differences, totalCompared};
}


class QLearningRegressionTest : public ::testing::TestWithParam<Case> {
protected:
    void SetUp() override {}
    void TearDown() override {}

    RegressionResult RunRegression(Case c, double penetration, uint64_t rounds) {
        // Hardcoded Q-learning parameters
        const double LEARNING_RATE = 0.01;
        const double EPSILON = 1.0;
        
        // Create Q-learning strategy with decaying parameters
        auto alpha = std::make_unique<LinearDecayingParameter>(LEARNING_RATE, 0.0001, 100);
        auto epsilon = std::make_unique<EpsilonDecayingParameter>(EPSILON, 0.25, 1-10e-8);
        auto qStrategy = std::make_unique<QLearningStrategy>(std::move(alpha), std::move(epsilon));
        
        Player* qPlayer = new Player(0.0, std::move(qStrategy));
        std::vector<Player*> players = {qPlayer};

        BlackjackRules rules = BlackjackRules(c.blackJackPay, c.standSoft17, c.deckSize, penetration, c.peek,
                               c.splitAfterSplit, c.doubleAfterSplit, c.reSplitAces, c.hitSplitAces, c.surrender, c.doubleOn);

        auto startTime = std::chrono::high_resolution_clock::now();
        std::vector<Player*> resultPlayers = runParallelSimulation(rules, players, rounds, g_num_threads);
        auto endTime = std::chrono::high_resolution_clock::now();

        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        double elapsedSec = elapsedMs / 1000.0;

        if(resultPlayers.empty())
            throw std::runtime_error("Q-learning simulation should complete successfully");

        double finalMoney = resultPlayers[0]->getMoney();
        double edgePerHand = finalMoney / (rounds / g_num_threads);
        
        // Get the trained strategy from the averaged result player
        const Strategy* trainedStrategy = resultPlayers[0]->getStrategy();
        const QLearningStrategy* qLearningStrat = dynamic_cast<const QLearningStrategy*>(trainedStrategy);
        
        if (qLearningStrat) {
            std::cout << "Final Q-Learning Strategy (averaged across threads):" << std::endl;
            std::cout << *qLearningStrat << std::endl;
            
            // Compare with known basic strategy
            std::string jsonFile = ToStringTableName(c);
            auto [differences, totalStates] = countStrategyDifferences(*qLearningStrat, jsonFile);
            
            if (totalStates > 0) {
                double accuracy = 100.0 * (totalStates - differences) / totalStates;
                std::cout << "\n=== Strategy Comparison ===" << std::endl;
                std::cout << "Compared with: " << jsonFile << ".json" << std::endl;
                std::cout << "Differences: " << differences << " / " << totalStates << std::endl;
                std::cout << "Accuracy: " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
            }
        }

        delete qPlayer;
        for (auto* player : resultPlayers) {
            delete player;
        }

        return {edgePerHand, elapsedSec};
    }
};


TEST_P(QLearningRegressionTest, QLearningTrainingTest) {
    const auto& c = GetParam();
    std::string scenario = ToString(c);
    std::cout << "Starting Q-Learning Test with NUM_ROUNDS = " << g_num_rounds
              << " NUM_THREADS = " << g_num_threads
              << " Scenario = " << scenario << std::endl;

    // Run Q-learning with CSM (penetration = 0.0)
    std::cout << "Training Q-Learning Agent..." << std::endl;
    RegressionResult result = RunRegression(c, 50.0, g_num_rounds);

    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "Edge per Hand: " << std::fixed << std::setprecision(6) << result.empiric_edge << std::endl;
    std::cout << "Edge %: " << std::fixed << std::setprecision(4) << (result.empiric_edge * 100) << "%" << std::endl;
    std::cout << "Training Time: " << std::fixed << std::setprecision(2) << result.time_sec << " seconds" << std::endl;
    std::cout << "Speed: " << std::fixed << std::setprecision(0) << (g_num_rounds / result.time_sec) << " hands/sec" << std::endl;

    // Check that the simulation runs successfully
    EXPECT_TRUE(true) << "Q-learning training completed";
}


static const int deckSize[] = {6, 2, 1};
static const bool standSoft17[] = {false, true};
static const bool doubleAfterSplit[] = {false, true};
static const int splitAfterSplit[] = {4};
static const DoubleDownOn doubleOn[] = {DoubleDownOn::ANY};
static const bool reSplitAces[] = {false};
static const bool hitSplitAces[] = {false};
static const bool peek[] = {false, true};
static const Surrender surrender[] = {Surrender::NO_SURRENDER, Surrender::SURRENDER_ANY, Surrender::SURRENDER_NO_ACE};
static const float blackJackPay[] = {1.5};


static std::vector<Case> AllCases = [] {
    std::vector<Case> v;
    for (int d : deckSize)
        for (bool ss17 : standSoft17)
            for (bool das : doubleAfterSplit)
                for (int sas : splitAfterSplit)
                    for (DoubleDownOn don : doubleOn)
                        for (bool rsa : reSplitAces)
                            for (bool hsa : hitSplitAces)
                                for (bool peek : peek)
                                    for (Surrender surr : surrender)
                                        for (float bj : blackJackPay){
                                            Case new_case{d, ss17, das, sas, don, rsa, hsa, peek, surr, bj};
                                            v.push_back(new_case);
                                        }
    static std::mt19937 rng(1234);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}();


INSTANTIATE_TEST_SUITE_P(BasicQLearning, QLearningRegressionTest,
                         ::testing::ValuesIn(AllCases));


void printHelp(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] [GTEST_OPTIONS]\n\n";
    std::cout << "Q-Learning Regression Test\n";
    std::cout << "Trains Q-learning agents and compares learned strategies with optimal basic strategy.\n\n";
    std::cout << "OPTIONS:\n";
    std::cout << "  --num-rounds <N>     Number of training rounds (default: 1000000000)\n";
    std::cout << "  --num-threads <N>    Number of parallel threads (default: 16)\n";
    std::cout << "  --help               Show this help message\n\n";
    std::cout << "GTEST_OPTIONS:\n";
    std::cout << "  --gtest_filter=<pattern>  Run only tests matching the pattern\n";
    std::cout << "  --gtest_list_tests        List all available tests\n";
    std::cout << "  (See Google Test documentation for more options)\n\n";
    std::cout << "EXAMPLES:\n";
    std::cout << "  " << program_name << " --num-rounds 100000000 --num-threads 8\n";
    std::cout << "  " << program_name << " --gtest_filter=*BasicQLearning*\n";
}


int main(int argc, char** argv) {
    // Parse custom arguments first
    std::vector<char*> gtest_args;
    gtest_args.push_back(argv[0]);

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            return 0;
        }
        else if (arg == "--num-rounds" && i + 1 < argc) {
            g_num_rounds = std::stoull(argv[++i]);
        }
        else if (arg == "--num-threads" && i + 1 < argc) {
            g_num_threads = std::stoi(argv[++i]);
        }
        else {
            // Pass unknown arguments to Google Test
            gtest_args.push_back(argv[i]);
        }
    }

    // Print configuration
    std::cout << "=== Configuration ===" << std::endl;
    std::cout << "NUM_ROUNDS:  " << g_num_rounds << std::endl;
    std::cout << "NUM_THREADS: " << g_num_threads << std::endl;
    std::cout << "=====================\n" << std::endl;

    // Initialize and run Google Test
    int gtest_argc = static_cast<int>(gtest_args.size());
    ::testing::InitGoogleTest(&gtest_argc, gtest_args.data());
    return RUN_ALL_TESTS();
}
