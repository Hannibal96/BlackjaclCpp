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
#include <sstream>

using json = nlohmann::json;


// Global configuration variables (can be overridden via command-line)
uint64_t g_num_rounds = 1'000'000'000ULL;
int g_num_threads = 16;
double g_penetration = 75.0;

// Game configuration parameters (defaults match all combinations)
std::vector<int> g_deck_sizes = {6, 2, 1};
std::vector<bool> g_stand_soft17 = {false, true};
std::vector<bool> g_double_after_split = {false, true};
std::vector<int> g_split_after_split = {4};
std::vector<std::string> g_double_on = {"ANY"};
std::vector<bool> g_resplit_aces = {false};
std::vector<bool> g_hit_split_aces = {false};
std::vector<bool> g_peek = {false, true};
std::vector<std::string> g_surrender = {"no", "yes", "2-10"};
std::vector<float> g_blackjack_pay = {1.5f};

// Q-learning hyperparameters
ExplorationMode g_exploration_mode = ExplorationMode::EPSILON_GREEDY;

double g_epsilon_start = 1.0;
double g_epsilon_min = 0.1;
double g_epsilon_decay = 0.99999;

double g_temperature_start = 1.0;
double g_temperature_min = 0.1;
double g_temperature_decay = 0.99999;

double g_alpha_start = 0.01;
double g_alpha_min = 0.0001;
double g_alpha_decay_steps = 100.0;


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
                        static const std::pair<Action, const char*> allActions[] = {
                            {Action::HIT,         "HIT"},
                            {Action::STAND,       "STAND"},
                            {Action::DOUBLE_DOWN, "DOUBLE_DOWN"},
                            {Action::SPLIT,       "SPLIT"},
                            {Action::SURRENDER,   "SURRENDER"},
                        };
                        static const char* handTypeStr[] = {"HARD", "SOFT", "PAIR"};
                        std::cout << "state {count=0, " << handTypeStr[static_cast<int>(handType)]
                                  << ", player=" << playerSum << ", dealer=" << dealerCard << "}"
                                  << " [expected=" << actionStr.get<std::string>() << "] :";
                        for (auto& [a, name] : allActions) {
                            double q = learned.getQValueDebug(handType, playerSum, dealerCard, a);
                            std::cout << "  " << name << " (" << std::fixed << std::setprecision(4) << q << ")";
                        }
                        std::cout << "\n";
                    }
                }
            }
        }
    }

    return {differences, totalCompared};
}


class QLearningRegressionTest {
public:
    RegressionResult RunRegression(Case c, double penetration, uint64_t rounds, int num_threads) {
        // Create Q-learning strategy with decaying parameters
        auto alpha = std::make_unique<LinearDecayingParameter>(g_alpha_start, g_alpha_min, g_alpha_decay_steps);
        std::unique_ptr<DecayingParameter> exploration;
        if (g_exploration_mode == ExplorationMode::EPSILON_GREEDY) {
            exploration = std::make_unique<EpsilonDecayingParameter>(g_epsilon_start, g_epsilon_min, g_epsilon_decay);
        } else {
            exploration = std::make_unique<EpsilonDecayingParameter>(g_temperature_start, g_temperature_min, g_temperature_decay);
        }
        auto qStrategy = std::make_unique<QLearningStrategy>(std::move(alpha), std::move(exploration), 1.0, g_exploration_mode);

        Player* qPlayer = new Player(0.0, std::move(qStrategy));
        std::vector<Player*> players = {qPlayer};

        BlackjackRules rules = BlackjackRules(c.blackJackPay, c.standSoft17, c.deckSize, penetration, c.peek,
                               c.splitAfterSplit, c.doubleAfterSplit, c.reSplitAces, c.hitSplitAces, c.surrender, c.doubleOn);

        auto startTime = std::chrono::high_resolution_clock::now();
        std::vector<Player*> resultPlayers = runParallelSimulation(rules, players, rounds, num_threads);
        auto endTime = std::chrono::high_resolution_clock::now();

        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        double elapsedSec = elapsedMs / 1000.0;

        if(resultPlayers.empty())
            throw std::runtime_error("Q-learning simulation should complete successfully");

        double finalMoney = resultPlayers[0]->getMoney();
        double handsPerThread = static_cast<double>(rounds) / static_cast<double>(num_threads);
        double edgePerHand = finalMoney / handsPerThread;

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

    void RunTest(const Case& c) {
        std::string scenario = ToString(c);
        std::cout << "\n=== Starting Q-Learning Test ===" << std::endl;
        std::cout << "NUM_ROUNDS:  " << g_num_rounds << std::endl;
        std::cout << "NUM_THREADS: " << g_num_threads << std::endl;
        std::cout << "Scenario:    " << scenario << std::endl;
        std::cout << "===============================\n" << std::endl;

        // Run Q-learning with specified penetration
        std::cout << "Training Q-Learning Agent (Penetration: " << g_penetration << "%)..." << std::endl;
        RegressionResult result = RunRegression(c, g_penetration, g_num_rounds, g_num_threads);

        std::cout << "\n=== Summary ===" << std::endl;
        std::cout << "Edge per Hand: " << std::fixed << std::setprecision(6) << result.empiric_edge << std::endl;
        std::cout << "Edge %: " << std::fixed << std::setprecision(4) << (result.empiric_edge * 100) << "%" << std::endl;
        std::cout << "Training Time: " << std::fixed << std::setprecision(2) << result.time_sec << " seconds" << std::endl;
        std::cout << "Speed: " << std::fixed << std::setprecision(0) << (g_num_rounds / result.time_sec) << " hands/sec" << std::endl;
    }
};




void printHelp(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Q-Learning Regression Test\n";
    std::cout << "Trains Q-learning agents and compares learned strategies with optimal basic strategy.\n\n";
    std::cout << "SIMULATION OPTIONS:\n";
    std::cout << "  --num-rounds <N>         Number of training rounds (default: 1000000000)\n";
    std::cout << "  --num-threads <N>        Number of parallel threads (default: 16)\n";
    std::cout << "  --penetration <val>      Shoe penetration percentage (default: 75.0)\n";
    std::cout << "                           Percentage of cards dealt before reshuffling.\n";
    std::cout << "                           75.0 means 75% of shoe is dealt (typical casino).\n";
    std::cout << "                           Higher penetration allows better card counting.\n\n";
    std::cout << "GAME CONFIGURATION OPTIONS (specify values to test):\n";
    std::cout << "  --decks <list>           Deck sizes to test (default: [6,2,1])\n";
    std::cout << "  --ss17 <list>            Stand on soft 17 (default: [true,false])\n";
    std::cout << "  --das <list>             Double after split (default: [true,false])\n";
    std::cout << "  --sas <list>             Max split after split (default: [4])\n";
    std::cout << "                           NOTE: Not relevant for strategy calculation\n";
    std::cout << "  --don <list>             Double on: ANY, 9-11, 10-11 (default: [ANY])\n";
    std::cout << "                           NOTE: Not relevant for strategy calculation\n";
    std::cout << "  --rsa <list>             Re-split aces (default: [false])\n";
    std::cout << "                           NOTE: Not relevant for strategy calculation\n";
    std::cout << "  --hsa <list>             Hit split aces (default: [false])\n";
    std::cout << "                           NOTE: Not relevant for strategy calculation\n";
    std::cout << "  --peek <list>            Dealer peeks (default: [true,false])\n";
    std::cout << "  --surr <list>            Surrender: no, yes, 2-10 (default: [no,yes,2-10])\n";
    std::cout << "  --bj <list>              Blackjack payout (default: [1.5])\n";
    std::cout << "                           NOTE: Not relevant for strategy calculation\n\n";
    std::cout << "Q-LEARNING HYPERPARAMETERS:\n";
    std::cout << "  --exploration <mode>     Exploration strategy: epsilon or boltzmann (default: epsilon)\n\n";
    std::cout << "  EPSILON-GREEDY params (--exploration epsilon):\n";
    std::cout << "  --epsilon-start <val>    Initial epsilon (default: 1.0)\n";
    std::cout << "  --epsilon-min <val>      Minimum epsilon (default: 0.1)\n";
    std::cout << "  --epsilon-decay <val>    Epsilon decay rate per hand (default: 0.99999)\n\n";
    std::cout << "  BOLTZMANN params (--exploration boltzmann):\n";
    std::cout << "  --temp-start <val>       Initial temperature (default: 1.0)\n";
    std::cout << "  --temp-min <val>         Minimum temperature (default: 0.1)\n";
    std::cout << "  --temp-decay <val>       Temperature decay rate per hand (default: 0.99999)\n\n";
    std::cout << "  SHARED params:\n";
    std::cout << "  --alpha-start <val>      Initial learning rate (default: 0.01)\n";
    std::cout << "  --alpha-min <val>        Minimum learning rate (default: 0.0001)\n";
    std::cout << "  --alpha-decay <val>      Alpha decay steps (default: 100)\n\n";
    std::cout << "OTHER OPTIONS:\n";
    std::cout << "  --help, -h               Show this help message\n\n";
    std::cout << "EXAMPLES:\n";
    std::cout << "  # Epsilon-greedy (default)\n";
    std::cout << "  " << program_name << " --exploration epsilon --epsilon-start 1.0 --epsilon-min 0.05 --epsilon-decay 0.99999\n\n";
    std::cout << "  # Boltzmann exploration\n";
    std::cout << "  " << program_name << " --exploration boltzmann --temp-start 1.0 --temp-min 0.1 --temp-decay 0.99999\n\n";
    std::cout << "  # Quick single-config test\n";
    std::cout << "  " << program_name << " --num-rounds 1000000 --decks [6] --ss17 [true] --das [true] --peek [false] --surr [no]\n\n";
    std::cout << "NOTE: List values can be specified with or without brackets: [1,2,3] or 1,2,3\n";
}


int main(int argc, char** argv) {
    // Parse command-line arguments
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
        else if (arg == "--penetration" && i + 1 < argc) {
            g_penetration = std::stod(argv[++i]);
        }
        else if (arg == "--decks" && i + 1 < argc) {
            g_deck_sizes = parseList<int>(argv[++i]);
        }
        else if (arg == "--ss17" && i + 1 < argc) {
            g_stand_soft17 = parseList<bool>(argv[++i]);
        }
        else if (arg == "--das" && i + 1 < argc) {
            g_double_after_split = parseList<bool>(argv[++i]);
        }
        else if (arg == "--sas" && i + 1 < argc) {
            g_split_after_split = parseList<int>(argv[++i]);
        }
        else if (arg == "--don" && i + 1 < argc) {
            g_double_on = parseList<std::string>(argv[++i]);
        }
        else if (arg == "--rsa" && i + 1 < argc) {
            g_resplit_aces = parseList<bool>(argv[++i]);
        }
        else if (arg == "--hsa" && i + 1 < argc) {
            g_hit_split_aces = parseList<bool>(argv[++i]);
        }
        else if (arg == "--peek" && i + 1 < argc) {
            g_peek = parseList<bool>(argv[++i]);
        }
        else if (arg == "--surr" && i + 1 < argc) {
            g_surrender = parseList<std::string>(argv[++i]);
        }
        else if (arg == "--bj" && i + 1 < argc) {
            g_blackjack_pay = parseList<float>(argv[++i]);
        }
        else if (arg == "--exploration" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "boltzmann") g_exploration_mode = ExplorationMode::BOLTZMANN;
            else if (mode == "epsilon") g_exploration_mode = ExplorationMode::EPSILON_GREEDY;
            else { std::cerr << "Unknown exploration mode: " << mode << " (use epsilon or boltzmann)\n"; return 1; }
        }
        else if (arg == "--epsilon-start" && i + 1 < argc) {
            g_epsilon_start = std::stod(argv[++i]);
        }
        else if (arg == "--epsilon-min" && i + 1 < argc) {
            g_epsilon_min = std::stod(argv[++i]);
        }
        else if (arg == "--epsilon-decay" && i + 1 < argc) {
            g_epsilon_decay = std::stod(argv[++i]);
        }
        else if (arg == "--temp-start" && i + 1 < argc) {
            g_temperature_start = std::stod(argv[++i]);
        }
        else if (arg == "--temp-min" && i + 1 < argc) {
            g_temperature_min = std::stod(argv[++i]);
        }
        else if (arg == "--temp-decay" && i + 1 < argc) {
            g_temperature_decay = std::stod(argv[++i]);
        }
        else if (arg == "--alpha-start" && i + 1 < argc) {
            g_alpha_start = std::stod(argv[++i]);
        }
        else if (arg == "--alpha-min" && i + 1 < argc) {
            g_alpha_min = std::stod(argv[++i]);
        }
        else if (arg == "--alpha-decay" && i + 1 < argc) {
            g_alpha_decay_steps = std::stod(argv[++i]);
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Use --help for usage information" << std::endl;
            return 1;
        }
    }

    // Generate test cases based on configuration
    std::vector<Case> allCases = generateTestCases(
        g_deck_sizes, g_stand_soft17, g_double_after_split, g_split_after_split,
        g_double_on, g_resplit_aces, g_hit_split_aces, g_peek, g_surrender, g_blackjack_pay
    );

    // Print configuration
    std::cout << "=== Global Configuration ===" << std::endl;
    std::cout << "NUM_ROUNDS:  " << g_num_rounds << std::endl;
    std::cout << "NUM_THREADS: " << g_num_threads << std::endl;
    std::cout << "Total test cases: " << allCases.size() << std::endl;
    if (g_exploration_mode == ExplorationMode::EPSILON_GREEDY) {
        std::cout << "\nQ-Learning Hyperparameters (Epsilon-Greedy):" << std::endl;
        std::cout << "  Epsilon: " << g_epsilon_start << " -> " << g_epsilon_min << " (decay: " << g_epsilon_decay << ")" << std::endl;
    } else {
        std::cout << "\nQ-Learning Hyperparameters (Boltzmann):" << std::endl;
        std::cout << "  Temperature: " << g_temperature_start << " -> " << g_temperature_min << " (decay: " << g_temperature_decay << ")" << std::endl;
    }
    std::cout << "  Alpha:       " << g_alpha_start << " -> " << g_alpha_min << " (steps: " << g_alpha_decay_steps << ")" << std::endl;
    std::cout << "============================\n" << std::endl;

    // Run all test cases
    QLearningRegressionTest test;
    int passed = 0;
    int failed = 0;

    for (size_t i = 0; i < allCases.size(); ++i) {
        std::cout << "\n[" << (i + 1) << "/" << allCases.size() << "] ";
        try {
            test.RunTest(allCases[i]);
            passed++;
        } catch (const std::exception& e) {
            std::cerr << "Test failed: " << e.what() << std::endl;
            failed++;
        }
    }

    // Print summary
    std::cout << "\n\n=== Test Summary ===" << std::endl;
    std::cout << "Total:  " << allCases.size() << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "====================" << std::endl;

    return (failed == 0) ? 0 : 1;
}
