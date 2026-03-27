#include "RegressionTestUtils.h"
#include "Game/BlackjackTable.h"
#include "Game/Player.h"
#include "RL/QLearningStrategy.h"
#include "RL/BasicStrategy.h"
#include "RL/DecayingParameter.h"
#include "Shoe/Shoe.h"
#include "Utils/GlobalRNG.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;

// Hyperparameters matching the Python reference implementation
// alpha = LinearDecayingParam(initial_value=0.01, final_value=0.00001, n=10)
// epsilon = ExpDecayingParam(initial_value=1, final_value=0.3, gamma=1 - 1e-8)
static double g_alpha_start       = 0.01;
static double g_alpha_min         = 0.00001;
static int    g_alpha_decay_steps = 10;

static double g_epsilon_start = 1.0;
static double g_epsilon_min   = 0.3;
static double g_epsilon_decay = 1.0 - 1e-8;

static uint64_t    g_num_rounds  = 10'000'000ULL;
static double      g_penetration = 100.0;
static uint32_t    g_seed        = 42;

// Game configuration (default: 6 decks, stand soft 17, DAS, no peek, no surrender)
static int         g_decks         = 6;
static bool        g_ss17          = true;
static bool        g_das           = true;
static bool        g_peek          = false;
static std::string g_surrender_str = "2-10";
static float       g_bj_pay        = 1.5f;


// ── reuse the JSON comparison from the existing regression test ──────────────
std::pair<int, int> countStrategyDifferencesDebug(const QLearningStrategy& learned,
                                                   const std::string& jsonFileName) {
    int differences = 0;
    int totalCompared = 0;

    std::string filepath = "basic_strategy_tables/" + jsonFileName + ".json";
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open strategy file: " << filepath << std::endl;
        return {0, 0};
    }

    json strategyJson;
    try { file >> strategyJson; }
    catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return {0, 0};
    }

    auto learnedBasic = learned.toBasicStrategy();

    for (auto& [countStr, handTypes] : strategyJson.items()) {
        int count = std::stoi(countStr);
        for (auto& [handTypeStr, playerSums] : handTypes.items()) {
            HandType handType;
            if      (handTypeStr == "HandType.HARD") handType = HandType::HARD;
            else if (handTypeStr == "HandType.SOFT") handType = HandType::SOFT;
            else if (handTypeStr == "HandType.PAIR") handType = HandType::PAIR;
            else continue;

            for (auto& [playerSumStr, dealerCards] : playerSums.items()) {
                int playerSum = std::stoi(playerSumStr);
                for (auto& [dealerCardStr, actionStr] : dealerCards.items()) {
                    int dealerCard = std::stoi(dealerCardStr);
                    Action knownAction = stringToAction(actionStr.get<std::string>());
                    Action learnedAction = learnedBasic->getActionFromTable(count, handType, playerSum, dealerCard);
                    totalCompared++;
                    if (learnedAction != knownAction) differences++;
                }
            }
        }
    }
    return {differences, totalCompared};
}


void runDebugTest(const Case& c) {
    std::cout << "\n=== QLearningDebugRegressionTest ===" << std::endl;
    std::cout << "Seed:        " << g_seed        << std::endl;
    std::cout << "Rounds:      " << g_num_rounds  << std::endl;
    std::cout << "Penetration: " << g_penetration << "%" << std::endl;
    std::cout << "Scenario:    " << ToString(c)   << std::endl;
    std::cout << "Alpha:       " << g_alpha_start << " -> " << g_alpha_min
              << "  (N=" << g_alpha_decay_steps << ")" << std::endl;
    std::cout << "Epsilon:     " << g_epsilon_start << " -> " << g_epsilon_min
              << "  (gamma=" << g_epsilon_decay << ")" << std::endl;
    std::cout << "====================================\n" << std::endl;

    // Single shared RNG — same instance drives both shoe and agent decisions
    GlobalRNG rng(g_seed);

    // DebugShoe uses the shared rng for card selection (matches Python DebugShoe.deal)
    auto shoe = std::make_unique<DebugShoe>(c.deckSize, g_penetration, &rng);

    // QLearningStrategy uses the same rng for epsilon-greedy (matches Python QLearningAgent)
    auto alpha   = std::make_unique<LinearDecayingParameter>(g_alpha_start, g_alpha_min, g_alpha_decay_steps);
    auto epsilon = std::make_unique<EpsilonDecayingParameter>(g_epsilon_start, g_epsilon_min, g_epsilon_decay);
    auto qStrat  = std::make_unique<QLearningStrategy>(std::move(alpha), std::move(epsilon), 1.0, &rng);

    Player* qPlayer = new Player(0.0, std::move(qStrat));
    std::vector<Player*> players = {qPlayer};

    BlackjackRules rules(c.blackJackPay, c.standSoft17, c.deckSize, g_penetration,
                         c.peek, c.splitAfterSplit, c.doubleAfterSplit,
                         c.reSplitAces, c.hitSplitAces, c.surrender, c.doubleOn);

    // Pass the pre-built DebugShoe directly into the table
    BlackjackTable table(rules, players, std::move(shoe));

    auto startTime = std::chrono::high_resolution_clock::now();
    for (uint64_t i = 0; i < g_num_rounds; ++i) {
        table.round();
        std::cout << table << std::endl;
    }
    auto endTime = std::chrono::high_resolution_clock::now();
    double elapsedSec = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0;

    // Print learned strategy and compare with basic strategy
    const Strategy* trainedStrategy = qPlayer->getStrategy();
    const QLearningStrategy* qLearned = dynamic_cast<const QLearningStrategy*>(trainedStrategy);

    if (qLearned) {
        std::cout << "Learned Strategy:" << std::endl;
        std::cout << *qLearned << std::endl;

        std::string jsonFile = ToStringTableName(c);
        auto [differences, totalStates] = countStrategyDifferencesDebug(*qLearned, jsonFile);

        if (totalStates > 0) {
            double accuracy = 100.0 * (totalStates - differences) / totalStates;
            std::cout << "\n=== Strategy Comparison ===" << std::endl;
            std::cout << "Compared with: " << jsonFile << ".json" << std::endl;
            std::cout << "Differences:   " << differences << " / " << totalStates << std::endl;
            std::cout << "Accuracy:      " << std::fixed << std::setprecision(2) << accuracy << "%" << std::endl;
        }
    }

    double finalMoney = qPlayer->getMoney();
    double edgePerHand = finalMoney / static_cast<double>(g_num_rounds);
    std::cout << "\nEdge per hand: " << std::fixed << std::setprecision(6) << edgePerHand << std::endl;
    std::cout << "Time: " << std::fixed << std::setprecision(2) << elapsedSec << "s" << std::endl;

    delete qPlayer;
}


void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Single-threaded Q-learning regression test with reproducible randomness.\n";
    std::cout << "Shares one SeededRNG between the shoe and the agent — identical to the Python implementation.\n\n";
    std::cout << "  --seed <N>            RNG seed (default: 42)\n";
    std::cout << "  --num-rounds <N>      Training rounds (default: 10000000)\n";
    std::cout << "  --penetration <val>   Shoe penetration % (default: 100.0)\n";
    std::cout << "  --decks <N>           Number of decks (default: 6)\n";
    std::cout << "  --ss17 <true|false>   Stand on soft 17 (default: true)\n";
    std::cout << "  --das <true|false>    Double after split (default: true)\n";
    std::cout << "  --peek <true|false>   Dealer peeks (default: false)\n";
    std::cout << "  --surr <no|yes|2-10>  Surrender (default: no)\n";
    std::cout << "  --alpha-start <val>   Initial alpha (default: 0.01)\n";
    std::cout << "  --alpha-min <val>     Final alpha (default: 0.00001)\n";
    std::cout << "  --alpha-decay <N>     Alpha decay steps N (default: 10)\n";
    std::cout << "  --epsilon-start <val> Initial epsilon (default: 1.0)\n";
    std::cout << "  --epsilon-min <val>   Final epsilon (default: 0.3)\n";
    std::cout << "  --epsilon-decay <val> Epsilon decay gamma (default: 1-1e-8)\n";
}


int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printHelp(argv[0]); return 0; }
        else if (arg == "--seed"          && i+1<argc) g_seed              = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (arg == "--num-rounds"    && i+1<argc) g_num_rounds        = std::stoull(argv[++i]);
        else if (arg == "--penetration"   && i+1<argc) g_penetration       = std::stod(argv[++i]);
        else if (arg == "--decks"         && i+1<argc) g_decks             = std::stoi(argv[++i]);
        else if (arg == "--ss17"          && i+1<argc) g_ss17              = (std::string(argv[++i]) == "true");
        else if (arg == "--das"           && i+1<argc) g_das               = (std::string(argv[++i]) == "true");
        else if (arg == "--peek"          && i+1<argc) g_peek              = (std::string(argv[++i]) == "true");
        else if (arg == "--surr"          && i+1<argc) g_surrender_str     = argv[++i];
        else if (arg == "--alpha-start"   && i+1<argc) g_alpha_start       = std::stod(argv[++i]);
        else if (arg == "--alpha-min"     && i+1<argc) g_alpha_min         = std::stod(argv[++i]);
        else if (arg == "--alpha-decay"   && i+1<argc) g_alpha_decay_steps = std::stoi(argv[++i]);
        else if (arg == "--epsilon-start" && i+1<argc) g_epsilon_start     = std::stod(argv[++i]);
        else if (arg == "--epsilon-min"   && i+1<argc) g_epsilon_min       = std::stod(argv[++i]);
        else if (arg == "--epsilon-decay" && i+1<argc) g_epsilon_decay     = std::stod(argv[++i]);
        else { std::cerr << "Unknown argument: " << arg << "\nUse --help for usage.\n"; return 1; }
    }

    Surrender surr = stringToSurrender(g_surrender_str);

    Case c;
    c.deckSize       = g_decks;
    c.standSoft17    = g_ss17;
    c.doubleAfterSplit = g_das;
    c.peek           = g_peek;
    c.surrender      = surr;
    c.blackJackPay   = g_bj_pay;
    c.splitAfterSplit = 4;
    c.doubleOn       = DoubleDownOn::ANY;
    c.reSplitAces    = false;
    c.hitSplitAces   = false;

    runDebugTest(c);
    return 0;
}
