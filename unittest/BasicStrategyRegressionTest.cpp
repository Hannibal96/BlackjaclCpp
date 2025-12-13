#include <gtest/gtest.h>
#include "RegressionTestUtils.h"
#include "Game/BlackjackTable.h"
#include "Game/Player.h"
#include "RL/BasicStrategy.h"
#include "RL/RandomStrategy.h"
#include "Utils/Utils.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>
#include <random>
#include <cstring>


// Global configuration variables (can be overridden via command-line)
uint64_t g_num_rounds = 1'000'000'000ULL;
int g_num_threads = 16;


struct Edge {
    double csm_edge;
    double cut_card_edge;
};


class BasicStrategyRegressionTest : public ::testing::TestWithParam<Case> {
protected:
    using json = nlohmann::json;

    json csm_data, cc_data;

    void SetUp() override {    
        namespace fs = std::filesystem;
        fs::path csm_full_path = fs::path(PROJECT_ROOT) / "basic_strategy_tables" / "json_wod_edge_csm.json";
        std::ifstream csm_file(csm_full_path);
        if (!csm_file.is_open()) {
            FAIL() << "Cannot open edge JSON file";
        }
        csm_file >> csm_data;

        fs::path cc_full_path = fs::path(PROJECT_ROOT) / "basic_strategy_tables" / "json_wod_edge.json";
        std::ifstream cc_file(cc_full_path);
        if (!cc_file.is_open()) {
            FAIL() << "Cannot open edge JSON file";
        }
        cc_file >> cc_data;
      }

    void TearDown() override {    }

    Edge GetEdge(Case c){
        std::ostringstream os;
        os << "d=" << c.deckSize
            << "_p=100"
            << "_ss17=" << (c.standSoft17 ? "True" : "False")
            << "_das=" << (c.doubleAfterSplit ? "True" : "False")
            << "_sas=" << c.splitAfterSplit
            << "_rsa=" << (c.reSplitAces ? "True" : "False")
            << "_hsa=" << (c.hitSplitAces ? "True" : "False")
            << "_don=" << (c.doubleOn == DoubleDownOn::ANY ? "Any" : c.doubleOn == DoubleDownOn::NINE_TEN_ELEVEN ? "9-11" : "10-11")
            << "_surr=" << (c.surrender == Surrender::NO_SURRENDER ? "no" : c.surrender == Surrender::SURRENDER_ANY ? "yes" : "2-10")
            << "_peek=" << (c.peek ? "True" : "False") 
            << "_bj=" << c.blackJackPay;
        
        std::string csm_val, cc_val;
        if (csm_data.contains(os.str()) && cc_data.contains(os.str())) {
            csm_val = csm_data[os.str()];
            cc_val = cc_data[os.str()];
        } else{
            std::cout << "Key not found: " << os.str() << std::endl;
            return {0, 0};
        } 
        if (!csm_val.empty() && csm_val.back() == '%')
            csm_val.pop_back();
        if (!cc_val.empty() && cc_val.back() == '%')
            cc_val.pop_back();
        
        return {std::stod(csm_val), std::stod(cc_val)};
    }

    RegressionResult RunRegression(Case c, double penetration, uint64_t rounds){
        auto parallelStrategy = std::make_unique<BasicStrategy>();
        bool loaded = parallelStrategy->loadFromJson(ToStringTableName(c));
        if(!loaded)
            throw std::runtime_error("Failed to load strategy file " + ToStringTableName(c) + ".json");

        Player* parallelPlayer = new Player(0.0, std::move(parallelStrategy));
        std::vector<Player*> parallelPlayers = {parallelPlayer};

        BlackjackRules parallelRules = BlackjackRules(c.blackJackPay, c.standSoft17, c.deckSize, penetration, c.peek,
                               c.splitAfterSplit, c.doubleAfterSplit, c.reSplitAces, c.hitSplitAces, c.surrender, c.doubleOn);

        auto startTime = std::chrono::high_resolution_clock::now();
        std::vector<Player*> resultPlayers = runParallelSimulation(parallelRules, parallelPlayers, rounds, g_num_threads);

        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        double elapsedSec = elapsedMs / 1000.0;

        if(resultPlayers.empty())
            throw std::runtime_error("Parallel simulation should complete successfully");

        double finalMoney = resultPlayers[0]->getMoney();
        double edgePerHand = finalMoney / (rounds / g_num_threads);

        delete parallelPlayer;
        for (auto* player : resultPlayers) {
            delete player;
        }

        return {edgePerHand, elapsedSec};
        }
};


TEST_P(BasicStrategyRegressionTest, ParallelSimulationRegressionTest) {
    const auto& c = GetParam();
    std::string scenario = ToString(c);
    std::cout << "Starting Regression Test with NUM_ROUNDS = " << g_num_rounds << " NUM_THREADS = " << g_num_threads << " Secnario = " << scenario << std::endl;

    Edge theory_edge = GetEdge(c);

    std::cout << "Cut Card Theoretical Edge %: " << std::fixed << std::setprecision(4) << theory_edge.cut_card_edge << "%";
    RegressionResult cc_res = RunRegression(c, 75.0, g_num_rounds);
    std::cout << " Cut Card Empiric Edge %: " << std::fixed << std::setprecision(4) << cc_res.empiric_edge * 100 << "%" << std::endl;
    double cc_error = cc_res.empiric_edge * 100 + theory_edge.cut_card_edge;
    // if(abs(cc_error) > 0.01){
    //     std::cout << "Cut Card Error is " << cc_error << " which is > 0.01... recalculating" << std::endl;
    //     cc_res = RunRegression(c, 75.0, g_num_rounds * 10);
    //     cc_error = cc_res.empiric_edge * 100 + theory_edge.cut_card_edge;
    // }

    std::cout << "CSM Theoretical Edge %: " << std::fixed << std::setprecision(4) << theory_edge.csm_edge << "%";
    RegressionResult csm_res = RunRegression(c, 0.0, g_num_rounds);
    std::cout << " CSM Empiric Edge %: " << std::fixed << std::setprecision(4) << csm_res.empiric_edge * 100 << "%" << std::endl;
    double csm_error = csm_res.empiric_edge * 100 + theory_edge.csm_edge;
    // if(abs(csm_error) > 0.01){
    //     std::cout << "CSM Error is " << csm_error << " which is > 0.01... recalculating" << std::endl;
    //     csm_res = RunRegression(c, 0.0, g_num_rounds * 10);
    //     csm_error = csm_res.empiric_edge * 100 + theory_edge.csm_edge;
    // }

    std::cout << "CSM Errors: " << csm_error << "%, ";
    std::cout << "Cut Card: " << cc_error << "%" << std::endl;

    std::cout << "CSM Speed: " << std::fixed << std::setprecision(0) << (g_num_rounds / csm_res.time_sec) << " hands/sec, ";
    std::cout << "Cut Card Speed: " << std::fixed << std::setprecision(0) << (g_num_rounds / cc_res.time_sec) << " hands/sec" << std::endl;
}

static const int deckSize[] = {6, 2, 1};
static const bool standSoft17[] = {false, true};
static const bool doubleAfterSplit[] = {false, true};
static const int splitAfterSplit[] = {2, 3, 4};
static const DoubleDownOn doubleOn[] = {DoubleDownOn::ANY, DoubleDownOn::NINE_TEN_ELEVEN, DoubleDownOn::TEN_ELEVEN };
static const bool reSplitAces[] = {false, true};
static const bool hitSplitAces[] = {false, true};
static const bool peek[] = {false, true};
static const Surrender surrender[] = {Surrender::NO_SURRENDER, Surrender::SURRENDER_NO_ACE};
// static const Surrender surrender[] = {Surrender::NO_SURRENDER, Surrender::SURRENDER_ANY, Surrender::SURRENDER_NO_ACE};
static const float blackJackPay[] = {1.2, 1.5};


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

INSTANTIATE_TEST_SUITE_P(AllCombos, BasicStrategyRegressionTest,
                         ::testing::ValuesIn(AllCases));


void printHelp(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS] [GTEST_OPTIONS]\n\n";
    std::cout << "Basic Strategy Regression Test\n";
    std::cout << "Tests basic strategy performance against theoretical edges.\n\n";
    std::cout << "OPTIONS:\n";
    std::cout << "  --num-rounds <N>     Number of simulation rounds (default: 1000000000)\n";
    std::cout << "  --num-threads <N>    Number of parallel threads (default: 16)\n";
    std::cout << "  --help               Show this help message\n\n";
    std::cout << "GTEST_OPTIONS:\n";
    std::cout << "  --gtest_filter=<pattern>  Run only tests matching the pattern\n";
    std::cout << "  --gtest_list_tests        List all available tests\n";
    std::cout << "  (See Google Test documentation for more options)\n\n";
    std::cout << "EXAMPLES:\n";
    std::cout << "  " << program_name << " --num-rounds 100000000 --num-threads 8\n";
    std::cout << "  " << program_name << " --gtest_filter=*AllCombos*\n";
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