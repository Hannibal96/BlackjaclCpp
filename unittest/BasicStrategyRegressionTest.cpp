#include <gtest/gtest.h>
#include "Game/BlackjackTable.h"
#include "Game/Player.h"
#include "RL/BasicStrategy.h"
#include "RL/RandomStrategy.h"
#include "Game/BlackjackRules.h"
#include "Utils/Utils.h"
#include <memory>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>


struct Case{
    int deckSize;
    bool standSoft17;
    bool doubleAfterSplit;
    int splitAfterSplit;
    DoubleDownOn doubleOn;
    bool reSplitAces;
    bool hitSplitAces;
    bool peek;
    Surrender surrender;
    float blackJackPay;
};

struct Edge{
    double csm_edge;
    double cut_card_edge;
};

struct RegressionResult{
    double empiric_edge;
    double time_sec;
};

inline auto as_tuple(const Case& s) {
    return std::tie(s.deckSize, s.standSoft17, s.doubleAfterSplit, s.splitAfterSplit, s.doubleOn, s.reSplitAces, s.hitSplitAces, s.peek, s.surrender, s.blackJackPay);
}

int caseDistance(Case& x, Case& y){
    auto tx = as_tuple(x);
    auto ty = as_tuple(y);

    std::size_t n = 0;

    std::apply([&](auto&&... xs){
        std::apply([&](auto&&... ys){
            ((n += (xs != ys ? 1 : 0)), ...);
        }, ty);
    }, tx);

    return n;

}

std::string ToStringTableName(const Case& c)
{
    std::ostringstream os;
    os << "decks=" << (c.deckSize > 4 ? 4 : c.deckSize)
       << "_ss17=" << (c.standSoft17 ? "True" : "False")
       << "_das=" << (c.doubleAfterSplit ? "True" : "False")
       << "_surr=" <<  (c.surrender == Surrender::NO_SURRENDER ? "no" : c.surrender == Surrender::SURRENDER_ANY ? "yes" : "2-10")
       << "_peek=" << (c.peek ? "True" : "False") ;
    return os.str();
}

std::string ToString(const Case& c)
{
    std::ostringstream os;
    os << "decks=" << c.deckSize
       << "_ss17=" << (c.standSoft17 ? "True" : "False")
       << "_das=" << (c.doubleAfterSplit ? "True" : "False")
       << "_sas=" << c.splitAfterSplit
       << "_don=" << (c.doubleOn == DoubleDownOn::ANY ? "Any" : c.doubleOn == DoubleDownOn::NINE_TEN_ELEVEN ? "9-11" : "10-11")
       << "_reSplitAces=" << (c.reSplitAces ? "True" : "False")
       << "_hitSplitAces=" << (c.hitSplitAces ? "True" : "False")
       << "_surr=" << (c.surrender == Surrender::NO_SURRENDER ? "no" : c.surrender == Surrender::SURRENDER_ANY ? "yes" : "2-10")
       << "_peek=" << (c.peek ? "True" : "False") 
       << "_bj-pays=" << c.blackJackPay;
    return os.str();
}


class BasicStrategyRegressionTest : public ::testing::TestWithParam<Case> {
protected:
    using json = nlohmann::json;

    const uint64_t NUM_ROUNDS = 1'000'000'000ULL;  
    const int NUM_THREADS = 16;
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
        std::vector<std::vector<Player*>> resultPlayers = runParallelSimulation(parallelRules, parallelPlayers, rounds, NUM_THREADS, false);
    
        auto endTime = std::chrono::high_resolution_clock::now();
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        double elapsedSec = elapsedMs / 1000.0;
        
        if(resultPlayers.empty())
            throw std::runtime_error("Parallel simulation should complete successfully");
    
        double finalMoney = 0;
        for(std::vector<Player*> threds_player : resultPlayers){
            finalMoney += threds_player[0]->getMoney();
        }
        double edgePerHand = finalMoney / rounds;

        delete parallelPlayer;
        for(std::vector<Player*> threds_players : resultPlayers){
            for (auto* player : threds_players) {
            delete player;
            }
        }

        return {edgePerHand, elapsedSec};
        }
};


TEST_P(BasicStrategyRegressionTest, ParallelSimulationRegressionTest) {
    const auto& c = GetParam();
    std::string scenario = ToString(c);
    std::cout << "Starting Regression Test with NUM_ROUNDS = " << NUM_ROUNDS << " NUM_THREADS = " << NUM_THREADS << " Secnario = " << scenario << std::endl;

    Edge theory_edge = GetEdge(c);

    std::cout << "Cut Card Theoretical Edge %: " << std::fixed << std::setprecision(4) << theory_edge.cut_card_edge << "%";
    RegressionResult cc_res = RunRegression(c, 75.0, NUM_ROUNDS);
    std::cout << " Cut Card Empiric Edge %: " << std::fixed << std::setprecision(4) << cc_res.empiric_edge * 100 << "%" << std::endl;
    double cc_error = cc_res.empiric_edge * 100 + theory_edge.cut_card_edge;
    // if(abs(cc_error) > 0.01){
    //     std::cout << "Cut Card Error is " << cc_error << " which is > 0.01... recalculating" << std::endl;
    //     cc_res = RunRegression(c, 75.0, NUM_ROUNDS * 10);
    //     cc_error = cc_res.empiric_edge * 100 + theory_edge.cut_card_edge;
    // }

    std::cout << "CSM Theoretical Edge %: " << std::fixed << std::setprecision(4) << theory_edge.csm_edge << "%";
    RegressionResult csm_res = RunRegression(c, 0.0, NUM_ROUNDS);
    std::cout << " CSM Empiric Edge %: " << std::fixed << std::setprecision(4) << csm_res.empiric_edge * 100 << "%" << std::endl;
    double csm_error = csm_res.empiric_edge * 100 + theory_edge.csm_edge;
    // if(abs(csm_error) > 0.01){
    //     std::cout << "CSM Error is " << csm_error << " which is > 0.01... recalculating" << std::endl;
    //     csm_res = RunRegression(c, 0.0, NUM_ROUNDS * 10);
    //     csm_error = csm_res.empiric_edge * 100 + theory_edge.csm_edge;
    // }

    std::cout << "CSM Errors: " << csm_error << "%, "; 
    std::cout << "Cut Card: " << cc_error << "%" << std::endl;
    
    std::cout << "CSM Speed: " << std::fixed << std::setprecision(0) << (NUM_ROUNDS / csm_res.time_sec) << " hands/sec, ";
    std::cout << "Cut Card Speed: " << std::fixed << std::setprecision(0) << (NUM_ROUNDS / cc_res.time_sec) << " hands/sec" << std::endl;
}

// static const int deckSize[] = {6};
// static const bool standSoft17[] = {true};
// static const bool doubleAfterSplit[] = {true};
// static const int splitAfterSplit[] = {4};
// static const DoubleDownOn doubleOn[] = {DoubleDownOn::ANY};
// static const bool reSplitAces[] = {false};
// static const bool hitSplitAces[] = {true};
// static const bool peek[] = {false};
// static const Surrender surrender[] = {Surrender::SURRENDER_NO_ACE};
// static const float blackJackPay[] = {1.5};

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
    // int deck_size = 6;
    // bool stand_soft17 = true;
    // bool double_after_split = false;
    // int split_after_split = 4;
    // DoubleDownOn double_on = DoubleDownOn::ANY;
    // bool re_split_aces = false;
    // bool hit_split_aces = true;
    // bool peek_ = false;
    // Surrender surrender_ = Surrender::SURRENDER_ANY;
    // float black_jack_pay = 1.5;

    std::vector<Case> v;

    // Case base_case{deck_size, stand_soft17, double_after_split, split_after_split, double_on, re_split_aces, hit_split_aces, peek_, surrender_, black_jack_pay};
    // v.push_back(base_case);

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
                                            // if (caseDistance(base_case, new_case) != 0)
                                            //     continue;
                                            // if (!peek && surr != Surrender::NO_SURRENDER)
                                            //     continue;
                                            v.push_back(new_case);
                                        }
    static std::mt19937 rng(1234);
    std::shuffle(v.begin(), v.end(), rng);
    return v;
}();

INSTANTIATE_TEST_SUITE_P(AllCombos, BasicStrategyRegressionTest,
                         ::testing::ValuesIn(AllCases));