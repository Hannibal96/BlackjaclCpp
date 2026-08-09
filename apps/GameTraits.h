#pragma once

// Game-specific glue shared by AlternatingOptimizationApp<Game> and
// CompareCountStrategiesApp<Game>. Everything that differs between classic
// blackjack and Double Down Madness lives here as a small "traits" struct;
// the two app engines are otherwise game-agnostic templates over `Game`.

#include "RegressionTestUtils.h"
#include "Game/BlackjackRules.h"
#include "Game/DoubleDownMadnessRules.h"
#include "RL/BasicStrategy.h"
#include "RL/DoubleDownMadnessStrategy.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Classic blackjack
// ---------------------------------------------------------------------------
struct BlackjackGame {
    using Rules = BlackjackRules;
    using Case  = ::Case;  // regression/RegressionTestUtils.h

    static constexpr bool kSupportsIllustrious18 = true;
    static constexpr bool kDefaultStandSoft17 = true;
    static constexpr const char* kBannerLabel        = "";
    static constexpr const char* kAlgorithmPrefix     = "";
    static constexpr const char* kCheckpointDirPrefix  = "";  // alternating-checkpoints
    static constexpr const char* kComparePrefix        = "";  // CompareCountStrategies

    // Extra rule dimensions beyond the shared --decks/--ss17.
    static inline std::vector<bool>        g_double_after_split = {true};
    static inline std::vector<int>         g_split_after_split  = {4};
    static inline std::vector<std::string> g_double_on          = {"ANY"};
    static inline std::vector<bool>        g_resplit_aces       = {false};
    static inline std::vector<bool>        g_hit_split_aces     = {false};
    static inline std::vector<bool>        g_peek               = {false};
    static inline std::vector<std::string> g_surrender          = {"2-10"};
    static inline std::vector<float>       g_blackjack_pay      = {1.5f};

    static Rules buildRules(const Case& c, double penetration, double minBet, double maxBet) {
        Rules rules(c.blackJackPay, c.standSoft17, c.deckSize, penetration,
                    c.peek, c.splitAfterSplit, c.doubleAfterSplit,
                    c.reSplitAces, c.hitSplitAces, c.surrender, c.doubleOn);
        rules.minBet = minBet;
        rules.maxBet = maxBet;
        return rules;
    }

    static std::string toString(const Case& c) { return ::ToString(c); }
    static std::string toStringTableName(const Case& c) { return ::ToStringTableName(c); }

    // AlternatingOptimization checkpoint folders must key on the *actual*
    // simulated deck count, not the <=4-deck cap toStringTableName() applies
    // for BasicStrategy JSON table lookups — otherwise big-shoe research runs
    // would collide with each other's folder names.
    static std::string checkpointFolderName(const Case& c) {
        std::string name = toStringTableName(c);
        const size_t firstSeparator = name.find('_');
        return "decks=" + std::to_string(c.deckSize) +
            (firstSeparator == std::string::npos ? std::string{} : name.substr(firstSeparator));
    }

    static std::unique_ptr<BasicStrategy> loadBasicStrategy(const Case& c) {
        auto strategy = std::make_unique<BasicStrategy>();
        std::string tableName = toStringTableName(c);
        if (!strategy->loadFromJson(tableName))
            throw std::runtime_error("Cannot load BasicStrategy: " + tableName);
        return strategy;
    }

    static std::vector<Case> generateCases(const std::vector<int>& decks,
                                           const std::vector<bool>& ss17) {
        return generateTestCases(decks, ss17, g_double_after_split, g_split_after_split,
                                  g_double_on, g_resplit_aces, g_hit_split_aces, g_peek,
                                  g_surrender, g_blackjack_pay);
    }

    static bool parseArg(const std::string& arg, int& i, int argc, char** argv) {
        if (arg == "--das" && i + 1 < argc)      { g_double_after_split = parseList<bool>(argv[++i]); return true; }
        if (arg == "--sas" && i + 1 < argc)      { g_split_after_split  = parseList<int>(argv[++i]);  return true; }
        if (arg == "--don" && i + 1 < argc)      { g_double_on          = parseList<std::string>(argv[++i]); return true; }
        if (arg == "--rsa" && i + 1 < argc)      { g_resplit_aces       = parseList<bool>(argv[++i]); return true; }
        if (arg == "--hsa" && i + 1 < argc)      { g_hit_split_aces     = parseList<bool>(argv[++i]); return true; }
        if (arg == "--peek" && i + 1 < argc)     { g_peek               = parseList<bool>(argv[++i]); return true; }
        if (arg == "--surr" && i + 1 < argc)     { g_surrender          = parseList<std::string>(argv[++i]); return true; }
        if (arg == "--bj" && i + 1 < argc)       { g_blackjack_pay      = parseList<float>(argv[++i]); return true; }
        return false;
    }

    static void printGameHelp() {
        std::cout << "  --das <list>              Double after split (default: true)\n";
        std::cout << "  --sas <list>              Max hands after split (default: 4)\n";
        std::cout << "  --don <list>              Double-down totals: ANY, 9-11, 10-11 (default: ANY)\n";
        std::cout << "  --rsa <list>              Resplit aces (default: false)\n";
        std::cout << "  --hsa <list>              Hit split aces (default: false)\n";
        std::cout << "  --peek <list>             Dealer peeks for blackjack (default: false)\n";
        std::cout << "  --surr <list>             Surrender: no, yes, 2-10 (default: 2-10)\n";
        std::cout << "  --bj <list>               Blackjack payout multiplier (default: 1.5)\n";
    }

    static void writeMetaGameSection(json& meta, const Case& c) {
        meta["game"]["das"]       = c.doubleAfterSplit;
        meta["game"]["sas"]       = c.splitAfterSplit;
        meta["game"]["don"]       = c.doubleOn == DoubleDownOn::ANY ? "ANY"
                                   : c.doubleOn == DoubleDownOn::NINE_TEN_ELEVEN ? "9-11" : "10-11";
        meta["game"]["rsa"]       = c.reSplitAces;
        meta["game"]["hsa"]       = c.hitSplitAces;
        meta["game"]["peek"]      = c.peek;
        meta["game"]["surrender"] = c.surrender == Surrender::NO_SURRENDER ? "no"
                                   : c.surrender == Surrender::SURRENDER_ANY ? "yes" : "2-10";
        meta["game"]["bj_pay"]    = c.blackJackPay;
    }

    static Case readMetaGameSection(const json& g) {
        Case c;
        c.deckSize         = g.at("decks").get<int>();
        c.standSoft17      = g.at("ss17").get<bool>();
        c.doubleAfterSplit = g.at("das").get<bool>();
        c.splitAfterSplit  = g.at("sas").get<int>();
        c.doubleOn         = stringToDoubleOn(g.at("don").get<std::string>());
        c.reSplitAces      = g.at("rsa").get<bool>();
        c.hitSplitAces     = g.at("hsa").get<bool>();
        c.peek             = g.at("peek").get<bool>();
        c.surrender        = stringToSurrender(g.at("surrender").get<std::string>());
        c.blackJackPay     = g.at("bj_pay").get<float>();
        return c;
    }

    static void appendMetaMismatch(const json& g, const Case& c, std::vector<std::string>& mismatches) {
        const bool das = g.at("das").get<bool>();
        if (das != c.doubleAfterSplit) {
            mismatches.push_back("das: simulation=" + std::string(c.doubleAfterSplit ? "true" : "false") +
                                 ", checkpoint=" + std::string(das ? "true" : "false"));
        }
    }
};

// ---------------------------------------------------------------------------
// Double Down Madness
// ---------------------------------------------------------------------------
struct DoubleDownMadnessGame {
    using Rules = DoubleDownMadnessRules;

    struct Case {
        int deckSize = 6;
        bool standSoft17 = false;
        DoubleDownMadnessVersion version = DoubleDownMadnessVersion::VERSION_1;
    };

    static constexpr bool kSupportsIllustrious18 = false;
    static constexpr bool kDefaultStandSoft17 = false;
    static constexpr const char* kBannerLabel        = "DoubleDownMadness";
    static constexpr const char* kAlgorithmPrefix     = "double_down_madness_";
    static constexpr const char* kCheckpointDirPrefix  = "double-down-madness-";
    static constexpr const char* kComparePrefix        = "DoubleDownMadness";

    static inline std::vector<int> g_versions = {1};

    static DoubleDownMadnessVersion versionFromInt(int value) {
        if (value < 1 || value > 3)
            throw std::invalid_argument("Double Down Madness version must be 1, 2, or 3");
        return static_cast<DoubleDownMadnessVersion>(value);
    }

    static Rules buildRules(const Case& c, double penetration, double minBet, double maxBet) {
        Rules rules(c.version, c.standSoft17, c.deckSize, penetration);
        rules.minBet = minBet;
        rules.maxBet = maxBet;
        return rules;
    }

    static std::string toString(const Case& c) {
        std::ostringstream os;
        os << "decks=" << c.deckSize
           << "_ss17=" << std::boolalpha << c.standSoft17
           << "_version=" << static_cast<int>(c.version);
        return os.str();
    }

    static std::string toStringTableName(const Case& c) { return toString(c); }
    static std::string checkpointFolderName(const Case& c) { return toString(c); }

    static std::unique_ptr<BasicStrategy> loadBasicStrategy(const Case& c) {
        return loadDoubleDownMadnessBasicStrategy(c.version);
    }

    static std::vector<Case> generateCases(const std::vector<int>& decks,
                                           const std::vector<bool>& ss17) {
        std::vector<Case> cases;
        for (int d : decks) {
            if (d <= 0)
                throw std::invalid_argument("Deck count must be positive");
            for (bool s : ss17)
                for (int v : g_versions)
                    cases.push_back({d, s, versionFromInt(v)});
        }
        return cases;
    }

    static bool parseArg(const std::string& arg, int& i, int argc, char** argv) {
        if (arg == "--version" && i + 1 < argc) { g_versions = parseList<int>(argv[++i]); return true; }
        return false;
    }

    static void printGameHelp() {
        std::cout << "  --version <list>          Double Down Madness versions 1-3 (default: 1)\n";
    }

    static void writeMetaGameSection(json& meta, const Case& c) {
        meta["game"]["version"] = static_cast<int>(c.version);
    }

    static Case readMetaGameSection(const json& g) {
        Case c;
        c.deckSize    = g.at("decks").get<int>();
        c.standSoft17 = g.at("ss17").get<bool>();
        c.version     = versionFromInt(g.at("version").get<int>());
        return c;
    }

    static void appendMetaMismatch(const json& g, const Case& c, std::vector<std::string>& mismatches) {
        const int version = g.at("version").get<int>();
        if (version != static_cast<int>(c.version)) {
            mismatches.push_back("version: simulation=" + std::to_string(static_cast<int>(c.version)) +
                                 ", checkpoint=" + std::to_string(version));
        }
    }
};
