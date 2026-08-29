#pragma once

// Game-specific glue shared by AlternatingOptimizationApp<Game> and
// CompareCountStrategiesApp<Game>. Everything that differs between classic
// blackjack and Double Down Madness lives here as a small "traits" struct;
// the two app engines are otherwise game-agnostic templates over `Game`.

#include "RegressionTestUtils.h"
#include "Game/BlackjackRules.h"
#include "Game/DoubleDownMadnessRules.h"
#include "Game/SpanishRules.h"
#include "RL/BasicStrategy.h"
#include "RL/DoubleDownMadnessStrategy.h"
#include "RL/SpanishStrategy.h"
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
    // Walker's method (CompareCountStrategies' 5-row HiLo/Walker comparison,
    // see WalkerStrategy.h) is Spanish-21-specific -- it corrects for that
    // game's structurally short (48-card, no TEN rank) shoe.
    static constexpr bool kSupportsWalker = false;
    static constexpr bool kDefaultStandSoft17 = true;
    // StateKey's cardCount field: only Spanish 21 needs it to carry the real
    // (capped-at-6) hand size (see Player::trackHandCardCount).
    static constexpr bool kTracksHandCardCount = false;
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
    static constexpr bool kSupportsWalker = false;
    static constexpr bool kDefaultStandSoft17 = false;
    static constexpr bool kTracksHandCardCount = false;
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

// ---------------------------------------------------------------------------
// Spanish 21
// ---------------------------------------------------------------------------
struct Spanish21Game {
    using Rules = SpanishRules;

    struct Case {
        int deckSize = 8;
        bool standSoft17 = true;
        int maxRedoubles = 0;  // 0 = redoubling off; see SpanishRules::maxRedoubles
        bool allowDoubleDownRescue = true;
        // Drives both SpanishRules::payCardCountBonuses and paySuitedBonus together,
        // and (via Spanish21Game::tracksHandCardCount()) whether Players carry the
        // real per-hand card count in StateKey at all. Off collapses Spanish 21 back
        // to Blackjack-sized state (StateKey's cardCount field pinned to the constant
        // 2) for Q-learning convergence -- see SpanishRules::payCardCountBonuses.
        bool cardCountBonuses = true;
    };

    static constexpr bool kSupportsIllustrious18 = false;
    // Katarina Walker's published Spanish 21 counting method -- CompareCountStrategies
    // shows a dedicated 5-row HiLo/Walker comparison for this game instead of the
    // Illustrious-18-style single-count flow (see WalkerStrategy.h).
    static constexpr bool kSupportsWalker = true;
    static constexpr bool kDefaultStandSoft17 = true;
    // Spanish 21's 5/6/7+-card 21 bonus makes card count strategically relevant,
    // so its Players must carry the real hand size in StateKey's cardCount field --
    // unless Case::cardCountBonuses turns that bonus (and the state cost that comes
    // with it) off; see tracksHandCardCount() below, which every call site should
    // use instead of this constant now that the decision is runtime, not per-game.
    static constexpr bool kTracksHandCardCount = true;
    static bool tracksHandCardCount(const Case& c) { return c.cardCountBonuses; }
    static constexpr const char* kBannerLabel        = "Spanish21";
    static constexpr const char* kAlgorithmPrefix     = "spanish21_";
    static constexpr const char* kCheckpointDirPrefix  = "spanish21-";
    static constexpr const char* kComparePrefix        = "Spanish21";

    // --redouble defaults off: it's WoO's separate "Variable Rule" with its
    // own published edge figures, and is the less commonly offered table
    // variant (see SpanishRules::maxRedoubles). --ddr defaults ON: WoO lists
    // it under "The Rules" (standard, always-on), so the published S17/H17
    // edges already assume it's in play (see SpanishRules' default
    // constructor). Both remain exposed for research/experimentation via
    // their flags.
    static inline std::vector<int> g_redouble = {0};
    static inline std::vector<bool> g_ddr = {true};
    static inline std::vector<bool> g_card_count_bonuses = {true};

    static Rules buildRules(const Case& c, double penetration, double minBet, double maxBet) {
        Rules rules(c.standSoft17, c.deckSize, penetration, /*maxSplit=*/4,
                    /*doubleAfterSplit=*/true, /*resplitAces=*/true, /*hitSplitAces=*/true,
                    /*surrender=*/true, /*peek=*/true, c.maxRedoubles,
                    c.allowDoubleDownRescue, /*suitedBonus=*/c.cardCountBonuses,
                    /*cardCountBonuses=*/c.cardCountBonuses);
        rules.minBet = minBet;
        rules.maxBet = maxBet;
        return rules;
    }

    static std::string toString(const Case& c) {
        std::ostringstream os;
        os << "decks=" << c.deckSize
           << "_ss17=" << std::boolalpha << c.standSoft17
           << "_redouble=" << c.maxRedoubles
           << "_ddr=" << c.allowDoubleDownRescue
           << "_cardCountBonuses=" << c.cardCountBonuses;
        return os.str();
    }

    static std::string toStringTableName(const Case& c) { return toString(c); }
    static std::string checkpointFolderName(const Case& c) { return toString(c); }

    static std::unique_ptr<BasicStrategy> loadBasicStrategy(const Case& c) {
        Rules rules(c.standSoft17, c.deckSize, 75.0, 4, true, true, true, true, true,
                    c.maxRedoubles, c.allowDoubleDownRescue, c.cardCountBonuses,
                    c.cardCountBonuses);
        return loadSpanishBasicStrategy(rules);
    }

    static std::vector<Case> generateCases(const std::vector<int>& decks,
                                           const std::vector<bool>& ss17) {
        std::vector<Case> cases;
        for (int d : decks) {
            if (d <= 0)
                throw std::invalid_argument("Deck count must be positive");
            for (bool s : ss17)
                for (int r : g_redouble)
                    for (bool ddr : g_ddr)
                        for (bool bonuses : g_card_count_bonuses)
                            cases.push_back({d, s, r, ddr, bonuses});
        }
        return cases;
    }

    static bool parseArg(const std::string& arg, int& i, int argc, char** argv) {
        if (arg == "--redouble" && i + 1 < argc) { g_redouble = parseList<int>(argv[++i]); return true; }
        if (arg == "--ddr" && i + 1 < argc)      { g_ddr       = parseList<bool>(argv[++i]); return true; }
        if (arg == "--card-count-bonuses" && i + 1 < argc) {
            g_card_count_bonuses = parseList<bool>(argv[++i]);
            return true;
        }
        return false;
    }

    static void printGameHelp() {
        std::cout << "  --redouble <list>         Number of allowed redoubles beyond the first\n";
        std::cout << "                             double, e.g. [0,2] (default: [0] -- off)\n";
        std::cout << "  --ddr <list>              Allow double-down rescue (default: true)\n";
        std::cout << "  --card-count-bonuses <list>  Pay the 5/6/7+-card 21 and 6-7-8/7-7-7\n";
        std::cout << "                             suited bonuses (default: true). false also\n";
        std::cout << "                             collapses StateKey's cardCount dimension back\n";
        std::cout << "                             to the constant 2 (Player::trackHandCardCount\n";
        std::cout << "                             off) -- the reward no longer depends on card\n";
        std::cout << "                             count once the bonuses are off, so this is a\n";
        std::cout << "                             much smaller, Blackjack-sized state space for\n";
        std::cout << "                             Q-learning to converge over.\n";
    }

    static void writeMetaGameSection(json& meta, const Case& c) {
        meta["game"]["redouble"] = c.maxRedoubles;
        meta["game"]["ddr"]      = c.allowDoubleDownRescue;
        meta["game"]["card_count_bonuses"] = c.cardCountBonuses;
    }

    static Case readMetaGameSection(const json& g) {
        Case c;
        c.deckSize             = g.at("decks").get<int>();
        c.standSoft17          = g.at("ss17").get<bool>();
        c.maxRedoubles          = g.at("redouble").get<int>();
        c.allowDoubleDownRescue = g.at("ddr").get<bool>();
        // .value(): older checkpoints predate this field -- default true (bonuses
        // on, matching the pre-existing behavior they were actually trained with).
        c.cardCountBonuses      = g.value("card_count_bonuses", true);
        return c;
    }

    static void appendMetaMismatch(const json& g, const Case& c, std::vector<std::string>& mismatches) {
        const int redouble = g.at("redouble").get<int>();
        if (redouble != c.maxRedoubles) {
            mismatches.push_back("redouble: simulation=" + std::to_string(c.maxRedoubles) +
                                 ", checkpoint=" + std::to_string(redouble));
        }
        const bool cardCountBonuses = g.value("card_count_bonuses", true);
        if (cardCountBonuses != c.cardCountBonuses) {
            mismatches.push_back("card_count_bonuses: simulation=" +
                                 std::string(c.cardCountBonuses ? "true" : "false") +
                                 ", checkpoint=" + std::string(cardCountBonuses ? "true" : "false"));
        }
        const bool ddr = g.at("ddr").get<bool>();
        if (ddr != c.allowDoubleDownRescue) {
            mismatches.push_back("ddr: simulation=" + std::string(c.allowDoubleDownRescue ? "true" : "false") +
                                 ", checkpoint=" + std::string(ddr ? "true" : "false"));
        }
    }
};
