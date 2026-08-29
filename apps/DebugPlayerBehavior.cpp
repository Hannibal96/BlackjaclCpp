// DebugPlayerBehavior — plays a small number of rounds against a given basic
// strategy/count and prints every hand decision (hand, action, count) and
// outcome (final hand, win/lose/push, balance) to stdout. Single-threaded, no
// training, no metrics: this is pure behavior debugging, so a human can watch
// a game unfold and confirm the engine is doing the right thing. Same
// GameTraits.h/GameAppDispatcher.h plumbing as AlternatingOptimization and
// CompareCountStrategies (--game blackjack|ddm|spanish21 plus each game's own
// rule flags), trimmed down to exactly what a single readable run needs.

#include "GameAppDispatcher.h"
#include "GameTraits.h"
#include "Game/Player.h"
#include "Game/CountingMethods.h"
#include "RL/BasicStrategy.h"
#include "RL/WalkerStrategy.h"
#include "Utils/Utils.h"
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

template <typename Game>
struct DebugPlayerBehaviorApp {
    using Rules = typename Game::Rules;
    using Case  = typename Game::Case;

    struct CountConfig {
        CountingSystem system;
        double resolution = 1.0;
        int minCount = -5;
        int maxCount = 5;
    };

    static inline uint64_t    g_num_rounds     = 20;
    static inline double      g_penetration    = 75.0;
    static inline double      g_starting_money = 0.0;

    static inline std::string g_count_name;
    static inline std::string g_count_weights_str;
    static inline double      g_count_resolution = 1.0;
    // Unset by default: a plain reference basic strategy table only has a
    // count=0 entry, so the range defaults to the loaded strategy's own
    // getCountRange() (see resolveCountConfig()) rather than a fixed [-5,5]
    // that would 404 on most lookups. Set explicitly to override (e.g. when
    // pointing --strategy-file at a full-deviations table with a real range).
    static inline std::optional<int> g_min_count;
    static inline std::optional<int> g_max_count;

    static inline std::string g_strategy_file;
    // Spanish21-only: play Katarina Walker's published index deviations
    // (WalkerStrategy.h) instead of the plain reference table. Mutually
    // exclusive with --strategy-file. Pair with --count walker to match
    // CompareCountStrategies' "Walker + deviations" row exactly.
    static inline bool g_walker_deviations = false;

    static inline std::vector<int>  g_deck_sizes   = {6};
    static inline std::vector<bool> g_stand_soft17 = {Game::kDefaultStandSoft17};

    static std::array<double, 13> parseWeights13(const std::string& s) {
        std::array<double, 13> w{};
        std::istringstream ss(s);
        std::string tok;
        int i = 0;
        while (std::getline(ss, tok, ',') && i < 13) w[i++] = std::stod(tok);
        if (i != 13)
            throw std::invalid_argument("--count-weights requires exactly 13 comma-separated values");
        return w;
    }

    static CountConfig resolveCountConfig(const BasicStrategy& strategy) {
        if (!g_count_name.empty() && !g_count_weights_str.empty())
            throw std::invalid_argument("Specify only one of --count or --count-weights");

        CountConfig cfg;
        cfg.resolution = g_count_resolution;
        const auto [tableMinCount, tableMaxCount] = strategy.getCountRange();
        cfg.minCount = g_min_count.value_or(tableMinCount);
        cfg.maxCount = g_max_count.value_or(tableMaxCount);

        if (!g_count_weights_str.empty()) {
            cfg.system.weights = parseWeights13(g_count_weights_str);
        } else {
            const std::string name = g_count_name.empty() ? "hilo" : g_count_name;
            auto opt = CountingMethods::fromName(name);
            if (!opt) throw std::runtime_error("Unknown count name: " + name);
            cfg.system = *opt;
        }
        return cfg;
    }

    static void printHelp(const char* prog) {
        std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
        std::cout << "Plays a small number of rounds against a given basic strategy and count,\n";
        std::cout << "printing every hand decision (hand, action, count) and outcome (final hand,\n";
        std::cout << "win/lose/push, balance) to stdout. No training, no metrics -- pure behavior\n";
        std::cout << "debugging, single-threaded so output stays in round order.\n\n";
        std::cout << "  --num-rounds <N>        Rounds to play and print (default: 20)\n";
        std::cout << "  --penetration <val>     Shoe penetration % (default: 75.0)\n";
        std::cout << "  --starting-money <val>  Player's starting balance (default: 0.0)\n\n";
        std::cout << "COUNT SPECIFICATION (pick one):\n";
        std::cout << "  --count <name>          none, hilo, ko, hiopt1, hiopt2, omega2, zen, halves (default: hilo)\n";
        std::cout << "  --count-weights <csv>   Explicit 13 comma-separated weights\n";
        std::cout << "  --count-resolution <v>  (default: 1.0)\n";
        std::cout << "  --min-count <N>         (default: the loaded strategy's own count range,\n";
        std::cout << "  --max-count <N>          e.g. [0,0] for a plain basic-strategy table)\n\n";
        std::cout << "STRATEGY:\n";
        std::cout << "  --strategy-file <path>  Load a BasicStrategy JSON file directly, e.g. a\n";
        std::cout << "                          full_deviations_strategy.json from CompareCountStrategies\n";
        std::cout << "                          or a Pk_agent.json from AlternatingOptimization.\n";
        std::cout << "                          Default: the game's own WoO/reference basic strategy.\n";
        if constexpr (Game::kSupportsWalker) {
            std::cout << "  --walker-deviations     Play Katarina Walker's published Spanish 21 index\n";
            std::cout << "                          deviations instead (mutually exclusive with\n";
            std::cout << "                          --strategy-file). Pair with --count walker.\n";
        }
        std::cout << "\n";
        std::cout << "GAME CONFIG:\n";
        std::cout << "  --decks <N>             Deck count (default: 6)\n";
        std::cout << "  --ss17 <bool>           Dealer stands on soft 17 (default: "
                  << (Game::kDefaultStandSoft17 ? "true" : "false") << ")\n";
        Game::printGameHelp();
    }

    static int run(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") { printHelp(argv[0]); return 0; }
            else if ((arg == "--game" || arg == "--table-type") && i + 1 < argc) { ++i; }

            else if (arg == "--num-rounds"       && i + 1 < argc) g_num_rounds = std::stoull(argv[++i]);
            else if (arg == "--penetration"      && i + 1 < argc) g_penetration = std::stod(argv[++i]);
            else if (arg == "--starting-money"   && i + 1 < argc) g_starting_money = std::stod(argv[++i]);

            else if (arg == "--count"            && i + 1 < argc) g_count_name = argv[++i];
            else if (arg == "--count-weights"    && i + 1 < argc) g_count_weights_str = argv[++i];
            else if (arg == "--count-resolution" && i + 1 < argc) g_count_resolution = std::stod(argv[++i]);
            else if (arg == "--min-count"        && i + 1 < argc) g_min_count = std::stoi(argv[++i]);
            else if (arg == "--max-count"        && i + 1 < argc) g_max_count = std::stoi(argv[++i]);

            else if (arg == "--strategy-file"    && i + 1 < argc) g_strategy_file = argv[++i];
            else if (arg == "--walker-deviations") g_walker_deviations = true;

            else if (arg == "--decks" && i + 1 < argc) g_deck_sizes   = parseList<int>(argv[++i]);
            else if (arg == "--ss17"  && i + 1 < argc) g_stand_soft17 = parseList<bool>(argv[++i]);
            else if (Game::parseArg(arg, i, argc, argv)) { /* consumed by game traits */ }
            else {
                std::cerr << "Unknown argument: " << arg << "\nUse --help.\n";
                return 1;
            }
        }

        try {
            std::vector<Case> cases = Game::generateCases(g_deck_sizes, g_stand_soft17);
            if (cases.empty())
                throw std::invalid_argument("No configuration resolved from the given flags");
            if (cases.size() > 1) {
                std::cerr << "WARNING: multiple configurations resolved from the given flags; "
                             "using the first (" << Game::toString(cases.front()) << ").\n";
            }
            const Case& c = cases.front();

            if (g_walker_deviations && !g_strategy_file.empty())
                throw std::invalid_argument("Specify only one of --strategy-file or --walker-deviations");
            if (g_walker_deviations && !Game::kSupportsWalker)
                throw std::invalid_argument("--walker-deviations is only supported for --game spanish21");

            Rules rules = Game::buildRules(c, g_penetration, /*minBet=*/0.0,
                                           /*maxBet=*/std::numeric_limits<double>::max());

            std::unique_ptr<BasicStrategy> strategy;
            std::string strategySource;
            if (!g_strategy_file.empty()) {
                strategy = std::make_unique<BasicStrategy>();
                if (!strategy->loadFromFile(g_strategy_file))
                    throw std::runtime_error("Failed to load strategy file: " + g_strategy_file);
                strategySource = g_strategy_file;
            } else if constexpr (Game::kSupportsWalker) {
                if (g_walker_deviations) {
                    strategy = loadWalkerBasicStrategy(rules, kWalkerMinCount, kWalkerMaxCount);
                    strategySource = "Katarina Walker's published index deviations";
                } else {
                    strategy = Game::loadBasicStrategy(c);
                    strategySource = "game reference basic strategy";
                }
            } else {
                strategy = Game::loadBasicStrategy(c);
                strategySource = "game reference basic strategy";
            }

            CountConfig count = resolveCountConfig(*strategy);

            auto player = std::make_unique<Player>(g_starting_money, strategy->clone(), "Debug");
            player->setNumDecks(c.deckSize);
            player->setCountSystem(count.system);
            player->setCountResolution(count.resolution);
            player->setCountRange(count.minCount, count.maxCount);
            if constexpr (Game::kTracksHandCardCount) {
                player->setTrackHandCardCount(Game::tracksHandCardCount(c));
            }

            std::cout << "=== DebugPlayerBehavior ===\n";
            std::cout << "Scenario:  " << Game::toString(c) << "  penetration=" << g_penetration << "%\n";
            std::cout << "Strategy:  " << strategySource << "\n";
            std::cout << "Count:     res=" << count.resolution
                      << "  range=[" << count.minCount << "," << count.maxCount << "]\n";
            std::cout << "Rounds:    " << g_num_rounds << "  starting money=" << g_starting_money << "\n";
            std::cout << "===========================\n\n";

            std::vector<Player*> players{player.get()};
            const bool ok = runSimulationVerbose(rules, players, g_num_rounds, std::cout);

            std::cout << "\n=== Final balance: " << player->getMoney() << " ===\n";
            return ok ? 0 : 1;
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    }
};

int runBlackjackDebugPlayerBehavior(int argc, char** argv) {
    return DebugPlayerBehaviorApp<BlackjackGame>::run(argc, argv);
}

int runDoubleDownMadnessDebugPlayerBehavior(int argc, char** argv) {
    return DebugPlayerBehaviorApp<DoubleDownMadnessGame>::run(argc, argv);
}

int runSpanish21DebugPlayerBehavior(int argc, char** argv) {
    return DebugPlayerBehaviorApp<Spanish21Game>::run(argc, argv);
}

int main(int argc, char** argv) {
    return dispatchGameApp(
        argc, argv, "DebugPlayerBehavior",
        runBlackjackDebugPlayerBehavior,
        runDoubleDownMadnessDebugPlayerBehavior,
        runSpanish21DebugPlayerBehavior);
}
