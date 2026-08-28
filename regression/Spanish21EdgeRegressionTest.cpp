#include "Game/SpanishRules.h"
#include "Game/Player.h"
#include "RL/SpanishStrategy.h"
#include "Utils/SimulationAnalysis.h"
#include "Utils/Utils.h"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Validates the transcribed WoO Spanish 21 basic-strategy tables by comparing
// simulated house edge against wizardofodds.com's published numbers. Output
// format mirrors BasicStrategyRegressionTest's (classic Blackjack): both a
// Cut Card run (default 75% penetration) and a CSM run (0% penetration,
// continuous shuffle) are measured and reported, each with its own speed.
//
// WoO publishes only one edge number per ruleset for Spanish 21 (no separate
// CSM-vs-cut-card split like Blackjack's two reference JSON files) -- both
// runs are compared against that same published number.
//
// The H17+redouble case is included and asserted like the others -- verified
// to land within the same tolerance band as S17/H17 (see the comment on
// SpanishRules::maxRedoubles). DDR (double-down rescue) is listed under WoO's
// "The Rules" (a standard, always-on rule), not "Variable Rules" like
// redoubling -- so the published S17/H17/H17+redouble edges above already
// assume DDR is in play, and ddr=true is the ground-truth default here.
// ddr=false (removing a standard rule) has no published edge of its own, so
// it's NOT part of the default sweep -- same reasoning as the S17+redouble
// skip below. --ddr [false] still runs (measured, not asserted -- see
// RunTest()'s exploratory branch and the Test Summary's separate
// "Exploratory" count) for anyone who wants to explore it manually.

uint64_t g_numRounds = 1'000'000'000ULL;
int g_numThreads = 10;
double g_penetration = 75.0;
std::vector<int> g_decks = {8};
std::vector<bool> g_standSoft17 = {true, false};
// 0 = redoubling off; N = up to N redoubles beyond the first (ordinary)
// double. 1 is the minimal "on" value right after the 0 sentinel; verified
// against WoO's published H17+redouble edge (0.42%) alongside maxRedoubles=2
// -- see SpanishRules::maxRedoubles for the exact accounting.
std::vector<int> g_redouble = {0, 1};
// Defaults to true only: DDR is a standard WoO rule already baked into the
// published edges above (see the file header comment) -- ddr=true is the
// ground-truth case; ddr=false has no published edge of its own, so it's
// not part of the default sweep. --ddr still accepts false for manual
// exploration -- see RunTest()'s exploratory branch.
std::vector<bool> g_ddr = {true};

struct Case {
    int deckSize;
    bool standSoft17;
    int maxRedoubles;
    bool ddr;
};

struct RegressionResult {
    double empiricEdge;   // fraction, not percent
    double stddev;
    uint64_t samples;
    double timeSec;
};

// Published edges from wizardofodds.com/games/spanish-21/. Deck count is not a
// WoO strategy/edge dimension for Spanish 21 (unlike Blackjack) -- the same
// published numbers apply across deck counts, so non-8-deck runs are
// exploratory (no published reference to assert against).
double publishedHouseEdge(const Case& c) {
    if (c.maxRedoubles > 0) return 0.0042;
    return c.standSoft17 ? 0.0040 : 0.0076;
}

std::string toString(const Case& c) {
    std::ostringstream os;
    os << "decks=" << c.deckSize << "_ss17=" << (c.standSoft17 ? "True" : "False")
       << "_redouble=" << c.maxRedoubles << "_ddr=" << (c.ddr ? "True" : "False");
    return os.str();
}

// The published edges assume DDR is in play (it's a standard WoO rule, not a
// variable one -- see the file header comment), so only ddr=true has a
// published number to compare against; ddr=false (a non-standard removal of
// that rule) has no edge of its own, regardless of deck/ss17/redouble.
bool hasPublishedReference(const Case& c) { return c.ddr; }

// Cross-products decks x standSoft17 x redouble x ddr. S17+redouble is
// skipped: no WoO chart exists for that combination (loadSpanishBasicStrategy()
// would silently fall back to the plain S17 base table), so there's no
// published number to assert against -- see CLAUDE.md's Spanish 21 section.
std::vector<Case> generateCases(const std::vector<int>& decks, const std::vector<bool>& ss17List,
                                const std::vector<int>& redoubleList, const std::vector<bool>& ddrList) {
    std::vector<Case> cases;
    for (int d : decks) {
        for (bool s : ss17List) {
            for (int r : redoubleList) {
                if (s && r > 0) {
                    std::cerr << "NOTE: skipping decks=" << d << " ss17=true redouble=" << r
                              << " -- no WoO reference exists for S17+redouble.\n";
                    continue;
                }
                for (bool ddr : ddrList) {
                    cases.push_back({d, s, r, ddr});
                }
            }
        }
    }
    return cases;
}

std::vector<int> parseIntList(std::string value) {
    if (!value.empty() && value.front() == '[') value.erase(value.begin());
    if (!value.empty() && value.back() == ']') value.pop_back();
    std::vector<int> values;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        values.push_back(std::stoi(value.substr(start, comma - start)));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return values;
}

std::vector<bool> parseBoolList(std::string value) {
    if (!value.empty() && value.front() == '[') value.erase(value.begin());
    if (!value.empty() && value.back() == ']') value.pop_back();
    std::vector<bool> values;
    size_t start = 0;
    while (start <= value.size()) {
        const size_t comma = value.find(',', start);
        const std::string item = value.substr(start, comma - start);
        if (item == "true" || item == "1" || item == "True") values.push_back(true);
        else if (item == "false" || item == "0" || item == "False") values.push_back(false);
        else throw std::invalid_argument("Invalid boolean value: " + item);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return values;
}

class Spanish21EdgeRegressionTest {
public:
    RegressionResult RunRegression(const Case& c, double penetration) {
        SpanishRules rules(c.standSoft17, c.deckSize, penetration, /*maxSplit=*/4,
                           /*doubleAfterSplit=*/true, /*resplitAces=*/true, /*hitSplitAces=*/true,
                           /*surrender=*/true, /*peek=*/true, c.maxRedoubles,
                           c.ddr, /*suitedBonus=*/true);
        rules.minBet = 1.0;
        rules.maxBet = 1.0;

        auto strategy = loadSpanishBasicStrategy(rules);
        auto* player = new Player(0.0, std::move(strategy));
        player->setNumDecks(c.deckSize);
        player->setTrackHandCardCount(true);
        player->enableRoundStats();

        const auto startTime = std::chrono::high_resolution_clock::now();
        std::vector<Player*> results =
            runParallelSimulation(rules, {player}, g_numRounds, g_numThreads);
        const auto endTime = std::chrono::high_resolution_clock::now();
        delete player;
        if (results.empty())
            throw std::runtime_error("Spanish 21 edge simulation failed");

        const EdgeStatistics stats = edgeStatisticsFromPlayer(*results.front());
        delete results.front();

        const double elapsedSec =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() / 1000.0;
        return {stats.mean, stats.stddev, stats.samples, elapsedSec};
    }

    // Returns true if this case has a WoO published reference and was actually
    // asserted against it; false if it ran as exploratory (measured only).
    bool RunTest(const Case& c) {
        std::string scenario = toString(c);
        std::cout << "\n=== Starting Regression Test ===" << std::endl;
        std::cout << "NUM_ROUNDS:  " << g_numRounds << std::endl;
        std::cout << "NUM_THREADS: " << g_numThreads << std::endl;
        std::cout << "Scenario:    " << scenario << std::endl;
        std::cout << "============================\n" << std::endl;

        // The dropped 6-7-8/Super-Bonus-reachability footnote nuances (documented
        // v1 simplification -- see the transcription script referenced from
        // CLAUDE.md's Basic Strategy Tables section) cost some EV, so the assertion
        // tolerance is widened beyond the pure statistical threshold; still tight
        // enough to catch a broken table.
        const double knownSimplificationTolerance = 0.30;  // percentage points

        const double theoreticalEdge = -publishedHouseEdge(c) * 100.0;

        std::cout << "Cut Card (Penetration: " << g_penetration << "%) Theoretical Edge %: "
                  << std::fixed << std::setprecision(4) << theoreticalEdge << "%";
        RegressionResult cc_res = RunRegression(c, g_penetration);
        std::cout << " Cut Card Empiric Edge %: " << std::fixed << std::setprecision(4)
                  << cc_res.empiricEdge * 100.0 << "%" << std::endl;
        const double cc_error = cc_res.empiricEdge * 100.0 - theoreticalEdge;

        std::cout << "CSM (Penetration: 0.0%) Theoretical Edge %: " << std::fixed
                  << std::setprecision(4) << theoreticalEdge << "%";
        RegressionResult csm_res = RunRegression(c, 0.0);
        std::cout << " CSM Empiric Edge %: " << std::fixed << std::setprecision(4)
                  << csm_res.empiricEdge * 100.0 << "%" << std::endl;
        const double csm_error = csm_res.empiricEdge * 100.0 - theoreticalEdge;

        std::cout << "CSM Error: " << csm_error << "%, Cut Card Error: " << cc_error << "%" << std::endl;

        std::cout << "CSM Speed: " << std::fixed << std::setprecision(0)
                  << (static_cast<double>(g_numRounds) / csm_res.timeSec) << " hands/sec, ";
        std::cout << "Cut Card Speed: " << std::fixed << std::setprecision(0)
                  << (static_cast<double>(g_numRounds) / cc_res.timeSec) << " hands/sec" << std::endl;

        const double cc_radius = cc_res.samples > 0
            ? 2.576 * cc_res.stddev / std::sqrt(static_cast<double>(cc_res.samples)) * 100.0 : 0.0;
        const double csm_radius = csm_res.samples > 0
            ? 2.576 * csm_res.stddev / std::sqrt(static_cast<double>(csm_res.samples)) * 100.0 : 0.0;
        std::cout << "CSM 99% radius: " << std::fixed << std::setprecision(4) << csm_radius
                  << "%, Cut Card 99% radius: " << cc_radius << "%" << std::endl;

        if (!hasPublishedReference(c)) {
            std::cout << "NOTE: ddr=false has no published edge of its own (the number above "
                      << "assumes standard DDR) -- measured for context only, not asserted.\n";
            return false;
        }

        if (std::abs(csm_error) >= csm_radius + knownSimplificationTolerance) {
            throw std::runtime_error(
                "CSM edge error " + std::to_string(csm_error) +
                "% exceeds threshold ±" + std::to_string(csm_radius + knownSimplificationTolerance) + "%");
        }
        if (std::abs(cc_error) >= cc_radius + knownSimplificationTolerance) {
            throw std::runtime_error(
                "Cut-card edge error " + std::to_string(cc_error) +
                "% exceeds threshold ±" + std::to_string(cc_radius + knownSimplificationTolerance) + "%");
        }
        return true;
    }
};

void printHelp(const char* program) {
    std::cout << "Usage: " << program << " [OPTIONS]\n\n"
              << "Measure Spanish 21 basic-strategy edge (S17, H17, and H17+redouble)\n"
              << "against wizardofodds.com's published house edges: S17=0.40%,\n"
              << "H17=0.76%, H17+redouble=0.42%. Both a Cut Card (default 75%\n"
              << "penetration) and a CSM (continuous shuffle) run are measured.\n\n"
              << "  --num-rounds <N>    Rounds per configuration (default: 1000000000)\n"
              << "  --num-threads <N>   Threads (default: 10)\n"
              << "  --penetration <P>   Cut-card penetration (default: 75)\n"
              << "  --decks <list>      Deck counts, e.g. [6,8] (default: [8])\n"
              << "  --ss17 <list>       Dealer stands on soft 17, e.g. [true,false] (default: [true,false])\n"
              << "  --redouble <list>   Redoubles allowed beyond the first double, e.g. [0,1]\n"
              << "                      (default: [0,1]; 0 = off). S17+redouble is always\n"
              << "                      skipped -- no WoO reference exists.\n"
              << "  --ddr <list>        Allow double-down rescue, e.g. [false,true]\n"
              << "                      (default: [true] -- DDR is a standard WoO rule already\n"
              << "                      baked into the published edges. ddr=false has no edge\n"
              << "                      of its own, so it's not part of the default sweep;\n"
              << "                      passing it runs measured-only, not asserted -- see the\n"
              << "                      printed NOTE for such cases).\n";
}

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printHelp(argv[0]);
            return 0;
        }
        if (arg == "--num-rounds" && i + 1 < argc)
            g_numRounds = std::stoull(argv[++i]);
        else if (arg == "--num-threads" && i + 1 < argc)
            g_numThreads = std::stoi(argv[++i]);
        else if (arg == "--penetration" && i + 1 < argc)
            g_penetration = std::stod(argv[++i]);
        else if (arg == "--decks" && i + 1 < argc)
            g_decks = parseIntList(argv[++i]);
        else if (arg == "--ss17" && i + 1 < argc)
            g_standSoft17 = parseBoolList(argv[++i]);
        else if (arg == "--redouble" && i + 1 < argc)
            g_redouble = parseIntList(argv[++i]);
        else if (arg == "--ddr" && i + 1 < argc)
            g_ddr = parseBoolList(argv[++i]);
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }

    std::vector<Case> allCases = generateCases(g_decks, g_standSoft17, g_redouble, g_ddr);

    std::cout << "=== Global Configuration ===" << std::endl;
    std::cout << "NUM_ROUNDS:  " << g_numRounds << std::endl;
    std::cout << "NUM_THREADS: " << g_numThreads << std::endl;
    std::cout << "Total test cases: " << allCases.size() << std::endl;
    std::cout << "============================\n" << std::endl;

    Spanish21EdgeRegressionTest test;
    int passed = 0;
    int failed = 0;
    int exploratory = 0;

    for (size_t i = 0; i < allCases.size(); ++i) {
        std::cout << "\n[" << (i + 1) << "/" << allCases.size() << "] ";
        try {
            if (test.RunTest(allCases[i])) passed++;
            else exploratory++;
        } catch (const std::exception& e) {
            std::cerr << "Test failed: " << e.what() << std::endl;
            failed++;
        }
    }

    std::cout << "\n\n=== Test Summary ===" << std::endl;
    std::cout << "Total:       " << allCases.size() << std::endl;
    std::cout << "Passed:      " << passed << std::endl;
    std::cout << "Exploratory: " << exploratory << " (no WoO reference, e.g. ddr=false -- measured only)"
              << std::endl;
    std::cout << "Failed:      " << failed << std::endl;
    std::cout << "====================" << std::endl;

    return (failed == 0) ? 0 : 1;
}
