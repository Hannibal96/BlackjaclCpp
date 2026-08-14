#include "Game/DoubleDownMadnessRules.h"
#include "Game/Player.h"
#include "RL/DoubleDownMadnessStrategy.h"
#include "Utils/SimulationAnalysis.h"
#include "Utils/Utils.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint64_t g_numRounds = 1'000'000'000ULL;
int g_numThreads = 10;
double g_penetration = 75.0;
std::vector<int> g_decks = {6};
std::vector<int> g_versions = {1, 2, 3};

DoubleDownMadnessVersion parseVersion(int value) {
    if (value < 1 || value > 3)
        throw std::invalid_argument("Version must be 1, 2, or 3");
    return static_cast<DoubleDownMadnessVersion>(value);
}

double publishedHouseEdge(DoubleDownMadnessVersion version) {
    switch (version) {
        case DoubleDownMadnessVersion::VERSION_1: return 0.0095;
        case DoubleDownMadnessVersion::VERSION_2: return 0.0207;
        case DoubleDownMadnessVersion::VERSION_3: return 0.0207;
    }
    throw std::logic_error("Unknown version");
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

void printHelp(const char* program) {
    std::cout
        << "Usage: " << program << " [OPTIONS]\n\n"
        << "Measure H17 Double Down Madness basic-strategy edge by deck count.\n"
        << "Six-deck runs are validated against published cut-card house edges:\n"
        << "V1=0.95%, V2=2.07%, V3=2.07%. Other deck counts are exploratory.\n\n"
        << "  --num-rounds <N>    Rounds per version (default: 1000000000)\n"
        << "  --num-threads <N>   Threads (default: 10)\n"
        << "  --penetration <P>   Cut-card penetration (default: 75)\n"
        << "  --decks <list>      Deck counts, e.g. [1,2,6,8] (default: [6])\n"
        << "  --versions <list>   Versions, e.g. [1,2,3] (default: all)\n";
}

void runVersion(DoubleDownMadnessVersion version, int decks) {
    if (decks <= 0)
        throw std::invalid_argument("Deck count must be positive");

    auto strategy = loadDoubleDownMadnessBasicStrategy(version);
    auto* player = new Player(0.0, std::move(strategy));
    player->setNumDecks(decks);
    player->enableRoundStats();

    DoubleDownMadnessRules rules(version, /*standS17=*/false, decks, g_penetration);
    rules.minBet = 1.0;
    rules.maxBet = 1.0;
    std::vector<Player*> results =
        runParallelSimulation(rules, {player}, g_numRounds, g_numThreads);
    delete player;
    if (results.empty())
        throw std::runtime_error("Double Down Madness edge simulation failed");

    const EdgeStatistics stats = edgeStatisticsFromPlayer(*results.front());
    delete results.front();

    const double confidenceRadius =
        stats.samples > 0
            ? 2.576 * stats.stddev / std::sqrt(static_cast<double>(stats.samples))
            : 0.0;
    std::cout << "\nDecks " << decks
              << "  Version " << static_cast<int>(version)
              << "  rounds=" << stats.samples
              << "  empirical player edge=" << std::fixed << std::setprecision(4)
              << stats.mean * 100.0 << "%"
              << "  std(X)=" << stats.stddev
              << "  99% radius=" << confidenceRadius * 100.0 << "%\n";

    if (decks != 6) return;

    const double expectedPlayerEdge = -publishedHouseEdge(version);
    const double sourceRoundingTolerance = 0.00005;
    const double error = std::abs(stats.mean - expectedPlayerEdge);
    std::cout << "  published six-deck player edge="
              << expectedPlayerEdge * 100.0 << "%\n";
    if (error > confidenceRadius + sourceRoundingTolerance) {
        throw std::runtime_error(
            "Empirical player edge differs from the published edge by " +
            std::to_string(error * 100.0) + "%");
    }
}

} // namespace

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
        else if (arg == "--versions" && i + 1 < argc)
            g_versions = parseIntList(argv[++i]);
        else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            return 1;
        }
    }

    int failures = 0;
    for (int decks : g_decks) {
        for (int value : g_versions) {
            try {
                runVersion(parseVersion(value), decks);
            } catch (const std::exception& e) {
                ++failures;
                std::cerr << "Decks " << decks << ", version " << value
                          << " failed: " << e.what() << "\n";
            }
        }
    }
    return failures == 0 ? 0 : 1;
}
