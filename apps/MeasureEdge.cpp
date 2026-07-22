#include "RegressionTestUtils.h"
#include "Game/BlackjackTable.h"
#include "Game/Player.h"
#include "Game/BettingStrategy.h"
#include "Game/CountingMethods.h"
#include "RL/BasicStrategy.h"
#include "RL/QLearningStrategy.h"
#include "Utils/RunLogger.h"
#include "Utils/Utils.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <array>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <limits>
#include <algorithm>

using json = nlohmann::json;

namespace {
constexpr const char* kQlearningCheckpointRoot = "checkpoints/checkpoints_QLearning";
constexpr const char* kOlsCheckpointRoot = "checkpoints/checkpoints_ols";
constexpr const char* kMeasureEdgeLogRoot = "checkpoints/MeasureEdge";
constexpr uint64_t kKellyMeasurementRounds = 1'000'000ULL;
}

static std::string currentTimestamp() {
    time_t t = time(nullptr);
    char buf[20];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&t));
    return buf;
}

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
uint64_t    g_num_rounds   = 1'000'000'000ULL;
int         g_num_threads  = 16;
double      g_penetration  = 75.0;
std::string g_mode         = "spread"; // "spread" or "kelly"
bool        g_mode_explicit = false;
int         g_kelly_measurements = 100;
std::string g_command_line;

// Count specification — exactly one of these should be set
std::string g_count_name;        // named system (none, hilo, ko, …)
std::string g_count_weights_str; // explicit "w0,…,w12"
std::string g_count_ols;         // folder under checkpoints/checkpoints_ols/ — loads OLS solution

// E[game] model overrides (NaN = use count system default)
double g_factor = std::numeric_limits<double>::quiet_NaN();
double g_bias   = std::numeric_limits<double>::quiet_NaN();

// Player counting parameters
double g_count_resolution = 1.0;

// Mode-specific parameters
std::string g_spread_str;        // "t1:b1,t2:b2,…"
double      g_kelly_fraction = 1.0;

// Table parameters
double g_min_bet      = 1.0;
double g_max_bet      = std::numeric_limits<double>::max();
bool   g_min_bet_explicit = false;
double g_initial_money = std::numeric_limits<double>::quiet_NaN(); // NaN = auto by mode

// Strategy selection
std::string g_strategy_checkpoint; // folder under checkpoints/checkpoints_QLearning/ (empty = use BasicStrategy)
std::string g_strategy_agent;      // agent name within checkpoint (empty = first in meta.json)
bool        g_strategy_verbose = false;

// Game configuration
std::vector<int>         g_deck_sizes         = {6};
std::vector<bool>        g_stand_soft17       = {true};
std::vector<bool>        g_double_after_split = {true};
std::vector<int>         g_split_after_split  = {4};
std::vector<std::string> g_double_on          = {"ANY"};
std::vector<bool>        g_resplit_aces       = {false};
std::vector<bool>        g_hit_split_aces     = {false};
std::vector<bool>        g_peek               = {false};
std::vector<std::string> g_surrender          = {"2-10"};
std::vector<float>       g_blackjack_pay      = {1.5f};

// ---------------------------------------------------------------------------
// OLS solver (14×14 Gauss-Jordan)
// ---------------------------------------------------------------------------
static std::array<double, 14> solveOLS(
    std::array<std::array<double, 14>, 14> A,
    std::array<double, 14> b)
{
    std::array<std::array<double, 15>, 14> M;
    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 14; ++j) M[i][j] = A[i][j];
        M[i][14] = b[i];
    }
    for (int col = 0; col < 14; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 14; ++row)
            if (std::abs(M[row][col]) > std::abs(M[pivot][col])) pivot = row;
        if (pivot != col) std::swap(M[col], M[pivot]);
        double diag = M[col][col];
        if (std::abs(diag) < 1e-15)
            throw std::runtime_error("Singular XtX in OLS checkpoint — insufficient data");
        for (int j = col; j <= 14; ++j) M[col][j] /= diag;
        for (int row = 0; row < 14; ++row) {
            if (row == col) continue;
            double f = M[row][col];
            for (int j = col; j <= 14; ++j) M[row][j] -= f * M[col][j];
        }
    }
    std::array<double, 14> w;
    for (int i = 0; i < 14; ++i) w[i] = M[i][14];
    return w;
}

// ---------------------------------------------------------------------------
// Parse helpers
// ---------------------------------------------------------------------------
static std::array<double, 13> parseWeights13(const std::string& s) {
    std::array<double, 13> w{};
    std::istringstream ss(s);
    std::string tok;
    int i = 0;
    while (std::getline(ss, tok, ',') && i < 13)
        w[i++] = std::stod(tok);
    if (i != 13)
        throw std::invalid_argument("--count-weights requires exactly 13 comma-separated values");
    return w;
}

static std::array<double, 14> loadOlsFromCheckpoint(const std::string& folder) {
    namespace fs = std::filesystem;
    fs::path root = fs::path(PROJECT_ROOT) / kOlsCheckpointRoot / folder;
    fs::path dataPath = root / "data.json";
    if (!fs::exists(dataPath))
        throw std::runtime_error("OLS checkpoint missing data.json: " + root.string());

    json data; { std::ifstream f(dataPath); f >> data; }
    if (!data.contains("XtX") || data["XtX"].size() != 14)
        throw std::runtime_error("Checkpoint XtX is not 14×14 — run FindOptimalCount first");

    std::array<std::array<double, 14>, 14> XtX{};
    std::array<double, 14> Xty{};
    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 14; ++j)
            XtX[i][j] = data["XtX"][i][j].get<double>();
        Xty[i] = data["Xty"][i].get<double>();
    }
    return solveOLS(XtX, Xty);
}

// ---------------------------------------------------------------------------
// Resolve the counting system from CLI params into a CountingSystem
// ---------------------------------------------------------------------------
static CountingSystem resolveCountingSystem() {
    int sources = (!g_count_name.empty() ? 1 : 0)
                + (!g_count_weights_str.empty() ? 1 : 0)
                + (!g_count_ols.empty() ? 1 : 0);
    if (sources > 1)
        throw std::invalid_argument("Specify only one of --count, --count-weights, --count-ols");

    CountingSystem sys;

    if (!g_count_ols.empty()) {
        auto w14 = loadOlsFromCheckpoint(g_count_ols);
        for (int i = 0; i < 13; ++i) sys.weights[i] = w14[i];
        sys.factor = 1.0;
        sys.bias   = w14[13];
    } else if (!g_count_weights_str.empty()) {
        sys.weights = parseWeights13(g_count_weights_str);
        sys.factor  = CountingMethods::kDefaultFactor;
        sys.bias    = 0.0;
    } else if (!g_count_name.empty()) {
        auto opt = CountingMethods::fromName(g_count_name);
        if (!opt) throw std::runtime_error("Unknown count name: " + g_count_name
                      + "\nAvailable: none, hilo, ko, hiopt1, hiopt2, omega2, zen, halves");
        sys = *opt;
    } else {
        // Default: Hi-Lo
        sys = *CountingMethods::fromName("hilo");
        std::cout << "Count:       Hi-Lo (default)\n";
    }

    // Apply CLI overrides
    if (!std::isnan(g_factor)) sys.factor = g_factor;
    if (!std::isnan(g_bias))   sys.bias   = g_bias;

    return sys;
}

static bool approxEqual(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

static std::string weightsToString(const std::array<double, 13>& weights) {
    std::ostringstream os;
    os << "[";
    for (size_t i = 0; i < weights.size(); ++i) {
        if (i > 0) os << ", ";
        os << weights[i];
    }
    os << "]";
    return os.str();
}

// Count configuration carried out of a checkpoint (must match training for correct state lookups).
struct CheckpointCountConfig {
    std::array<double, 13> weights{};
    double resolution = 1.0;
    int    minCount   = 0;
    int    maxCount   = 0;
};

static void warnIfCountConfigDiffers(const CountingSystem& requestedCount,
                                     const CheckpointCountConfig& loadedCount) {
    bool weightsDiffer = false;
    for (size_t i = 0; i < requestedCount.weights.size(); ++i) {
        if (!approxEqual(requestedCount.weights[i], loadedCount.weights[i])) {
            weightsDiffer = true;
            break;
        }
    }

    bool resolutionDiffers = !approxEqual(g_count_resolution, loadedCount.resolution);
    if (!weightsDiffer && !resolutionDiffers) return;

    std::cout << "WARNING: requested count parameters differ from the loaded strategy:\n";
    if (weightsDiffer) {
        std::cout << "  requested weights = " << weightsToString(requestedCount.weights) << "\n";
        std::cout << "  strategy weights  = " << weightsToString(loadedCount.weights) << "\n";
    }
    if (resolutionDiffers) {
        std::cout << "  requested resolution = " << g_count_resolution << "\n";
        std::cout << "  strategy resolution  = " << loadedCount.resolution << "\n";
    }
    std::cout << "  Play decisions will use the strategy's stored count config"
              << " (resolution=" << loadedCount.resolution
              << ", range=[" << loadedCount.minCount << ", " << loadedCount.maxCount << "]).\n";
}

// ---------------------------------------------------------------------------
// Load a learned policy from a Q-learning checkpoint folder.
// The checkpoint is converted to a fixed BasicStrategy so MeasureEdge evaluates
// the learned greedy action per state, not the training-time exploration policy.
// Fills outCount with the count config used during training (caller must apply
// it to the player so that stateToKey generates the same keys as during training).
// Reads meta.json, warns on stdout if game rules differ from the current Case.
// ---------------------------------------------------------------------------
static std::unique_ptr<Strategy> loadStrategyFromCheckpoint(
    const std::string& folder, const std::string& agentNameHint,
    const Case& c, CheckpointCountConfig& outCount)
{
    namespace fs = std::filesystem;
    fs::path ckptRoot = fs::path(PROJECT_ROOT) / kQlearningCheckpointRoot / folder;
    fs::path metaPath = ckptRoot / "meta.json";
    if (!fs::exists(metaPath))
        throw std::runtime_error(
            "Checkpoint '" + folder + "' not found or missing meta.json");

    json meta; { std::ifstream f(metaPath); f >> meta; }

    // Validate and resolve agent name
    if (!meta.contains("agents") || meta["agents"].empty())
        throw std::runtime_error("meta.json in '" + folder + "' has no agents");

    std::string agentName = agentNameHint;
    if (agentName.empty()) {
        agentName = meta["agents"][0]["name"].get<std::string>();
    } else {
        bool found = false;
        for (const auto& a : meta["agents"])
            if (a["name"].get<std::string>() == agentName) { found = true; break; }
        if (!found) {
            std::cout << "WARNING: agent '" << agentName << "' not found in checkpoint. Available:";
            for (const auto& a : meta["agents"])
                std::cout << " " << a["name"].get<std::string>();
            std::cout << "\n";
            throw std::runtime_error("Agent '" + agentName + "' not found in checkpoint '" + folder + "'");
        }
    }

    // Extract count config for this agent (required for correct stateToKey lookups)
    for (const auto& a : meta["agents"]) {
        if (a["name"].get<std::string>() != agentName) continue;
        if (!a.contains("counting")) break;
        const auto& cnt = a["counting"];
        if (cnt.contains("weights")) {
            const auto& w = cnt["weights"];
            for (int i = 0; i < 13 && i < (int)w.size(); ++i)
                outCount.weights[i] = w[i].get<double>();
        }
        if (cnt.contains("resolution")) outCount.resolution = cnt["resolution"].get<double>();
        if (cnt.contains("min_count"))  outCount.minCount   = cnt["min_count"].get<int>();
        if (cnt.contains("max_count"))  outCount.maxCount   = cnt["max_count"].get<int>();
        break;
    }

    // Compare game rules — print to stdout so they're never missed
    if (meta.contains("game")) {
        const auto& g = meta["game"];
        std::vector<std::string> mismatches;
        auto warnIf = [&](const char* key, const std::string& sim, const std::string& ckpt) {
            if (sim != ckpt)
                mismatches.push_back(std::string(key) + ": simulation=" + sim + ", checkpoint=" + ckpt);
        };
        if (g.contains("decks"))
            warnIf("decks", std::to_string(c.deckSize), std::to_string(g["decks"].get<int>()));
        if (g.contains("ss17"))
            warnIf("ss17", c.standSoft17 ? "true" : "false", g["ss17"].get<bool>() ? "true" : "false");
        if (g.contains("das"))
            warnIf("das", c.doubleAfterSplit ? "true" : "false", g["das"].get<bool>() ? "true" : "false");
        if (g.contains("sas"))
            warnIf("sas", std::to_string(c.splitAfterSplit), std::to_string(g["sas"].get<int>()));
        if (g.contains("rsa"))
            warnIf("rsa", c.reSplitAces ? "true" : "false", g["rsa"].get<bool>() ? "true" : "false");
        if (g.contains("hsa"))
            warnIf("hsa", c.hitSplitAces ? "true" : "false", g["hsa"].get<bool>() ? "true" : "false");
        if (g.contains("peek"))
            warnIf("peek", c.peek ? "true" : "false", g["peek"].get<bool>() ? "true" : "false");
        if (g.contains("don")) {
            std::string simDon = (c.doubleOn == DoubleDownOn::ANY) ? "ANY" :
                                 (c.doubleOn == DoubleDownOn::NINE_TEN_ELEVEN) ? "9-11" : "10-11";
            warnIf("don", simDon, g["don"].get<std::string>());
        }
        if (g.contains("surrender")) {
            std::string simSurr = (c.surrender == Surrender::NO_SURRENDER) ? "no" :
                                  (c.surrender == Surrender::SURRENDER_ANY) ? "yes" : "2-10";
            warnIf("surrender", simSurr, g["surrender"].get<std::string>());
        }
        if (g.contains("bj_pay")) {
            float ckptBj = g["bj_pay"].get<float>();
            if (std::abs(c.blackJackPay - ckptBj) > 0.001f)
                mismatches.push_back("bj_pay: simulation=" + std::to_string(c.blackJackPay)
                                     + ", checkpoint=" + std::to_string(ckptBj));
        }
        if (!mismatches.empty()) {
            std::cout << "WARNING: simulation rules differ from checkpoint training rules:\n";
            for (const auto& m : mismatches)
                std::cout << "  " << m << "\n";
            std::cout << "  Strategy may not be optimal for this game configuration.\n";
        }
    }

    // Load the agent file — pass relative path so loadFromFile prepends PROJECT_ROOT correctly
    std::string relPath = std::string(kQlearningCheckpointRoot) + "/" + folder + "/" + agentName + "_agent.json";
    auto qstrat = QLearningStrategy::loadFromFile(relPath);
    if (!qstrat)
        throw std::runtime_error("Failed to load agent file: " + relPath);

    auto learnedPolicy = qstrat->toBasicStrategy();
    if (!learnedPolicy)
        throw std::runtime_error("Failed to derive greedy policy from agent file: " + relPath);

    std::cout << "Strategy:    Learned policy from " << relPath << "\n";
    std::cout << "Count range: [" << outCount.minCount << ", " << outCount.maxCount
              << "]  resolution=" << outCount.resolution << "\n";
    return learnedPolicy;
}

// ---------------------------------------------------------------------------
// Run one case
// ---------------------------------------------------------------------------
static void runCase(const Case& c) {
    RunLogger logger(std::filesystem::path(PROJECT_ROOT) / kMeasureEdgeLogRoot,
                     currentTimestamp() + "_" + ToStringTableName(c));
    std::cout << "\n=== MeasureEdge ===\n";
    std::cout << "Command:     " << g_command_line << "\n";
    bool noModeFlatBet = (!g_mode_explicit && g_mode == "spread" && g_spread_str.empty());
    std::cout << "Mode:        " << (noModeFlatBet ? "none (flat unit betting)" : g_mode) << "\n";
    std::cout << "Scenario:    " << ToString(c)
              << "_penetration=" << g_penetration << "%\n";
    if (g_mode == "kelly") {
        std::cout << "Kelly eval:  " << g_kelly_measurements
                  << " x " << kKellyMeasurementRounds
                  << " rounds  Threads: " << g_num_threads << "\n";
    } else {
        std::cout << "Rounds:      " << g_num_rounds
                  << "  Threads: " << g_num_threads << "\n";
    }
    std::cout << "Penetration: " << g_penetration << "%\n";

    CountingSystem sys = resolveCountingSystem();

    if (!g_count_name.empty())        std::cout << "Count:       " << g_count_name << "\n";
    else if (!g_count_weights_str.empty()) std::cout << "Count:       custom weights\n";
    else if (!g_count_ols.empty())    std::cout << "Count:       OLS from " << kOlsCheckpointRoot << "/" << g_count_ols << "/\n";

    std::cout << "Factor:      " << sys.factor << "\n";
    std::cout << "Bias:        " << sys.bias   << "\n";
    std::cout << "Min bet:     " << g_min_bet  << "\n";
    if (g_max_bet < std::numeric_limits<double>::max())
        std::cout << "Max bet:     " << g_max_bet << "\n";

    // Load play strategy — Q-learning checkpoint if specified, else BasicStrategy
    CheckpointCountConfig ckptCount;
    std::unique_ptr<Strategy> strat;
    if (!g_strategy_checkpoint.empty()) {
        strat = loadStrategyFromCheckpoint(g_strategy_checkpoint, g_strategy_agent, c, ckptCount);
        warnIfCountConfigDiffers(sys, ckptCount);
    } else {
        std::string tableName = ToStringTableName(c);
        auto bs = std::make_unique<BasicStrategy>();
        if (!bs->loadFromJson(tableName))
            throw std::runtime_error("Cannot load BasicStrategy: " + tableName);
        std::cout << "Strategy:    BasicStrategy (" << tableName << ")\n";
        strat = std::move(bs);
    }

    if (g_strategy_verbose) {
        std::cout << "\n=== Strategy Table ===\n";
        std::cout << *strat;
        std::cout << "======================\n";
    }

    double startMoney = std::isnan(g_initial_money)
                        ? (g_mode == "kelly" ? 1.0 : 0.0)
                        : g_initial_money;

    auto makeConfiguredPlayer = [&](double initialMoney) -> Player* {
        auto* player = new Player(initialMoney, strat->clone());
        player->setNumDecks(c.deckSize);

        if (!g_strategy_checkpoint.empty()) {
            // Count weights/resolution/range MUST match training or state key lookups will miss.
            // Apply checkpoint config first, then override factor/bias from CLI for E[game] model.
            player->setCountWeights(ckptCount.weights);
            player->setCountResolution(ckptCount.resolution);
            player->setCountRange(ckptCount.minCount, ckptCount.maxCount);
            player->setCountFactor(sys.factor);
            player->setCountBias(sys.bias);
        } else {
            player->setCountSystem(sys);
            player->setCountResolution(g_count_resolution);
        }

        if (g_mode == "spread") {
            if (!g_spread_str.empty()) {
                player->setBettingStrategy(
                    std::make_unique<SpreadBetting>(SpreadBetting::fromString(g_spread_str)));
            }
        } else if (g_mode == "kelly") {
            player->setBettingStrategy(std::make_unique<KellyBetting>(g_kelly_fraction));
        } else {
            delete player;
            throw std::invalid_argument("Unknown mode '" + g_mode + "': use spread or kelly");
        }

        return player;
    };

    if (g_mode == "spread") {
        if (g_spread_str.empty()) {
            if (noModeFlatBet) {
                std::cout << "Spread:      none (flat unit betting)\n";
            } else {
                throw std::invalid_argument("--spread is required in spread mode (e.g. \"0.5:1,1:2,2:5,3:10\")");
            }
        } else {
            std::cout << "Spread:      " << g_spread_str << "\n";
        }
    } else if (g_mode == "kelly") {
        std::cout << "Kelly frac:  " << g_kelly_fraction << "\n";
        std::cout << "Start money: " << startMoney << "\n";
    } else {
        throw std::invalid_argument("Unknown mode '" + g_mode + "': use spread or kelly");
    }

    std::cout << "========================\n";

    BlackjackRules rules(c.blackJackPay, c.standSoft17, c.deckSize, g_penetration,
                         c.peek, c.splitAfterSplit, c.doubleAfterSplit,
                         c.reSplitAces, c.hitSplitAces, c.surrender, c.doubleOn);
    rules.minBet = (g_mode == "kelly" && !g_min_bet_explicit) ? 0.0 : g_min_bet;
    rules.maxBet = g_max_bet;

    auto t0 = std::chrono::high_resolution_clock::now();

    if (g_mode == "spread") {
        auto* player = makeConfiguredPlayer(startMoney);
        std::vector<Player*> results =
            runParallelSimulation(rules, {player}, g_num_rounds, g_num_threads);
        double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count() / 1000.0;

        delete player;
        Player* result = results[0];

        std::cout << "\n=== Results (" << g_num_rounds << " rounds, "
                  << std::fixed << std::setprecision(0)
                  << (g_num_rounds / secs) << " hands/sec) ===\n";
        std::cout << std::setprecision(6);

        uint64_t roundsPerThread = g_num_rounds / static_cast<uint64_t>(g_num_threads);
        double netPerRound = result->getMoney() / static_cast<double>(roundsPerThread);
        std::cout << "Average net per round: " << netPerRound << "\n";
        std::cout << "  (positive = player edge per unit bet)\n";

        delete result;
        return;
    }

    double growthSum = 0.0;
    double growthSqSum = 0.0;
    double avgFinalSum = 0.0;
    double avgFinalSqSum = 0.0;
    double avgLogFinalSum = 0.0;
    double avgLogFinalSqSum = 0.0;
    for (int rep = 0; rep < g_kelly_measurements; ++rep) {
        auto* player = makeConfiguredPlayer(startMoney);
        std::vector<Player*> results =
            runParallelSimulation(rules, {player}, kKellyMeasurementRounds, g_num_threads);
        delete player;
        if (results.empty())
            throw std::runtime_error("Kelly measurement failed: no result players");

        Player* result = results[0];
        uint64_t roundsPerThread = kKellyMeasurementRounds / static_cast<uint64_t>(g_num_threads);
        double avgFinal = result->getMoney();
        double avgLogFinal = result->getLogMoney();
        double growthRate = (startMoney > 0.0 && std::isfinite(avgLogFinal))
            ? std::exp((avgLogFinal - std::log(startMoney)) / static_cast<double>(roundsPerThread))
            : 0.0;

        growthSum += growthRate;
        growthSqSum += growthRate * growthRate;
        avgFinalSum += avgFinal;
        avgFinalSqSum += avgFinal * avgFinal;
        avgLogFinalSum += avgLogFinal;
        avgLogFinalSqSum += avgLogFinal * avgLogFinal;
        delete result;
    }

    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - t0).count() / 1000.0;

    uint64_t totalKellyRounds = kKellyMeasurementRounds * static_cast<uint64_t>(g_kelly_measurements);
    const double n = static_cast<double>(g_kelly_measurements);
    auto sampleStddev = [n](double sum, double sqSum) {
        if (n <= 1.0) return 0.0;
        double sampleVariance = (sqSum - (sum * sum / n)) / (n - 1.0);
        return std::sqrt(std::max(0.0, sampleVariance));
    };
    double avgFinalMean = avgFinalSum / n;
    double avgFinalStddev = sampleStddev(avgFinalSum, avgFinalSqSum);
    double avgLogFinalMean = avgLogFinalSum / n;
    double avgLogFinalStddev = sampleStddev(avgLogFinalSum, avgLogFinalSqSum);
    double growthMean = growthSum / n;
    double growthStddev = sampleStddev(growthSum, growthSqSum);

    std::cout << "\n=== Results (" << g_kelly_measurements
              << " experiments x " << kKellyMeasurementRounds << " rounds, "
              << std::fixed << std::setprecision(0)
              << (totalKellyRounds / secs) << " hands/sec) ===\n";
    std::cout << std::setprecision(6);
    std::cout << "Average final bankroll: "
              << avgFinalMean << "  stddev: " << avgFinalStddev << "\n";
    std::cout << "Average log bankroll:   "
              << avgLogFinalMean << "  stddev: " << avgLogFinalStddev << "\n";
    std::cout << "Growth rate per round:  "
              << growthMean << "  stddev: " << growthStddev << "\n";
    std::cout << "  (>1.0 = bankroll grows; <1.0 = bankroll shrinks)\n";
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------
static void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " --mode <spread|kelly> [OPTIONS]\n\n";
    std::cout << "Measure the edge of a card counter under a fixed BasicStrategy.\n\n";

    std::cout << "MODES:\n";
    std::cout << "  spread   Step-function bet sizing keyed on true count.\n";
    std::cout << "           Metric: average net profit per round.\n";
    std::cout << "  kelly    Bet a fraction of bankroll proportional to E[game].\n";
    std::cout << "           Metric: average growth rate per round across 100 experiments of 1,000,000 rounds each.\n\n";

    std::cout << "SIMULATION:\n";
    std::cout << "  --num-rounds <N>       (default: 1000000000)\n";
    std::cout << "  --num-threads <N>      (default: 16)\n";
    std::cout << "  --penetration <val>    (default: 75.0)\n";
    std::cout << "  --kelly-measurements <N>  Number of Kelly experiments of 1,000,000 rounds each (default: 100)\n\n";

    std::cout << "COUNT SPECIFICATION (pick one):\n";
    std::cout << "  --count <name>         Named system: none, hilo, ko, hiopt1, hiopt2, omega2, zen, halves\n";
    std::cout << "  --count-weights <csv>  Explicit 13 weights: \"1,1,1,1,1,0,0,0,-1,-1,-1,-1,-1\"\n";
    std::cout << "  --count-ols <dir>      Load OLS weights from checkpoints/checkpoints_ols/<dir>/data.json\n";
    std::cout << "                         (sets factor=1.0, bias=w[13] automatically)\n\n";

    std::cout << "COUNT MODEL OVERRIDES:\n";
    std::cout << "  --factor <val>         Override E[game] factor (default from count system)\n";
    std::cout << "  --bias <val>           Override E[game] bias   (default from count system)\n";
    std::cout << "  --count-resolution <v> True-count discretization step (default: 1.0)\n\n";

    std::cout << "TABLE PARAMETERS:\n";
    std::cout << "  --min-bet <val>        Table minimum bet (default: 1.0, or 0.0 in kelly mode)\n";
    std::cout << "  --max-bet <val>        Table maximum bet (default: unlimited)\n";
    std::cout << "  --initial-money <val>  Starting bankroll (default: 0 for spread, 1 for kelly)\n\n";

    std::cout << "MODE OPTIONS:\n";
    std::cout << "  --spread <map>         Spread map: \"1:1,2:4,3:8,4:12\"  (threshold:bet,...)\n";
    std::cout << "                         Bet B when count strictly exceeds threshold T.\n";
    std::cout << "                         Count below the lowest threshold returns 0 (table clamps to --min-bet).\n";
    std::cout << "                         If --mode is omitted and no spread is given, MeasureEdge uses flat unit betting.\n";
    std::cout << "  --kelly-fraction <val> Scale applied to Kelly fraction (default: 1.0)\n\n";

    std::cout << "STRATEGY (default: BasicStrategy):\n";
    std::cout << "  --strategy <dir>         Load a learned policy from checkpoints/checkpoints_QLearning/<dir>/\n";
    std::cout << "                           The checkpoint is converted to a fixed greedy strategy table\n";
    std::cout << "                           before simulation; no exploration is used in MeasureEdge.\n";
    std::cout << "                           A warning is printed if the checkpoint's training rules\n";
    std::cout << "                           differ from the current simulation rules.\n";
    std::cout << "  --strategy-agent <name>  Agent name within the checkpoint (default: first agent)\n\n";
    std::cout << "  --strategy-verbose       Print the resolved strategy table before simulation\n\n";

    std::cout << "GAME CONFIG:\n";
    std::cout << "  --decks, --ss17, --das, --sas, --don, --rsa, --hsa,\n";
    std::cout << "  --peek, --surr, --bj   (comma-sep lists, same as FindDeviations)\n\n";

    std::cout << "EXAMPLES:\n";
    std::cout << "  # Hi-Lo spread betting (1-12 spread) on a 6-deck S17 game:\n";
    std::cout << "  " << prog << " --mode spread --count hilo \\\n";
    std::cout << "      --spread \"1:1,2:4,3:8,4:12\" --min-bet 1 --max-bet 12 \\\n";
    std::cout << "      --decks 6 --ss17 true --peek true\n\n";

    std::cout << "  # Kelly betting using OLS-derived count (from FindOptimalCount checkpoint):\n";
    std::cout << "  " << prog << " --mode kelly --count-ols my_checkpoint \\\n";
    std::cout << "      --kelly-fraction 0.5 --min-bet 1 --initial-money 1000 \\\n";
    std::cout << "      --decks 6 --ss17 true --peek true\n\n";

    std::cout << "  # Hi-Lo Kelly with explicit bias override for a known house edge:\n";
    std::cout << "  " << prog << " --mode kelly --count hilo \\\n";
    std::cout << "      --bias -0.0043 --kelly-fraction 1.0 --min-bet 1 \\\n";
    std::cout << "      --decks 6 --ss17 true --peek true\n\n";

    std::cout << "  # Kelly with a trained Q-learning strategy from a checkpoint:\n";
    std::cout << "  " << prog << " --mode kelly --count hilo \\\n";
    std::cout << "      --strategy my_training_run --strategy-agent agent_0 \\\n";
    std::cout << "      --kelly-fraction 0.5 --min-bet 1 \\\n";
    std::cout << "      --decks 6 --ss17 true --peek true\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    g_command_line = commandLineFromArgs(argc, argv);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printHelp(argv[0]); return 0; }

        else if (arg == "--mode"             && i+1<argc) { g_mode = argv[++i]; g_mode_explicit = true; }
        else if (arg == "--num-rounds"       && i+1<argc) g_num_rounds       = std::stoull(argv[++i]);
        else if (arg == "--num-threads"      && i+1<argc) g_num_threads      = std::stoi(argv[++i]);
        else if (arg == "--penetration"      && i+1<argc) g_penetration      = std::stod(argv[++i]);
        else if (arg == "--kelly-measurements" && i+1<argc) g_kelly_measurements = std::stoi(argv[++i]);

        else if (arg == "--count"            && i+1<argc) g_count_name       = argv[++i];
        else if (arg == "--count-weights"    && i+1<argc) g_count_weights_str= argv[++i];
        else if (arg == "--count-ols"        && i+1<argc) g_count_ols        = argv[++i];
        else if (arg == "--factor"           && i+1<argc) g_factor           = std::stod(argv[++i]);
        else if (arg == "--bias"             && i+1<argc) g_bias             = std::stod(argv[++i]);
        else if (arg == "--count-resolution" && i+1<argc) g_count_resolution = std::stod(argv[++i]);

        else if (arg == "--min-bet"          && i+1<argc) { g_min_bet = std::stod(argv[++i]); g_min_bet_explicit = true; }
        else if (arg == "--max-bet"          && i+1<argc) g_max_bet          = std::stod(argv[++i]);
        else if (arg == "--initial-money"    && i+1<argc) g_initial_money    = std::stod(argv[++i]);

        else if (arg == "--spread"                && i+1<argc) g_spread_str          = argv[++i];
        else if (arg == "--kelly-fraction"        && i+1<argc) g_kelly_fraction      = std::stod(argv[++i]);

        else if (arg == "--strategy"   && i+1<argc) g_strategy_checkpoint = argv[++i];
        else if (arg == "--strategy-agent"        && i+1<argc) g_strategy_agent      = argv[++i];
        else if (arg == "--strategy-verbose")                g_strategy_verbose   = true;

        else if (arg == "--decks" && i+1<argc) g_deck_sizes         = parseList<int>(argv[++i]);
        else if (arg == "--ss17"  && i+1<argc) g_stand_soft17       = parseList<bool>(argv[++i]);
        else if (arg == "--das"   && i+1<argc) g_double_after_split = parseList<bool>(argv[++i]);
        else if (arg == "--sas"   && i+1<argc) g_split_after_split  = parseList<int>(argv[++i]);
        else if (arg == "--don"   && i+1<argc) g_double_on          = parseList<std::string>(argv[++i]);
        else if (arg == "--rsa"   && i+1<argc) g_resplit_aces       = parseList<bool>(argv[++i]);
        else if (arg == "--hsa"   && i+1<argc) g_hit_split_aces     = parseList<bool>(argv[++i]);
        else if (arg == "--peek"  && i+1<argc) g_peek               = parseList<bool>(argv[++i]);
        else if (arg == "--surr"  && i+1<argc) g_surrender          = parseList<std::string>(argv[++i]);
        else if (arg == "--bj"    && i+1<argc) g_blackjack_pay      = parseList<float>(argv[++i]);

        else { std::cerr << "Unknown argument: " << arg << "\nUse --help.\n"; return 1; }
    }

    if (!g_strategy_agent.empty() && g_strategy_checkpoint.empty()) {
        std::cerr << "Error: --strategy-agent requires --strategy.\n";
        return 1;
    }
    if (g_kelly_measurements < 1) {
        std::cerr << "Error: --kelly-measurements must be >= 1.\n";
        return 1;
    }

    auto cases = generateTestCases(
        g_deck_sizes, g_stand_soft17, g_double_after_split, g_split_after_split,
        g_double_on, g_resplit_aces, g_hit_split_aces, g_peek, g_surrender, g_blackjack_pay);

    try {
        for (const auto& c : cases)
            runCase(c);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
