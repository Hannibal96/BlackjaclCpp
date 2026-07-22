#include "RegressionTestUtils.h"
#include "Game/BlackjackTable.h"
#include "Game/Player.h"
#include "RL/BasicStrategy.h"
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
#include <numeric>
#include <sstream>
#include <limits>

using json = nlohmann::json;

namespace {
constexpr const char* kOlsCheckpointRoot = "checkpoints/checkpoints_ols";
}

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
uint64_t    g_num_rounds          = 1'000'000'000ULL;
int         g_num_threads         = 16;
double      g_penetration         = 75.0;
uint64_t    g_checkpoint_interval = 0;        // 0 = checkpoint only at end
bool        g_no_save             = false;
std::string g_checkpoint_name;
std::string g_load_checkpoint;
std::string g_command_line;
uint64_t    g_sample_every        = 1;        // record every N-th round

// Game configuration (single-case defaults matching most common casino)
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

static const char* kRankNames[] = {
    "2","3","4","5","6","7","8","9","10","J","Q","K","A"
};

// ---------------------------------------------------------------------------
// Checkpoint helpers
// ---------------------------------------------------------------------------
static std::string currentTimestamp() {
    time_t t = time(nullptr);
    char buf[20];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&t));
    return buf;
}

static std::string doubleOnStr(DoubleDownOn d) {
    if (d == DoubleDownOn::ANY)             return "ANY";
    if (d == DoubleDownOn::NINE_TEN_ELEVEN) return "9-11";
    return "10-11";
}
static std::string surrenderStr(Surrender s) {
    if (s == Surrender::NO_SURRENDER)  return "no";
    if (s == Surrender::SURRENDER_ANY) return "yes";
    return "2-10";
}

static json buildMeta(const Case& c, uint64_t totalRounds, int numThreads,
                      uint64_t checkpointInterval, uint64_t sampleEvery) {
    json m;
    m["game"]["decks"]              = c.deckSize;
    m["game"]["ss17"]               = c.standSoft17;
    m["game"]["das"]                = c.doubleAfterSplit;
    m["game"]["sas"]                = c.splitAfterSplit;
    m["game"]["don"]                = doubleOnStr(c.doubleOn);
    m["game"]["rsa"]                = c.reSplitAces;
    m["game"]["hsa"]                = c.hitSplitAces;
    m["game"]["peek"]               = c.peek;
    m["game"]["surrender"]          = surrenderStr(c.surrender);
    m["game"]["bj_pay"]             = c.blackJackPay;
    m["game"]["penetration"]        = g_penetration;
    m["sim"]["total_rounds"]        = totalRounds;
    m["sim"]["num_threads"]         = numThreads;
    m["sim"]["checkpoint_interval"] = checkpointInterval;
    m["sim"]["sample_every"]        = sampleEvery;
    return m;
}

static void saveCheckpoint(const std::string& folder, const Player& player,
                           const Case& c, uint64_t roundsPlayed, uint64_t roundsRecorded) {
    namespace fs = std::filesystem;
    fs::path root = fs::path(PROJECT_ROOT) / kOlsCheckpointRoot / folder;
    fs::create_directories(root);

    // meta.json
    json meta = buildMeta(c, roundsPlayed, g_num_threads, g_checkpoint_interval, g_sample_every);
    {
        std::ofstream f(root / "meta.json");
        if (f.is_open()) f << meta.dump(2);
    }

    // data.json: XtX, Xty, rounds_played, rounds_recorded
    json data;
    data["rounds_played"]   = roundsPlayed;
    data["rounds_recorded"] = roundsRecorded;
    json xtx = json::array();
    for (int i = 0; i < 14; ++i) {
        json row = json::array();
        for (int j = 0; j < 14; ++j) row.push_back(player.getXtX()[i][j]);
        xtx.push_back(row);
    }
    data["XtX"] = xtx;
    json xty = json::array();
    for (int i = 0; i < 14; ++i) xty.push_back(player.getXty()[i]);
    data["Xty"] = xty;

    std::ofstream f(root / "data.json");
    if (f.is_open()) f << data.dump(2);
}

// ---------------------------------------------------------------------------
// CSM house edge lookup (for bias sanity check)
// ---------------------------------------------------------------------------
static std::string edgeKey(const Case& c) {
    std::ostringstream os;
    os << "d=" << c.deckSize
       << "_p=100"
       << "_ss17=" << (c.standSoft17 ? "True" : "False")
       << "_das=" << (c.doubleAfterSplit ? "True" : "False")
       << "_sas=" << c.splitAfterSplit
       << "_rsa=" << (c.reSplitAces ? "True" : "False")
       << "_hsa=" << (c.hitSplitAces ? "True" : "False")
       << "_don=" << (c.doubleOn == DoubleDownOn::ANY ? "Any" :
                      c.doubleOn == DoubleDownOn::NINE_TEN_ELEVEN ? "9-11" : "10-11")
       << "_surr=" << (c.surrender == Surrender::NO_SURRENDER ? "no" :
                       c.surrender == Surrender::SURRENDER_ANY ? "yes" : "2-10")
       << "_peek=" << (c.peek ? "True" : "False")
       << "_bj=" << c.blackJackPay;
    return os.str();
}

// Returns CSM edge as a fraction (e.g. -0.0061 for -0.61%), or NaN on lookup failure.
static double lookupCsmEdge(const Case& c) {
    namespace fs = std::filesystem;
    fs::path p = fs::path(PROJECT_ROOT) / "basic_strategy_tables" / "json_wod_edge_csm.json";
    std::ifstream f(p);
    if (!f.is_open()) return std::numeric_limits<double>::quiet_NaN();

    json data; f >> data;
    std::string key = edgeKey(c);
    if (!data.contains(key)) return std::numeric_limits<double>::quiet_NaN();

    std::string val = data[key].get<std::string>();
    if (!val.empty() && val.back() == '%') val.pop_back();
    return std::stod(val) / 100.0;
}

// ---------------------------------------------------------------------------
// OLS solver — Gauss-Jordan elimination (14×14 augmented system)
// w[0..12] = counting weights, w[13] = bias/intercept
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
            if (std::abs(M[row][col]) > std::abs(M[pivot][col]))
                pivot = row;
        if (pivot != col) std::swap(M[col], M[pivot]);

        double diag = M[col][col];
        if (std::abs(diag) < 1e-15)
            throw std::runtime_error("Singular XtX — not enough variation in the data");

        for (int j = col; j <= 14; ++j) M[col][j] /= diag;
        for (int row = 0; row < 14; ++row) {
            if (row == col) continue;
            double f = M[row][col];
            for (int j = col; j <= 14; ++j)
                M[row][j] -= f * M[col][j];
        }
    }

    std::array<double, 14> w;
    for (int i = 0; i < 14; ++i) w[i] = M[i][14];
    return w;
}

// ---------------------------------------------------------------------------
// Print results + sanity checks
// ---------------------------------------------------------------------------
static void printResults(const std::array<double, 14>& w, const Case& c) {
    std::cout << "\n=== Optimal Counting Weights ===\n";
    std::cout << std::fixed << std::setprecision(6);

    double scale = std::abs(w[12]) > 1e-9 ? std::abs(w[12]) : 1.0;

    std::cout << std::left;
    std::cout << "  " << std::setw(6) << "Rank"
              << std::setw(14) << "Weight (raw)"
              << "Weight (norm to |Ace|=1)\n";
    std::cout << "  " << std::string(40, '-') << "\n";
    for (int i = 0; i < 13; ++i) {
        std::cout << "  " << std::setw(6) << kRankNames[i]
                  << std::setw(14) << w[i]
                  << (w[i] / scale) << "\n";
    }
    std::cout << "  " << std::string(40, '-') << "\n";
    std::cout << "  " << std::setw(6) << "Bias"
              << std::setw(14) << w[13] << "(intercept — expected ≈ house edge)\n";

    // Sanity check 1: 10-value cards (10, J, Q, K = indices 8..11) should be ~equal
    double w10 = w[8], wJ = w[9], wQ = w[10], wK = w[11];
    double tenMean = (w10 + wJ + wQ + wK) / 4.0;
    double tenMaxDev = std::max({std::abs(w10 - tenMean), std::abs(wJ - tenMean),
                                  std::abs(wQ - tenMean), std::abs(wK - tenMean)});
    double tenRelDev = (std::abs(tenMean) > 1e-12) ? tenMaxDev / std::abs(tenMean) : tenMaxDev;

    std::cout << "\n--- Sanity Check 1: 10-value cards (10/J/Q/K should be ~equal) ---\n";
    std::cout << "  Mean: " << tenMean
              << "  Max deviation: " << tenMaxDev
              << "  Relative: " << std::setprecision(4) << (tenRelDev * 100.0) << "%\n";
    if (tenRelDev < 0.05)
        std::cout << "  PASS (relative deviation < 5%)\n";
    else
        std::cout << "  WARN (relative deviation >= 5% — may need more rounds)\n";

    // Sanity check 2: sum of count weights (w[0..12] only) should be ~0
    double wSum = 0.0;
    for (int i = 0; i < 13; ++i) wSum += w[i];
    double sumRelScale = scale > 1e-12 ? std::abs(wSum) / scale : std::abs(wSum);

    std::cout << "\n--- Sanity Check 2: sum of count weights (excl. bias) should be ~0 ---\n";
    std::cout << "  Sum(w[0..12]): " << std::setprecision(6) << wSum
              << "  (relative to |Ace| weight: " << std::setprecision(4)
              << (sumRelScale * 100.0) << "%)\n";
    if (sumRelScale < 0.05)
        std::cout << "  PASS (sum < 5% of Ace weight)\n";
    else
        std::cout << "  WARN (sum >= 5% of Ace weight — may need more rounds)\n";

    // Sanity check 3: bias ≈ player's expected net per hand = -(house edge)
    // JSON stores house edge (positive = house wins), bias is player's net (opposite sign).
    double csmHouseEdge = lookupCsmEdge(c);
    std::cout << "\n--- Sanity Check 3: bias should be ≈ -(CSM house edge) ---\n";
    std::cout << "  Bias (w[13]):         " << std::setprecision(6) << w[13] << "\n";
    if (std::isnan(csmHouseEdge)) {
        std::cout << "  Expected player net:  (house edge not found in table)\n";
    } else {
        double expectedBias = -csmHouseEdge;  // player net = -(house edge)
        double biasDiff = std::abs(w[13] - expectedBias);
        std::cout << "  CSM house edge:       " << std::setprecision(6) << csmHouseEdge
                  << "  (" << std::setprecision(4) << (csmHouseEdge * 100.0) << "%)\n";
        std::cout << "  Expected player net:  " << std::setprecision(6) << expectedBias << "\n";
        std::cout << "  Difference:           " << biasDiff
                  << "  (relative: " << std::setprecision(4)
                  << (std::abs(expectedBias) > 1e-12 ? biasDiff / std::abs(expectedBias) * 100.0 : biasDiff * 100.0)
                  << "%)\n";
        if (biasDiff < 0.002)
            std::cout << "  PASS (difference < 0.2%)\n";
        else
            std::cout << "  WARN (difference >= 0.2% — may need more rounds)\n";
    }
}

// ---------------------------------------------------------------------------
// Run one case
// ---------------------------------------------------------------------------
static void runCase(const Case& c, const std::string& ckptFolder,
                    std::array<std::array<double, 14>, 14> prevXtX,
                    std::array<double, 14> prevXty,
                    uint64_t prevRecordedRounds,
                    uint64_t prevPlayedRounds) {
    RunLogger logger(std::filesystem::path(PROJECT_ROOT) / kOlsCheckpointRoot, ckptFolder);

    std::cout << "\n=== FindOptimalCount ===\n";
    std::cout << "Command:     " << g_command_line << "\n";
    std::cout << "Scenario:    " << ToString(c)
              << "_penetration=" << g_penetration << "%\n";
    std::cout << "Rounds:      " << g_num_rounds
              << "  Threads: " << g_num_threads << "\n";
    std::cout << "Penetration: " << g_penetration << "%\n";
    std::cout << "Sample-every: " << g_sample_every << "\n";
    std::cout << "Folder:      " << kOlsCheckpointRoot << "/" << ckptFolder << "/\n";
    if (prevRecordedRounds > 0)
        std::cout << "Resuming from " << prevRecordedRounds << " recorded rounds ("
                  << prevPlayedRounds << " played).\n";
    std::cout << "========================\n";

    std::string tablePath = ToStringTableName(c);
    auto strat = std::make_unique<BasicStrategy>();
    if (!strat->loadFromJson(tablePath))
        throw std::runtime_error("Cannot load BasicStrategy from: " + tablePath);

    auto* player = new Player(0.0, std::move(strat));
    player->setNumDecks(c.deckSize);
    player->enableRegression();
    player->setRegressionSampleEvery(g_sample_every);

    BlackjackRules rules(c.blackJackPay, c.standSoft17, c.deckSize, g_penetration,
                         c.peek, c.splitAfterSplit, c.doubleAfterSplit,
                         c.reSplitAces, c.hitSplitAces, c.surrender, c.doubleOn);

    uint64_t remaining = g_num_rounds;
    uint64_t playedTotal = prevPlayedRounds;
    uint64_t batchSize = (g_checkpoint_interval > 0) ? g_checkpoint_interval : g_num_rounds;

    while (remaining > 0) {
        uint64_t batch = std::min(batchSize, remaining);

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<Player*> results =
            runParallelSimulation(rules, {player}, batch, g_num_threads);
        double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::high_resolution_clock::now() - t0).count() / 1000.0;

        delete player;
        player = results[0];

        playedTotal += batch;
        remaining   -= batch;

        uint64_t recordedTotal = player->getRegressionRounds() + prevRecordedRounds;

        std::cout << "\nCheckpoint: " << playedTotal << " played, "
                  << recordedTotal << " recorded"
                  << "  (" << std::fixed << std::setprecision(0)
                  << (batch / secs) << " hands/sec)\n";

        std::array<std::array<double, 14>, 14> totalXtX = player->getXtX();
        std::array<double, 14> totalXty = player->getXty();
        for (int i = 0; i < 14; ++i) {
            for (int j = 0; j < 14; ++j) totalXtX[i][j] += prevXtX[i][j];
            totalXty[i] += prevXty[i];
        }

        if (!g_no_save)
            saveCheckpoint(ckptFolder, *player, c, playedTotal, recordedTotal);

        try {
            auto w = solveOLS(totalXtX, totalXty);
            std::cout << "(intermediate result after " << recordedTotal << " recorded rounds)\n";
            printResults(w, c);
        } catch (const std::exception& e) {
            std::cout << "OLS not yet solvable: " << e.what() << "\n";
        }
    }

    delete player;
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------
static void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Find optimal card counting weights using OLS regression under a fixed BasicStrategy.\n\n";
    std::cout << "METHOD:\n";
    std::cout << "  Runs many rounds of blackjack with the known-optimal BasicStrategy.\n";
    std::cout << "  For each round, records x = [removedCards[i]/remainingDecks (13), 1.0 bias]\n";
    std::cout << "  and y = net outcome.  Solves w = (X^T X)^-1 X^T y via Gauss-Jordan.\n";
    std::cout << "  w[0..12] = card counting weights, w[13] = bias (expected ≈ house edge).\n\n";
    std::cout << "CHECKPOINT STRUCTURE:\n";
    std::cout << "    checkpoints/checkpoints_ols/<folder>/meta.json     Game + sim config\n";
    std::cout << "    checkpoints/checkpoints_ols/<folder>/data.json     XtX (14x14), Xty (14), rounds\n\n";
    std::cout << "SIMULATION:\n";
    std::cout << "  --num-rounds <N>              Rounds to run (default: 1000000000)\n";
    std::cout << "  --num-threads <N>             Threads (default: 16)\n";
    std::cout << "  --penetration <val>           Shoe penetration % (default: 75.0)\n";
    std::cout << "  --sample-every <N>            Record every N-th round (default: 1)\n";
    std::cout << "                                Use >1 to reduce autocorrelation between rounds\n";
    std::cout << "  --checkpoint-interval <N>     Save every N rounds (default: end only)\n";
    std::cout << "  --checkpoint-name <name>      Name the folder (default: timestamp)\n";
    std::cout << "  --no-save                     Disable checkpoint saving\n\n";
    std::cout << "CHECKPOINT RESUME:\n";
    std::cout << "  --load-checkpoint <folder>    Resume from checkpoints/checkpoints_ols/<folder>/\n";
    std::cout << "                                All game flags are ignored; config comes from meta.json\n\n";
    std::cout << "GAME CONFIG (ignored when --load-checkpoint is used):\n";
    std::cout << "  --decks, --ss17, --das, --sas, --don, --rsa, --hsa,\n";
    std::cout << "  --peek, --surr, --bj   (same as FindDeviations, comma-sep lists)\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    g_command_line = commandLineFromArgs(argc, argv);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printHelp(argv[0]); return 0; }

        else if (arg == "--num-rounds"          && i+1<argc) g_num_rounds          = std::stoull(argv[++i]);
        else if (arg == "--num-threads"         && i+1<argc) g_num_threads         = std::stoi(argv[++i]);
        else if (arg == "--penetration"         && i+1<argc) g_penetration         = std::stod(argv[++i]);
        else if (arg == "--sample-every"        && i+1<argc) g_sample_every        = std::stoull(argv[++i]);
        else if (arg == "--checkpoint-interval" && i+1<argc) g_checkpoint_interval = std::stoull(argv[++i]);
        else if (arg == "--checkpoint-name"     && i+1<argc) g_checkpoint_name     = argv[++i];
        else if (arg == "--no-save")                         g_no_save             = true;
        else if (arg == "--load-checkpoint"     && i+1<argc) g_load_checkpoint     = argv[++i];

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

    std::vector<Case> cases;
    std::array<std::array<double, 14>, 14> prevXtX{};
    std::array<double, 14> prevXty{};
    uint64_t prevRecordedRounds = 0;
    uint64_t prevPlayedRounds   = 0;
    std::string ckptFolder;

    if (!g_load_checkpoint.empty()) {
        Case c;
        namespace fs = std::filesystem;
        fs::path root = fs::path(PROJECT_ROOT) / kOlsCheckpointRoot / g_load_checkpoint;
        fs::path metaPath = root / "meta.json";
        fs::path dataPath = root / "data.json";
        if (!fs::exists(metaPath))
            throw std::runtime_error("Checkpoint missing meta.json");
        if (!fs::exists(dataPath))
            throw std::runtime_error("Checkpoint missing data.json");

        json meta; { std::ifstream f(metaPath); f >> meta; }
        auto& g = meta["game"];
        c.deckSize         = g["decks"].get<int>();
        c.standSoft17      = g["ss17"].get<bool>();
        c.doubleAfterSplit = g["das"].get<bool>();
        c.splitAfterSplit  = g["sas"].get<int>();
        c.doubleOn         = stringToDoubleOn(g["don"].get<std::string>());
        c.reSplitAces      = g["rsa"].get<bool>();
        c.hitSplitAces     = g["hsa"].get<bool>();
        c.peek             = g["peek"].get<bool>();
        c.surrender        = stringToSurrender(g["surrender"].get<std::string>());
        c.blackJackPay     = g["bj_pay"].get<float>();
        g_penetration      = g["penetration"].get<double>();

        json data; { std::ifstream f(dataPath); f >> data; }

        // Validate checkpoint is 14-dim (bias-augmented)
        if (!data.contains("XtX") || data["XtX"].size() != 14)
            throw std::runtime_error(
                "Checkpoint XtX is not 14×14 — it may be from an older run without the bias term. "
                "Please start a new checkpoint.");

        // rounds_recorded is the count that actually went into XtX/Xty.
        // rounds_completed was the old name (pre-played/recorded split) — fall back to it.
        prevRecordedRounds = data.contains("rounds_recorded")
            ? data["rounds_recorded"].get<uint64_t>()
            : data.value("rounds_completed", uint64_t(0));
        prevPlayedRounds   = data.value("rounds_played", uint64_t(0));

        for (int i = 0; i < 14; ++i) {
            for (int j = 0; j < 14; ++j)
                prevXtX[i][j] = data["XtX"][i][j].get<double>();
            prevXty[i] = data["Xty"][i].get<double>();
        }

        std::cout << "\n=== Loaded OLS checkpoint: " << kOlsCheckpointRoot << "/" << g_load_checkpoint << "/ ===\n";
        std::cout << "  decks=" << c.deckSize << "  ss17=" << (c.standSoft17?"true":"false")
                  << "  das=" << (c.doubleAfterSplit?"true":"false")
                  << "  sas=" << c.splitAfterSplit << "\n"
                  << "  don=" << doubleOnStr(c.doubleOn)
                  << "  rsa=" << (c.reSplitAces?"true":"false")
                  << "  hsa=" << (c.hitSplitAces?"true":"false")
                  << "  peek=" << (c.peek?"true":"false") << "\n"
                  << "  surr=" << surrenderStr(c.surrender)
                  << "  bj_pay=" << c.blackJackPay
                  << "  penetration=" << g_penetration << "%\n"
                  << "  Recorded rounds so far: " << prevRecordedRounds
                  << "  (played: " << prevPlayedRounds << ")\n"
                  << "=== (CLI game flags are ignored) ===\n\n";

        ckptFolder = g_load_checkpoint;
        cases.push_back(c);
    } else {
        cases = generateTestCases(
            g_deck_sizes, g_stand_soft17, g_double_after_split, g_split_after_split,
            g_double_on, g_resplit_aces, g_hit_split_aces, g_peek, g_surrender, g_blackjack_pay);
        ckptFolder = g_checkpoint_name.empty() ? currentTimestamp() : g_checkpoint_name;
    }

    for (const auto& c : cases) {
        std::string folder = (cases.size() == 1)
                             ? ckptFolder
                             : (ckptFolder + "_" + ToStringTableName(c));
        runCase(c, folder, prevXtX, prevXty, prevRecordedRounds, prevPlayedRounds);
    }

    return 0;
}
