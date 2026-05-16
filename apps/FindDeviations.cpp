#include "RegressionTestUtils.h"
#include "Game/BlackjackTable.h"
#include "Game/Player.h"
#include "RL/QLearningStrategy.h"
#include "RL/BasicStrategy.h"
#include "RL/DecayingParameter.h"
#include "Utils/Utils.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <ctime>
#include <memory>
#include <array>
#include <filesystem>
#include <sstream>
#include <limits>
#include <cstdlib>
#include <unistd.h>

using json = nlohmann::json;

namespace {
constexpr const char* kQlearningCheckpointRoot = "checkpoints/checkpoints_QLearning";
enum class TrainingStopMode { FIXED_ROUNDS, TABLE_DIFF };
}

// ---------------------------------------------------------------------------
// Hi-Lo weights  (index 0=TWO … 12=ACE)
// ---------------------------------------------------------------------------
static const std::array<double, 13> HI_LO_WEIGHTS = {
     1,  1,  1,  1,  1,   // 2-6  : +1
     0,  0,  0,           // 7-9  :  0
    -1, -1, -1, -1, -1    // T,J,Q,K,A : -1
};

// ---------------------------------------------------------------------------
// Agent configuration — one per player slot
// ---------------------------------------------------------------------------
struct AgentConfig {
    std::string name = "agent";

    ExplorationMode explorationMode = ExplorationMode::EPSILON_GREEDY;
    double epsilonStart  = 1.0,   epsilonMin  = 0.1,    epsilonDecay    = 0.99999;
    double tempStart     = 1.0,   tempMin     = 0.1,    tempDecay       = 0.99999;
    double alphaStart    = 0.01,  alphaMin    = 0.0001, alphaDecaySteps = 100.0;

    double countResolution = 1.0;
    int    minCount        = -5;
    int    maxCount        =  5;

    // Set by runCase — not from CLI
    std::string loadFrom;
};

// ---------------------------------------------------------------------------
// Simulation parameters
// ---------------------------------------------------------------------------
uint64_t    g_num_rounds          = 1'000'000'000ULL;
int         g_num_threads         = 16;
double      g_penetration         = 75.0;
uint64_t    g_checkpoint_interval = 0;        // 0 = checkpoint only at end
uint64_t    g_sample_rounds       = 100'000'000ULL;
double      g_diff_threshold      = 0.001;
TrainingStopMode g_training_stop_mode = TrainingStopMode::FIXED_ROUNDS;
bool        g_full_verbose        = false;
bool        g_no_save             = false;
std::string g_checkpoint_name;               // custom folder name (empty = timestamp)
std::string g_load_checkpoint;              // folder name to resume from (empty = fresh)

// Game configuration (single-case defaults)
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

// Agent configs — default is a single Hi-Lo agent; --agent-config overrides all
std::vector<AgentConfig> g_agents = {AgentConfig{}};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string currentTimestamp() {
    time_t t = time(nullptr);
    char buf[20];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&t));
    return buf;
}

static std::string doubleOnToString(DoubleDownOn d) {
    if (d == DoubleDownOn::ANY)             return "ANY";
    if (d == DoubleDownOn::NINE_TEN_ELEVEN) return "9-11";
    return "10-11";
}

static std::string surrenderToString(Surrender s) {
    if (s == Surrender::NO_SURRENDER)  return "no";
    if (s == Surrender::SURRENDER_ANY) return "yes";
    return "2-10";
}

static const char* stopModeToString(TrainingStopMode mode) {
    return (mode == TrainingStopMode::TABLE_DIFF) ? "diff" : "rounds";
}

static void saveMeta(const std::string& ckptFolder, const json& meta);

std::string buildRunHeader(const Case& c, const std::string& ckptFolder) {
    std::ostringstream os;
    os << "\n=== FindDeviations ===\n";
    os << "Scenario:   " << ToString(c) << "\n";
    if (g_training_stop_mode == TrainingStopMode::FIXED_ROUNDS) {
        os << "Rounds:     " << g_num_rounds << "  Threads: " << g_num_threads << "\n";
    } else {
        os << "Stop mode:  diff  sample=" << g_sample_rounds
           << "  threshold=" << g_diff_threshold
           << "  Threads: " << g_num_threads << "\n";
    }
    os << "Penetration:" << g_penetration << "%\n";
    if (g_checkpoint_interval > 0)
        os << "Checkpoint every " << g_checkpoint_interval << " rounds\n";
    os << "Agents:     " << g_agents.size() << "\n";
    os << "Folder:     " << kQlearningCheckpointRoot << "/" << ckptFolder << "/\n";
    os << "======================\n";
    return os.str();
}

// ---------------------------------------------------------------------------
// Build meta.json that captures the full training identity
// ---------------------------------------------------------------------------
json buildMetaJson(const Case& c, const std::vector<AgentConfig>& agents,
                   uint64_t totalRounds, int numThreads, uint64_t checkpointInterval) {
    json meta;

    meta["game"]["decks"]       = c.deckSize;
    meta["game"]["ss17"]        = c.standSoft17;
    meta["game"]["das"]         = c.doubleAfterSplit;
    meta["game"]["sas"]         = c.splitAfterSplit;
    meta["game"]["don"]         = doubleOnToString(c.doubleOn);
    meta["game"]["rsa"]         = c.reSplitAces;
    meta["game"]["hsa"]         = c.hitSplitAces;
    meta["game"]["peek"]        = c.peek;
    meta["game"]["surrender"]   = surrenderToString(c.surrender);
    meta["game"]["bj_pay"]      = c.blackJackPay;
    meta["game"]["penetration"] = g_penetration;

    meta["agents"] = json::array();
    for (const auto& ag : agents) {
        json a;
        a["name"] = ag.name;

        a["counting"]["weights"]    = HI_LO_WEIGHTS;
        a["counting"]["resolution"] = ag.countResolution;
        a["counting"]["min_count"]  = ag.minCount;
        a["counting"]["max_count"]  = ag.maxCount;

        a["training"]["exploration_mode"] =
            (ag.explorationMode == ExplorationMode::EPSILON_GREEDY ? "epsilon" : "boltzmann");
        a["training"]["alpha_start"]       = ag.alphaStart;
        a["training"]["alpha_min"]         = ag.alphaMin;
        a["training"]["alpha_decay_steps"] = ag.alphaDecaySteps;
        if (ag.explorationMode == ExplorationMode::EPSILON_GREEDY) {
            a["training"]["epsilon_start"] = ag.epsilonStart;
            a["training"]["epsilon_min"]   = ag.epsilonMin;
            a["training"]["epsilon_decay"] = ag.epsilonDecay;
        } else {
            a["training"]["temp_start"] = ag.tempStart;
            a["training"]["temp_min"]   = ag.tempMin;
            a["training"]["temp_decay"] = ag.tempDecay;
        }
        a["training"]["total_rounds"]        = totalRounds;
        a["training"]["num_threads"]         = numThreads;
        a["training"]["checkpoint_interval"] = checkpointInterval;
        a["training"]["stop_mode"]           = stopModeToString(g_training_stop_mode);
        a["training"]["sample_rounds"]       = g_sample_rounds;
        a["training"]["diff_threshold"]      = g_diff_threshold;

        meta["agents"].push_back(a);
    }

    return meta;
}

// ---------------------------------------------------------------------------
// Load checkpoint folder: reads meta.json, populates g_agents, returns Case
// ---------------------------------------------------------------------------
Case loadCheckpointFolder(const std::string& folderName) {
    namespace fs = std::filesystem;
    fs::path metaPath = fs::path(PROJECT_ROOT) / kQlearningCheckpointRoot / folderName / "meta.json";
    if (!fs::exists(metaPath))
        throw std::runtime_error(
            "Checkpoint folder '" + folderName + "' not found or missing meta.json");

    std::ifstream f(metaPath);
    if (!f.is_open())
        throw std::runtime_error(
            "Cannot open meta.json in checkpoint '" + folderName + "'");

    json j; f >> j;

    // Reconstruct Case from game config
    auto& g = j["game"];
    Case c;
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

    // Rebuild agent configs from meta
    g_agents.clear();
    for (const auto& a : j["agents"]) {
        AgentConfig ag;
        ag.name            = a["name"].get<std::string>();
        ag.countResolution = a["counting"]["resolution"].get<double>();
        ag.minCount        = a["counting"]["min_count"].get<int>();
        ag.maxCount        = a["counting"]["max_count"].get<int>();

        const auto& t  = a["training"];
        std::string mode = t["exploration_mode"].get<std::string>();
        ag.explorationMode = (mode == "boltzmann")
                             ? ExplorationMode::BOLTZMANN
                             : ExplorationMode::EPSILON_GREEDY;
        ag.alphaStart       = t["alpha_start"].get<double>();
        ag.alphaMin         = t["alpha_min"].get<double>();
        ag.alphaDecaySteps  = t["alpha_decay_steps"].get<double>();
        if (ag.explorationMode == ExplorationMode::EPSILON_GREEDY) {
            ag.epsilonStart = t.value("epsilon_start", 1.0);
            ag.epsilonMin   = t.value("epsilon_min",   0.1);
            ag.epsilonDecay = t.value("epsilon_decay",  0.99999);
        } else {
            ag.tempStart = t.value("temp_start", 1.0);
            ag.tempMin   = t.value("temp_min",   0.1);
            ag.tempDecay = t.value("temp_decay",  0.99999);
        }
        g_agents.push_back(ag);
    }

    // Print everything that was loaded so there is no confusion about what is
    // actually being used (all game + counting + training params come from meta.json,
    // not from the CLI when --load-checkpoint is active).
    std::cout << "\n=== Loaded checkpoint: " << kQlearningCheckpointRoot << "/" << folderName << "/ ===\n";
    std::cout << "Game config:\n";
    std::cout << "  decks=" << c.deckSize
              << "  ss17=" << (c.standSoft17 ? "true" : "false")
              << "  das=" << (c.doubleAfterSplit ? "true" : "false")
              << "  sas=" << c.splitAfterSplit << "\n";
    std::cout << "  don=" << doubleOnToString(c.doubleOn)
              << "  rsa=" << (c.reSplitAces ? "true" : "false")
              << "  hsa=" << (c.hitSplitAces ? "true" : "false")
              << "  peek=" << (c.peek ? "true" : "false") << "\n";
    std::cout << "  surr=" << surrenderToString(c.surrender)
              << "  bj_pay=" << c.blackJackPay
              << "  penetration=" << g_penetration << "%\n";
    for (const auto& ag : g_agents) {
        std::cout << "Agent '" << ag.name << "':\n";
        std::cout << "  counting: resolution=" << ag.countResolution
                  << "  count=[" << ag.minCount << "," << ag.maxCount << "]\n";
        std::string mode = (ag.explorationMode == ExplorationMode::BOLTZMANN)
                           ? "boltzmann" : "epsilon";
        std::cout << "  training: mode=" << mode;
        if (ag.explorationMode == ExplorationMode::EPSILON_GREEDY)
            std::cout << "  eps=[" << ag.epsilonStart << "->" << ag.epsilonMin
                      << " decay=" << ag.epsilonDecay << "]";
        else
            std::cout << "  temp=[" << ag.tempStart << "->" << ag.tempMin
                      << " decay=" << ag.tempDecay << "]";
        std::cout << "  alpha=[" << ag.alphaStart << "->" << ag.alphaMin
                  << " steps=" << ag.alphaDecaySteps << "]\n";
    }
    const auto& firstTraining = j["agents"].at(0).at("training");
    std::string stopMode = firstTraining.value("stop_mode", std::string("rounds"));
    g_training_stop_mode = (stopMode == "diff") ? TrainingStopMode::TABLE_DIFF
                                                : TrainingStopMode::FIXED_ROUNDS;
    g_sample_rounds = firstTraining.value("sample_rounds", 100'000'000ULL);
    g_diff_threshold = firstTraining.value("diff_threshold", 0.001);
    std::cout << "Stop mode: " << stopModeToString(g_training_stop_mode);
    if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF) {
        std::cout << "  sample_rounds=" << g_sample_rounds
                  << "  diff_threshold=" << g_diff_threshold;
    }
    std::cout << "\n";
    std::cout << "=== (CLI game/agent flags are ignored) ===\n\n";

    return c;
}

// ---------------------------------------------------------------------------
// Build a Player from an AgentConfig.
// warnOnMissing: print a warning if ag.loadFrom file is not found.
// ---------------------------------------------------------------------------
Player* buildPlayer(const AgentConfig& ag, int deckSize, bool warnOnMissing = false) {
    std::unique_ptr<QLearningStrategy> qStrat;

    if (!ag.loadFrom.empty()) {
        qStrat = QLearningStrategy::loadFromFile(ag.loadFrom);
        if (!qStrat && warnOnMissing) {
            std::cerr << "Warning: failed to load checkpoint '" << ag.loadFrom
                      << "', starting fresh.\n";
        }
    }

    if (!qStrat) {
        auto alpha = std::make_unique<LinearDecayingParameter>(
            ag.alphaStart, ag.alphaMin, static_cast<int>(ag.alphaDecaySteps));
        std::unique_ptr<DecayingParameter> exploration;
        if (ag.explorationMode == ExplorationMode::EPSILON_GREEDY) {
            exploration = std::make_unique<EpsilonDecayingParameter>(
                ag.epsilonStart, ag.epsilonMin, ag.epsilonDecay);
        } else {
            exploration = std::make_unique<EpsilonDecayingParameter>(
                ag.tempStart, ag.tempMin, ag.tempDecay);
        }
        qStrat = std::make_unique<QLearningStrategy>(
            std::move(alpha), std::move(exploration), 1.0, ag.explorationMode);
    }

    auto* p = new Player(0.0, std::move(qStrat));
    p->setCountWeights(HI_LO_WEIGHTS);
    p->setCountResolution(ag.countResolution);
    p->setCountRange(ag.minCount, ag.maxCount);
    p->setNumDecks(deckSize);
    return p;
}

void beginLiveProgressFrame(const std::string& header) {
    if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF && !g_full_verbose) {
        if (::isatty(STDOUT_FILENO)) {
            int rc = std::system("clear");
            if (rc != 0) {
                std::cout << "\x1b[2J\x1b[H";
            }
        } else {
            std::cout << "\x1b[2J\x1b[H";
        }
        std::cout << header << std::flush;
    }
}

void printAgentProgress(const AgentConfig& ag,
                        const QLearningStrategy& qStrat,
                        uint64_t completed,
                        double handsPerSec,
                        double avgDiff,
                        bool showDiff) {
    std::cout << "\n--- Agent: " << ag.name
              << "  checkpoint at " << completed << " rounds"
              << "  (count [" << ag.minCount << "," << ag.maxCount
              << "]  res=" << ag.countResolution << ")"
              << "  speed: " << std::fixed << std::setprecision(0)
              << handsPerSec << " hands/sec";
    if (showDiff) {
        std::cout << "  avg|ΔQ|=" << std::setprecision(6) << avgDiff
                  << "  threshold=" << g_diff_threshold;
    }
    std::cout << " ---\n";
    std::cout << qStrat << "\n";
}

bool shouldSaveCheckpoint(uint64_t completed, uint64_t lastSavedAt) {
    if (g_no_save) return false;
    if (lastSavedAt == 0) return true;
    if (g_checkpoint_interval == 0) return true;
    return (completed - lastSavedAt) >= g_checkpoint_interval;
}

void saveCheckpointBatch(const Case& c,
                         const std::string& ckptFolder,
                         const std::vector<Player*>& players,
                         uint64_t completed) {
    for (size_t i = 0; i < players.size(); ++i) {
        const auto& ag = g_agents[i];
        const auto* qStrat = dynamic_cast<const QLearningStrategy*>(players[i]->getStrategy());
        if (!qStrat) continue;

        std::string agentFile  = std::string(kQlearningCheckpointRoot) + "/" + ckptFolder + "/" + ag.name + "_agent.json";
        std::string stratFile  = std::string(kQlearningCheckpointRoot) + "/" + ckptFolder + "/" + ag.name + "_strategy.json";

        qStrat->saveToFile(agentFile, completed);

        auto policy = qStrat->toBasicStrategy();
        policy->saveToJson(stratFile);
    }

    json meta = buildMetaJson(c, g_agents, completed, g_num_threads, g_checkpoint_interval);
    saveMeta(ckptFolder, meta);
    std::cout << "Checkpoint saved to: " << kQlearningCheckpointRoot << "/" << ckptFolder << "/\n";
}

// ---------------------------------------------------------------------------
// Load agent configs from JSON file (--agent-config)
// ---------------------------------------------------------------------------
std::vector<AgentConfig> loadAgentConfigs(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open agent config: " + path);
    json j;
    f >> j;

    std::vector<AgentConfig> configs;
    for (const auto& item : j) {
        AgentConfig ag;
        if (item.contains("name"))             ag.name            = item["name"];
        if (item.contains("exploration_mode")) {
            ag.explorationMode = (item["exploration_mode"] == "boltzmann")
                                     ? ExplorationMode::BOLTZMANN
                                     : ExplorationMode::EPSILON_GREEDY;
        }
        if (item.contains("epsilon_start"))     ag.epsilonStart    = item["epsilon_start"];
        if (item.contains("epsilon_min"))       ag.epsilonMin      = item["epsilon_min"];
        if (item.contains("epsilon_decay"))     ag.epsilonDecay    = item["epsilon_decay"];
        if (item.contains("temp_start"))        ag.tempStart       = item["temp_start"];
        if (item.contains("temp_min"))          ag.tempMin         = item["temp_min"];
        if (item.contains("temp_decay"))        ag.tempDecay       = item["temp_decay"];
        if (item.contains("alpha_start"))       ag.alphaStart      = item["alpha_start"];
        if (item.contains("alpha_min"))         ag.alphaMin        = item["alpha_min"];
        if (item.contains("alpha_decay_steps")) ag.alphaDecaySteps = item["alpha_decay_steps"];
        if (item.contains("count_resolution"))  ag.countResolution = item["count_resolution"];
        if (item.contains("min_count"))         ag.minCount        = item["min_count"];
        if (item.contains("max_count"))         ag.maxCount        = item["max_count"];
        configs.push_back(ag);
    }
    return configs;
}

// ---------------------------------------------------------------------------
// Save meta.json to folder
// ---------------------------------------------------------------------------
static void saveMeta(const std::string& ckptFolder, const json& meta) {
    namespace fs = std::filesystem;
    fs::path full = fs::path(PROJECT_ROOT) / kQlearningCheckpointRoot / ckptFolder / "meta.json";
    std::ofstream f(full);
    if (f.is_open()) f << meta.dump(2);
}

// ---------------------------------------------------------------------------
// Run one game configuration with all agent configs
// ---------------------------------------------------------------------------
void runCase(const Case& c) {
    // Determine checkpoint folder — fixed for the entire session
    std::string ckptFolder;
    if (!g_load_checkpoint.empty()) {
        ckptFolder = g_load_checkpoint;
    } else {
        ckptFolder = g_checkpoint_name.empty() ? currentTimestamp() : g_checkpoint_name;
    }

    const std::string runHeader = buildRunHeader(c, ckptFolder);
    std::cout << runHeader;

    // Create the checkpoint folder up front
    if (!g_no_save) {
        namespace fs = std::filesystem;
        fs::create_directories(fs::path(PROJECT_ROOT) / kQlearningCheckpointRoot / ckptFolder);
    }

    BlackjackRules rules(c.blackJackPay, c.standSoft17, c.deckSize, g_penetration, c.peek,
                         c.splitAfterSplit, c.doubleAfterSplit, c.reSplitAces, c.hitSplitAces,
                         c.surrender, c.doubleOn);

    // Build one player per agent — load from folder when resuming
    bool resuming = !g_load_checkpoint.empty();
    std::vector<Player*> players;
    for (auto ag : g_agents) {
        if (resuming) {
            ag.loadFrom = std::string(kQlearningCheckpointRoot) + "/" + ckptFolder + "/" + ag.name + "_agent.json";
        }
        players.push_back(buildPlayer(ag, c.deckSize, /*warnOnMissing=*/resuming));
    }

    // Determine starting round count from existing checkpoint files
    uint64_t completedOffset = 0;
    if (resuming) {
        namespace fs = std::filesystem;
        for (const auto& ag : g_agents) {
            fs::path full = fs::path(PROJECT_ROOT) / kQlearningCheckpointRoot / ckptFolder
                            / (ag.name + "_agent.json");
            if (fs::exists(full)) {
                try {
                    std::ifstream f(full);
                    json j; f >> j;
                    uint64_t prev = j.value("rounds_completed", uint64_t(0));
                    completedOffset = std::max(completedOffset, prev);
                } catch (...) {}
            }
        }
    }

    uint64_t completed = completedOffset;
    uint64_t trainedThisRun = 0;
    uint64_t lastSavedAt = completedOffset;
    uint64_t batchSize = 0;
    if (g_training_stop_mode == TrainingStopMode::FIXED_ROUNDS) {
        batchSize = (g_checkpoint_interval > 0) ? g_checkpoint_interval : g_num_rounds;
    } else {
        batchSize = g_sample_rounds;
    }
    uint64_t remaining = (g_training_stop_mode == TrainingStopMode::FIXED_ROUNDS) ? g_num_rounds : 0;
    std::vector<QLearningStrategy::QTableSnapshot> previousTables(players.size());
    std::vector<double> lastDiffs(players.size(), std::numeric_limits<double>::infinity());

    auto wallStart = std::chrono::high_resolution_clock::now();

    while (g_training_stop_mode == TrainingStopMode::TABLE_DIFF || remaining > 0) {
        uint64_t batch = batchSize;
        if (g_training_stop_mode == TrainingStopMode::FIXED_ROUNDS)
            batch = std::min(batchSize, remaining);
        if (batch == 0)
            break;

        auto batchStart = std::chrono::high_resolution_clock::now();
        std::vector<Player*> results =
            runParallelSimulation(rules, players, batch, g_num_threads);
        auto batchEnd = std::chrono::high_resolution_clock::now();
        double batchSecs = std::chrono::duration_cast<std::chrono::milliseconds>(
            batchEnd - batchStart).count() / 1000.0;
        double handsPerSec = (batchSecs > 0) ? batch / batchSecs : 0.0;

        completed += batch;
        trainedThisRun += batch;
        if (g_training_stop_mode == TrainingStopMode::FIXED_ROUNDS)
            remaining -= batch;

        // Replace each player with the averaged result from this batch
        for (size_t i = 0; i < players.size(); ++i) {
            delete players[i];
            players[i] = results[i];
        }

        bool allDiffsBelowThreshold = true;
        beginLiveProgressFrame(runHeader);

        // Per-agent: print strategy table + save checkpoint
        for (size_t i = 0; i < players.size(); ++i) {
            const auto& ag = g_agents[i];
            const auto* qStrat = dynamic_cast<const QLearningStrategy*>(
                players[i]->getStrategy());
            if (!qStrat) continue;

            double avgDiff = std::numeric_limits<double>::infinity();
            if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF) {
                auto currentSnapshot = qStrat->snapshotQTable();
                avgDiff = QLearningStrategy::averageAbsDifference(currentSnapshot, previousTables[i]);
                previousTables[i] = std::move(currentSnapshot);
                lastDiffs[i] = avgDiff;
                if (!(avgDiff <= g_diff_threshold))
                    allDiffsBelowThreshold = false;
            }
            printAgentProgress(ag, *qStrat, completed, handsPerSec, avgDiff,
                               g_training_stop_mode == TrainingStopMode::TABLE_DIFF);
        }

        if (shouldSaveCheckpoint(completed, lastSavedAt)) {
            saveCheckpointBatch(c, ckptFolder, players, completed);
            lastSavedAt = completed;
        }

        if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF && allDiffsBelowThreshold) {
            if (!g_no_save && lastSavedAt != completed) {
                saveCheckpointBatch(c, ckptFolder, players, completed);
                lastSavedAt = completed;
            }
            std::cout << "\nConverged: average absolute Q-table change is below threshold for all agents.\n";
            std::cout << std::flush;
            break;
        }
        std::cout << std::flush;
    }

    if (!g_no_save && lastSavedAt != completed) {
        saveCheckpointBatch(c, ckptFolder, players, completed);
    }

    auto wallEnd = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        wallEnd - wallStart).count() / 1000.0;

    std::cout << "\nTotal time: " << std::fixed << std::setprecision(2) << elapsed << "s  ("
              << std::setprecision(0) << ((elapsed > 0.0 ? trainedThisRun / elapsed : 0.0)) << " hands/sec)\n";

    for (size_t i = 0; i < players.size(); ++i) {
        const auto& ag = g_agents[i];
        double edge = 0.0;
        if (trainedThisRun > 0) {
            edge = players[i]->getMoney() /
                   (static_cast<double>(trainedThisRun) / g_num_threads);
        }
        std::cout << "Agent: " << ag.name
                  << "  edge: " << std::fixed << std::setprecision(6)
                  << edge << "  (" << std::setprecision(4) << (edge * 100.0) << "%)\n";
        if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF) {
            std::cout << "         final avg|ΔQ|: " << std::setprecision(6)
                      << lastDiffs[i] << "\n";
        }
    }

    for (auto* p : players) delete p;
}

// ---------------------------------------------------------------------------
// Help
// ---------------------------------------------------------------------------
void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Find count-dependent deviations using Q-learning + Hi-Lo count.\n\n";

    std::cout << "CHECKPOINT FOLDER STRUCTURE:\n";
    std::cout << "  Each training session saves to one folder under checkpoints/checkpoints_QLearning/:\n";
    std::cout << "    checkpoints/checkpoints_QLearning/<folder>/\n";
    std::cout << "      meta.json               Game config + counting + training params for all agents\n";
    std::cout << "      <agentname>_agent.json  Full Q-table and per-state alpha/epsilon state\n";
    std::cout << "      <agentname>_strategy.json  Argmax policy table (ready to play)\n";
    std::cout << "  The folder name is a timestamp by default, or set via --checkpoint-name.\n";
    std::cout << "  All checkpoints within a run go to the same folder.\n\n";

    std::cout << "SIMULATION:\n";
    std::cout << "  --num-rounds <N>              Training rounds (default: 1000000000)\n";
    std::cout << "  --stop-mode <rounds|diff>     Stop after fixed rounds or by table-diff convergence (default: rounds)\n";
    std::cout << "  --sample-rounds <N>           In diff mode, compare Q-tables every N rounds (default: 100000000)\n";
    std::cout << "  --diff-threshold <val>        Stop diff mode when avg abs table change is <= threshold (default: 0.001)\n";
    std::cout << "  --num-threads <N>             Threads (default: 16)\n";
    std::cout << "  --penetration <val>           Shoe penetration % (default: 75.0)\n";
    std::cout << "  --checkpoint-interval <N>     Save every N rounds (0=end only, default: 0)\n";
    std::cout << "  --checkpoint-name <name>      Name the checkpoint folder (default: timestamp)\n";
    std::cout << "  --full-verbose                Keep every sampled table in the console instead of live-refreshing it\n";
    std::cout << "  --no-save                     Disable all checkpoint saving\n\n";

    std::cout << "CHECKPOINT RESUME:\n";
    std::cout << "  --load-checkpoint <folder>    Resume from checkpoints/checkpoints_QLearning/<folder>/\n";
    std::cout << "                                Reads meta.json for game config (overrides all game\n";
    std::cout << "                                flags), loads each agent's Q-table and training state.\n";
    std::cout << "                                Subsequent saves go back to the same folder.\n\n";

    std::cout << "AGENT CONFIGS:\n";
    std::cout << "  --agent-config <file.json>    Load multiple agent configs from a JSON array file.\n";
    std::cout << "                                Each agent gets its own _agent.json and _strategy.json.\n";
    std::cout << "                                Ignored when --load-checkpoint is used.\n\n";
    std::cout << "  Agent config JSON format:\n";
    std::cout << "    [{\"name\":\"baseline\",\"epsilon_start\":1.0,\"min_count\":-5,\"max_count\":5,...},\n";
    std::cout << "     {\"name\":\"aggressive\",\"epsilon_decay\":0.9999,...}]\n\n";

    std::cout << "DEFAULT AGENT (when no --agent-config or --load-checkpoint):\n";
    std::cout << "  --count-resolution <val>      True-count discretization step (default: 1.0)\n";
    std::cout << "  --min-count <N>               Clamp true count to this minimum (default: -5)\n";
    std::cout << "  --max-count <N>               Clamp true count to this maximum (default: 5)\n";
    std::cout << "  --exploration <mode>          epsilon or boltzmann (default: epsilon)\n";
    std::cout << "  --epsilon-start <val>         Initial epsilon (default: 1.0)\n";
    std::cout << "  --epsilon-min <val>           Minimum epsilon (default: 0.1)\n";
    std::cout << "  --epsilon-decay <val>         Per-step epsilon decay factor (default: 0.99999)\n";
    std::cout << "  --temp-start/min/decay        Boltzmann temperature params (defaults: 1.0/0.1/0.99999)\n";
    std::cout << "  --alpha-start <val>           Initial learning rate (default: 0.01)\n";
    std::cout << "  --alpha-min <val>             Minimum learning rate (default: 0.0001)\n";
    std::cout << "  --alpha-decay <N>             Steps to decay alpha to min (default: 100)\n\n";

    std::cout << "GAME CONFIG (ignored when --load-checkpoint is used):\n";
    std::cout << "  --decks <N>                   Number of decks (default: 6)\n";
    std::cout << "  --ss17 <bool>                 Stand on soft 17 (default: true)\n";
    std::cout << "  --das <bool>                  Double after split (default: true)\n";
    std::cout << "  --sas <N>                     Max splits allowed (default: 4)\n";
    std::cout << "  --don <ANY|9-11|10-11>        Double down restriction (default: ANY)\n";
    std::cout << "  --rsa <bool>                  Resplit aces (default: false)\n";
    std::cout << "  --hsa <bool>                  Hit split aces (default: false)\n";
    std::cout << "  --peek <bool>                 Dealer peeks for blackjack (default: false)\n";
    std::cout << "  --surr <no|yes|2-10>          Surrender rule (default: 2-10)\n";
    std::cout << "  --bj <val>                    Blackjack payout multiplier (default: 1.5)\n";
    std::cout << "  All game flags accept comma-separated lists to run multiple configurations.\n\n";

    std::cout << "EXAMPLES:\n";
    std::cout << "  # Fresh run with a named folder:\n";
    std::cout << "  " << prog << " --num-rounds 1000000000 --checkpoint-name Ambassador\n\n";
    std::cout << "  # Resume from that folder and train 500M more rounds:\n";
    std::cout << "  " << prog << " --num-rounds 500000000 --load-checkpoint Ambassador\n\n";
    std::cout << "  # Fresh run with intermediate checkpoints every 100M rounds:\n";
    std::cout << "  " << prog << " --num-rounds 1000000000 --checkpoint-interval 100000000 \\\n";
    std::cout << "      --checkpoint-name MGMGrand\n\n";
    std::cout << "  # Multi-agent run (compare two exploration strategies):\n";
    std::cout << "  " << prog << " --num-rounds 1000000000 --checkpoint-name MyRun \\\n";
    std::cout << "      --agent-config agents.json\n\n";
    std::cout << "  # Resume multi-agent run (all agents loaded from folder):\n";
    std::cout << "  " << prog << " --num-rounds 500000000 --load-checkpoint MyRun\n\n";
    std::cout << "  # Inspect a checkpoint without saving:\n";
    std::cout << "  " << prog << " --num-rounds 50000000 --no-save --load-checkpoint Ambassador\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    bool agentConfigLoaded = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printHelp(argv[0]); return 0; }

        // Simulation
        else if (arg == "--num-rounds"          && i+1<argc) g_num_rounds           = std::stoull(argv[++i]);
        else if (arg == "--stop-mode"           && i+1<argc) {
            std::string mode = argv[++i];
            g_training_stop_mode = (mode == "diff") ? TrainingStopMode::TABLE_DIFF
                                                    : TrainingStopMode::FIXED_ROUNDS;
        }
        else if (arg == "--sample-rounds"       && i+1<argc) g_sample_rounds        = std::stoull(argv[++i]);
        else if (arg == "--diff-threshold"      && i+1<argc) g_diff_threshold       = std::stod(argv[++i]);
        else if (arg == "--num-threads"         && i+1<argc) g_num_threads          = std::stoi(argv[++i]);
        else if (arg == "--penetration"         && i+1<argc) g_penetration          = std::stod(argv[++i]);
        else if (arg == "--checkpoint-interval" && i+1<argc) g_checkpoint_interval  = std::stoull(argv[++i]);
        else if (arg == "--checkpoint-name"     && i+1<argc) g_checkpoint_name      = argv[++i];
        else if (arg == "--full-verbose")                    g_full_verbose         = true;
        else if (arg == "--no-save")                         g_no_save              = true;

        // Resume from checkpoint folder
        else if (arg == "--load-checkpoint" && i+1<argc) g_load_checkpoint = argv[++i];

        // Agent config file (overrides default single agent)
        else if (arg == "--agent-config" && i+1<argc) {
            g_agents = loadAgentConfigs(argv[++i]);
            agentConfigLoaded = true;
        }

        // Default single-agent params (ignored if --agent-config or --load-checkpoint used)
        else if (arg == "--count-resolution" && i+1<argc && !agentConfigLoaded)
            g_agents[0].countResolution = std::stod(argv[++i]);
        else if (arg == "--min-count"        && i+1<argc && !agentConfigLoaded)
            g_agents[0].minCount = std::stoi(argv[++i]);
        else if (arg == "--max-count"        && i+1<argc && !agentConfigLoaded)
            g_agents[0].maxCount = std::stoi(argv[++i]);
        else if (arg == "--exploration"      && i+1<argc && !agentConfigLoaded) {
            std::string m = argv[++i];
            if (m == "boltzmann") g_agents[0].explorationMode = ExplorationMode::BOLTZMANN;
            else                  g_agents[0].explorationMode = ExplorationMode::EPSILON_GREEDY;
        }
        else if (arg == "--epsilon-start"    && i+1<argc && !agentConfigLoaded)
            g_agents[0].epsilonStart     = std::stod(argv[++i]);
        else if (arg == "--epsilon-min"      && i+1<argc && !agentConfigLoaded)
            g_agents[0].epsilonMin       = std::stod(argv[++i]);
        else if (arg == "--epsilon-decay"    && i+1<argc && !agentConfigLoaded)
            g_agents[0].epsilonDecay     = std::stod(argv[++i]);
        else if (arg == "--temp-start"       && i+1<argc && !agentConfigLoaded)
            g_agents[0].tempStart        = std::stod(argv[++i]);
        else if (arg == "--temp-min"         && i+1<argc && !agentConfigLoaded)
            g_agents[0].tempMin          = std::stod(argv[++i]);
        else if (arg == "--temp-decay"       && i+1<argc && !agentConfigLoaded)
            g_agents[0].tempDecay        = std::stod(argv[++i]);
        else if (arg == "--alpha-start"      && i+1<argc && !agentConfigLoaded)
            g_agents[0].alphaStart       = std::stod(argv[++i]);
        else if (arg == "--alpha-min"        && i+1<argc && !agentConfigLoaded)
            g_agents[0].alphaMin         = std::stod(argv[++i]);
        else if (arg == "--alpha-decay"      && i+1<argc && !agentConfigLoaded)
            g_agents[0].alphaDecaySteps  = std::stod(argv[++i]);

        // Game config (only used for fresh runs — ignored if --load-checkpoint)
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

    for (const auto& ag : g_agents)
        if (ag.minCount > ag.maxCount) {
            std::cerr << "Error: min-count > max-count for agent '" << ag.name << "'\n";
            return 1;
        }
    if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF && g_sample_rounds == 0) {
        std::cerr << "Error: --sample-rounds must be >= 1 in diff mode\n";
        return 1;
    }

    // Build case list — load from folder if resuming, otherwise from CLI game config
    std::vector<Case> cases;
    if (!g_load_checkpoint.empty()) {
        try {
            cases.push_back(loadCheckpointFolder(g_load_checkpoint));
        } catch (const std::exception& e) {
            std::cerr << "Error: " << e.what() << "\n";
            return 1;
        }
    } else {
        cases = generateTestCases(
            g_deck_sizes, g_stand_soft17, g_double_after_split, g_split_after_split,
            g_double_on, g_resplit_aces, g_hit_split_aces, g_peek, g_surrender, g_blackjack_pay);
    }

    for (const auto& c : cases)
        runCase(c);

    return 0;
}
