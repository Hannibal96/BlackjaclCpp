#include "RegressionTestUtils.h"
#include "Game/BlackjackTable.h"
#include "Game/Player.h"
#include "Game/BettingStrategy.h"
#include "Game/CountingMethods.h"
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
#include <stdexcept>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <unistd.h>

using json = nlohmann::json;

namespace {

constexpr std::array<double, 13> ZERO_WEIGHTS = {};
constexpr double kEps = 1e-12;
constexpr const char* kAlternatingCheckpointRoot = "checkpoints/alternating-checkpoints";
constexpr double kLearnedCountFactorMultiplier = 180.0;
constexpr double kKellyInitialMoney = 1.0;
constexpr double kKellyFraction = 1.0;
constexpr uint64_t kKellyMeasurementRounds = 1'000'000ULL;
enum class TrainingStopMode { FIXED_ROUNDS, TABLE_DIFF };

struct AgentConfig {
    ExplorationMode explorationMode = ExplorationMode::EPSILON_GREEDY;
    double epsilonStart  = 1.0;
    double epsilonMin    = 0.1;
    double epsilonDecay  = 0.99999;
    double tempStart     = 1.0;
    double tempMin       = 0.1;
    double tempDecay     = 0.99999;
    double alphaStart    = 0.01;
    double alphaMin      = 0.0001;
    double alphaDecaySteps = 100.0;
    double countResolution = 1.0;
    int minCount = -5;
    int maxCount = 5;
};

struct CountConfig {
    CountingSystem system;
    double resolution = 1.0;
    int minCount = -5;
    int maxCount = 5;
};

struct PolicyArtifact {
    std::unique_ptr<QLearningStrategy> qStrategy;
    std::unique_ptr<BasicStrategy> policy;
    double trainingEdge = 0.0;
    double evaluationEdge = 0.0;
    double handsPerSec = 0.0;
    uint64_t trainingRounds = 0;
};

struct CountArtifact {
    CountConfig count;
    std::array<double, 14> rawSolution{};
    std::array<std::array<double, 14>, 14> XtX{};
    std::array<double, 14> Xty{};
    uint64_t recordedRounds = 0;
    double evaluationEdge = 0.0;
    double evaluationKellyGrowth = 1.0;
    double handsPerSec = 0.0;
    double threshold = 0.0;
    double normalizationScale = 1.0;
    double referenceFlatEdge = 0.0;
};

struct EvCountGraphPoint {
    double count = 0.0;
    uint64_t n = 0;
    double meanReward = 0.0;
    double stddevReward = 0.0;
    double confidenceLower = 0.0;
    double confidenceUpper = 0.0;
    double regressionReward = 0.0;
};

struct EvCountGraphArtifact {
    double resolution = 0.25;
    uint64_t rounds = 0;
    double handsPerSec = 0.0;
    double minCount = 0.0;
    double maxCount = 0.0;
    std::vector<EvCountGraphPoint> points;
};

struct NamedEvCountGraphArtifact {
    std::string label;
    EvCountGraphArtifact graph;
};

enum class Phase { POLICY, COUNT, DONE };

struct ResumeState {
    Phase nextPhase = Phase::POLICY;
    uint64_t nextPolicyIndex = 0;
    CountConfig currentCount;
    std::string currentPolicyLabel;
    std::string currentPolicyPath;
    double currentPolicyFlatEdge = 0.0;
};

uint64_t    g_num_rounds          = 1'000'000'000ULL;
uint64_t    g_eval_rounds         = 0;   // 0 => use g_num_rounds
uint64_t    g_iterations          = 5;   // number of loop iterations k producing P_k from W_k, then W_{k+1}
int         g_num_threads         = 16;
double      g_penetration         = 75.0;
uint64_t    g_sample_every        = 1;
uint64_t    g_sample_rounds       = 100'000'000ULL;
double      g_diff_threshold      = 0.001;
TrainingStopMode g_training_stop_mode = TrainingStopMode::FIXED_ROUNDS;
int         g_kelly_measurements  = 100;
bool        g_verbose             = false;
bool        g_full_verbose        = false;
bool        g_no_save             = false;
std::string g_checkpoint_name;
std::string g_load_checkpoint;

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

AgentConfig g_agent;

std::string currentTimestamp() {
    time_t t = time(nullptr);
    char buf[20];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&t));
    return buf;
}

std::string doubleOnToString(DoubleDownOn d) {
    if (d == DoubleDownOn::ANY)             return "ANY";
    if (d == DoubleDownOn::NINE_TEN_ELEVEN) return "9-11";
    return "10-11";
}

std::string surrenderToString(Surrender s) {
    if (s == Surrender::NO_SURRENDER)  return "no";
    if (s == Surrender::SURRENDER_ANY) return "yes";
    return "2-10";
}

const char* stopModeToString(TrainingStopMode mode) {
    return (mode == TrainingStopMode::TABLE_DIFF) ? "diff" : "rounds";
}

uint64_t countPhasePlayedRounds();

std::string buildRunHeader(const Case& c, const std::string& folder, uint64_t evalRounds) {
    std::ostringstream os;
    os << "\n=== AlternatingOptimization ===\n";
    os << "Scenario:      " << ToString(c) << "\n";
    if (g_training_stop_mode == TrainingStopMode::FIXED_ROUNDS) {
        os << "Train rounds:  " << g_num_rounds << "  Threads: " << g_num_threads << "\n";
    } else {
        os << "Policy stop:   diff  sample=" << g_sample_rounds
           << "  threshold=" << g_diff_threshold
           << "  Threads: " << g_num_threads << "\n";
    }
    os << "Count rounds:  " << countPhasePlayedRounds()
       << "  (record every " << g_sample_every << " => target "
       << g_num_rounds << " recorded samples)\n";
    os << "Eval rounds:   " << evalRounds << "\n";
    os << "Iterations:    " << g_iterations << "  (each iteration learns P_k then W_{k+1})\n";
    os << "Penetration:   " << g_penetration << "%\n";
    os << "Sample-every:  " << g_sample_every << "\n";
    os << "Folder:        " << kAlternatingCheckpointRoot << "/" << folder << "/\n";
    os << "Initial count: none (all zeros)\n";
    os << "Count factor x:" << kLearnedCountFactorMultiplier << "\n";
    os << "===============================\n";
    return os.str();
}

json buildMetaJson(const Case& c) {
    json meta;
    meta["algorithm"] = "alternating_optimization";
    meta["artifacts"]["naming"] = "W0 (implicit zeros), P0, W1, P1, W2, P2, ...";

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

    meta["algorithm_config"]["num_rounds_per_phase"] = g_num_rounds;
    meta["algorithm_config"]["eval_rounds"]          = (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds);
    meta["algorithm_config"]["iterations"]           = g_iterations;
    meta["algorithm_config"]["sample_every"]         = g_sample_every;
    meta["algorithm_config"]["num_threads"]          = g_num_threads;
    meta["algorithm_config"]["policy_stop_mode"]     = stopModeToString(g_training_stop_mode);
    meta["algorithm_config"]["policy_sample_rounds"] = g_sample_rounds;
    meta["algorithm_config"]["policy_diff_threshold"] = g_diff_threshold;
    meta["algorithm_config"]["initial_count"]["weights"] = ZERO_WEIGHTS;
    meta["algorithm_config"]["initial_count"]["factor"]  = 1.0;
    meta["algorithm_config"]["initial_count"]["bias"]    = 0.0;
    meta["algorithm_config"]["learned_count_factor_multiplier"] = kLearnedCountFactorMultiplier;
    meta["algorithm_config"]["initial_count"]["resolution"] = g_agent.countResolution;
    meta["algorithm_config"]["initial_count"]["min_count"]  = g_agent.minCount;
    meta["algorithm_config"]["initial_count"]["max_count"]  = g_agent.maxCount;

    meta["rl"]["exploration_mode"] =
        (g_agent.explorationMode == ExplorationMode::EPSILON_GREEDY ? "epsilon" : "boltzmann");
    meta["rl"]["epsilon_start"] = g_agent.epsilonStart;
    meta["rl"]["epsilon_min"]   = g_agent.epsilonMin;
    meta["rl"]["epsilon_decay"] = g_agent.epsilonDecay;
    meta["rl"]["temp_start"]    = g_agent.tempStart;
    meta["rl"]["temp_min"]      = g_agent.tempMin;
    meta["rl"]["temp_decay"]    = g_agent.tempDecay;
    meta["rl"]["alpha_start"]   = g_agent.alphaStart;
    meta["rl"]["alpha_min"]     = g_agent.alphaMin;
    meta["rl"]["alpha_decay_steps"] = g_agent.alphaDecaySteps;
    meta["rl"]["count_resolution"]  = g_agent.countResolution;
    meta["rl"]["min_count"]         = g_agent.minCount;
    meta["rl"]["max_count"]         = g_agent.maxCount;

    return meta;
}

BlackjackRules buildRules(const Case& c, double minBet = 1.0, double maxBet = 1.0) {
    BlackjackRules rules(c.blackJackPay, c.standSoft17, c.deckSize, g_penetration,
                         c.peek, c.splitAfterSplit, c.doubleAfterSplit,
                         c.reSplitAces, c.hitSplitAces, c.surrender, c.doubleOn);
    rules.minBet = minBet;
    rules.maxBet = maxBet;
    return rules;
}

std::unique_ptr<QLearningStrategy> makeQStrategy() {
    auto alpha = std::make_unique<LinearDecayingParameter>(
        g_agent.alphaStart, g_agent.alphaMin, static_cast<int>(g_agent.alphaDecaySteps));

    std::unique_ptr<DecayingParameter> exploration;
    if (g_agent.explorationMode == ExplorationMode::EPSILON_GREEDY) {
        exploration = std::make_unique<EpsilonDecayingParameter>(
            g_agent.epsilonStart, g_agent.epsilonMin, g_agent.epsilonDecay);
    } else {
        exploration = std::make_unique<EpsilonDecayingParameter>(
            g_agent.tempStart, g_agent.tempMin, g_agent.tempDecay);
    }

    return std::make_unique<QLearningStrategy>(
        std::move(alpha), std::move(exploration), 1.0, g_agent.explorationMode);
}

double averageRoundsPerThread(uint64_t rounds, int threads) {
    return static_cast<double>(rounds) / static_cast<double>(threads);
}

double netPerRoundFromPlayer(const Player& player, uint64_t rounds, int threads) {
    return player.getMoney() / averageRoundsPerThread(rounds, threads);
}

double roundTo(double value, int digits) {
    double scale = std::pow(10.0, digits);
    return std::round(value * scale) / scale;
}

uint64_t countPhasePlayedRounds() {
    if (g_sample_every == 0)
        throw std::invalid_argument("--sample-every must be >= 1");
    if (g_num_rounds > std::numeric_limits<uint64_t>::max() / g_sample_every)
        throw std::overflow_error("count phase rounds overflow: num-rounds * sample-every is too large");
    return g_num_rounds * g_sample_every;
}

double evaluateEdge(const Case& c,
                    const BasicStrategy& strategy,
                    const CountConfig& count,
                    std::unique_ptr<BettingStrategy> betting,
                    double minBet,
                    double maxBet,
                    uint64_t rounds) {
    auto cloned = strategy.clone();
    auto* player = new Player(0.0, std::move(cloned));
    player->setNumDecks(c.deckSize);
    player->setCountSystem(count.system);
    player->setCountResolution(count.resolution);
    auto [strategyMinCount, strategyMaxCount] = strategy.getCountRange();
    player->setCountRange(strategyMinCount, strategyMaxCount);
    if (betting) player->setBettingStrategy(std::move(betting));

    BlackjackRules rules = buildRules(c, minBet, maxBet);
    std::vector<Player*> results = runParallelSimulation(rules, {player}, rounds, g_num_threads);

    delete player;
    if (results.empty())
        throw std::runtime_error("Edge evaluation failed: no result players");

    Player* result = results[0];
    double edge = netPerRoundFromPlayer(*result, rounds, g_num_threads);
    delete result;
    return edge;
}

double evaluateKellyGrowth(const Case& c,
                           const BasicStrategy& strategy,
                           const CountConfig& count,
                           uint64_t rounds,
                           double initialMoney = kKellyInitialMoney,
                           double kellyFraction = kKellyFraction) {
    (void)rounds;
    BlackjackRules rules = buildRules(c, 0.0, std::numeric_limits<double>::max());
    double growthSum = 0.0;

    for (int rep = 0; rep < g_kelly_measurements; ++rep) {
        auto cloned = strategy.clone();
        auto* player = new Player(initialMoney, std::move(cloned));
        player->setNumDecks(c.deckSize);
        player->setCountSystem(count.system);
        player->setCountResolution(count.resolution);
        auto [strategyMinCount, strategyMaxCount] = strategy.getCountRange();
        player->setCountRange(strategyMinCount, strategyMaxCount);
        player->setBettingStrategy(std::make_unique<KellyBetting>(kellyFraction));

        std::vector<Player*> results =
            runParallelSimulation(rules, {player}, kKellyMeasurementRounds, g_num_threads);
        delete player;
        if (results.empty())
            throw std::runtime_error("Kelly evaluation failed: no result players");

        Player* result = results[0];
        double avgLogFinal = result->getLogMoney();
        double growthRate = 0.0;
        double roundsPerThread = averageRoundsPerThread(kKellyMeasurementRounds, g_num_threads);
        if (initialMoney > 0.0 && std::isfinite(avgLogFinal) && roundsPerThread > 0.0) {
            growthRate = std::exp((avgLogFinal - std::log(initialMoney)) / roundsPerThread);
        }
        growthSum += growthRate;
        delete result;
    }
    return growthSum / static_cast<double>(g_kelly_measurements);
}

std::array<double, 14> solveOLS(std::array<std::array<double, 14>, 14> A,
                                std::array<double, 14> b) {
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

CountArtifact normalizeCount(const std::array<double, 14>& raw) {
    CountArtifact artifact;
    artifact.rawSolution = raw;

    for (int i = 0; i < 13; ++i)
        artifact.count.system.weights[i] = raw[i] * kLearnedCountFactorMultiplier;
    artifact.count.system.factor = 1.0 / kLearnedCountFactorMultiplier;
    artifact.count.system.bias   = raw[13];
    artifact.count.resolution    = g_agent.countResolution;
    artifact.count.minCount      = g_agent.minCount;
    artifact.count.maxCount      = g_agent.maxCount;
    artifact.normalizationScale  = kLearnedCountFactorMultiplier;

    return artifact;
}

double spreadThresholdFromCount(const CountConfig& count) {
    if (std::abs(count.system.factor) < kEps)
        return std::numeric_limits<double>::infinity();
    return -count.system.bias / count.system.factor;
}

json countConfigToJson(const CountConfig& count, bool roundWeights = true) {
    json j;
    json weights = json::array();
    for (double w : count.system.weights)
        weights.push_back(roundWeights ? roundTo(w, 4) : w);
    j["weights"] = weights;
    j["factor"] = count.system.factor;
    j["bias"] = count.system.bias;
    j["resolution"] = count.resolution;
    j["min_count"] = count.minCount;
    j["max_count"] = count.maxCount;
    return j;
}

json evCountGraphToJson(const EvCountGraphArtifact& graph) {
    json j;
    j["resolution"] = graph.resolution;
    j["rounds"] = graph.rounds;
    j["hands_per_sec"] = graph.handsPerSec;
    j["min_count"] = graph.minCount;
    j["max_count"] = graph.maxCount;
    j["points"] = json::array();
    for (const auto& p : graph.points) {
        json row;
        row["count"] = p.count;
        row["n"] = p.n;
        if (p.n > 0) row["mean_reward"] = p.meanReward;
        else         row["mean_reward"] = nullptr;
        row["stddev_reward"] = p.stddevReward;
        if (p.n > 0) {
            row["confidence_lower"] = p.confidenceLower;
            row["confidence_upper"] = p.confidenceUpper;
        } else {
            row["confidence_lower"] = nullptr;
            row["confidence_upper"] = nullptr;
        }
        row["regression_reward"] = p.regressionReward;
        j["points"].push_back(row);
    }
    return j;
}

EvCountGraphArtifact evCountGraphFromJson(const json& j) {
    EvCountGraphArtifact graph;
    graph.resolution = j.value("resolution", 0.25);
    graph.rounds = j.value("rounds", 0ULL);
    graph.handsPerSec = j.value("hands_per_sec", 0.0);
    graph.minCount = j.value("min_count", 0.0);
    graph.maxCount = j.value("max_count", 0.0);
    for (const auto& row : j.at("points")) {
        EvCountGraphPoint point;
        point.count = row.at("count").get<double>();
        point.n = row.value("n", 0ULL);
        if (!row.at("mean_reward").is_null())
            point.meanReward = row.at("mean_reward").get<double>();
        point.stddevReward = row.value("stddev_reward", 0.0);
        if (row.contains("confidence_lower") && !row.at("confidence_lower").is_null())
            point.confidenceLower = row.at("confidence_lower").get<double>();
        else
            point.confidenceLower = point.meanReward;
        if (row.contains("confidence_upper") && !row.at("confidence_upper").is_null())
            point.confidenceUpper = row.at("confidence_upper").get<double>();
        else
            point.confidenceUpper = point.meanReward;
        point.regressionReward = row.value("regression_reward", 0.0);
        graph.points.push_back(point);
    }
    return graph;
}

std::string evCountGraphsToJson(const std::vector<NamedEvCountGraphArtifact>& graphs) {
    json j;
    j["graphs"] = json::array();
    for (const auto& named : graphs) {
        json entry;
        entry["label"] = named.label;
        entry["graph"] = evCountGraphToJson(named.graph);
        j["graphs"].push_back(entry);
    }
    return j.dump(2);
}

std::string colorForGraph(size_t index) {
    static const char* kColors[] = {
        "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"
    };
    return kColors[index % (sizeof(kColors) / sizeof(kColors[0]))];
}

std::string evCountGraphToSvg(const std::string& label,
                              const EvCountGraphArtifact& graph) {
    const double width = 1000.0;
    const double height = 600.0;
    const double left = 90.0;
    const double right = 30.0;
    const double top = 40.0;
    const double bottom = 70.0;
    const double plotWidth = width - left - right;
    const double plotHeight = height - top - bottom;

    double minX = graph.minCount;
    double maxX = graph.maxCount;
    if (std::abs(maxX - minX) < kEps) maxX = minX + graph.resolution;

    double minY = std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const auto& p : graph.points) {
        minY = std::min(minY, p.regressionReward);
        maxY = std::max(maxY, p.regressionReward);
        if (p.n > 0) {
            minY = std::min(minY, p.confidenceLower);
            maxY = std::max(maxY, p.confidenceUpper);
        }
    }
    if (!std::isfinite(minY) || !std::isfinite(maxY)) {
        minY = -1.0;
        maxY = 1.0;
    }
    if (std::abs(maxY - minY) < kEps) {
        minY -= 1.0;
        maxY += 1.0;
    } else {
        double padding = 0.1 * (maxY - minY);
        minY -= padding;
        maxY += padding;
    }

    auto mapX = [&](double x) {
        return left + ((x - minX) / (maxX - minX)) * plotWidth;
    };
    auto mapY = [&](double y) {
        return top + (1.0 - ((y - minY) / (maxY - minY))) * plotHeight;
    };

    std::ostringstream empiricalPath;
    bool empiricalStarted = false;
    std::vector<std::pair<double, double>> upperBandPoints;
    std::vector<std::pair<double, double>> lowerBandPoints;
    std::ostringstream regressionPath;
    bool regressionStarted = false;
    for (const auto& p : graph.points) {
        const double x = mapX(p.count);
        const double regressionY = mapY(p.regressionReward);
        regressionPath << (regressionStarted ? " L " : "M ")
                       << x << " " << regressionY;
        regressionStarted = true;

        if (p.n > 0) {
            const double empiricalY = mapY(p.meanReward);
            empiricalPath << (empiricalStarted ? " L " : "M ")
                          << x << " " << empiricalY;
            empiricalStarted = true;
            upperBandPoints.emplace_back(x, mapY(p.confidenceUpper));
            lowerBandPoints.emplace_back(x, mapY(p.confidenceLower));
        }
    }

    std::ostringstream confidenceBand;
    bool confidenceStarted = false;
    if (!upperBandPoints.empty()) {
        confidenceBand << "M " << upperBandPoints.front().first << " " << upperBandPoints.front().second;
        for (size_t i = 1; i < upperBandPoints.size(); ++i) {
            confidenceBand << " L " << upperBandPoints[i].first << " " << upperBandPoints[i].second;
        }
        for (size_t i = lowerBandPoints.size(); i-- > 0;) {
            confidenceBand << " L " << lowerBandPoints[i].first << " " << lowerBandPoints[i].second;
        }
        confidenceBand << " Z";
        confidenceStarted = true;
    }

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    svg << "<defs>\n";
    svg << "  <pattern id=\"minorGrid\" width=\"" << (plotWidth / 20.0) << "\" height=\"" << (plotHeight / 20.0)
        << "\" patternUnits=\"userSpaceOnUse\">\n";
    svg << "    <path d=\"M " << (plotWidth / 20.0) << " 0 L 0 0 0 " << (plotHeight / 20.0)
        << "\" fill=\"none\" stroke=\"#dddddd\" stroke-width=\"1.1\"/>\n";
    svg << "  </pattern>\n";
    svg << "  <pattern id=\"majorGrid\" width=\"" << (plotWidth / 10.0) << "\" height=\"" << (plotHeight / 10.0)
        << "\" patternUnits=\"userSpaceOnUse\">\n";
    svg << "    <rect width=\"" << (plotWidth / 10.0) << "\" height=\"" << (plotHeight / 10.0)
        << "\" fill=\"url(#minorGrid)\"/>\n";
    svg << "    <path d=\"M " << (plotWidth / 10.0) << " 0 L 0 0 0 " << (plotHeight / 10.0)
        << "\" fill=\"none\" stroke=\"#b8b8b8\" stroke-width=\"1.5\"/>\n";
    svg << "  </pattern>\n";
    svg << "</defs>\n";
    svg << "<text x=\"" << left << "\" y=\"24\" font-family=\"sans-serif\" font-size=\"20\">"
        << label << " EV vs Count</text>\n";
    svg << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plotWidth
        << "\" height=\"" << plotHeight << "\" fill=\"url(#majorGrid)\"/>\n";
    svg << "<line x1=\"" << left << "\" y1=\"" << (top + plotHeight) << "\" x2=\"" << (left + plotWidth)
        << "\" y2=\"" << (top + plotHeight) << "\" stroke=\"black\" stroke-width=\"1.2\"/>\n";
    svg << "<line x1=\"" << left << "\" y1=\"" << top << "\" x2=\"" << left
        << "\" y2=\"" << (top + plotHeight) << "\" stroke=\"black\" stroke-width=\"1.2\"/>\n";

    for (int i = 0; i <= 16; ++i) {
        double xValue = minX + (maxX - minX) * (static_cast<double>(i) / 16.0);
        double x = mapX(xValue);
        svg << "<line x1=\"" << x << "\" y1=\"" << (top + plotHeight)
            << "\" x2=\"" << x << "\" y2=\"" << (top + plotHeight + 6)
            << "\" stroke=\"black\"/>\n";
        if (i % 2 == 0) {
            svg << "<text x=\"" << x << "\" y=\"" << (top + plotHeight + 24)
                << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">"
                << std::fixed << std::setprecision(2) << xValue << "</text>\n";
        }
    }
    for (int i = 0; i <= 12; ++i) {
        double yValue = minY + (maxY - minY) * (static_cast<double>(i) / 12.0);
        double y = mapY(yValue);
        svg << "<line x1=\"" << (left - 6) << "\" y1=\"" << y
            << "\" x2=\"" << left << "\" y2=\"" << y << "\" stroke=\"black\"/>\n";
        if (i % 2 == 0) {
            svg << "<text x=\"" << (left - 10) << "\" y=\"" << (y + 4)
                << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">"
                << std::fixed << std::setprecision(4) << yValue << "</text>\n";
        }
    }

    if (minX <= 0.0 && maxX >= 0.0) {
        double zeroX = mapX(0.0);
        svg << "<line x1=\"" << zeroX << "\" y1=\"" << top
            << "\" x2=\"" << zeroX << "\" y2=\"" << (top + plotHeight)
            << "\" stroke=\"#666666\" stroke-width=\"1.8\"/>\n";
        svg << "<text x=\"" << zeroX << "\" y=\"" << (top + plotHeight + 40)
            << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#444444\">0</text>\n";
    }
    if (minY <= 0.0 && maxY >= 0.0) {
        double zeroY = mapY(0.0);
        svg << "<line x1=\"" << left << "\" y1=\"" << zeroY
            << "\" x2=\"" << (left + plotWidth) << "\" y2=\"" << zeroY
            << "\" stroke=\"#666666\" stroke-width=\"1.8\"/>\n";
        svg << "<text x=\"" << (left - 14) << "\" y=\"" << (zeroY + 4)
            << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#444444\">0</text>\n";
    }

    if (confidenceStarted) {
        svg << "<path d=\"" << confidenceBand.str()
            << "\" fill=\"#1f77b4\" fill-opacity=\"0.28\" stroke=\"none\"/>\n";
        std::ostringstream upperBandPath;
        std::ostringstream lowerBandPath;
        for (size_t i = 0; i < upperBandPoints.size(); ++i) {
            upperBandPath << (i == 0 ? "M " : " L ")
                          << upperBandPoints[i].first << " " << upperBandPoints[i].second;
            lowerBandPath << (i == 0 ? "M " : " L ")
                          << lowerBandPoints[i].first << " " << lowerBandPoints[i].second;
        }
        svg << "<path d=\"" << upperBandPath.str()
            << "\" fill=\"none\" stroke=\"#0f4c81\" stroke-opacity=\"0.45\" stroke-width=\"1.2\" stroke-dasharray=\"4,3\"/>\n";
        svg << "<path d=\"" << lowerBandPath.str()
            << "\" fill=\"none\" stroke=\"#0f4c81\" stroke-opacity=\"0.45\" stroke-width=\"1.2\" stroke-dasharray=\"4,3\"/>\n";
    }
    if (empiricalStarted) {
        svg << "<path d=\"" << empiricalPath.str()
            << "\" fill=\"none\" stroke=\"#1f77b4\" stroke-width=\"2.5\"/>\n";
    }
    if (regressionStarted) {
        svg << "<path d=\"" << regressionPath.str()
            << "\" fill=\"none\" stroke=\"#d62728\" stroke-width=\"2\"/>\n";
    }

    svg << "<text x=\"" << (left + plotWidth / 2.0) << "\" y=\"" << (height - 18)
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Count</text>\n";
    svg << "<text x=\"20\" y=\"" << (top + plotHeight / 2.0)
        << "\" transform=\"rotate(-90 20 " << (top + plotHeight / 2.0)
        << ")\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">EV per round</text>\n";

    svg << "<line x1=\"" << (width - 220) << "\" y1=\"20\" x2=\"" << (width - 180)
        << "\" y2=\"20\" stroke=\"#1f77b4\" stroke-width=\"2.5\"/>\n";
    svg << "<text x=\"" << (width - 170) << "\" y=\"24\" font-family=\"sans-serif\" font-size=\"12\">Empirical EV</text>\n";
    svg << "<rect x=\"" << (width - 220) << "\" y=\"31\" width=\"40\" height=\"10\" fill=\"#1f77b4\" fill-opacity=\"0.28\"/>\n";
    svg << "<text x=\"" << (width - 170) << "\" y=\"40\" font-family=\"sans-serif\" font-size=\"12\">95% confidence band</text>\n";
    svg << "<line x1=\"" << (width - 220) << "\" y1=\"56\" x2=\"" << (width - 180)
        << "\" y2=\"56\" stroke=\"#d62728\" stroke-width=\"2\"/>\n";
    svg << "<text x=\"" << (width - 170) << "\" y=\"60\" font-family=\"sans-serif\" font-size=\"12\">Regression line</text>\n";
    svg << "</svg>\n";
    return svg.str();
}

std::string evCountOverlaySvg(const std::string& label,
                              const std::vector<NamedEvCountGraphArtifact>& graphs) {
    const double width = 1100.0;
    const double height = 680.0;
    const double left = 90.0;
    const double right = 240.0;
    const double top = 40.0;
    const double bottom = 70.0;
    const double plotWidth = width - left - right;
    const double plotHeight = height - top - bottom;

    double minX = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const auto& named : graphs) {
        minX = std::min(minX, named.graph.minCount);
        maxX = std::max(maxX, named.graph.maxCount);
        for (const auto& p : named.graph.points) {
            minY = std::min(minY, p.regressionReward);
            maxY = std::max(maxY, p.regressionReward);
            if (p.n > 0) {
                minY = std::min(minY, p.meanReward);
                maxY = std::max(maxY, p.meanReward);
            }
        }
    }
    if (!std::isfinite(minX) || !std::isfinite(maxX) || std::abs(maxX - minX) < kEps) {
        minX = -5.0;
        maxX = 5.0;
    }
    if (!std::isfinite(minY) || !std::isfinite(maxY)) {
        minY = -1.0;
        maxY = 1.0;
    }
    if (std::abs(maxY - minY) < kEps) {
        minY -= 1.0;
        maxY += 1.0;
    } else {
        double padding = 0.1 * (maxY - minY);
        minY -= padding;
        maxY += padding;
    }

    auto mapX = [&](double x) {
        return left + ((x - minX) / (maxX - minX)) * plotWidth;
    };
    auto mapY = [&](double y) {
        return top + (1.0 - ((y - minY) / (maxY - minY))) * plotHeight;
    };

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    svg << "<defs>\n";
    svg << "  <pattern id=\"minorGridOverlay\" width=\"" << (plotWidth / 20.0) << "\" height=\"" << (plotHeight / 20.0)
        << "\" patternUnits=\"userSpaceOnUse\">\n";
    svg << "    <path d=\"M " << (plotWidth / 20.0) << " 0 L 0 0 0 " << (plotHeight / 20.0)
        << "\" fill=\"none\" stroke=\"#dddddd\" stroke-width=\"1.1\"/>\n";
    svg << "  </pattern>\n";
    svg << "  <pattern id=\"majorGridOverlay\" width=\"" << (plotWidth / 10.0) << "\" height=\"" << (plotHeight / 10.0)
        << "\" patternUnits=\"userSpaceOnUse\">\n";
    svg << "    <rect width=\"" << (plotWidth / 10.0) << "\" height=\"" << (plotHeight / 10.0)
        << "\" fill=\"url(#minorGridOverlay)\"/>\n";
    svg << "    <path d=\"M " << (plotWidth / 10.0) << " 0 L 0 0 0 " << (plotHeight / 10.0)
        << "\" fill=\"none\" stroke=\"#b8b8b8\" stroke-width=\"1.5\"/>\n";
    svg << "  </pattern>\n";
    svg << "</defs>\n";
    svg << "<text x=\"" << left << "\" y=\"24\" font-family=\"sans-serif\" font-size=\"20\">"
        << label << " cumulative EV vs Count</text>\n";
    svg << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plotWidth
        << "\" height=\"" << plotHeight << "\" fill=\"url(#majorGridOverlay)\"/>\n";
    svg << "<line x1=\"" << left << "\" y1=\"" << (top + plotHeight) << "\" x2=\"" << (left + plotWidth)
        << "\" y2=\"" << (top + plotHeight) << "\" stroke=\"black\" stroke-width=\"1.2\"/>\n";
    svg << "<line x1=\"" << left << "\" y1=\"" << top << "\" x2=\"" << left
        << "\" y2=\"" << (top + plotHeight) << "\" stroke=\"black\" stroke-width=\"1.2\"/>\n";

    for (int i = 0; i <= 16; ++i) {
        double xValue = minX + (maxX - minX) * (static_cast<double>(i) / 16.0);
        double x = mapX(xValue);
        svg << "<line x1=\"" << x << "\" y1=\"" << (top + plotHeight)
            << "\" x2=\"" << x << "\" y2=\"" << (top + plotHeight + 6)
            << "\" stroke=\"black\"/>\n";
        if (i % 2 == 0) {
            svg << "<text x=\"" << x << "\" y=\"" << (top + plotHeight + 24)
                << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">"
                << std::fixed << std::setprecision(2) << xValue << "</text>\n";
        }
    }
    for (int i = 0; i <= 12; ++i) {
        double yValue = minY + (maxY - minY) * (static_cast<double>(i) / 12.0);
        double y = mapY(yValue);
        svg << "<line x1=\"" << (left - 6) << "\" y1=\"" << y
            << "\" x2=\"" << left << "\" y2=\"" << y << "\" stroke=\"black\"/>\n";
        if (i % 2 == 0) {
            svg << "<text x=\"" << (left - 10) << "\" y=\"" << (y + 4)
                << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">"
                << std::fixed << std::setprecision(4) << yValue << "</text>\n";
        }
    }
    if (minX <= 0.0 && maxX >= 0.0) {
        double zeroX = mapX(0.0);
        svg << "<line x1=\"" << zeroX << "\" y1=\"" << top
            << "\" x2=\"" << zeroX << "\" y2=\"" << (top + plotHeight)
            << "\" stroke=\"#666666\" stroke-width=\"1.8\"/>\n";
    }
    if (minY <= 0.0 && maxY >= 0.0) {
        double zeroY = mapY(0.0);
        svg << "<line x1=\"" << left << "\" y1=\"" << zeroY
            << "\" x2=\"" << (left + plotWidth) << "\" y2=\"" << zeroY
            << "\" stroke=\"#666666\" stroke-width=\"1.8\"/>\n";
    }

    for (size_t graphIndex = 0; graphIndex < graphs.size(); ++graphIndex) {
        const auto& named = graphs[graphIndex];
        const std::string color = colorForGraph(graphIndex);
        std::ostringstream empiricalPath;
        std::ostringstream regressionPath;
        bool empiricalStarted = false;
        bool regressionStarted = false;
        for (const auto& p : named.graph.points) {
            const double x = mapX(p.count);
            regressionPath << (regressionStarted ? " L " : "M ")
                           << x << " " << mapY(p.regressionReward);
            regressionStarted = true;
            if (p.n > 0) {
                empiricalPath << (empiricalStarted ? " L " : "M ")
                              << x << " " << mapY(p.meanReward);
                empiricalStarted = true;
            }
        }
        if (empiricalStarted) {
            svg << "<path d=\"" << empiricalPath.str()
                << "\" fill=\"none\" stroke=\"" << color
                << "\" stroke-width=\"2.4\"/>\n";
        }
        if (regressionStarted) {
            svg << "<path d=\"" << regressionPath.str()
                << "\" fill=\"none\" stroke=\"" << color
                << "\" stroke-width=\"1.5\" stroke-dasharray=\"6,4\" stroke-opacity=\"0.85\"/>\n";
        }
    }

    svg << "<text x=\"" << (left + plotWidth / 2.0) << "\" y=\"" << (height - 18)
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Count</text>\n";
    svg << "<text x=\"20\" y=\"" << (top + plotHeight / 2.0)
        << "\" transform=\"rotate(-90 20 " << (top + plotHeight / 2.0)
        << ")\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">EV per round</text>\n";

    double legendY = 24.0;
    for (size_t graphIndex = 0; graphIndex < graphs.size(); ++graphIndex) {
        const auto& named = graphs[graphIndex];
        const std::string color = colorForGraph(graphIndex);
        svg << "<line x1=\"" << (width - 210) << "\" y1=\"" << legendY
            << "\" x2=\"" << (width - 170) << "\" y2=\"" << legendY
            << "\" stroke=\"" << color << "\" stroke-width=\"2.4\"/>\n";
        svg << "<line x1=\"" << (width - 210) << "\" y1=\"" << (legendY + 10)
            << "\" x2=\"" << (width - 170) << "\" y2=\"" << (legendY + 10)
            << "\" stroke=\"" << color
            << "\" stroke-width=\"1.5\" stroke-dasharray=\"6,4\" stroke-opacity=\"0.85\"/>\n";
        svg << "<text x=\"" << (width - 160) << "\" y=\"" << (legendY + 4)
            << "\" font-family=\"sans-serif\" font-size=\"12\">"
            << named.label << " empirical</text>\n";
        svg << "<text x=\"" << (width - 160) << "\" y=\"" << (legendY + 14)
            << "\" font-family=\"sans-serif\" font-size=\"12\">"
            << named.label << " regression</text>\n";
        legendY += 30.0;
    }
    svg << "</svg>\n";
    return svg.str();
}

CountConfig countConfigFromJson(const json& j) {
    CountConfig count;
    for (size_t i = 0; i < count.system.weights.size(); ++i)
        count.system.weights[i] = j.at("weights").at(i).get<double>();
    count.system.factor = j.value("factor", 1.0);
    count.system.bias = j.value("bias", 0.0);
    count.resolution = j.value("resolution", 1.0);
    count.minCount = j.value("min_count", -5);
    count.maxCount = j.value("max_count", 5);
    return count;
}

CountConfig makeZeroCount() {
    CountConfig count;
    count.system.weights = ZERO_WEIGHTS;
    count.system.factor = 1.0;
    count.system.bias = 0.0;
    count.resolution = g_agent.countResolution;
    count.minCount = g_agent.minCount;
    count.maxCount = g_agent.maxCount;
    return count;
}

const char* phaseToString(Phase phase) {
    switch (phase) {
        case Phase::POLICY: return "policy";
        case Phase::COUNT: return "count";
        case Phase::DONE: return "done";
    }
    return "policy";
}

Phase stringToPhase(const std::string& phase) {
    if (phase == "count") return Phase::COUNT;
    if (phase == "done") return Phase::DONE;
    return Phase::POLICY;
}

void saveMeta(const std::string& folder, const json& meta) {
    namespace fs = std::filesystem;
    fs::create_directories(fs::path(PROJECT_ROOT) / kAlternatingCheckpointRoot / folder);
    std::ofstream f(fs::path(PROJECT_ROOT) / kAlternatingCheckpointRoot / folder / "meta.json");
    if (f.is_open()) f << meta.dump(2);
}

void saveState(const std::string& folder, const ResumeState& state) {
    namespace fs = std::filesystem;
    fs::path root = fs::path(PROJECT_ROOT) / kAlternatingCheckpointRoot / folder;
    fs::create_directories(root);

    json j;
    j["next_phase"] = phaseToString(state.nextPhase);
    j["next_policy_index"] = state.nextPolicyIndex;
    j["current_count"] = countConfigToJson(state.currentCount, false);
    j["current_policy_label"] = state.currentPolicyLabel;
    j["current_policy_path"] = state.currentPolicyPath;
    j["current_policy_flat_edge"] = state.currentPolicyFlatEdge;

    std::ofstream f(root / "state.json");
    if (f.is_open()) f << j.dump(2);
}

ResumeState inferResumeState(const std::string& folder) {
    namespace fs = std::filesystem;
    fs::path root = fs::path(PROJECT_ROOT) / kAlternatingCheckpointRoot / folder;

    int maxP = -1;
    int maxW = 0;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_regular_file()) continue;
        std::string name = entry.path().filename().string();
        if (name.size() > 15 && name[0] == 'P' && name.find("_strategy.json") != std::string::npos) {
            size_t end = name.find("_strategy.json");
            maxP = std::max(maxP, std::stoi(name.substr(1, end - 1)));
        } else if (name.size() > 5 && name[0] == 'W' && name.find(".json") != std::string::npos
                   && name.find("_data.json") == std::string::npos) {
            size_t end = name.find(".json");
            maxW = std::max(maxW, std::stoi(name.substr(1, end - 1)));
        }
    }

    ResumeState state;
    state.currentCount = makeZeroCount();
    if (maxW > 0) {
        std::ifstream wf(root / ("W" + std::to_string(maxW) + ".json"));
        if (wf.is_open()) {
            json wj;
            wf >> wj;
            state.currentCount = countConfigFromJson(wj.at("count_config"));
        }
    }

    if (maxP >= 0 && maxW < maxP + 1) {
        state.nextPhase = Phase::COUNT;
        state.nextPolicyIndex = static_cast<uint64_t>(maxP);
        state.currentPolicyLabel = "P" + std::to_string(maxP);
        state.currentPolicyPath = std::string(kAlternatingCheckpointRoot) + "/" + folder + "/" +
                                  state.currentPolicyLabel + "_strategy.json";
        std::ifstream pf(root / (state.currentPolicyLabel + ".json"));
        if (pf.is_open()) {
            json pj;
            pf >> pj;
            state.currentPolicyFlatEdge = pj.value("evaluation_edge_flat", 0.0);
        }
    } else {
        state.nextPhase = Phase::POLICY;
        state.nextPolicyIndex = static_cast<uint64_t>(maxW);
    }
    return state;
}

ResumeState loadState(const std::string& folder) {
    namespace fs = std::filesystem;
    fs::path path = fs::path(PROJECT_ROOT) / kAlternatingCheckpointRoot / folder / "state.json";
    if (!fs::exists(path)) {
        return inferResumeState(folder);
    }

    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open alternating checkpoint state.json");

    json j;
    f >> j;

    ResumeState state;
    state.nextPhase = stringToPhase(j.value("next_phase", "policy"));
    state.nextPolicyIndex = j.value("next_policy_index", uint64_t(0));
    state.currentCount = countConfigFromJson(j.at("current_count"));
    state.currentPolicyLabel = j.value("current_policy_label", "");
    state.currentPolicyPath = j.value("current_policy_path", "");
    state.currentPolicyFlatEdge = j.value("current_policy_flat_edge", 0.0);
    return state;
}

Case loadCheckpointFolder(const std::string& folder) {
    namespace fs = std::filesystem;
    fs::path metaPath = fs::path(PROJECT_ROOT) / kAlternatingCheckpointRoot / folder / "meta.json";
    if (!fs::exists(metaPath))
        throw std::runtime_error("Checkpoint folder '" + folder + "' missing meta.json");

    json meta;
    { std::ifstream f(metaPath); f >> meta; }

    auto& g = meta.at("game");
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
    g_penetration      = g.at("penetration").get<double>();

    auto& a = meta.at("algorithm_config");
    g_num_rounds = a.at("num_rounds_per_phase").get<uint64_t>();
    g_eval_rounds = a.at("eval_rounds").get<uint64_t>();
    g_iterations = a.at("iterations").get<uint64_t>();
    g_sample_every = a.at("sample_every").get<uint64_t>();
    g_num_threads = a.at("num_threads").get<int>();
    std::string stopMode = a.value("policy_stop_mode", std::string("rounds"));
    g_training_stop_mode = (stopMode == "diff") ? TrainingStopMode::TABLE_DIFF
                                                : TrainingStopMode::FIXED_ROUNDS;
    g_sample_rounds = a.value("policy_sample_rounds", 100'000'000ULL);
    g_diff_threshold = a.value("policy_diff_threshold", 0.001);

    auto& rl = meta.at("rl");
    std::string mode = rl.at("exploration_mode").get<std::string>();
    g_agent.explorationMode = (mode == "boltzmann")
        ? ExplorationMode::BOLTZMANN
        : ExplorationMode::EPSILON_GREEDY;
    g_agent.epsilonStart = rl.value("epsilon_start", 1.0);
    g_agent.epsilonMin = rl.value("epsilon_min", 0.1);
    g_agent.epsilonDecay = rl.value("epsilon_decay", 0.99999);
    g_agent.tempStart = rl.value("temp_start", 1.0);
    g_agent.tempMin = rl.value("temp_min", 0.1);
    g_agent.tempDecay = rl.value("temp_decay", 0.99999);
    g_agent.alphaStart = rl.value("alpha_start", 0.01);
    g_agent.alphaMin = rl.value("alpha_min", 0.0001);
    g_agent.alphaDecaySteps = rl.value("alpha_decay_steps", 100.0);
    g_agent.countResolution = rl.value("count_resolution", 1.0);
    g_agent.minCount = rl.value("min_count", -5);
    g_agent.maxCount = rl.value("max_count", 5);

    std::cout << "\n=== Loaded checkpoint: " << kAlternatingCheckpointRoot << "/" << folder << "/ ===\n";
    std::cout << "Scenario:      " << ToString(c) << "\n";
    std::cout << "Train rounds:  " << g_num_rounds << "  Threads: " << g_num_threads << "\n";
    std::cout << "Eval rounds:   " << g_eval_rounds << "\n";
    std::cout << "Iterations:    " << g_iterations << "\n";
    std::cout << "Penetration:   " << g_penetration << "%\n";
    std::cout << "Sample-every:  " << g_sample_every << "\n";
    std::cout << "Policy stop:   " << stopModeToString(g_training_stop_mode);
    if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF) {
        std::cout << "  sample=" << g_sample_rounds
                  << "  threshold=" << g_diff_threshold;
    }
    std::cout << "\n";
    std::cout << "RL mode:       " << (g_agent.explorationMode == ExplorationMode::BOLTZMANN ? "boltzmann" : "epsilon") << "\n";
    std::cout << "Count grid:    res=" << g_agent.countResolution
              << "  range=[" << g_agent.minCount << "," << g_agent.maxCount << "]\n";
    std::cout << "=== (CLI game/algorithm/RL flags are ignored) ===\n\n";
    return c;
}

void savePolicyArtifact(const std::string& folder,
                        const std::string& label,
                        const PolicyArtifact& artifact,
                        const CountConfig& count) {
    namespace fs = std::filesystem;
    fs::path root = fs::path(PROJECT_ROOT) / kAlternatingCheckpointRoot / folder;
    fs::create_directories(root);

    artifact.qStrategy->saveToFile(
        (std::string(kAlternatingCheckpointRoot) + "/" + folder + "/" + label + "_agent.json"),
        artifact.trainingRounds);
    artifact.policy->saveToJson(
        (std::string(kAlternatingCheckpointRoot) + "/" + folder + "/" + label + "_strategy.json"));

    json summary;
    summary["label"] = label;
    summary["type"] = "policy";
    summary["training_rounds"] = artifact.trainingRounds;
    summary["training_edge_flat"] = artifact.trainingEdge;
    summary["evaluation_edge_flat"] = artifact.evaluationEdge;
    summary["hands_per_sec"] = artifact.handsPerSec;
    summary["count_config"] = countConfigToJson(count, true);
    std::ofstream f(root / (label + ".json"));
    if (f.is_open()) f << summary.dump(2);
}

void saveCountArtifact(const std::string& folder,
                       const std::string& label,
                       const CountArtifact& artifact,
                       uint64_t rounds) {
    namespace fs = std::filesystem;
    fs::path root = fs::path(PROJECT_ROOT) / kAlternatingCheckpointRoot / folder;
    fs::create_directories(root);

    json summary;
    summary["label"] = label;
    summary["type"] = "count";
    summary["regression_rounds"] = rounds;
    summary["recorded_rounds"] = artifact.recordedRounds;
    summary["hands_per_sec"] = artifact.handsPerSec;
    summary["spread_threshold"] = artifact.threshold;
    summary["evaluation_edge_spread_1_10"] = artifact.evaluationEdge;
    summary["evaluation_kelly_growth"] = artifact.evaluationKellyGrowth;
    summary["normalization_scale"] = artifact.normalizationScale;
    summary["learned_count_factor_multiplier"] = kLearnedCountFactorMultiplier;
    summary["reference_flat_edge"] = artifact.referenceFlatEdge;
    summary["raw_solution"] = artifact.rawSolution;
    summary["count_config"] = countConfigToJson(artifact.count, true);

    std::ofstream f(root / (label + ".json"));
    if (f.is_open()) f << summary.dump(2);

    json data;
    data["XtX"] = json::array();
    for (int i = 0; i < 14; ++i) {
        json row = json::array();
        for (int j = 0; j < 14; ++j) row.push_back(artifact.XtX[i][j]);
        data["XtX"].push_back(row);
    }
    data["Xty"] = artifact.Xty;
    data["recorded_rounds"] = artifact.recordedRounds;
    std::ofstream df(root / (label + "_data.json"));
    if (df.is_open()) df << data.dump(2);
}

void saveEvCountGraphArtifact(const std::string& folder,
                              const std::string& label,
                              const EvCountGraphArtifact& graph) {
    namespace fs = std::filesystem;
    fs::path root = fs::path(PROJECT_ROOT) / kAlternatingCheckpointRoot / folder;
    fs::create_directories(root);
    std::ofstream f(root / (label + "_graph.json"));
    if (f.is_open()) f << evCountGraphToJson(graph).dump(2);
    std::ofstream svg(root / (label + "_graph.svg"));
    if (svg.is_open()) svg << evCountGraphToSvg(label, graph);
}

void saveCumulativeEvCountGraphArtifact(const std::string& folder,
                                        int upToWeightIndex) {
    namespace fs = std::filesystem;
    fs::path root = fs::path(PROJECT_ROOT) / kAlternatingCheckpointRoot / folder;

    std::vector<NamedEvCountGraphArtifact> graphs;
    for (int i = 1; i <= upToWeightIndex; ++i) {
        fs::path graphPath = root / ("W" + std::to_string(i) + "_graph.json");
        if (!fs::exists(graphPath))
            continue;
        std::ifstream f(graphPath);
        if (!f.is_open())
            continue;
        json j;
        f >> j;
        graphs.push_back({"W" + std::to_string(i), evCountGraphFromJson(j)});
    }

    if (graphs.empty())
        return;

    std::ofstream jf(root / ("W" + std::to_string(upToWeightIndex) + "_graph_overlay.json"));
    if (jf.is_open()) jf << evCountGraphsToJson(graphs);
    std::ofstream sf(root / ("W" + std::to_string(upToWeightIndex) + "_graph_overlay.svg"));
    if (sf.is_open()) sf << evCountOverlaySvg("W1..W" + std::to_string(upToWeightIndex), graphs);
}

PolicyArtifact learnPolicy(const Case& c,
                           const CountConfig& count,
                           uint64_t rounds,
                           const std::string& label,
                           const std::string& runHeader) {
    (void)runHeader;
    auto* player = new Player(0.0, makeQStrategy());
    player->setNumDecks(c.deckSize);
    player->setCountSystem(count.system);
    player->setCountResolution(count.resolution);
    player->setCountRange(count.minCount, count.maxCount);

    BlackjackRules rules = buildRules(c, 1.0, 1.0);
    Player* result = nullptr;
    uint64_t trainedRounds = 0;
    auto wallStart = std::chrono::high_resolution_clock::now();
    std::vector<QLearningStrategy::QTableSnapshot> previousTables(1);

    while (true) {
        uint64_t batch = rounds;
        if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF) {
            batch = g_sample_rounds;
        } else if (trainedRounds >= rounds) {
            break;
        } else {
            batch = std::min(rounds - trainedRounds, rounds);
        }

        std::vector<Player*> results = runParallelSimulation(rules, {player}, batch, g_num_threads);
        delete player;
        if (results.empty())
            throw std::runtime_error("Policy learning failed: no result players");
        player = results[0];
        trainedRounds += batch;

        const auto* sampled = dynamic_cast<const QLearningStrategy*>(player->getStrategy());
        if (!sampled)
            throw std::runtime_error("Policy learning failed: result strategy is not QLearningStrategy");

        double avgDiff = std::numeric_limits<double>::infinity();
        if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF) {
            auto currentSnapshot = sampled->snapshotQTable();
            avgDiff = QLearningStrategy::averageAbsDifference(currentSnapshot, previousTables[0]);
            previousTables[0] = std::move(currentSnapshot);
        }

        if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF) {
            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - wallStart).count() / 1000.0;
            double handsPerSec = (elapsed > 0.0 ? trainedRounds / elapsed : 0.0);
            std::cout << label
                      << " sample rounds=" << trainedRounds
                      << "  speed=" << std::fixed << std::setprecision(0) << handsPerSec << " hands/sec"
                      << "  avg|ΔQ|=" << std::setprecision(6) << avgDiff
                      << "  threshold=" << g_diff_threshold << "\n";
        }

        if (g_training_stop_mode == TrainingStopMode::FIXED_ROUNDS) {
            if (trainedRounds >= rounds) break;
        } else if (avgDiff <= g_diff_threshold) {
            break;
        }
    }

    result = player;
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - wallStart).count() / 1000.0;
    const auto* learned = dynamic_cast<const QLearningStrategy*>(result->getStrategy());
    if (!learned) {
        delete result;
        throw std::runtime_error("Policy learning failed: result strategy is not QLearningStrategy");
    }

    auto policy = learned->toBasicStrategy();
    auto clonedBase = learned->clone();
    auto clonedQ = std::unique_ptr<QLearningStrategy>(
        dynamic_cast<QLearningStrategy*>(clonedBase.release()));
    if (!clonedQ) {
        delete result;
        throw std::runtime_error("Policy learning failed: could not clone learned Q strategy");
    }

    PolicyArtifact artifact;
    artifact.trainingRounds = trainedRounds;
    artifact.trainingEdge = netPerRoundFromPlayer(*result, trainedRounds, g_num_threads);
    artifact.handsPerSec  = (secs > 0.0 ? trainedRounds / secs : 0.0);
    artifact.evaluationEdge = evaluateEdge(c, *policy, count, nullptr, 1.0, 1.0,
                                           (g_eval_rounds == 0 ? trainedRounds : g_eval_rounds));
    artifact.qStrategy = std::move(clonedQ);
    artifact.policy = std::move(policy);

    delete result;
    return artifact;
}

CountArtifact learnCount(const Case& c,
                         const BasicStrategy& fixedPolicy,
                         const CountConfig& strategyCount,
                         uint64_t rounds) {
    auto cloned = fixedPolicy.clone();
    auto* player = new Player(0.0, std::move(cloned));
    player->setNumDecks(c.deckSize);
    player->setCountSystem(strategyCount.system);
    player->setCountResolution(strategyCount.resolution);
    player->setCountRange(strategyCount.minCount, strategyCount.maxCount);
    player->enableRegression();
    player->setRegressionSampleEvery(g_sample_every);

    BlackjackRules rules = buildRules(c, 1.0, 1.0);
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Player*> results = runParallelSimulation(rules, {player}, rounds, g_num_threads);
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count() / 1000.0;

    delete player;
    if (results.empty())
        throw std::runtime_error("Count learning failed: no result players");

    Player* result = results[0];
    CountArtifact artifact = normalizeCount(solveOLS(result->getXtX(), result->getXty()));
    artifact.XtX = result->getXtX();
    artifact.Xty = result->getXty();
    artifact.recordedRounds = result->getRegressionRounds();
    artifact.handsPerSec = (secs > 0.0 ? rounds / secs : 0.0);
    artifact.threshold = spreadThresholdFromCount(artifact.count);

    auto betting = std::make_unique<SpreadBetting>(
        std::vector<std::pair<double, double>>{{artifact.threshold, 10.0}});
    artifact.evaluationEdge = evaluateEdge(c, fixedPolicy, artifact.count, std::move(betting),
                                           1.0, 10.0, (g_eval_rounds == 0 ? rounds : g_eval_rounds));
    artifact.evaluationKellyGrowth = evaluateKellyGrowth(
        c, fixedPolicy, artifact.count, (g_eval_rounds == 0 ? rounds : g_eval_rounds));

    delete result;
    return artifact;
}

EvCountGraphArtifact measureEvCountGraph(const Case& c,
                                         const BasicStrategy& fixedPolicy,
                                         const CountConfig& count,
                                         uint64_t rounds) {
    auto cloned = fixedPolicy.clone();
    auto* player = new Player(0.0, std::move(cloned));
    player->setNumDecks(c.deckSize);
    player->setCountSystem(count.system);
    player->setCountResolution(count.resolution);
    auto [strategyMinCount, strategyMaxCount] = fixedPolicy.getCountRange();
    player->setCountRange(strategyMinCount, strategyMaxCount);
    double graphResolution = (count.resolution > 0.0) ? (count.resolution / 4.0) : 0.25;
    player->enableCountGraph(graphResolution);

    BlackjackRules rules = buildRules(c, 1.0, 1.0);
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Player*> results = runParallelSimulation(rules, {player}, rounds, g_num_threads);
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count() / 1000.0;

    delete player;
    if (results.empty())
        throw std::runtime_error("EV-count graph failed: no result players");

    Player* result = results[0];
    EvCountGraphArtifact graph;
    graph.resolution = graphResolution;
    graph.rounds = rounds;
    graph.handsPerSec = (secs > 0.0 ? rounds / secs : 0.0);
    graph.minCount = count.minCount;
    graph.maxCount = count.maxCount;
    const int minBinIndex = static_cast<int>(std::llround(graph.minCount / graphResolution));
    const int maxBinIndex = static_cast<int>(std::llround(graph.maxCount / graphResolution));
    const auto& bins = result->getCountGraphBins();
    for (int binIndex = minBinIndex; binIndex <= maxBinIndex; ++binIndex) {
        EvCountGraphPoint point;
        point.count = static_cast<double>(binIndex) * graphResolution;
        point.regressionReward = count.system.bias + count.system.factor * point.count;
        auto it = bins.find(binIndex);
        if (it != bins.end() && it->second.n > 0) {
            const auto& stats = it->second;
            point.n = stats.n;
            point.meanReward = stats.sumReward / static_cast<double>(stats.n);
            double meanSq = stats.sumRewardSq / static_cast<double>(stats.n);
            double variance = std::max(0.0, meanSq - point.meanReward * point.meanReward);
            point.stddevReward = std::sqrt(variance);
            double stderr = point.stddevReward / std::sqrt(static_cast<double>(stats.n));
            double z95 = 1.96;
            point.confidenceLower = point.meanReward - z95 * stderr;
            point.confidenceUpper = point.meanReward + z95 * stderr;
        }
        graph.points.push_back(point);
    }
    delete result;
    return graph;
}

void printCountSummary(const std::string& label, const CountArtifact& artifact) {
    static const char* kRankNames[] = {
        "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"
    };

    double tenMean = (artifact.count.system.weights[8] + artifact.count.system.weights[9]
                    + artifact.count.system.weights[10] + artifact.count.system.weights[11]) / 4.0;
    double tenMaxDev = std::max({
        std::abs(artifact.count.system.weights[8]  - tenMean),
        std::abs(artifact.count.system.weights[9]  - tenMean),
        std::abs(artifact.count.system.weights[10] - tenMean),
        std::abs(artifact.count.system.weights[11] - tenMean)
    });
    double tenRelDev = (std::abs(tenMean) > kEps) ? tenMaxDev / std::abs(tenMean) : tenMaxDev;

    double sumWeights = 0.0;
    for (double w : artifact.count.system.weights) sumWeights += w;

    std::cout << "\n--- " << label << " ---\n";
    std::cout << "Recorded rounds: " << artifact.recordedRounds
              << "  speed: " << std::fixed << std::setprecision(0)
              << artifact.handsPerSec << " hands/sec\n";
    std::cout << std::setprecision(6);
    std::cout << "Count table:\n";
    std::cout << "  " << std::left << std::setw(10) << "Card"
              << std::setw(12) << "Value" << "\n";
    std::cout << "  " << std::string(22, '-') << "\n";
    for (size_t i = 0; i < artifact.count.system.weights.size(); ++i) {
        std::cout << "  " << std::left << std::setw(10) << kRankNames[i]
                  << std::setw(12) << std::fixed << std::setprecision(4)
                  << artifact.count.system.weights[i] << "\n";
    }
    std::cout << "  " << std::left << std::setw(10) << "Bias"
              << std::setw(12) << std::fixed << std::setprecision(6)
              << artifact.count.system.bias << "\n";
    std::cout << "  " << std::left << std::setw(10) << "EV>0 TC"
              << std::setw(12) << std::fixed << std::setprecision(6)
              << artifact.threshold << "\n";
    std::cout << "  " << std::left << std::setw(10) << "Factor"
              << std::setw(12) << std::fixed << std::setprecision(6)
              << artifact.count.system.factor << "\n";
    std::cout << std::setprecision(6);
    std::cout << "\nSpread 1:10 evaluation edge: " << artifact.evaluationEdge << "\n";
    std::cout << "Kelly growth per round:      " << artifact.evaluationKellyGrowth << "\n";
    std::cout << "Sanity checks:\n";
    std::cout << "  Sum(weights): " << sumWeights
              << "  (" << (std::abs(sumWeights) < 0.05 ? "PASS" : "WARN") << ", expected near 0)\n";
    std::cout << "  10-value consistency: mean=" << tenMean
              << "  max_dev=" << tenMaxDev
              << "  rel=" << (tenRelDev * 100.0) << "%";
    std::cout << "  (" << (tenRelDev < 0.05 ? "PASS" : "WARN") << ")\n";
    std::cout << "  Bias vs flat edge: bias=" << artifact.count.system.bias
              << "  flat_edge=" << artifact.referenceFlatEdge
              << "  diff=" << std::abs(artifact.count.system.bias - artifact.referenceFlatEdge) << "\n";
}

void printEvCountGraphSummary(const std::string& label, const EvCountGraphArtifact& graph) {
    std::cout << "EV-count graph:         " << graph.points.size()
              << " points  res=" << graph.resolution
              << "  speed: " << std::fixed << std::setprecision(0)
              << graph.handsPerSec << " hands/sec\n";
    std::cout << std::setprecision(6);
}

void printPolicySummary(const std::string& label, const PolicyArtifact& artifact) {
    std::cout << "\n--- " << label << " ---\n";
    std::cout << "Training rounds: " << artifact.trainingRounds << "\n";
    std::cout << "Training speed: " << std::fixed << std::setprecision(0)
              << artifact.handsPerSec << " hands/sec\n";
    std::cout << std::setprecision(6);
    std::cout << "Training flat edge:   " << artifact.trainingEdge << "\n";
    std::cout << "Evaluation flat edge: " << artifact.evaluationEdge << "\n";
}

void runCase(const Case& c) {
    const bool resuming = !g_load_checkpoint.empty();
    const std::string folder = resuming
        ? g_load_checkpoint
        : (g_checkpoint_name.empty()
            ? currentTimestamp() + "_" + ToStringTableName(c)
            : g_checkpoint_name + "_" + ToStringTableName(c));
    const uint64_t evalRounds = (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds);
    const std::string runHeader = buildRunHeader(c, folder, evalRounds);

    std::cout << runHeader;

    ResumeState state = resuming ? loadState(folder) : ResumeState{};
    if (!resuming) {
        state.currentCount = makeZeroCount();
        if (!g_no_save) {
            saveMeta(folder, buildMetaJson(c));
            saveState(folder, state);
        }
    }

    std::unique_ptr<BasicStrategy> currentPolicy;
    while (state.nextPhase != Phase::DONE) {
        if (state.nextPhase == Phase::POLICY) {
            if (state.nextPolicyIndex >= g_iterations) {
                state.nextPhase = Phase::DONE;
                if (!g_no_save) saveState(folder, state);
                break;
            }

            std::string pLabel = "P" + std::to_string(state.nextPolicyIndex);
            PolicyArtifact learnedPolicy = learnPolicy(c, state.currentCount, g_num_rounds, pLabel, runHeader);
            printPolicySummary(pLabel, learnedPolicy);
            if (g_verbose) {
                std::cout << "\n=== " << pLabel << " Strategy ===\n";
                std::cout << *learnedPolicy.policy;
                std::cout << "========================\n";
            }
            if (!g_no_save)
                savePolicyArtifact(folder, pLabel, learnedPolicy, state.currentCount);

            currentPolicy = std::move(learnedPolicy.policy);
            state.currentPolicyLabel = pLabel;
            state.currentPolicyPath = std::string(kAlternatingCheckpointRoot) + "/" + folder + "/" +
                                      pLabel + "_strategy.json";
            state.currentPolicyFlatEdge = learnedPolicy.evaluationEdge;
            state.nextPhase = Phase::COUNT;
            if (!g_no_save) saveState(folder, state);
            continue;
        }

        if (!currentPolicy) {
            currentPolicy = std::make_unique<BasicStrategy>();
            if (state.currentPolicyPath.empty() || !currentPolicy->loadFromFile(state.currentPolicyPath)) {
                throw std::runtime_error("Cannot resume count step: failed to load policy from " +
                                         state.currentPolicyPath);
            }
        }

        std::string wLabel = "W" + std::to_string(state.nextPolicyIndex + 1);
        const uint64_t countRounds = countPhasePlayedRounds();
        CountArtifact countArtifact = learnCount(c, *currentPolicy, state.currentCount, countRounds);
        countArtifact.referenceFlatEdge = state.currentPolicyFlatEdge;
        printCountSummary(wLabel, countArtifact);
        if (!g_no_save)
            saveCountArtifact(folder, wLabel, countArtifact, countRounds);

        const uint64_t graphRounds = (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds);
        EvCountGraphArtifact graphArtifact = measureEvCountGraph(c, *currentPolicy, countArtifact.count, graphRounds);
        printEvCountGraphSummary(wLabel, graphArtifact);
        if (!g_no_save) {
            saveEvCountGraphArtifact(folder, wLabel, graphArtifact);
            saveCumulativeEvCountGraphArtifact(folder, static_cast<int>(state.nextPolicyIndex + 1));
        }

        state.currentCount = countArtifact.count;
        state.currentPolicyLabel.clear();
        state.currentPolicyPath.clear();
        state.currentPolicyFlatEdge = 0.0;
        state.nextPolicyIndex += 1;
        state.nextPhase = Phase::POLICY;
        currentPolicy.reset();
        if (!g_no_save) saveState(folder, state);
    }
}

void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Alternating optimization between policy learning (RL) and count learning (OLS).\n\n";

    std::cout << "METHOD:\n";
    std::cout << "  1. Start from implicit W0 = zero count.\n";
    std::cout << "  2. For each loop index k = 0,1,2,... : learn Pk from fixed Wk, then learn W{k+1} from fixed Pk.\n\n";

    std::cout << "POLICY RL STOP MODES:\n";
    std::cout << "  rounds : learn each policy Pk for exactly --num-rounds hands.\n";
    std::cout << "  diff   : learn each policy Pk in chunks of --sample-rounds hands, compare the\n";
    std::cout << "           sampled Q-table against the previous sampled table, and stop when the\n";
    std::cout << "           average absolute entry change is <= --diff-threshold.\n";
    std::cout << "           Missing Q-table entries are treated as 0 in the comparison.\n\n";

    std::cout << "CHECKPOINT STRUCTURE:\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/meta.json      Run config\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/state.json     Resume state\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/P*.json        Policy summaries\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/P*_agent.json  Learned Q-tables\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/P*_strategy.json  Greedy policy tables\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/W*.json               Count summaries\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/W*_data.json          Regression matrices\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/W*_graph.json/svg     Single EV-vs-count graph\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/W*_graph_overlay.*    Cumulative W1..Wk comparison graph\n\n";

    std::cout << "SIMULATION:\n";
    std::cout << "  --num-rounds <N>      Rounds used in each policy/count phase (default: 1000000000)\n";
    std::cout << "  --stop-mode <rounds|diff>  Policy RL stopping rule (default: rounds)\n";
    std::cout << "  --sample-rounds <N>   In diff mode, compare policy Q-tables every N rounds (default: 100000000)\n";
    std::cout << "  --diff-threshold <v>  In diff mode, stop when avg abs Q-table change <= v (default: 0.001)\n";
    std::cout << "  --eval-rounds <N>     Edge evaluation rounds after each step (default: num-rounds)\n";
    std::cout << "                        Policy steps report flat edge; count steps report spread edge and Kelly growth.\n";
    std::cout << "  --iterations <N>      Number of loop iterations k (default: 5)\n";
    std::cout << "  --num-threads <N>     Threads (default: 16)\n";
    std::cout << "  --penetration <val>   Shoe penetration % (default: 75.0)\n";
    std::cout << "  --sample-every <N>    Record every N-th round for OLS (default: 1)\n";
    std::cout << "                        The count phase plays N * num-rounds rounds so OLS still gets about num-rounds samples.\n";
    std::cout << "  --kelly-measurements <N>  Number of Kelly experiments of 1,000,000 rounds each (default: 100)\n";
    std::cout << "  --checkpoint-name <name>  Folder prefix under checkpoints/alternating-checkpoints/ (default: timestamp)\n";
    std::cout << "  --load-checkpoint <name>  Resume from checkpoints/alternating-checkpoints/<name>/\n";
    std::cout << "  --verbose             Print the final learned policy table after each policy step completes\n";
    std::cout << "  --full-verbose        Alias for --verbose\n";
    std::cout << "  --no-save             Disable artifact saving\n\n";

    std::cout << "RL PARAMETERS:\n";
    std::cout << "  --count-resolution <val>  True-count discretization step (default: 1.0)\n";
    std::cout << "  --min-count <N>           Clamp true count minimum (default: -5)\n";
    std::cout << "  --max-count <N>           Clamp true count maximum (default: 5)\n";
    std::cout << "  --exploration <mode>      epsilon or boltzmann (default: epsilon)\n";
    std::cout << "  --epsilon-start <val>     Initial epsilon (default: 1.0)\n";
    std::cout << "  --epsilon-min <val>       Minimum epsilon (default: 0.1)\n";
    std::cout << "  --epsilon-decay <val>     Per-step epsilon decay factor (default: 0.99999)\n";
    std::cout << "  --temp-start <val>        Initial Boltzmann temperature (default: 1.0)\n";
    std::cout << "  --temp-min <val>          Minimum Boltzmann temperature (default: 0.1)\n";
    std::cout << "  --temp-decay <val>        Per-step Boltzmann decay factor (default: 0.99999)\n";
    std::cout << "  --alpha-start <val>       Initial learning rate (default: 0.01)\n";
    std::cout << "  --alpha-min <val>         Minimum learning rate (default: 0.0001)\n";
    std::cout << "  --alpha-decay <N>         Steps to decay alpha to min (default: 100)\n\n";

    std::cout << "LEARNED COUNT SCALING:\n";
    std::cout << "  Learned count tags are temporarily scaled by "
              << kLearnedCountFactorMultiplier
              << ", and the EV factor is scaled inversely.\n\n";

    std::cout << "GAME CONFIG:\n";
    std::cout << "  --decks, --ss17, --das, --sas, --don, --rsa, --hsa,\n";
    std::cout << "  --peek, --surr, --bj   (same format as the existing apps)\n";
    std::cout << "  These are ignored when --load-checkpoint is used.\n\n";
}

} // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printHelp(argv[0]); return 0; }

        else if (arg == "--num-rounds"       && i + 1 < argc) g_num_rounds = std::stoull(argv[++i]);
        else if (arg == "--stop-mode"        && i + 1 < argc) {
            std::string mode = argv[++i];
            g_training_stop_mode = (mode == "diff") ? TrainingStopMode::TABLE_DIFF
                                                    : TrainingStopMode::FIXED_ROUNDS;
        }
        else if (arg == "--sample-rounds"    && i + 1 < argc) g_sample_rounds = std::stoull(argv[++i]);
        else if (arg == "--diff-threshold"   && i + 1 < argc) g_diff_threshold = std::stod(argv[++i]);
        else if (arg == "--eval-rounds"      && i + 1 < argc) g_eval_rounds = std::stoull(argv[++i]);
        else if (arg == "--iterations"       && i + 1 < argc) g_iterations = std::stoull(argv[++i]);
        else if (arg == "--num-threads"      && i + 1 < argc) g_num_threads = std::stoi(argv[++i]);
        else if (arg == "--penetration"      && i + 1 < argc) g_penetration = std::stod(argv[++i]);
        else if (arg == "--sample-every"     && i + 1 < argc) g_sample_every = std::stoull(argv[++i]);
        else if (arg == "--kelly-measurements" && i + 1 < argc) g_kelly_measurements = std::stoi(argv[++i]);
        else if (arg == "--checkpoint-name"  && i + 1 < argc) g_checkpoint_name = argv[++i];
        else if (arg == "--load-checkpoint"  && i + 1 < argc) g_load_checkpoint = argv[++i];
        else if (arg == "--verbose")                       g_verbose = true;
        else if (arg == "--full-verbose")                  { g_full_verbose = true; g_verbose = true; }
        else if (arg == "--no-save")                         g_no_save = true;

        else if (arg == "--count-resolution" && i + 1 < argc) g_agent.countResolution = std::stod(argv[++i]);
        else if (arg == "--min-count"        && i + 1 < argc) g_agent.minCount = std::stoi(argv[++i]);
        else if (arg == "--max-count"        && i + 1 < argc) g_agent.maxCount = std::stoi(argv[++i]);
        else if (arg == "--exploration"      && i + 1 < argc) {
            std::string mode = argv[++i];
            g_agent.explorationMode = (mode == "boltzmann")
                ? ExplorationMode::BOLTZMANN
                : ExplorationMode::EPSILON_GREEDY;
        }
        else if (arg == "--epsilon-start"    && i + 1 < argc) g_agent.epsilonStart = std::stod(argv[++i]);
        else if (arg == "--epsilon-min"      && i + 1 < argc) g_agent.epsilonMin = std::stod(argv[++i]);
        else if (arg == "--epsilon-decay"    && i + 1 < argc) g_agent.epsilonDecay = std::stod(argv[++i]);
        else if (arg == "--temp-start"       && i + 1 < argc) g_agent.tempStart = std::stod(argv[++i]);
        else if (arg == "--temp-min"         && i + 1 < argc) g_agent.tempMin = std::stod(argv[++i]);
        else if (arg == "--temp-decay"       && i + 1 < argc) g_agent.tempDecay = std::stod(argv[++i]);
        else if (arg == "--alpha-start"      && i + 1 < argc) g_agent.alphaStart = std::stod(argv[++i]);
        else if (arg == "--alpha-min"        && i + 1 < argc) g_agent.alphaMin = std::stod(argv[++i]);
        else if (arg == "--alpha-decay"      && i + 1 < argc) g_agent.alphaDecaySteps = std::stod(argv[++i]);

        else if (arg == "--decks" && i + 1 < argc) g_deck_sizes = parseList<int>(argv[++i]);
        else if (arg == "--ss17"  && i + 1 < argc) g_stand_soft17 = parseList<bool>(argv[++i]);
        else if (arg == "--das"   && i + 1 < argc) g_double_after_split = parseList<bool>(argv[++i]);
        else if (arg == "--sas"   && i + 1 < argc) g_split_after_split = parseList<int>(argv[++i]);
        else if (arg == "--don"   && i + 1 < argc) g_double_on = parseList<std::string>(argv[++i]);
        else if (arg == "--rsa"   && i + 1 < argc) g_resplit_aces = parseList<bool>(argv[++i]);
        else if (arg == "--hsa"   && i + 1 < argc) g_hit_split_aces = parseList<bool>(argv[++i]);
        else if (arg == "--peek"  && i + 1 < argc) g_peek = parseList<bool>(argv[++i]);
        else if (arg == "--surr"  && i + 1 < argc) g_surrender = parseList<std::string>(argv[++i]);
        else if (arg == "--bj"    && i + 1 < argc) g_blackjack_pay = parseList<float>(argv[++i]);
        else {
            std::cerr << "Unknown argument: " << arg << "\nUse --help.\n";
            return 1;
        }
    }

    std::vector<Case> cases;
    if (g_kelly_measurements < 1) {
        std::cerr << "Error: --kelly-measurements must be >= 1.\n";
        return 1;
    }
    if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF && g_sample_rounds == 0) {
        std::cerr << "Error: --sample-rounds must be >= 1 in diff mode.\n";
        return 1;
    }
    if (!g_load_checkpoint.empty()) {
        cases.push_back(loadCheckpointFolder(g_load_checkpoint));
    } else {
        cases = generateTestCases(
            g_deck_sizes, g_stand_soft17, g_double_after_split, g_split_after_split,
            g_double_on, g_resplit_aces, g_hit_split_aces, g_peek, g_surrender, g_blackjack_pay);
    }

    try {
        for (const auto& c : cases)
            runCase(c);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
