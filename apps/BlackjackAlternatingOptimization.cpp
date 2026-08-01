#include "RegressionTestUtils.h"
#include "Game/BlackjackTable.h"
#include "Game/Player.h"
#include "Game/BettingStrategy.h"
#include "Game/CountingMethods.h"
#include "RL/QLearningStrategy.h"
#include "RL/BasicStrategy.h"
#include "RL/DecayingParameter.h"
#include "Utils/RunLogger.h"
#include "Utils/SimulationAnalysis.h"
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
#include <optional>
#include <vector>
#include <unistd.h>

using json = nlohmann::json;

namespace {

constexpr std::array<double, 13> ZERO_WEIGHTS = {};
constexpr double kEps = 1e-12;
constexpr const char* kAlternatingCheckpointRoot = "checkpoints/alternating-checkpoints";
constexpr double kTargetTenValueTag = -1.0;
constexpr double kKellyInitialMoney = 1.0;
constexpr double kKellyFraction = 1.0;
constexpr uint64_t kKellyMeasurementRounds = 1'000'000ULL;
enum class TrainingStopMode { FIXED_ROUNDS, TABLE_DIFF };
enum class CountRegressionObjective {
    CLASSICAL_OLS,
    QUADRATIC_KELLY
};
enum class CountRegressionConstraint {
    NONE,
    SUM_ZERO,
    SUM_ZERO_FIXED_W1_BIAS,
    SUM_ZERO_FIXED_P0_FLAT_EDGE,
    FIXED_ZERO_BIAS
};

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
    double evaluationEdgeStddev = 0.0;
    double evaluationEdgeSecondMoment = 0.0;
    bool hasCountBettingEvaluation = false;
    double evaluationSpreadEdge = 0.0;
    double evaluationSpreadStddev = 0.0;
    double evaluationSpreadSecondMoment = 0.0;
    double evaluationKellyGrowth = 1.0;
    double evaluationKellyGrowthStddev = 0.0;
    double handsPerSec = 0.0;
    uint64_t trainingRounds = 0;
};

struct CountArtifact {
    CountConfig count;
    std::array<double, 14> rawSolution{};
    std::array<double, 13> rawNormalizedWeights{};
    std::array<std::array<double, 14>, 14> XtX{};
    std::array<double, 14> Xty{};
    uint64_t recordedRounds = 0;
    double evaluationEdge = 0.0;
    double evaluationEdgeStddev = 0.0;
    double evaluationEdgeSecondMoment = 0.0;
    double evaluationKellyGrowth = 1.0;
    double evaluationKellyGrowthStddev = 0.0;
    double evaluationKellyFraction = 1.0;
    KellyGrowthCurve kellyCurve;
    double handsPerSec = 0.0;
    double threshold = 0.0;
    double normalizationScale = 1.0;
    double referenceFlatEdge = 0.0;
    CountRegressionObjective regressionObjective = CountRegressionObjective::CLASSICAL_OLS;
    CountRegressionConstraint regressionConstraint =
        CountRegressionConstraint::SUM_ZERO_FIXED_W1_BIAS;
    std::optional<double> forcedBias;
};

struct KellyEvaluationResult {
    double growthMean = 1.0;
    double growthStddev = 0.0;
};

struct EvCountGraphPoint {
    double count = 0.0;
    uint64_t n = 0;
    double meanReward = 0.0;
    double secondMomentReward = 0.0;
    double stddevReward = 0.0;
    double confidenceLower = 0.0;
    double confidenceUpper = 0.0;
    double regressionReward = 0.0;
};

struct EvCountGraphArtifact {
    double resolution = 0.25;
    uint64_t rounds = 0;
    double handsPerSec = 0.0;
    bool hasEvRegressionLine = true;
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
    double currentPolicyFlatStddev = 0.0;
    double currentPolicyFlatSecondMoment = 0.0;
};

uint64_t    g_num_rounds          = 1'000'000'000ULL;
uint64_t    g_eval_rounds         = 0;   // 0 => use g_num_rounds
uint64_t    g_iterations          = 3;   // number of loop iterations k producing P_k from W_k, then W_{k+1}
int         g_num_threads         = 10;
double      g_penetration         = 75.0;
uint64_t    g_sample_every        = 1;
uint64_t    g_sample_rounds       = 100'000'000ULL;
double      g_diff_threshold      = 0.001;
TrainingStopMode g_training_stop_mode = TrainingStopMode::FIXED_ROUNDS;
CountRegressionObjective g_count_regression_objective =
    CountRegressionObjective::CLASSICAL_OLS;
CountRegressionConstraint g_count_regression_constraint =
    CountRegressionConstraint::SUM_ZERO_FIXED_W1_BIAS;
int         g_kelly_measurements  = 10;
std::optional<double> g_kelly_fraction_min;
std::optional<double> g_kelly_fraction_max;
double      g_kelly_fraction_step = 0.05;
bool        g_verbose             = false;
bool        g_full_verbose        = false;
bool        g_no_save             = false;
std::string g_checkpoint_name;
std::string g_load_checkpoint;
std::string g_command_line;

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

const char* countRegressionObjectiveToString(CountRegressionObjective objective) {
    switch (objective) {
        case CountRegressionObjective::CLASSICAL_OLS: return "classical_ols";
        case CountRegressionObjective::QUADRATIC_KELLY: return "quadratic_kelly";
    }
    return "classical_ols";
}

const char* countRegressionConstraintToString(CountRegressionConstraint constraint) {
    switch (constraint) {
        case CountRegressionConstraint::NONE: return "none";
        case CountRegressionConstraint::SUM_ZERO: return "sum_zero";
        case CountRegressionConstraint::SUM_ZERO_FIXED_W1_BIAS:
            return "sum_zero_fixed_w1_bias";
        case CountRegressionConstraint::SUM_ZERO_FIXED_P0_FLAT_EDGE:
            return "sum_zero_fixed_p0_flat_edge";
        case CountRegressionConstraint::FIXED_ZERO_BIAS: return "fixed_zero_bias";
    }
    return "sum_zero_fixed_w1_bias";
}

CountRegressionObjective stringToCountRegressionObjective(const std::string& value) {
    return value == "quadratic_kelly"
        ? CountRegressionObjective::QUADRATIC_KELLY
        : CountRegressionObjective::CLASSICAL_OLS;
}

CountRegressionConstraint stringToCountRegressionConstraint(const std::string& value) {
    if (value == "none") return CountRegressionConstraint::NONE;
    if (value == "sum_zero") return CountRegressionConstraint::SUM_ZERO;
    if (value == "sum_zero_fixed_p0_flat_edge")
        return CountRegressionConstraint::SUM_ZERO_FIXED_P0_FLAT_EDGE;
    if (value == "fixed_zero_bias") return CountRegressionConstraint::FIXED_ZERO_BIAS;
    return CountRegressionConstraint::SUM_ZERO_FIXED_W1_BIAS;
}

std::string currentTimestamp() {
    time_t t = time(nullptr);
    char buf[20];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&t));
    return buf;
}

std::string checkpointTableName(const Case& c) {
    // Strategy-table lookup caps large shoes at four decks. Checkpoint identity
    // must retain the actual simulation deck count so research runs cannot mix.
    std::string name = ToStringTableName(c);
    const size_t firstSeparator = name.find('_');
    return "decks=" + std::to_string(c.deckSize) +
        (firstSeparator == std::string::npos ? std::string{} : name.substr(firstSeparator));
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
    os << "Command:       " << g_command_line << "\n";
    os << "Scenario:      " << ToString(c)
       << "_penetration=" << g_penetration << "%\n";
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
    os << "Count objective: "
       << countRegressionObjectiveToString(g_count_regression_objective) << "\n";
    os << "Count constraint: "
       << countRegressionConstraintToString(g_count_regression_constraint) << "\n";
    os << "Penetration:   " << g_penetration << "%\n";
    os << "Sample-every:  " << g_sample_every << "\n";
    os << "Kelly sweep:  centered at "
       << (g_count_regression_objective == CountRegressionObjective::QUADRATIC_KELLY
               ? "fitted-signal multiplier 1.0"
               : "nearest step to 1/E[X^2]")
       << " +/- 0.25 (minimum 0)";
    if (g_kelly_fraction_min) os << "  min override=" << *g_kelly_fraction_min;
    if (g_kelly_fraction_max) os << "  max override=" << *g_kelly_fraction_max;
    os << "  step " << g_kelly_fraction_step << "\n";
    os << "Folder:        " << kAlternatingCheckpointRoot << "/" << folder << "/\n";
    os << "Initial count: none (all zeros)\n";
    os << "Count scaling:  average 10/J/Q/K tag = " << kTargetTenValueTag << "\n";
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
    meta["algorithm_config"]["kelly_fraction_range_mode"] =
        g_count_regression_objective == CountRegressionObjective::QUADRATIC_KELLY
            ? "direct_fraction_multiplier_one_plus_minus_0.25"
            : "rounded_predicted_1_over_ex2_plus_minus_0.25";
    meta["algorithm_config"]["kelly_fraction_min"] = g_kelly_fraction_min
        ? json(*g_kelly_fraction_min) : json(nullptr);
    meta["algorithm_config"]["kelly_fraction_max"] = g_kelly_fraction_max
        ? json(*g_kelly_fraction_max) : json(nullptr);
    meta["algorithm_config"]["kelly_fraction_step"] = g_kelly_fraction_step;
    meta["algorithm_config"]["kelly_measurements"] = g_kelly_measurements;
    meta["algorithm_config"]["initial_count"]["weights"] = ZERO_WEIGHTS;
    meta["algorithm_config"]["initial_count"]["factor"]  = 1.0;
    meta["algorithm_config"]["initial_count"]["bias"]    = 0.0;
    meta["algorithm_config"]["learned_count_normalization"] = "ten_value_average";
    meta["algorithm_config"]["learned_count_target_ten_value_tag"] = kTargetTenValueTag;
    meta["algorithm_config"]["count_regression_objective"] =
        countRegressionObjectiveToString(g_count_regression_objective);
    meta["algorithm_config"]["count_regression_constraint"] =
        countRegressionConstraintToString(g_count_regression_constraint);
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

EdgeStatistics evaluateEdge(const Case& c,
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
    player->enableRoundStats();

    BlackjackRules rules = buildRules(c, minBet, maxBet);
    std::vector<Player*> results = runParallelSimulation(rules, {player}, rounds, g_num_threads);

    delete player;
    if (results.empty())
        throw std::runtime_error("Edge evaluation failed: no result players");

    Player* result = results[0];
    EdgeStatistics edge = edgeStatisticsFromPlayer(*result);
    delete result;
    return edge;
}

KellyEvaluationResult evaluateKellyGrowth(const Case& c,
                                          const BasicStrategy& strategy,
                                          const CountConfig& count,
                                          uint64_t rounds,
                                          double initialMoney = kKellyInitialMoney,
                                          double kellyFraction = kKellyFraction) {
    (void)rounds;
    BlackjackRules rules = buildRules(c, 0.0, std::numeric_limits<double>::max());
    double growthSum = 0.0;
    double growthSqSum = 0.0;

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
        growthSqSum += growthRate * growthRate;
        delete result;
    }

    KellyEvaluationResult result;
    const double n = static_cast<double>(g_kelly_measurements);
    result.growthMean = growthSum / n;
    if (g_kelly_measurements > 1) {
        double sampleVariance = (growthSqSum - (growthSum * growthSum / n)) / (n - 1.0);
        result.growthStddev = std::sqrt(std::max(0.0, sampleVariance));
    }
    return result;
}

KellyGrowthCurve evaluateKellyGrowthCurve(const std::string& label,
                                          const Case& c,
                                          const BasicStrategy& strategy,
                                          const CountConfig& count,
                                          double predictedOptimalFraction) {
    KellyGrowthCurve curve;
    curve.label = label;
    curve.predictedOptimalFraction = predictedOptimalFraction;
    const KellyFractionRange range = resolveKellyFractionRange(
        predictedOptimalFraction, g_kelly_fraction_min, g_kelly_fraction_max,
        g_kelly_fraction_step);
    for (double fraction : makeKellyFractionGrid(
             range.minimum, range.maximum, g_kelly_fraction_step)) {
        const KellyEvaluationResult result = evaluateKellyGrowth(
            c, strategy, count, kKellyMeasurementRounds, kKellyInitialMoney, fraction);
        curve.points.push_back({fraction, result.growthMean, result.growthStddev});
    }
    return curve;
}

// Let theta=[w,b], with 13 rank weights w and intercept b. Both objectives are
// quadratic forms 0.5 theta' A theta - d' theta, represented by the same
// streaming arrays:
//   OLS:             A=E[cc'],       d=E[Xc]
//   Quadratic Kelly: A=E[X^2 cc'],   d=E[Xc]
// The latter follows log(1+theta'c X) ~= theta'c X - 0.5(theta'c X)^2.
// Without constraints, either objective solves A theta=d.
std::array<double, 14> solveUnconstrainedNormalEquation(
        const std::array<std::array<double, 14>, 14>& A,
        const std::array<double, 14>& d) {
    std::array<std::array<double, 15>, 14> M{};
    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 14; ++j) M[i][j] = A[i][j];
        M[i][14] = d[i];
    }

    for (int col = 0; col < 14; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 14; ++row)
            if (std::abs(M[row][col]) > std::abs(M[pivot][col]))
                pivot = row;
        if (pivot != col) std::swap(M[col], M[pivot]);

        double diag = M[col][col];
        if (std::abs(diag) < 1e-15)
            throw std::runtime_error("Singular unconstrained normal-equation system");

        for (int j = col; j <= 14; ++j) M[col][j] /= diag;
        for (int row = 0; row < 14; ++row) {
            if (row == col) continue;
            double f = M[row][col];
            for (int j = col; j <= 14; ++j)
                M[row][j] -= f * M[col][j];
        }
    }

    std::array<double, 14> result{};
    for (int i = 0; i < 14; ++i) result[i] = M[i][14];
    return result;
}

// For either objective, sum-zero with free bias adds q'theta=0 where
// q=[1,...,1,0]. The KKT system is [A q; q' 0][theta;lambda]=[d;0].
std::array<double, 14> solveSumZeroNormalEquation(
        const std::array<std::array<double, 14>, 14>& A,
        const std::array<double, 14>& b) {
    // Solve the KKT system [X'X c; c' 0][w; lambda] = [X'y; 0].
    // c constrains the 13 rank weights to sum to zero and excludes the bias.
    std::array<std::array<double, 16>, 15> M{};
    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 14; ++j) M[i][j] = A[i][j];
        M[i][14] = (i < 13) ? 1.0 : 0.0;
        M[i][15] = b[i];
    }
    for (int j = 0; j < 13; ++j) M[14][j] = 1.0;

    for (int col = 0; col < 15; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 15; ++row)
            if (std::abs(M[row][col]) > std::abs(M[pivot][col]))
                pivot = row;
        if (pivot != col) std::swap(M[col], M[pivot]);

        double diag = M[col][col];
        if (std::abs(diag) < 1e-15)
            throw std::runtime_error("Singular sum-zero KKT system - not enough variation in the data");

        for (int j = col; j <= 15; ++j) M[col][j] /= diag;
        for (int row = 0; row < 15; ++row) {
            if (row == col) continue;
            double f = M[row][col];
            for (int j = col; j <= 15; ++j)
                M[row][j] -= f * M[col][j];
        }
    }

    std::array<double, 14> w;
    for (int i = 0; i < 14; ++i) w[i] = M[i][15];
    return w;
}

// Fixing b to b0 reduces either objective to the rank-weight system with
// adjusted right side d_w-A_wb*b0. Adding 1'w=0 gives
// [A_ww 1; 1' 0][w;lambda]=[d_w-A_wb*b0;0]. This uses only streamed A,d.
std::array<double, 14> solveSumZeroFixedBiasNormalEquation(
        const std::array<std::array<double, 14>, 14>& XtX,
        const std::array<double, 14>& Xty,
        double fixedBias) {
    // With X = [Z 1] and fixed bias b, solve
    // min ||Z w - (Y - 1 b)||^2 subject to 1' w = 0.
    // Z'1 is already stored in the intercept column of X'X.
    std::array<std::array<double, 15>, 14> M{};
    for (int i = 0; i < 13; ++i) {
        for (int j = 0; j < 13; ++j) M[i][j] = XtX[i][j];
        M[i][13] = 1.0;
        M[i][14] = Xty[i] - fixedBias * XtX[i][13];
    }
    for (int j = 0; j < 13; ++j) M[13][j] = 1.0;

    for (int col = 0; col < 14; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 14; ++row)
            if (std::abs(M[row][col]) > std::abs(M[pivot][col]))
                pivot = row;
        if (pivot != col) std::swap(M[col], M[pivot]);

        double diag = M[col][col];
        if (std::abs(diag) < 1e-15)
            throw std::runtime_error("Singular fixed-bias sum-zero KKT system");

        for (int j = col; j <= 14; ++j) M[col][j] /= diag;
        for (int row = 0; row < 14; ++row) {
            if (row == col) continue;
            double f = M[row][col];
            for (int j = col; j <= 14; ++j)
                M[row][j] -= f * M[col][j];
        }
    }

    std::array<double, 14> result{};
    for (int i = 0; i < 13; ++i) result[i] = M[i][14];
    result[13] = fixedBias;
    return result;
}

CountArtifact normalizeCount(const std::array<double, 14>& raw) {
    CountArtifact artifact;
    artifact.rawSolution = raw;

    const double normalizationScale =
        learnedCountNormalizationScale(raw, kTargetTenValueTag);

    for (int i = 0; i < 13; ++i) {
        artifact.rawNormalizedWeights[i] = raw[i] * normalizationScale;
        artifact.count.system.weights[i] = artifact.rawNormalizedWeights[i];
    }

    // By symmetry, 10/J/Q/K should share the same tag. Keep the original fitted
    // values for reporting, but use their average in the learned count.
    double tenValueAverage =
        (artifact.rawNormalizedWeights[8] + artifact.rawNormalizedWeights[9] +
         artifact.rawNormalizedWeights[10] + artifact.rawNormalizedWeights[11]) / 4.0;
    artifact.count.system.weights[8] = tenValueAverage;
    artifact.count.system.weights[9] = tenValueAverage;
    artifact.count.system.weights[10] = tenValueAverage;
    artifact.count.system.weights[11] = tenValueAverage;

    artifact.count.system.factor = 1.0 / normalizationScale;
    artifact.count.system.bias   = raw[13];
    artifact.count.system.continuousBettingCount =
        g_count_regression_objective == CountRegressionObjective::QUADRATIC_KELLY;
    artifact.count.resolution    = g_agent.countResolution;
    artifact.count.minCount      = g_agent.minCount;
    artifact.count.maxCount      = g_agent.maxCount;
    artifact.normalizationScale  = normalizationScale;

    return artifact;
}

double spreadThresholdFromCount(const CountConfig& count) {
    if (std::abs(count.system.factor) < kEps)
        return std::numeric_limits<double>::infinity();
    const double threshold = -count.system.bias / count.system.factor;
    return count.system.continuousBettingCount
        ? std::nextafter(threshold, std::numeric_limits<double>::infinity())
        : threshold;
}

json countConfigToJson(const CountConfig& count, bool roundWeights = true) {
    json j;
    json weights = json::array();
    for (double w : count.system.weights)
        weights.push_back(roundWeights ? roundTo(w, 4) : w);
    j["weights"] = weights;
    j["factor"] = count.system.factor;
    j["bias"] = count.system.bias;
    j["continuous_betting_count"] = count.system.continuousBettingCount;
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
    j["has_ev_regression_line"] = graph.hasEvRegressionLine;
    j["min_count"] = graph.minCount;
    j["max_count"] = graph.maxCount;
    j["points"] = json::array();
    for (const auto& p : graph.points) {
        json row;
        row["count"] = p.count;
        row["n"] = p.n;
        if (p.n > 0) row["mean_reward"] = p.meanReward;
        else         row["mean_reward"] = nullptr;
        if (p.n > 0) row["conditional_second_moment"] = p.secondMomentReward;
        else         row["conditional_second_moment"] = nullptr;
        row["stddev_reward"] = p.stddevReward;
        if (p.n > 0) {
            row["confidence_lower"] = p.confidenceLower;
            row["confidence_upper"] = p.confidenceUpper;
        } else {
            row["confidence_lower"] = nullptr;
            row["confidence_upper"] = nullptr;
        }
        if (graph.hasEvRegressionLine) {
            row["regression_reward"] = p.regressionReward;
        } else {
            row["regression_reward"] = nullptr;
            row["fitted_bet_fraction"] = p.regressionReward;
        }
        j["points"].push_back(row);
    }
    return j;
}

json countHistogramToJson(const EvCountGraphArtifact& graph) {
    json j;
    j["resolution"] = graph.resolution;
    j["min_count"] = graph.minCount;
    j["max_count"] = graph.maxCount;
    j["points"] = json::array();
    for (const auto& p : graph.points) {
        json row;
        row["count"] = p.count;
        row["n"] = p.n;
        j["points"].push_back(row);
    }
    return j;
}

EvCountGraphArtifact evCountGraphFromJson(const json& j) {
    EvCountGraphArtifact graph;
    graph.resolution = j.value("resolution", 0.25);
    graph.rounds = j.value("rounds", 0ULL);
    graph.handsPerSec = j.value("hands_per_sec", 0.0);
    graph.hasEvRegressionLine = j.value("has_ev_regression_line", true);
    graph.minCount = j.value("min_count", 0.0);
    graph.maxCount = j.value("max_count", 0.0);
    for (const auto& row : j.at("points")) {
        EvCountGraphPoint point;
        point.count = row.at("count").get<double>();
        point.n = row.value("n", 0ULL);
        if (!row.at("mean_reward").is_null())
            point.meanReward = row.at("mean_reward").get<double>();
        point.stddevReward = row.value("stddev_reward", 0.0);
        if (row.contains("conditional_second_moment") &&
            !row.at("conditional_second_moment").is_null()) {
            point.secondMomentReward =
                row.at("conditional_second_moment").get<double>();
        } else {
            point.secondMomentReward =
                point.stddevReward * point.stddevReward +
                point.meanReward * point.meanReward;
        }
        if (row.contains("confidence_lower") && !row.at("confidence_lower").is_null())
            point.confidenceLower = row.at("confidence_lower").get<double>();
        else
            point.confidenceLower = point.meanReward;
        if (row.contains("confidence_upper") && !row.at("confidence_upper").is_null())
            point.confidenceUpper = row.at("confidence_upper").get<double>();
        else
            point.confidenceUpper = point.meanReward;
        if (row.contains("regression_reward") && row.at("regression_reward").is_number())
            point.regressionReward = row.at("regression_reward").get<double>();
        else
            point.regressionReward = row.value("fitted_bet_fraction", 0.0);
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

ConditionalSecondMomentCurve conditionalSecondMomentCurve(
    const std::string& label,
    const EvCountGraphArtifact& graph) {
    ConditionalSecondMomentCurve curve;
    curve.label = label;
    curve.points.reserve(graph.points.size());
    for (const auto& point : graph.points) {
        curve.points.push_back({
            point.count,
            point.n,
            point.secondMomentReward
        });
    }
    return curve;
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
        if (graph.hasEvRegressionLine) {
            minY = std::min(minY, p.regressionReward);
            maxY = std::max(maxY, p.regressionReward);
        }
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
        if (graph.hasEvRegressionLine) {
            const double regressionY = mapY(p.regressionReward);
            regressionPath << (regressionStarted ? " L " : "M ")
                           << x << " " << regressionY;
            regressionStarted = true;
        }

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
    if (graph.hasEvRegressionLine) {
        svg << "<line x1=\"" << (width - 220) << "\" y1=\"56\" x2=\"" << (width - 180)
            << "\" y2=\"56\" stroke=\"#d62728\" stroke-width=\"2\"/>\n";
        svg << "<text x=\"" << (width - 170) << "\" y=\"60\" font-family=\"sans-serif\" font-size=\"12\">Regression line</text>\n";
    }
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
            if (named.graph.hasEvRegressionLine) {
                minY = std::min(minY, p.regressionReward);
                maxY = std::max(maxY, p.regressionReward);
            }
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
            if (named.graph.hasEvRegressionLine) {
                regressionPath << (regressionStarted ? " L " : "M ")
                               << x << " " << mapY(p.regressionReward);
                regressionStarted = true;
            }
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

std::string countHistogramToSvg(const std::string& label,
                                const EvCountGraphArtifact& graph) {
    const double width = 1000.0;
    const double height = 560.0;
    const double left = 90.0;
    const double right = 30.0;
    const double top = 40.0;
    const double bottom = 70.0;
    const double plotWidth = width - left - right;
    const double plotHeight = height - top - bottom;

    double minX = graph.minCount;
    double maxX = graph.maxCount;
    if (std::abs(maxX - minX) < kEps) maxX = minX + graph.resolution;

    uint64_t maxN = 1;
    for (const auto& p : graph.points) maxN = std::max(maxN, p.n);

    auto mapX = [&](double x) {
        return left + ((x - minX) / (maxX - minX)) * plotWidth;
    };
    auto mapY = [&](double n) {
        return top + (1.0 - (n / static_cast<double>(maxN))) * plotHeight;
    };

    const double barWidth = std::max(2.0, plotWidth / std::max<size_t>(1, graph.points.size()) * 0.75);

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    svg << "<text x=\"" << left << "\" y=\"24\" font-family=\"sans-serif\" font-size=\"20\">"
        << label << " count histogram</text>\n";
    svg << "<line x1=\"" << left << "\" y1=\"" << (top + plotHeight) << "\" x2=\"" << (left + plotWidth)
        << "\" y2=\"" << (top + plotHeight) << "\" stroke=\"black\" stroke-width=\"1.2\"/>\n";
    svg << "<line x1=\"" << left << "\" y1=\"" << top << "\" x2=\"" << left
        << "\" y2=\"" << (top + plotHeight) << "\" stroke=\"black\" stroke-width=\"1.2\"/>\n";

    for (int i = 0; i <= 12; ++i) {
        double yValue = static_cast<double>(maxN) * (static_cast<double>(i) / 12.0);
        double y = mapY(yValue);
        svg << "<line x1=\"" << left << "\" y1=\"" << y
            << "\" x2=\"" << (left + plotWidth) << "\" y2=\"" << y
            << "\" stroke=\"#e1e1e1\" stroke-dasharray=\"2,3\"/>\n";
        if (i % 2 == 0) {
            svg << "<text x=\"" << (left - 10) << "\" y=\"" << (y + 4)
                << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">"
                << static_cast<uint64_t>(std::llround(yValue)) << "</text>\n";
        }
    }

    for (int i = 0; i <= 16; ++i) {
        double xValue = minX + (maxX - minX) * (static_cast<double>(i) / 16.0);
        double x = mapX(xValue);
        svg << "<line x1=\"" << x << "\" y1=\"" << top
            << "\" x2=\"" << x << "\" y2=\"" << (top + plotHeight)
            << "\" stroke=\"#ececec\" stroke-dasharray=\"2,3\"/>\n";
        if (i % 2 == 0) {
            svg << "<text x=\"" << x << "\" y=\"" << (top + plotHeight + 24)
                << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">"
                << std::fixed << std::setprecision(2) << xValue << "</text>\n";
        }
    }

    for (const auto& p : graph.points) {
        double x = mapX(p.count);
        double y = mapY(static_cast<double>(p.n));
        svg << "<rect x=\"" << (x - barWidth / 2.0) << "\" y=\"" << y
            << "\" width=\"" << barWidth << "\" height=\"" << ((top + plotHeight) - y)
            << "\" fill=\"#4c78a8\" fill-opacity=\"0.85\"/>\n";
    }

    if (minX <= 0.0 && maxX >= 0.0) {
        double zeroX = mapX(0.0);
        svg << "<line x1=\"" << zeroX << "\" y1=\"" << top
            << "\" x2=\"" << zeroX << "\" y2=\"" << (top + plotHeight)
            << "\" stroke=\"#666666\" stroke-width=\"1.8\"/>\n";
    }

    svg << "<text x=\"" << (left + plotWidth / 2.0) << "\" y=\"" << (height - 18)
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Count</text>\n";
    svg << "<text x=\"20\" y=\"" << (top + plotHeight / 2.0)
        << "\" transform=\"rotate(-90 20 " << (top + plotHeight / 2.0)
        << ")\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Encountered rounds</text>\n";
    svg << "</svg>\n";
    return svg.str();
}

CountConfig countConfigFromJson(const json& j) {
    CountConfig count;
    for (size_t i = 0; i < count.system.weights.size(); ++i)
        count.system.weights[i] = j.at("weights").at(i).get<double>();
    count.system.factor = j.value("factor", 1.0);
    count.system.bias = j.value("bias", 0.0);
    count.system.continuousBettingCount =
        j.value("continuous_betting_count", false);
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
    j["current_policy_flat_stddev"] = state.currentPolicyFlatStddev;
    j["current_policy_flat_second_moment"] = state.currentPolicyFlatSecondMoment;

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
            state.currentPolicyFlatStddev = pj.value("evaluation_edge_flat_stddev", 0.0);
            state.currentPolicyFlatSecondMoment =
                pj.value("evaluation_edge_flat_second_moment", 0.0);
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
    state.currentPolicyFlatStddev = j.value("current_policy_flat_stddev", 0.0);
    state.currentPolicyFlatSecondMoment =
        j.value("current_policy_flat_second_moment", 0.0);
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
    if (a.contains("kelly_fraction_range_mode")) {
        g_kelly_fraction_min = a.contains("kelly_fraction_min") &&
                a.at("kelly_fraction_min").is_number()
            ? std::optional<double>(a.at("kelly_fraction_min").get<double>())
            : std::nullopt;
        g_kelly_fraction_max = a.contains("kelly_fraction_max") &&
                a.at("kelly_fraction_max").is_number()
            ? std::optional<double>(a.at("kelly_fraction_max").get<double>())
            : std::nullopt;
    } else {
        // Legacy checkpoints stored the old fixed defaults; preserve them on resume.
        g_kelly_fraction_min = a.value("kelly_fraction_min", 0.65);
        g_kelly_fraction_max = a.value("kelly_fraction_max", 1.0);
    }
    g_kelly_fraction_step = a.value("kelly_fraction_step", 0.05);
    g_kelly_measurements = a.value("kelly_measurements", 10);
    if (g_kelly_measurements < 1)
        throw std::runtime_error("Checkpoint kelly_measurements must be >= 1");
    (void)makeKellyFractionGrid(0.0, 0.0, g_kelly_fraction_step);
    if (g_kelly_fraction_min && g_kelly_fraction_max) {
        (void)makeKellyFractionGrid(
            *g_kelly_fraction_min, *g_kelly_fraction_max, g_kelly_fraction_step);
    }
    if (a.contains("count_regression_objective")) {
        g_count_regression_objective = stringToCountRegressionObjective(
            a.at("count_regression_objective").get<std::string>());
        g_count_regression_constraint = stringToCountRegressionConstraint(
            a.value("count_regression_constraint", "sum_zero_fixed_w1_bias"));
    } else if (a.contains("count_regression_mode")) {
        const std::string legacyMode = a.at("count_regression_mode").get<std::string>();
        if (legacyMode == "quadratic_kelly") {
            g_count_regression_objective = CountRegressionObjective::QUADRATIC_KELLY;
            g_count_regression_constraint = CountRegressionConstraint::NONE;
        } else {
            g_count_regression_objective = CountRegressionObjective::CLASSICAL_OLS;
            g_count_regression_constraint =
                stringToCountRegressionConstraint(legacyMode == "classical_ols"
                    ? "none" : legacyMode);
        }
    } else {
        const std::string legacyConstraint =
            a.value("count_regression_constraint", std::string{});
        if (legacyConstraint.find("W2+") != std::string::npos) {
            g_count_regression_constraint =
                CountRegressionConstraint::SUM_ZERO_FIXED_W1_BIAS;
        } else if (legacyConstraint.find("sum") != std::string::npos) {
            g_count_regression_constraint = CountRegressionConstraint::SUM_ZERO;
        } else {
            g_count_regression_constraint = CountRegressionConstraint::NONE;
        }
        g_count_regression_objective = CountRegressionObjective::CLASSICAL_OLS;
    }

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
    std::cout << "Count objective: "
              << countRegressionObjectiveToString(g_count_regression_objective) << "\n";
    std::cout << "Count constraint: "
              << countRegressionConstraintToString(g_count_regression_constraint) << "\n";
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
    summary["evaluation_edge_flat_stddev"] = artifact.evaluationEdgeStddev;
    summary["evaluation_edge_flat_second_moment"] = artifact.evaluationEdgeSecondMoment;
    if (artifact.hasCountBettingEvaluation) {
        summary["evaluation_edge_spread_1_10"] = artifact.evaluationSpreadEdge;
        summary["evaluation_edge_spread_1_10_stddev"] = artifact.evaluationSpreadStddev;
        summary["evaluation_edge_spread_1_10_second_moment"] =
            artifact.evaluationSpreadSecondMoment;
        summary["evaluation_kelly_growth"] = artifact.evaluationKellyGrowth;
        summary["evaluation_kelly_growth_stddev"] = artifact.evaluationKellyGrowthStddev;
    } else {
        summary["evaluation_edge_spread_1_10"] = nullptr;
        summary["evaluation_edge_spread_1_10_stddev"] = nullptr;
        summary["evaluation_edge_spread_1_10_second_moment"] = nullptr;
        summary["evaluation_kelly_growth"] = nullptr;
        summary["evaluation_kelly_growth_stddev"] = nullptr;
    }
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
    summary["evaluation_edge_spread_1_10_stddev"] = artifact.evaluationEdgeStddev;
    summary["evaluation_edge_spread_1_10_second_moment"] =
        artifact.evaluationEdgeSecondMoment;
    summary["evaluation_kelly_growth"] = artifact.evaluationKellyGrowth;
    summary["evaluation_kelly_growth_stddev"] = artifact.evaluationKellyGrowthStddev;
    summary["evaluation_kelly_fraction"] = artifact.evaluationKellyFraction;
    summary["kelly_sweep_center"] = artifact.kellyCurve.predictedOptimalFraction;
    if (artifact.regressionObjective != CountRegressionObjective::QUADRATIC_KELLY) {
        summary["predicted_kelly_fraction_1_over_ex2"] =
            artifact.kellyCurve.predictedOptimalFraction;
    }
    summary["normalization_scale"] = artifact.normalizationScale;
    summary["learned_count_target_ten_value_tag"] = kTargetTenValueTag;
    summary["regression_objective_type"] =
        countRegressionObjectiveToString(artifact.regressionObjective);
    summary["regression_mode"] =
        countRegressionObjectiveToString(artifact.regressionObjective);
    summary["regression_constraint"] =
        countRegressionConstraintToString(artifact.regressionConstraint);
    summary["regression_objective"] =
        artifact.regressionObjective == CountRegressionObjective::QUADRATIC_KELLY
            ? "maximize_second_order_expected_log_growth"
            : "minimize_expected_squared_error";
    std::string biasConstraint = "free";
    if (artifact.forcedBias.has_value()) {
        biasConstraint =
            artifact.regressionConstraint ==
                    CountRegressionConstraint::SUM_ZERO_FIXED_P0_FLAT_EDGE
                ? "fixed_to_P0_flat_edge"
                : "fixed_to_W1_bias";
    }
    summary["bias_constraint"] = biasConstraint;
    if (artifact.forcedBias.has_value())
        summary["forced_bias"] = *artifact.forcedBias;
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
    data["regression_objective"] =
        artifact.regressionObjective == CountRegressionObjective::QUADRATIC_KELLY
            ? "quadratic_kelly"
            : "expected_value_ols";
    data["regression_constraint"] =
        countRegressionConstraintToString(artifact.regressionConstraint);
    data["bias_constraint"] = biasConstraint;
    data["recorded_rounds"] = artifact.recordedRounds;
    std::ofstream df(root / (label + "_data.json"));
    if (df.is_open()) df << data.dump(2);
}

void saveEvCountGraphArtifact(const std::filesystem::path& root,
                              const std::string& label,
                              const EvCountGraphArtifact& graph) {
    namespace fs = std::filesystem;
    fs::create_directories(root);
    std::ofstream f(root / (label + "_graph.json"));
    if (f.is_open()) f << evCountGraphToJson(graph).dump(2);
    std::ofstream svg(root / (label + "_graph.svg"));
    if (svg.is_open()) svg << evCountGraphToSvg(label, graph);
    std::ofstream hf(root / (label + "_histogram.json"));
    if (hf.is_open()) hf << countHistogramToJson(graph).dump(2);
    std::ofstream hsv(root / (label + "_histogram.svg"));
    if (hsv.is_open()) hsv << countHistogramToSvg(label, graph);
    const std::vector<ConditionalSecondMomentCurve> secondMomentCurves{
        conditionalSecondMomentCurve(label, graph)
    };
    std::ofstream(root / (label + "_second_moment_graph.json"))
        << std::setw(2) << conditionalSecondMomentCurvesToJson(secondMomentCurves) << "\n";
    std::ofstream(root / (label + "_second_moment_graph.svg"))
        << conditionalSecondMomentCurvesToSvg(
               label + " conditional second moment", secondMomentCurves);
}

void saveCumulativeEvCountGraphArtifact(const std::filesystem::path& root,
                                        int upToWeightIndex) {
    namespace fs = std::filesystem;

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

    std::vector<ConditionalSecondMomentCurve> secondMomentCurves;
    secondMomentCurves.reserve(graphs.size());
    for (const auto& named : graphs) {
        secondMomentCurves.push_back(
            conditionalSecondMomentCurve(named.label, named.graph));
    }
    std::ofstream(root / ("W" + std::to_string(upToWeightIndex) +
                          "_second_moment_graph_overlay.json"))
        << std::setw(2)
        << conditionalSecondMomentCurvesToJson(secondMomentCurves) << "\n";
    std::ofstream(root / ("W" + std::to_string(upToWeightIndex) +
                          "_second_moment_graph_overlay.svg"))
        << conditionalSecondMomentCurvesToSvg(
               "W1..W" + std::to_string(upToWeightIndex) +
                   " conditional second moment",
               secondMomentCurves);
}

void saveKellyGrowthGraphArtifact(const std::filesystem::path& root,
                                  const std::string& label,
                                  const KellyGrowthCurve& curve) {
    const std::vector<KellyGrowthCurve> curves{curve};
    std::ofstream(root / (label + "_kelly_graph.json"))
        << std::setw(2) << kellyGrowthCurvesToJson(curves) << "\n";
    std::ofstream(root / (label + "_kelly_graph.svg"))
        << kellyGrowthCurvesToSvg(label + " Kelly growth", curves);
}

void saveCumulativeKellyGrowthGraphArtifact(const std::filesystem::path& root,
                                            int upToWeightIndex) {
    namespace fs = std::filesystem;
    std::vector<KellyGrowthCurve> curves;
    for (int i = 1; i <= upToWeightIndex; ++i) {
        const fs::path path = root / ("W" + std::to_string(i) + "_kelly_graph.json");
        if (!fs::exists(path)) continue;
        std::ifstream input(path);
        if (!input.is_open()) continue;
        json value;
        input >> value;
        curves.push_back(kellyGrowthCurveFromJson(value));
    }
    if (curves.empty()) return;

    const std::string prefix = "W" + std::to_string(upToWeightIndex) + "_kelly_graph_overlay";
    std::ofstream(root / (prefix + ".json"))
        << std::setw(2) << kellyGrowthCurvesToJson(curves) << "\n";
    std::ofstream(root / (prefix + ".svg"))
        << kellyGrowthCurvesToSvg(
               "W1..W" + std::to_string(upToWeightIndex) + " Kelly growth", curves);
}

PolicyArtifact learnPolicy(const Case& c,
                           const CountConfig& count,
                           uint64_t rounds,
                           const std::string& label,
                           const std::string& runHeader,
                           bool evaluateCountBetting) {
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
    const EdgeStatistics flat = evaluateEdge(
        c, *policy, count, nullptr, 1.0, 1.0,
        (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds));
    artifact.evaluationEdge = flat.mean;
    artifact.evaluationEdgeStddev = flat.stddev;
    artifact.evaluationEdgeSecondMoment = flat.secondMoment;
    if (evaluateCountBetting) {
        const double threshold = spreadThresholdFromCount(count);
        auto betting = std::make_unique<SpreadBetting>(
            std::vector<std::pair<double, double>>{{threshold, 10.0}});
        const EdgeStatistics spread = evaluateEdge(
            c, *policy, count, std::move(betting), 1.0, 10.0,
            (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds));
        artifact.evaluationSpreadEdge = spread.mean;
        artifact.evaluationSpreadStddev = spread.stddev;
        artifact.evaluationSpreadSecondMoment = spread.secondMoment;
        KellyEvaluationResult kelly = evaluateKellyGrowth(
            c, *policy, count, (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds));
        artifact.evaluationKellyGrowth = kelly.growthMean;
        artifact.evaluationKellyGrowthStddev = kelly.growthStddev;
        artifact.hasCountBettingEvaluation = true;
    }
    artifact.qStrategy = std::move(clonedQ);
    artifact.policy = std::move(policy);

    delete result;
    return artifact;
}

CountArtifact learnCount(const Case& c,
                         const std::string& label,
                         const BasicStrategy& fixedPolicy,
                         const CountConfig& strategyCount,
                         uint64_t rounds,
                         std::optional<double> forcedBias,
                         double predictedKellyFraction) {
    auto cloned = fixedPolicy.clone();
    auto* player = new Player(0.0, std::move(cloned));
    player->setNumDecks(c.deckSize);
    player->setCountSystem(strategyCount.system);
    player->setCountResolution(strategyCount.resolution);
    player->setCountRange(strategyCount.minCount, strategyCount.maxCount);
    player->enableRegression(
        g_count_regression_objective == CountRegressionObjective::QUADRATIC_KELLY
            ? RegressionObjective::QUADRATIC_KELLY
            : RegressionObjective::EXPECTED_VALUE_OLS);
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
    std::array<double, 14> rawSolution{};
    switch (g_count_regression_constraint) {
        case CountRegressionConstraint::NONE:
            rawSolution =
                g_count_regression_objective == CountRegressionObjective::QUADRATIC_KELLY
                    ? solveQuadraticKellyRegression(result->getXtX(), result->getXty())
                    : solveUnconstrainedNormalEquation(
                          result->getXtX(), result->getXty());
            break;
        case CountRegressionConstraint::SUM_ZERO:
            rawSolution = solveSumZeroNormalEquation(
                result->getXtX(), result->getXty());
            break;
        case CountRegressionConstraint::SUM_ZERO_FIXED_W1_BIAS:
        case CountRegressionConstraint::SUM_ZERO_FIXED_P0_FLAT_EDGE:
            rawSolution = forcedBias.has_value()
                ? solveSumZeroFixedBiasNormalEquation(
                      result->getXtX(), result->getXty(), *forcedBias)
                : solveSumZeroNormalEquation(result->getXtX(), result->getXty());
            break;
        case CountRegressionConstraint::FIXED_ZERO_BIAS:
            rawSolution = solveQuadraticKellyRegressionWithFixedBias(
                result->getXtX(), result->getXty(), 0.0);
            break;
    }
    CountArtifact artifact = normalizeCount(rawSolution);
    artifact.regressionObjective = g_count_regression_objective;
    artifact.regressionConstraint = g_count_regression_constraint;
    artifact.forcedBias = forcedBias;
    artifact.XtX = result->getXtX();
    artifact.Xty = result->getXty();
    artifact.recordedRounds = result->getRegressionRounds();
    artifact.handsPerSec = (secs > 0.0 ? rounds / secs : 0.0);
    artifact.threshold = spreadThresholdFromCount(artifact.count);

    auto betting = std::make_unique<SpreadBetting>(
        std::vector<std::pair<double, double>>{{artifact.threshold, 10.0}});
    const EdgeStatistics spread = evaluateEdge(
        c, fixedPolicy, artifact.count, std::move(betting), 1.0, 10.0,
        (g_eval_rounds == 0 ? rounds : g_eval_rounds));
    artifact.evaluationEdge = spread.mean;
    artifact.evaluationEdgeStddev = spread.stddev;
    artifact.evaluationEdgeSecondMoment = spread.secondMoment;
    const double predictedMultiplier =
        g_count_regression_objective == CountRegressionObjective::QUADRATIC_KELLY
            ? 1.0
            : predictedKellyFraction;
    artifact.kellyCurve = evaluateKellyGrowthCurve(
        label, c, fixedPolicy, artifact.count, predictedMultiplier);
    if (const auto* optimum = artifact.kellyCurve.optimalPoint()) {
        artifact.evaluationKellyFraction = optimum->fraction;
        artifact.evaluationKellyGrowth = optimum->growthMean;
        artifact.evaluationKellyGrowthStddev = optimum->growthStddev;
    }

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
    graph.hasEvRegressionLine =
        g_count_regression_objective != CountRegressionObjective::QUADRATIC_KELLY;
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
            point.secondMomentReward = meanSq;
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

    double tenMean = (artifact.rawNormalizedWeights[8] + artifact.rawNormalizedWeights[9]
                    + artifact.rawNormalizedWeights[10] + artifact.rawNormalizedWeights[11]) / 4.0;
    double tenMaxDev = std::max({
        std::abs(artifact.rawNormalizedWeights[8]  - tenMean),
        std::abs(artifact.rawNormalizedWeights[9]  - tenMean),
        std::abs(artifact.rawNormalizedWeights[10] - tenMean),
        std::abs(artifact.rawNormalizedWeights[11] - tenMean)
    });
    double tenRelDev = (std::abs(tenMean) > kEps) ? tenMaxDev / std::abs(tenMean) : tenMaxDev;

    double sumWeights = 0.0;
    for (double w : artifact.count.system.weights) sumWeights += w;
    const uint64_t evalRounds = (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds);

    std::cout << "\n--- " << label << " ---\n";
    std::cout << "Recorded rounds: " << artifact.recordedRounds
              << "  speed: " << std::fixed << std::setprecision(0)
              << artifact.handsPerSec << " hands/sec\n";
    std::cout << std::setprecision(6);
    std::cout << "Count table:\n";
    std::cout << "  " << std::left << std::setw(10) << "Card"
              << std::setw(14) << "Original"
              << std::setw(14) << "Used" << "\n";
    std::cout << "  " << std::string(38, '-') << "\n";
    for (size_t i = 0; i < artifact.count.system.weights.size(); ++i) {
        std::cout << "  " << std::left << std::setw(10) << kRankNames[i]
                  << std::setw(14) << std::fixed << std::setprecision(4)
                  << artifact.rawNormalizedWeights[i]
                  << std::setw(14) << artifact.count.system.weights[i] << "\n";
    }
    std::cout << "  " << std::left << std::setw(10) << "Bias"
              << std::setw(12) << std::fixed << std::setprecision(6)
              << artifact.count.system.bias << "\n";
    std::cout << "  " << std::left << std::setw(10)
              << (artifact.regressionObjective == CountRegressionObjective::QUADRATIC_KELLY
                      ? "Bet>0 TC" : "EV>0 TC")
              << std::setw(12) << std::fixed << std::setprecision(6)
              << artifact.threshold << "\n";
    std::cout << "  " << std::left << std::setw(10) << "Factor"
              << std::setw(12) << std::fixed << std::setprecision(6)
              << artifact.count.system.factor << "\n";
    std::cout << std::setprecision(6);
    std::cout << "\nSpread 1:10 evaluation edge (" << evalRounds << " rounds): "
              << artifact.evaluationEdge << "  std(X): " << artifact.evaluationEdgeStddev
              << "  E[X^2]: " << artifact.evaluationEdgeSecondMoment << "\n";
    std::cout << "Optimal Kelly multiplier:    " << artifact.evaluationKellyFraction
              << (artifact.regressionObjective == CountRegressionObjective::QUADRATIC_KELLY
                      ? "  fitted-signal target: " : "  predicted 1/E[X^2]: ")
              << artifact.kellyCurve.predictedOptimalFraction << "\n";
    std::cout << "Optimal Kelly growth/round:  " << artifact.evaluationKellyGrowth
              << "  stddev: " << artifact.evaluationKellyGrowthStddev << "\n";
    std::cout << "Sanity checks:\n";
    std::cout << "  Sum(weights): " << sumWeights;
    if (artifact.regressionConstraint == CountRegressionConstraint::NONE ||
        artifact.regressionConstraint == CountRegressionConstraint::FIXED_ZERO_BIAS)
        std::cout << "  (unconstrained)\n";
    else
        std::cout << "  (" << (std::abs(sumWeights) < 0.05 ? "PASS" : "WARN") << ", expected near 0)\n";
    std::cout << "  10-value consistency: mean=" << tenMean
              << "  max_dev=" << tenMaxDev
              << "  rel=" << (tenRelDev * 100.0) << "%";
    std::cout << "  (" << (tenRelDev < 0.05 ? "PASS" : "WARN") << ")\n";
    std::cout << "  Bias vs flat edge: bias=" << artifact.count.system.bias
              << "  flat_edge=" << artifact.referenceFlatEdge
              << "  diff=" << std::abs(artifact.count.system.bias - artifact.referenceFlatEdge)
              << "  rel_to_0.5%="
              << (std::abs(artifact.count.system.bias - artifact.referenceFlatEdge) / 0.005 * 100.0)
              << "%\n";
    std::cout << "  Bias constraint: ";
    if (!artifact.forcedBias.has_value()) {
        std::cout << "free\n";
    } else if (artifact.regressionConstraint ==
               CountRegressionConstraint::SUM_ZERO_FIXED_P0_FLAT_EDGE) {
        std::cout << "fixed to P0 flat edge\n";
    } else if (artifact.regressionConstraint == CountRegressionConstraint::FIXED_ZERO_BIAS) {
        std::cout << "fixed to zero\n";
    } else {
        std::cout << "fixed to W1 bias\n";
    }
}

void printEvCountGraphSummary(const std::string& label, const EvCountGraphArtifact& graph) {
    std::map<double, uint64_t> countsByBin;
    uint64_t totalObserved = 0;
    for (const auto& p : graph.points) {
        countsByBin[p.count] = p.n;
        totalObserved += p.n;
    }

    uint64_t symmetryMass = 0;
    uint64_t symmetryDiff = 0;
    for (const auto& [count, n] : countsByBin) {
        if (count <= 0.0) continue;
        auto it = countsByBin.find(-count);
        if (it == countsByBin.end()) continue;
        symmetryMass += n + it->second;
        symmetryDiff += static_cast<uint64_t>(std::llabs(static_cast<long long>(n) - static_cast<long long>(it->second)));
    }
    double symmetryRel = (symmetryMass > 0)
        ? static_cast<double>(symmetryDiff) / static_cast<double>(symmetryMass)
        : 0.0;

    std::cout << "EV-count graph:         " << graph.points.size()
              << " points  res=" << graph.resolution
              << "  speed: " << std::fixed << std::setprecision(0)
              << graph.handsPerSec << " hands/sec\n";
    std::cout << "Count histogram:        total=" << totalObserved
              << "  symmetry_rel_diff=" << std::setprecision(4) << (symmetryRel * 100.0) << "%";
    std::cout << "  (" << (symmetryRel < 0.10 ? "PASS" : "WARN")
              << ", expected roughly symmetric)\n";
    std::cout << std::setprecision(6);
}

void printPolicySummary(const std::string& label, const PolicyArtifact& artifact) {
    const uint64_t evalRounds = (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds);
    std::cout << "\n--- " << label << " ---\n";
    std::cout << "Training rounds: " << artifact.trainingRounds << "\n";
    std::cout << "Training speed: " << std::fixed << std::setprecision(0)
              << artifact.handsPerSec << " hands/sec\n";
    std::cout << std::setprecision(6);
    std::cout << "Training flat edge:   " << artifact.trainingEdge << "\n";
    std::cout << "Evaluation flat edge (" << evalRounds << " rounds): "
              << artifact.evaluationEdge << "  std(X): " << artifact.evaluationEdgeStddev
              << "  E[X^2]: " << artifact.evaluationEdgeSecondMoment << "\n";
    if (artifact.hasCountBettingEvaluation) {
        std::cout << "Spread 1:10 evaluation edge (" << evalRounds << " rounds): "
                  << artifact.evaluationSpreadEdge
                  << "  std(X): " << artifact.evaluationSpreadStddev
                  << "  E[X^2]: " << artifact.evaluationSpreadSecondMoment << "\n";
        std::cout << "Kelly growth per round:      " << artifact.evaluationKellyGrowth
                  << "  stddev: " << artifact.evaluationKellyGrowthStddev << "\n";
    } else {
        std::cout << "Spread/Kelly evaluation skipped for P0: W0 is the zero count, "
                     "so no positive-count betting signal is expected.\n";
    }
}

void runCase(const Case& c) {
    const bool resuming = !g_load_checkpoint.empty();
    const std::string folder = resuming
        ? g_load_checkpoint
        : (g_checkpoint_name.empty()
            ? currentTimestamp() + "_" + checkpointTableName(c)
            : g_checkpoint_name + "_" + checkpointTableName(c));
    const uint64_t evalRounds = (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds);
    RunLogger logger(std::filesystem::path(PROJECT_ROOT) / kAlternatingCheckpointRoot, folder);
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
            PolicyArtifact learnedPolicy =
                learnPolicy(c, state.currentCount, g_num_rounds, pLabel, runHeader,
                            state.nextPolicyIndex > 0);
            printPolicySummary(pLabel, learnedPolicy);
            logger.fileStream() << "\n=== " << pLabel << " Strategy ===\n";
            logger.fileStream() << *learnedPolicy.policy;
            logger.fileStream() << "========================\n";
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
            state.currentPolicyFlatStddev = learnedPolicy.evaluationEdgeStddev;
            state.currentPolicyFlatSecondMoment = learnedPolicy.evaluationEdgeSecondMoment;
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
        std::optional<double> forcedBias;
        if (g_count_regression_constraint ==
            CountRegressionConstraint::SUM_ZERO_FIXED_W1_BIAS) {
            // W1 estimates b1 freely; W2+ reuse that conserved value.
            if (state.nextPolicyIndex > 0)
                forcedBias = state.currentCount.system.bias;
        } else if (g_count_regression_constraint ==
                   CountRegressionConstraint::SUM_ZERO_FIXED_P0_FLAT_EDGE) {
            // P0 is evaluated before W1, so W1 and every later fit can use its
            // measured flat edge as the conserved bias.
            forcedBias = state.nextPolicyIndex == 0
                ? state.currentPolicyFlatEdge
                : state.currentCount.system.bias;
        } else if (g_count_regression_constraint ==
                   CountRegressionConstraint::FIXED_ZERO_BIAS) {
            forcedBias = 0.0;
        }
        const double predictedKellyFraction = state.currentPolicyFlatSecondMoment > 0.0
            ? 1.0 / state.currentPolicyFlatSecondMoment
            : 0.0;
        CountArtifact countArtifact = learnCount(
            c, wLabel, *currentPolicy, state.currentCount, countRounds, forcedBias,
            predictedKellyFraction);
        countArtifact.referenceFlatEdge = state.currentPolicyFlatEdge;
        printCountSummary(wLabel, countArtifact);
        if (!g_no_save)
            saveCountArtifact(folder, wLabel, countArtifact, countRounds);

        const uint64_t graphRounds = (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds);
        EvCountGraphArtifact graphArtifact = measureEvCountGraph(c, *currentPolicy, countArtifact.count, graphRounds);
        printEvCountGraphSummary(wLabel, graphArtifact);
        saveEvCountGraphArtifact(logger.runDir(), wLabel, graphArtifact);
        saveCumulativeEvCountGraphArtifact(logger.runDir(), static_cast<int>(state.nextPolicyIndex + 1));
        saveKellyGrowthGraphArtifact(logger.runDir(), wLabel, countArtifact.kellyCurve);
        saveCumulativeKellyGrowthGraphArtifact(
            logger.runDir(), static_cast<int>(state.nextPolicyIndex + 1));

        state.currentCount = countArtifact.count;
        state.currentPolicyLabel.clear();
        state.currentPolicyPath.clear();
        state.currentPolicyFlatEdge = 0.0;
        state.currentPolicyFlatStddev = 0.0;
        state.currentPolicyFlatSecondMoment = 0.0;
        state.nextPolicyIndex += 1;
        state.nextPhase = Phase::POLICY;
        currentPolicy.reset();
        if (!g_no_save) saveState(folder, state);
    }
}

void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Alternating optimization between policy learning (RL) and configurable count regression.\n\n";

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
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/W*_second_moment_graph.json/svg\n";
    std::cout << "                                                                      Unit-wager E[X^2 | count]\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/W*_graph_overlay.*    Cumulative W1..Wk comparison graph\n\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/W*_kelly_graph.*     Kelly multiplier sweep\n";
    std::cout << "    checkpoints/alternating-checkpoints/<folder>/W*_kelly_graph_overlay.*  Cumulative Kelly comparison\n\n";

    std::cout << "SIMULATION:\n";
    std::cout << "  --num-rounds <N>      Rounds used in each policy/count phase (default: 1000000000)\n";
    std::cout << "  --stop-mode <rounds|diff>  Policy RL stopping rule (default: rounds)\n";
    std::cout << "  --sample-rounds <N>   In diff mode, compare policy Q-tables every N rounds (default: 100000000)\n";
    std::cout << "  --diff-threshold <v>  In diff mode, stop when avg abs Q-table change <= v (default: 0.001)\n";
    std::cout << "  --eval-rounds <N>     Edge evaluation rounds after each step (default: num-rounds)\n";
    std::cout << "                        P0 reports flat edge; P1+ also report spread edge and Kelly growth.\n";
    std::cout << "                        Count steps report spread edge and Kelly growth.\n";
    std::cout << "  --iterations <N>      Number of loop iterations k (default: 3)\n";
    std::cout << "  --num-threads <N>     Threads (default: 10)\n";
    std::cout << "  --penetration <val>   Shoe penetration % (default: 75.0)\n";
    std::cout << "  --sample-every <N>    Record every N-th round for count regression (default: 1)\n";
    std::cout << "                        The count phase plays N * num-rounds rounds to retain about num-rounds samples.\n";
    std::cout << "  --kelly-measurements <N>  Experiments per Kelly fraction, 1,000,000 rounds each (default: 10)\n";
    std::cout << "  --kelly-fraction-min <v>   Override sweep minimum\n";
    std::cout << "  --kelly-fraction-max <v>   Override sweep maximum\n";
    std::cout << "                              Default: nearest step to 1/E[X^2] +/- 0.25, clamped at 0\n";
    std::cout << "  --kelly-fraction-step <v>  Kelly multiplier sweep step (default: 0.05)\n";
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

    std::cout << "COUNT OBJECTIVE (choose at most one; default: --count-classical-ols):\n";
    std::cout << "  --count-classical-ols       Minimize E[(X-w'c)^2].\n";
    std::cout << "  --count-quadratic-kelly     Maximize the second-order approximation to E[log(1+w'c X)].\n\n";
    std::cout << "COUNT CONSTRAINT (choose at most one; default: --count-sum-zero-fixed-b1):\n";
    std::cout << "  --count-unconstrained       Fit all rank weights and the bias freely.\n";
    std::cout << "  --count-sum-zero            Require sum(rank weights)=0; bias is free.\n";
    std::cout << "  --count-sum-zero-fixed-b1   Require sum(rank weights)=0; W1 estimates b1, fixed for W2+.\n";
    std::cout << "  --count-sum-zero-fixed-p0-edge\n";
    std::cout << "                                Require sum(rank weights)=0; P0 flat edge is fixed for W1+.\n\n";

    std::cout << "LEARNED COUNT SCALING:\n";
    std::cout << "  Each learned count is scaled so the average 10/J/Q/K tag is "
              << kTargetTenValueTag
              << "; the betting factor is scaled inversely.\n\n";

    std::cout << "GAME CONFIG:\n";
    std::cout << "  --decks, --ss17, --das, --sas, --don, --rsa, --hsa,\n";
    std::cout << "  --peek, --surr, --bj   (same format as the existing apps)\n";
    std::cout << "  These are ignored when --load-checkpoint is used.\n\n";
}

} // namespace

int runBlackjackAlternatingOptimization(int argc, char** argv) {
    g_command_line = commandLineFromArgs(argc, argv);
    int countObjectiveFlags = 0;
    int countConstraintFlags = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printHelp(argv[0]); return 0; }
        else if ((arg == "--game" || arg == "--table-type") && i + 1 < argc) { ++i; }

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
        else if (arg == "--kelly-fraction-min" && i + 1 < argc) g_kelly_fraction_min = std::stod(argv[++i]);
        else if (arg == "--kelly-fraction-max" && i + 1 < argc) g_kelly_fraction_max = std::stod(argv[++i]);
        else if (arg == "--kelly-fraction-step" && i + 1 < argc) g_kelly_fraction_step = std::stod(argv[++i]);
        else if (arg == "--checkpoint-name"  && i + 1 < argc) g_checkpoint_name = argv[++i];
        else if (arg == "--load-checkpoint"  && i + 1 < argc) g_load_checkpoint = argv[++i];
        else if (arg == "--verbose")                       g_verbose = true;
        else if (arg == "--full-verbose")                  { g_full_verbose = true; g_verbose = true; }
        else if (arg == "--no-save")                         g_no_save = true;
        else if (arg == "--count-classical-ols") {
            g_count_regression_objective = CountRegressionObjective::CLASSICAL_OLS;
            ++countObjectiveFlags;
        }
        else if (arg == "--count-quadratic-kelly") {
            g_count_regression_objective = CountRegressionObjective::QUADRATIC_KELLY;
            ++countObjectiveFlags;
        }
        else if (arg == "--count-unconstrained") {
            g_count_regression_constraint = CountRegressionConstraint::NONE;
            ++countConstraintFlags;
        }
        else if (arg == "--count-sum-zero") {
            g_count_regression_constraint = CountRegressionConstraint::SUM_ZERO;
            ++countConstraintFlags;
        }
        else if (arg == "--count-sum-zero-fixed-b1" ||
                 arg == "--count-sum-zero-fixed-bias") {
            g_count_regression_constraint =
                CountRegressionConstraint::SUM_ZERO_FIXED_W1_BIAS;
            ++countConstraintFlags;
        }
        else if (arg == "--count-sum-zero-fixed-p0-edge") {
            g_count_regression_constraint =
                CountRegressionConstraint::SUM_ZERO_FIXED_P0_FLAT_EDGE;
            ++countConstraintFlags;
        }

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
    try {
        (void)makeKellyFractionGrid(0.0, 0.0, g_kelly_fraction_step);
        if ((g_kelly_fraction_min &&
             (!std::isfinite(*g_kelly_fraction_min) || *g_kelly_fraction_min < 0.0)) ||
            (g_kelly_fraction_max &&
             (!std::isfinite(*g_kelly_fraction_max) || *g_kelly_fraction_max < 0.0))) {
            throw std::invalid_argument("Kelly fraction overrides must be finite and non-negative");
        }
        if (g_kelly_fraction_min && g_kelly_fraction_max) {
            (void)makeKellyFractionGrid(
                *g_kelly_fraction_min, *g_kelly_fraction_max, g_kelly_fraction_step);
        }
    } catch (const std::invalid_argument& e) {
        std::cerr << "Error: " << e.what() << ".\n";
        return 1;
    }
    if (countObjectiveFlags > 1) {
        std::cerr << "Error: choose only one count objective flag.\n";
        return 1;
    }
    if (countConstraintFlags > 1) {
        std::cerr << "Error: choose only one count constraint flag.\n";
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
