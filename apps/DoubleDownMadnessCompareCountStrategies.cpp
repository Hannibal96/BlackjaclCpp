#include "RegressionTestUtils.h"
#include "Game/DoubleDownMadnessTable.h"
#include "Game/Player.h"
#include "Game/BettingStrategy.h"
#include "Game/CountingMethods.h"
#include "RL/BasicStrategy.h"
#include "RL/DoubleDownMadnessStrategy.h"
#include "RL/QLearningStrategy.h"
#include "RL/DecayingParameter.h"
#include "Utils/RunLogger.h"
#include "Utils/SimulationAnalysis.h"
#include "Utils/Utils.h"
#include <nlohmann/json.hpp>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <ctime>
#include <array>
#include <map>
#include <stdexcept>
#include <cmath>
#include <sstream>
#include <limits>
#include <optional>
#include <vector>

using json = nlohmann::json;

namespace {
constexpr const char* kQlearningCheckpointRoot = "checkpoints/double-down-madness-alternating-checkpoints";
constexpr const char* kOlsCheckpointRoot = "checkpoints/checkpoints_ols";
constexpr const char* kCompareCheckpointRoot = "checkpoints/DoubleDownMadnessCompareCountStrategies";
constexpr uint64_t kKellyMeasurementRounds = 1'000'000ULL;
double g_referenceBias = -0.005;
double g_referenceFactor = 0.005;

enum class TrainingStopMode { FIXED_ROUNDS, TABLE_DIFF };

struct MadnessCase {
    int deckSize = 6;
    bool standSoft17 = false;
    DoubleDownMadnessVersion version = DoubleDownMadnessVersion::VERSION_1;
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
};

struct StrategyCountConfig {
    CountingSystem system;
    double resolution = 1.0;
    int minCount = -5;
    int maxCount = 5;
};

struct EvalResult {
    std::string label;
    std::string note;
    double spreadEdge = 0.0;
    double spreadStddev = 0.0;
    double spreadSecondMoment = 0.0;
    double kellyGrowth = 1.0;
    double kellyGrowthStddev = 0.0;
    double kellyFraction = 1.0;
    KellyGrowthCurve kellyCurve;
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
};

struct StrategyGraphArtifact {
    std::string label;
    EvCountGraphPoint referenceTemplate;
    double resolution = 0.25;
    double minCount = 0.0;
    double maxCount = 0.0;
    uint64_t rounds = 0;
    double handsPerSec = 0.0;
    EdgeStatistics flatStatistics;
    std::vector<EvCountGraphPoint> points;
};

struct CheckpointCountConfig {
    std::array<double, 13> weights{};
    double factor = 1.0;
    double bias = 0.0;
    double resolution = 1.0;
    int    minCount   = 0;
    int    maxCount   = 0;
};

uint64_t    g_num_rounds   = 1'000'000'000ULL;   // training rounds for full deviations
uint64_t    g_eval_rounds  = 0;                  // 0 => use g_num_rounds for spread eval
int         g_num_threads  = 10;
double      g_penetration  = 75.0;
int         g_kelly_measurements = 10;
std::optional<double> g_kelly_fraction_min;
std::optional<double> g_kelly_fraction_max;
double      g_kelly_fraction_step = 0.05;
double      g_count_resolution = 1.0;
int         g_min_count = -5;
int         g_max_count = 5;
TrainingStopMode g_training_stop_mode = TrainingStopMode::FIXED_ROUNDS;
uint64_t    g_sample_rounds = 100'000'000ULL;
double      g_diff_threshold = 0.001;
bool        g_verbose = false;
std::string g_command_line;

std::string g_count_name;
std::string g_count_weights_str;
std::string g_count_ols;
double g_factor = std::numeric_limits<double>::quiet_NaN();
double g_bias   = std::numeric_limits<double>::quiet_NaN();

std::string g_deviations_checkpoint;
std::string g_deviations_agent;

AgentConfig g_agent;

std::vector<int>  g_deck_sizes   = {6};
std::vector<bool> g_stand_soft17 = {false};
std::vector<int>  g_versions     = {1};

std::string currentTimestamp() {
    time_t t = time(nullptr);
    char buf[20];
    strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", localtime(&t));
    return buf;
}

DoubleDownMadnessVersion versionFromInt(int value) {
    if (value < 1 || value > 3)
        throw std::invalid_argument("Double Down Madness version must be 1, 2, or 3");
    return static_cast<DoubleDownMadnessVersion>(value);
}

std::string ToString(const MadnessCase& c) {
    std::ostringstream os;
    os << "decks=" << c.deckSize
       << "_ss17=" << std::boolalpha << c.standSoft17
       << "_version=" << static_cast<int>(c.version);
    return os.str();
}

std::string ToStringTableName(const MadnessCase& c) {
    return ToString(c);
}

std::vector<MadnessCase> generateMadnessCases() {
    std::vector<MadnessCase> cases;
    for (int decks : g_deck_sizes) {
        if (decks <= 0)
            throw std::invalid_argument("Deck count must be positive");
        for (bool standSoft17 : g_stand_soft17) {
            for (int version : g_versions)
                cases.push_back({decks, standSoft17, versionFromInt(version)});
        }
    }
    return cases;
}

bool countWasExplicitlyConfigured() {
    return !g_count_name.empty() ||
           !g_count_weights_str.empty() ||
           !g_count_ols.empty() ||
           !std::isnan(g_factor) ||
           !std::isnan(g_bias) ||
           g_count_resolution != 1.0 ||
           g_min_count != -5 ||
           g_max_count != 5;
}

std::array<double, 14> solveOLS(
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
            throw std::runtime_error("Singular XtX in OLS checkpoint");
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

std::array<double, 13> parseWeights13(const std::string& s) {
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

std::array<double, 14> loadOlsFromCheckpoint(const std::string& folder) {
    namespace fs = std::filesystem;
    fs::path root = fs::path(PROJECT_ROOT) / kOlsCheckpointRoot / folder;
    fs::path dataPath = root / "data.json";
    if (!fs::exists(dataPath))
        throw std::runtime_error("OLS checkpoint missing data.json: " + root.string());

    json data; { std::ifstream f(dataPath); f >> data; }
    std::array<std::array<double, 14>, 14> XtX{};
    std::array<double, 14> Xty{};
    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 14; ++j)
            XtX[i][j] = data["XtX"][i][j].get<double>();
        Xty[i] = data["Xty"][i].get<double>();
    }
    return solveOLS(XtX, Xty);
}

StrategyCountConfig resolveCountConfig() {
    int sources = (!g_count_name.empty() ? 1 : 0)
                + (!g_count_weights_str.empty() ? 1 : 0)
                + (!g_count_ols.empty() ? 1 : 0);
    if (sources > 1)
        throw std::invalid_argument("Specify only one of --count, --count-weights, --count-ols");

    StrategyCountConfig cfg;
    cfg.resolution = g_count_resolution;
    cfg.minCount = g_min_count;
    cfg.maxCount = g_max_count;

    if (!g_count_ols.empty()) {
        auto w14 = loadOlsFromCheckpoint(g_count_ols);
        for (int i = 0; i < 13; ++i) cfg.system.weights[i] = w14[i];
        cfg.system.factor = 1.0;
        cfg.system.bias = w14[13];
    } else if (!g_count_weights_str.empty()) {
        cfg.system.weights = parseWeights13(g_count_weights_str);
        cfg.system.factor = CountingMethods::kDefaultFactor;
        cfg.system.bias = CountingMethods::kDefaultBias;
    } else if (!g_count_name.empty()) {
        auto opt = CountingMethods::fromName(g_count_name);
        if (!opt)
            throw std::runtime_error("Unknown count name: " + g_count_name);
        cfg.system = *opt;
    } else {
        cfg.system = *CountingMethods::fromName("hilo");
        if (g_deviations_checkpoint.empty())
            std::cout << "Count:       Hi-Lo (default)\n";
    }

    if (!std::isnan(g_factor)) cfg.system.factor = g_factor;
    if (!std::isnan(g_bias))   cfg.system.bias   = g_bias;
    g_referenceBias = cfg.system.bias;
    g_referenceFactor = cfg.system.factor;
    return cfg;
}

bool approxEqual(double a, double b, double eps = 1e-9) {
    return std::abs(a - b) <= eps;
}

DoubleDownMadnessRules buildRules(const MadnessCase& c, double minBet, double maxBet) {
    DoubleDownMadnessRules rules(c.version, c.standSoft17, c.deckSize, g_penetration);
    rules.minBet = minBet;
    rules.maxBet = maxBet;
    return rules;
}

json graphsToJson(const std::vector<StrategyGraphArtifact>& graphs) {
    json root;
    root["reference_line"]["bias"] = g_referenceBias;
    root["reference_line"]["factor"] = g_referenceFactor;
    root["graphs"] = json::array();
    for (const auto& graph : graphs) {
        json g;
        g["label"] = graph.label;
        g["resolution"] = graph.resolution;
        g["min_count"] = graph.minCount;
        g["max_count"] = graph.maxCount;
        g["rounds"] = graph.rounds;
        g["hands_per_sec"] = graph.handsPerSec;
        g["flat_edge"] = graph.flatStatistics.mean;
        g["flat_stddev"] = graph.flatStatistics.stddev;
        g["flat_second_moment"] = graph.flatStatistics.secondMoment;
        g["points"] = json::array();
        for (const auto& p : graph.points) {
            json row;
            row["count"] = p.count;
            row["n"] = p.n;
            row["reference_reward"] =
                g_referenceBias + g_referenceFactor * p.count;
            row["mean_reward"] = (p.n > 0 ? json(p.meanReward) : json(nullptr));
            row["conditional_second_moment"] =
                (p.n > 0 ? json(p.secondMomentReward) : json(nullptr));
            row["stddev_reward"] = p.stddevReward;
            row["confidence_lower"] = (p.n > 0 ? json(p.confidenceLower) : json(nullptr));
            row["confidence_upper"] = (p.n > 0 ? json(p.confidenceUpper) : json(nullptr));
            g["points"].push_back(row);
        }
        root["graphs"].push_back(g);
    }
    return root;
}

std::string colorForGraph(size_t index) {
    static const char* kColors[] = {"#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd"};
    return kColors[index % (sizeof(kColors) / sizeof(kColors[0]))];
}

ConditionalSecondMomentCurve conditionalSecondMomentCurve(
    const StrategyGraphArtifact& graph) {
    ConditionalSecondMomentCurve curve;
    curve.label = graph.label;
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

std::string graphsToSvg(const std::vector<StrategyGraphArtifact>& graphs) {
    const double width = 1200.0;
    const double height = 720.0;
    const double left = 90.0;
    const double right = 280.0;
    const double top = 40.0;
    const double bottom = 70.0;
    const double plotWidth = width - left - right;
    const double plotHeight = height - top - bottom;

    double minX = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const auto& graph : graphs) {
        minX = std::min(minX, graph.minCount);
        maxX = std::max(maxX, graph.maxCount);
        for (const auto& p : graph.points) {
            minY = std::min(minY, g_referenceBias + g_referenceFactor * p.count);
            maxY = std::max(maxY, g_referenceBias + g_referenceFactor * p.count);
            if (p.n > 0) {
                minY = std::min(minY, p.confidenceLower);
                maxY = std::max(maxY, p.confidenceUpper);
            }
        }
    }
    if (!std::isfinite(minX) || !std::isfinite(maxX) || std::abs(maxX - minX) < 1e-12) {
        minX = -5.0; maxX = 5.0;
    }
    if (!std::isfinite(minY) || !std::isfinite(maxY)) {
        minY = -0.03; maxY = 0.03;
    }
    double padding = 0.1 * std::max(1e-6, maxY - minY);
    minY -= padding;
    maxY += padding;

    auto mapX = [&](double x) { return left + ((x - minX) / (maxX - minX)) * plotWidth; };
    auto mapY = [&](double y) { return top + (1.0 - ((y - minY) / (maxY - minY))) * plotHeight; };

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    svg << "<defs>\n";
    svg << "  <pattern id=\"minorGridCompare\" width=\"" << (plotWidth / 20.0) << "\" height=\"" << (plotHeight / 20.0)
        << "\" patternUnits=\"userSpaceOnUse\">\n";
    svg << "    <path d=\"M " << (plotWidth / 20.0) << " 0 L 0 0 0 " << (plotHeight / 20.0)
        << "\" fill=\"none\" stroke=\"#dddddd\" stroke-width=\"1.1\"/>\n";
    svg << "  </pattern>\n";
    svg << "  <pattern id=\"majorGridCompare\" width=\"" << (plotWidth / 10.0) << "\" height=\"" << (plotHeight / 10.0)
        << "\" patternUnits=\"userSpaceOnUse\">\n";
    svg << "    <rect width=\"" << (plotWidth / 10.0) << "\" height=\"" << (plotHeight / 10.0)
        << "\" fill=\"url(#minorGridCompare)\"/>\n";
    svg << "    <path d=\"M " << (plotWidth / 10.0) << " 0 L 0 0 0 " << (plotHeight / 10.0)
        << "\" fill=\"none\" stroke=\"#b8b8b8\" stroke-width=\"1.5\"/>\n";
    svg << "  </pattern>\n";
    svg << "</defs>\n";
    svg << "<text x=\"" << left << "\" y=\"24\" font-family=\"sans-serif\" font-size=\"20\">EV vs Count comparison</text>\n";
    svg << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plotWidth
        << "\" height=\"" << plotHeight << "\" fill=\"url(#majorGridCompare)\"/>\n";
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

    std::ostringstream refPath;
    bool refStarted = false;
    if (!graphs.empty()) {
        for (const auto& p : graphs.front().points) {
            double x = mapX(p.count);
            double y = mapY(g_referenceBias + g_referenceFactor * p.count);
            refPath << (refStarted ? " L " : "M ") << x << " " << y;
            refStarted = true;
        }
    }
    if (refStarted) {
        svg << "<path d=\"" << refPath.str()
            << "\" fill=\"none\" stroke=\"#444444\" stroke-width=\"2\" stroke-dasharray=\"6,4\"/>\n";
    }

    for (size_t gi = 0; gi < graphs.size(); ++gi) {
        const auto& graph = graphs[gi];
        std::string color = colorForGraph(gi);
        std::vector<std::pair<double,double>> upper, lower;
        std::ostringstream empiricalPath;
        bool empiricalStarted = false;
        for (const auto& p : graph.points) {
            if (p.n == 0) continue;
            double x = mapX(p.count);
            upper.emplace_back(x, mapY(p.confidenceUpper));
            lower.emplace_back(x, mapY(p.confidenceLower));
            empiricalPath << (empiricalStarted ? " L " : "M ") << x << " " << mapY(p.meanReward);
            empiricalStarted = true;
        }
        if (!upper.empty()) {
            std::ostringstream band;
            std::ostringstream upperPath;
            std::ostringstream lowerPath;
            band << "M " << upper.front().first << " " << upper.front().second;
            for (size_t i = 1; i < upper.size(); ++i) band << " L " << upper[i].first << " " << upper[i].second;
            for (size_t i = lower.size(); i-- > 0;) band << " L " << lower[i].first << " " << lower[i].second;
            band << " Z";
            for (size_t i = 0; i < upper.size(); ++i) {
                upperPath << (i == 0 ? "M " : " L ") << upper[i].first << " " << upper[i].second;
                lowerPath << (i == 0 ? "M " : " L ") << lower[i].first << " " << lower[i].second;
            }
            svg << "<path d=\"" << band.str() << "\" fill=\"" << color << "\" fill-opacity=\"0.16\" stroke=\"none\"/>\n";
            svg << "<path d=\"" << upperPath.str()
                << "\" fill=\"none\" stroke=\"" << color
                << "\" stroke-opacity=\"0.45\" stroke-width=\"1.2\" stroke-dasharray=\"4,3\"/>\n";
            svg << "<path d=\"" << lowerPath.str()
                << "\" fill=\"none\" stroke=\"" << color
                << "\" stroke-opacity=\"0.45\" stroke-width=\"1.2\" stroke-dasharray=\"4,3\"/>\n";
        }
        if (empiricalStarted) {
            svg << "<path d=\"" << empiricalPath.str()
                << "\" fill=\"none\" stroke=\"" << color << "\" stroke-width=\"2.5\"/>\n";
        }
    }

    svg << "<text x=\"" << (left + plotWidth / 2.0) << "\" y=\"" << (height - 18)
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Count</text>\n";
    svg << "<text x=\"20\" y=\"" << (top + plotHeight / 2.0)
        << "\" transform=\"rotate(-90 20 " << (top + plotHeight / 2.0)
        << ")\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">EV per round</text>\n";
    double legendY = 24.0;
    for (size_t gi = 0; gi < graphs.size(); ++gi) {
        std::string color = colorForGraph(gi);
        svg << "<line x1=\"" << (width - 240) << "\" y1=\"" << legendY
            << "\" x2=\"" << (width - 200) << "\" y2=\"" << legendY
            << "\" stroke=\"" << color << "\" stroke-width=\"2.5\"/>\n";
        svg << "<rect x=\"" << (width - 240) << "\" y=\"" << (legendY + 8) << "\" width=\"40\" height=\"8\" fill=\"" << color << "\" fill-opacity=\"0.16\"/>\n";
        svg << "<text x=\"" << (width - 190) << "\" y=\"" << (legendY + 4)
            << "\" font-family=\"sans-serif\" font-size=\"12\">" << graphs[gi].label << " empirical</text>\n";
        legendY += 26.0;
    }
    svg << "<line x1=\"" << (width - 240) << "\" y1=\"" << legendY
        << "\" x2=\"" << (width - 200) << "\" y2=\"" << legendY
        << "\" stroke=\"#444444\" stroke-width=\"2\" stroke-dasharray=\"6,4\"/>\n";
    svg << "<text x=\"" << (width - 190) << "\" y=\"" << (legendY + 4)
        << "\" font-family=\"sans-serif\" font-size=\"12\">Reference: -0.005 + 0.005 * count</text>\n";
    svg << "</svg>\n";
    return svg.str();
}

json histogramToJson(const StrategyGraphArtifact& graph) {
    json root;
    root["label"] = graph.label;
    root["resolution"] = graph.resolution;
    root["min_count"] = graph.minCount;
    root["max_count"] = graph.maxCount;
    root["points"] = json::array();
    for (const auto& p : graph.points) {
        json row;
        row["count"] = p.count;
        row["n"] = p.n;
        root["points"].push_back(row);
    }
    return root;
}

std::string histogramToSvg(const StrategyGraphArtifact& graph) {
    const double width = 1200.0;
    const double height = 680.0;
    const double left = 90.0;
    const double right = 260.0;
    const double top = 40.0;
    const double bottom = 70.0;
    const double plotWidth = width - left - right;
    const double plotHeight = height - top - bottom;

    double minX = graph.minCount;
    double maxX = graph.maxCount;
    uint64_t maxN = 1;
    for (const auto& p : graph.points) maxN = std::max(maxN, p.n);
    if (!std::isfinite(minX) || !std::isfinite(maxX) || std::abs(maxX - minX) < 1e-12) {
        minX = -5.0;
        maxX = 5.0;
    }
    auto mapX = [&](double x) { return left + ((x - minX) / (maxX - minX)) * plotWidth; };
    auto mapY = [&](double n) { return top + (1.0 - n / static_cast<double>(maxN)) * plotHeight; };
    const double barWidth = std::max(2.0, plotWidth / std::max<size_t>(1, graph.points.size()) * 0.75);

    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    svg << "<defs>\n";
    svg << "  <pattern id=\"minorGridCompareHist\" width=\"" << (plotWidth / 20.0) << "\" height=\"" << (plotHeight / 20.0)
        << "\" patternUnits=\"userSpaceOnUse\">\n";
    svg << "    <path d=\"M " << (plotWidth / 20.0) << " 0 L 0 0 0 " << (plotHeight / 20.0)
        << "\" fill=\"none\" stroke=\"#dddddd\" stroke-width=\"1.1\"/>\n";
    svg << "  </pattern>\n";
    svg << "  <pattern id=\"majorGridCompareHist\" width=\"" << (plotWidth / 10.0) << "\" height=\"" << (plotHeight / 10.0)
        << "\" patternUnits=\"userSpaceOnUse\">\n";
    svg << "    <rect width=\"" << (plotWidth / 10.0) << "\" height=\"" << (plotHeight / 10.0)
        << "\" fill=\"url(#minorGridCompareHist)\"/>\n";
    svg << "    <path d=\"M " << (plotWidth / 10.0) << " 0 L 0 0 0 " << (plotHeight / 10.0)
        << "\" fill=\"none\" stroke=\"#b8b8b8\" stroke-width=\"1.5\"/>\n";
    svg << "  </pattern>\n";
    svg << "</defs>\n";
    svg << "<text x=\"" << left << "\" y=\"24\" font-family=\"sans-serif\" font-size=\"20\">Count histogram</text>\n";
    svg << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plotWidth
        << "\" height=\"" << plotHeight << "\" fill=\"url(#majorGridCompareHist)\"/>\n";
    svg << "<line x1=\"" << left << "\" y1=\"" << (top + plotHeight) << "\" x2=\"" << (left + plotWidth)
        << "\" y2=\"" << (top + plotHeight) << "\" stroke=\"black\" stroke-width=\"1.2\"/>\n";
    svg << "<line x1=\"" << left << "\" y1=\"" << top << "\" x2=\"" << left
        << "\" y2=\"" << (top + plotHeight) << "\" stroke=\"black\" stroke-width=\"1.2\"/>\n";
    for (int i = 0; i <= 12; ++i) {
        double yValue = static_cast<double>(maxN) * (static_cast<double>(i) / 12.0);
        double y = mapY(yValue);
        svg << "<line x1=\"" << (left - 6) << "\" y1=\"" << y
            << "\" x2=\"" << left << "\" y2=\"" << y << "\" stroke=\"black\"/>\n";
        if (i % 2 == 0) {
            svg << "<text x=\"" << (left - 10) << "\" y=\"" << (y + 4)
                << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">"
                << static_cast<uint64_t>(std::llround(yValue)) << "</text>\n";
        }
    }
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
    if (minX <= 0.0 && maxX >= 0.0) {
        double zeroX = mapX(0.0);
        svg << "<line x1=\"" << zeroX << "\" y1=\"" << top
            << "\" x2=\"" << zeroX << "\" y2=\"" << (top + plotHeight)
            << "\" stroke=\"#666666\" stroke-width=\"1.8\"/>\n";
        svg << "<text x=\"" << zeroX << "\" y=\"" << (top + plotHeight + 40)
            << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\" fill=\"#444444\">0</text>\n";
    }
    for (const auto& p : graph.points) {
        double x = mapX(p.count);
        double y = mapY(static_cast<double>(p.n));
        svg << "<rect x=\"" << (x - barWidth / 2.0) << "\" y=\"" << y
            << "\" width=\"" << barWidth << "\" height=\"" << ((top + plotHeight) - y)
            << "\" fill=\"#4c78a8\" fill-opacity=\"0.85\"/>\n";
    }
    svg << "<rect x=\"" << (width - 220) << "\" y=\"16\" width=\"40\" height=\"12\" fill=\"#4c78a8\" fill-opacity=\"0.85\"/>\n";
    svg << "<text x=\"" << (width - 170) << "\" y=\"28\" font-family=\"sans-serif\" font-size=\"12\">"
        << graph.label << "</text>\n";
    svg << "<text x=\"" << (left + plotWidth / 2.0) << "\" y=\"" << (height - 18)
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Count</text>\n";
    svg << "<text x=\"20\" y=\"" << (top + plotHeight / 2.0)
        << "\" transform=\"rotate(-90 20 " << (top + plotHeight / 2.0)
        << ")\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Occurrences</text>\n";
    svg << "</svg>\n";
    return svg.str();
}

double averageRoundsPerThread(uint64_t rounds, int threads) {
    return static_cast<double>(rounds) / static_cast<double>(threads);
}

StrategyGraphArtifact measureEvCountGraph(const MadnessCase& c,
                                          const std::string& label,
                                          const BasicStrategy& strategy,
                                          const StrategyCountConfig& count,
                                          uint64_t rounds) {
    auto* player = new Player(0.0, strategy.clone());
    player->setNumDecks(c.deckSize);
    player->setCountSystem(count.system);
    player->setCountResolution(count.resolution);
    auto [strategyMinCount, strategyMaxCount] = strategy.getCountRange();
    player->setCountRange(strategyMinCount, strategyMaxCount);
    double graphResolution = (count.resolution > 0.0) ? (count.resolution / 4.0) : 0.25;
    player->enableCountGraph(graphResolution);
    player->enableRoundStats();

    DoubleDownMadnessRules rules = buildRules(c, 1.0, 1.0);
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Player*> results = runParallelSimulation(rules, {player}, rounds, g_num_threads);
    double secs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start).count() / 1000.0;

    delete player;
    if (results.empty())
        throw std::runtime_error("EV-count graph failed");
    Player* result = results[0];

    StrategyGraphArtifact graph;
    graph.label = label;
    graph.resolution = graphResolution;
    graph.rounds = rounds;
    graph.handsPerSec = (secs > 0.0 ? rounds / secs : 0.0);
    graph.flatStatistics = edgeStatisticsFromPlayer(*result);
    graph.minCount = count.minCount;
    graph.maxCount = count.maxCount;
    const int minBinIndex = static_cast<int>(std::llround(graph.minCount / graphResolution));
    const int maxBinIndex = static_cast<int>(std::llround(graph.maxCount / graphResolution));
    const auto& bins = result->getCountGraphBins();
    for (int binIndex = minBinIndex; binIndex <= maxBinIndex; ++binIndex) {
        EvCountGraphPoint point;
        point.count = static_cast<double>(binIndex) * graphResolution;
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
            point.confidenceLower = point.meanReward - 1.96 * stderr;
            point.confidenceUpper = point.meanReward + 1.96 * stderr;
        }
        graph.points.push_back(point);
    }
    delete result;
    return graph;
}

EdgeStatistics evaluateSpreadEdge(const MadnessCase& c,
                                  const BasicStrategy& strategy,
                                  const StrategyCountConfig& count,
                                  uint64_t rounds) {
    double threshold = (std::abs(count.system.factor) < 1e-12)
        ? std::numeric_limits<double>::infinity()
        : (-count.system.bias / count.system.factor);

    auto* player = new Player(0.0, strategy.clone());
    player->setNumDecks(c.deckSize);
    player->setCountSystem(count.system);
    player->setCountResolution(count.resolution);
    auto [strategyMinCount, strategyMaxCount] = strategy.getCountRange();
    player->setCountRange(strategyMinCount, strategyMaxCount);
    player->setBettingStrategy(
        std::make_unique<SpreadBetting>(std::vector<std::pair<double, double>>{{threshold, 10.0}}));
    player->enableRoundStats();

    DoubleDownMadnessRules rules = buildRules(c, 1.0, 10.0);
    std::vector<Player*> results = runParallelSimulation(rules, {player}, rounds, g_num_threads);
    delete player;
    if (results.empty())
        throw std::runtime_error("Spread evaluation failed");
    Player* result = results[0];
    EdgeStatistics edge = edgeStatisticsFromPlayer(*result);
    delete result;
    return edge;
}

KellyEvaluationResult evaluateKellyGrowth(const std::string& label,
                                          const MadnessCase& c,
                                          const BasicStrategy& strategy,
                                          const StrategyCountConfig& count,
                                          double kellyFraction) {
    (void)label;
    DoubleDownMadnessRules rules = buildRules(c, 0.0, std::numeric_limits<double>::max());
    double growthSum = 0.0;
    double growthSqSum = 0.0;
    for (int rep = 0; rep < g_kelly_measurements; ++rep) {
        auto* player = new Player(1.0, strategy.clone());
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
            throw std::runtime_error("Kelly evaluation failed");
        Player* result = results[0];
        double avgLogFinal = result->getLogMoney();
        double roundsPerThread = averageRoundsPerThread(kKellyMeasurementRounds, g_num_threads);
        double growthRate = std::isfinite(avgLogFinal)
            ? std::exp((avgLogFinal - std::log(1.0)) / roundsPerThread)
            : 0.0;
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
                                          const MadnessCase& c,
                                          const BasicStrategy& strategy,
                                          const StrategyCountConfig& count,
                                          double predictedOptimalFraction) {
    KellyGrowthCurve curve;
    curve.label = label;
    curve.predictedOptimalFraction = predictedOptimalFraction;
    const KellyFractionRange range = resolveKellyFractionRange(
        predictedOptimalFraction, g_kelly_fraction_min, g_kelly_fraction_max,
        g_kelly_fraction_step);
    for (double fraction : makeKellyFractionGrid(
             range.minimum, range.maximum, g_kelly_fraction_step)) {
        const KellyEvaluationResult result =
            evaluateKellyGrowth(label, c, strategy, count, fraction);
        curve.points.push_back({fraction, result.growthMean, result.growthStddev});
    }
    return curve;
}

std::unique_ptr<BasicStrategy> loadBasicStrategyForCase(const MadnessCase& c) {
    return loadDoubleDownMadnessBasicStrategy(c.version);
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

std::unique_ptr<BasicStrategy> trainFullDeviations(const MadnessCase& c,
                                                   const StrategyCountConfig& count) {
    auto* player = new Player(0.0, makeQStrategy());
    player->setNumDecks(c.deckSize);
    player->setCountSystem(count.system);
    player->setCountResolution(count.resolution);
    player->setCountRange(count.minCount, count.maxCount);

    DoubleDownMadnessRules rules = buildRules(c, 1.0, 1.0);
    std::vector<QLearningStrategy::QTableSnapshot> previousTables(1);
    uint64_t trainedRounds = 0;
    auto wallStart = std::chrono::high_resolution_clock::now();

    while (true) {
        uint64_t batch = (g_training_stop_mode == TrainingStopMode::TABLE_DIFF)
            ? g_sample_rounds
            : std::min(g_num_rounds - trainedRounds, g_num_rounds);
        if (batch == 0) break;

        std::vector<Player*> results = runParallelSimulation(rules, {player}, batch, g_num_threads);
        delete player;
        if (results.empty())
            throw std::runtime_error("Training full deviations failed");
        player = results[0];
        trainedRounds += batch;

        const auto* q = dynamic_cast<const QLearningStrategy*>(player->getStrategy());
        if (!q)
            throw std::runtime_error("Training full deviations failed: result is not QLearningStrategy");

        double avgDiff = std::numeric_limits<double>::infinity();
        if (g_training_stop_mode == TrainingStopMode::TABLE_DIFF) {
            auto currentSnapshot = q->snapshotQTable();
            avgDiff = QLearningStrategy::averageAbsDifference(currentSnapshot, previousTables[0]);
            previousTables[0] = std::move(currentSnapshot);

            double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::high_resolution_clock::now() - wallStart).count() / 1000.0;
            double handsPerSec = (elapsed > 0.0 ? trainedRounds / elapsed : 0.0);
            std::cout << "Full deviations sample rounds=" << trainedRounds
                      << "  speed=" << std::fixed << std::setprecision(0) << handsPerSec << " hands/sec"
                      << "  avg|ΔQ|=" << std::setprecision(6) << avgDiff
                      << "  threshold=" << g_diff_threshold << "\n";
            if (avgDiff <= g_diff_threshold) break;
        } else if (trainedRounds >= g_num_rounds) {
            break;
        }
    }

    const auto* q = dynamic_cast<const QLearningStrategy*>(player->getStrategy());
    if (!q) {
        delete player;
        throw std::runtime_error("Training full deviations failed: final strategy is not QLearningStrategy");
    }
    auto strategy = q->toBasicStrategy();
    delete player;
    return strategy;
}

std::unique_ptr<BasicStrategy> loadFullDeviationsFromCheckpoint(
    const std::string& folder,
    const std::string& agentNameHint,
    const MadnessCase& c,
    CheckpointCountConfig& outCount) {
    namespace fs = std::filesystem;
    fs::path ckptRoot = fs::path(PROJECT_ROOT) / kQlearningCheckpointRoot / folder;
    fs::path metaPath = ckptRoot / "meta.json";
    if (!fs::exists(metaPath))
        throw std::runtime_error("Checkpoint '" + folder + "' not found or missing meta.json");

    json meta; { std::ifstream f(metaPath); f >> meta; }
    const std::string agentName = agentNameHint;
    const bool validPolicyName =
        agentName.size() >= 2 &&
        agentName.front() == 'P' &&
        std::all_of(agentName.begin() + 1, agentName.end(),
                    [](char ch) { return ch >= '0' && ch <= '9'; });
    if (!validPolicyName)
        throw std::invalid_argument(
            "--deviations-agent must name an alternating policy such as P1");
    const int policyIndex = std::stoi(agentName.substr(1));

    json countJson;
    if (policyIndex == 0) {
        countJson = meta.at("algorithm_config").at("initial_count");
    } else {
        const fs::path countPath =
            ckptRoot / ("W" + std::to_string(policyIndex) + ".json");
        if (!fs::exists(countPath)) {
            throw std::runtime_error(
                "Policy " + agentName + " requires missing " +
                countPath.filename().string());
        }
        json countArtifact;
        { std::ifstream f(countPath); f >> countArtifact; }
        countJson = countArtifact.at("count_config");
    }
    for (int i = 0; i < 13; ++i)
        outCount.weights[i] = countJson.at("weights")[i].get<double>();
    outCount.factor = countJson.at("factor").get<double>();
    outCount.bias = countJson.at("bias").get<double>();
    outCount.resolution = countJson.at("resolution").get<double>();
    outCount.minCount = countJson.at("min_count").get<int>();
    outCount.maxCount = countJson.at("max_count").get<int>();

    std::string relPath = std::string(kQlearningCheckpointRoot) + "/" + folder + "/" + agentName + "_agent.json";
    auto q = QLearningStrategy::loadFromFile(relPath);
    if (!q)
        throw std::runtime_error("Failed to load agent file: " + relPath);

    auto strategy = q->toBasicStrategy();
    if (!strategy)
        throw std::runtime_error("Failed to derive greedy policy from agent file: " + relPath);

    if (meta.contains("game")) {
        const auto& g = meta["game"];
        std::vector<std::string> mismatches;
        auto warnIf = [&](const char* key, const std::string& sim, const std::string& ckpt) {
            if (sim != ckpt)
                mismatches.push_back(std::string(key) + ": simulation=" + sim + ", checkpoint=" + ckpt);
        };
        warnIf("decks", std::to_string(c.deckSize), std::to_string(g["decks"].get<int>()));
        warnIf("ss17", c.standSoft17 ? "true" : "false", g["ss17"].get<bool>() ? "true" : "false");
        warnIf("version", std::to_string(static_cast<int>(c.version)),
               std::to_string(g["version"].get<int>()));
        if (!mismatches.empty()) {
            std::cout << "WARNING: simulation rules differ from deviations checkpoint:\n";
            for (const auto& m : mismatches) std::cout << "  " << m << "\n";
        }
    }

    return strategy;
}

EvalResult evaluateCase(const std::string& label,
                        const std::string& note,
                        const MadnessCase& c,
                        const BasicStrategy& strategy,
                        const StrategyCountConfig& count,
                        double predictedKellyFraction) {
    EvalResult result;
    result.label = label;
    result.note = note;
    const uint64_t evalRounds = (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds);
    const EdgeStatistics spread = evaluateSpreadEdge(c, strategy, count, evalRounds);
    result.spreadEdge = spread.mean;
    result.spreadStddev = spread.stddev;
    result.spreadSecondMoment = spread.secondMoment;
    result.kellyCurve = evaluateKellyGrowthCurve(
        label, c, strategy, count, predictedKellyFraction);
    if (const auto* optimum = result.kellyCurve.optimalPoint()) {
        result.kellyFraction = optimum->fraction;
        result.kellyGrowth = optimum->growthMean;
        result.kellyGrowthStddev = optimum->growthStddev;
    }
    return result;
}

void printResultsTable(const std::vector<EvalResult>& results) {
    std::cout << "\n=== Comparison ===\n";
    std::cout << std::left << std::setw(34) << "Policy"
              << std::right << std::setw(18) << "Spread 1:10"
              << std::setw(18) << "Spread std(X)"
              << std::setw(16) << "Kelly fraction"
              << std::setw(18) << "Kelly growth"
              << std::setw(18) << "Kelly stddev" << "\n";
    std::cout << std::string(122, '-') << "\n";
    for (const auto& r : results) {
        std::cout << std::left << std::setw(34) << r.label
                  << std::right << std::setw(18) << std::fixed << std::setprecision(6) << r.spreadEdge
                  << std::setw(18) << std::fixed << std::setprecision(6) << r.spreadStddev
                  << std::setw(16) << std::fixed << std::setprecision(2) << r.kellyFraction
                  << std::setw(18) << std::fixed << std::setprecision(6) << r.kellyGrowth
                  << std::setw(18) << std::fixed << std::setprecision(6) << r.kellyGrowthStddev << "\n";
        if (!r.kellyCurve.points.empty()) {
            std::cout << "  Spread E[X^2]=" << std::setprecision(6) << r.spreadSecondMoment
                      << "  predicted Kelly multiplier from flat 1/E[X^2]="
                      << std::setprecision(6) << r.kellyCurve.predictedOptimalFraction << "\n";
        }
        if (!r.note.empty())
            std::cout << "  " << r.note << "\n";
    }
}

void runCase(const MadnessCase& c) {
    const std::string runName = currentTimestamp() + "_" + ToStringTableName(c);
    RunLogger logger(std::filesystem::path(PROJECT_ROOT) / kCompareCheckpointRoot, runName);
    const uint64_t evalRounds = (g_eval_rounds == 0 ? g_num_rounds : g_eval_rounds);
    std::cout << "\n=== DoubleDownMadnessCompareCountStrategies ===\n";
    std::cout << "Command:       " << g_command_line << "\n";
    std::cout << "Scenario:      " << ToString(c)
              << "_penetration=" << g_penetration << "%\n";
    std::cout << "Eval rounds:   " << evalRounds << "  Threads: " << g_num_threads << "\n";
    std::cout << "Kelly eval:    " << g_kelly_measurements << " x " << kKellyMeasurementRounds << " rounds\n";
    std::cout << "Kelly sweep:   centered at nearest step to 1/E[X^2] +/- 0.25 (minimum 0)";
    if (g_kelly_fraction_min) std::cout << "  min override=" << *g_kelly_fraction_min;
    if (g_kelly_fraction_max) std::cout << "  max override=" << *g_kelly_fraction_max;
    std::cout << "  step " << g_kelly_fraction_step << "\n";
    std::cout << "Train rounds:  " << g_num_rounds << "\n";
    std::cout << "Penetration:   " << g_penetration << "%\n";
    std::cout << "Count grid:    res=" << g_count_resolution
              << "  range=[" << g_min_count << "," << g_max_count << "]\n";
    std::cout << "===============================\n";

    StrategyCountConfig requestedCount = resolveCountConfig();
    std::vector<EvalResult> results;
    std::vector<StrategyGraphArtifact> graphs;
    std::unique_ptr<BasicStrategy> fullDeviations;
    std::string fullNote;

    if (!g_deviations_checkpoint.empty()) {
        CheckpointCountConfig ckptCount;
        fullDeviations = loadFullDeviationsFromCheckpoint(g_deviations_checkpoint, g_deviations_agent, c, ckptCount);
        fullNote = "Loaded from checkpoint " + g_deviations_checkpoint;

        StrategyCountConfig checkpointCount;
        checkpointCount.system.weights = ckptCount.weights;
        checkpointCount.system.factor = ckptCount.factor;
        checkpointCount.system.bias = ckptCount.bias;
        checkpointCount.resolution = ckptCount.resolution;
        checkpointCount.minCount = ckptCount.minCount;
        checkpointCount.maxCount = ckptCount.maxCount;

        if (!countWasExplicitlyConfigured()) {
            requestedCount = checkpointCount;
            g_referenceBias = requestedCount.system.bias;
            g_referenceFactor = requestedCount.system.factor;
            std::cout << "Count:       loaded W" << g_deviations_agent.substr(1)
                      << " for " << g_deviations_agent << "\n";
        }

        bool countMismatch = false;
        for (size_t i = 0; i < ckptCount.weights.size(); ++i) {
            if (!approxEqual(ckptCount.weights[i], requestedCount.system.weights[i])) {
                countMismatch = true;
                break;
            }
        }
        if (!approxEqual(ckptCount.factor, requestedCount.system.factor) ||
            !approxEqual(ckptCount.bias, requestedCount.system.bias) ||
            !approxEqual(ckptCount.resolution, requestedCount.resolution) ||
            ckptCount.minCount != requestedCount.minCount ||
            ckptCount.maxCount != requestedCount.maxCount) {
            countMismatch = true;
        }
        if (countMismatch) {
            std::cout << "WARNING: loaded deviations checkpoint was trained with a different count setup.\n";
            std::cout << "         Both policies will be evaluated on the explicitly requested count config.\n";
            fullNote += " (evaluated on requested count config; checkpoint training count differed)";
        }
    }

    std::cout << "Effective count grid: res=" << requestedCount.resolution
              << "  range=[" << requestedCount.minCount << ","
              << requestedCount.maxCount << "]"
              << "  factor=" << requestedCount.system.factor
              << "  bias=" << requestedCount.system.bias << "\n";

    auto basicStrategy = loadBasicStrategyForCase(c);
    graphs.push_back(measureEvCountGraph(
        c, "Known basic strategy", *basicStrategy, requestedCount, evalRounds));
    results.push_back(evaluateCase(
        "Known basic strategy", "", c, *basicStrategy, requestedCount,
        graphs.back().flatStatistics.secondMoment > 0.0
            ? 1.0 / graphs.back().flatStatistics.secondMoment : 0.0));

    if (!fullDeviations) {
        fullDeviations = trainFullDeviations(c, requestedCount);
        fullNote = (g_training_stop_mode == TrainingStopMode::TABLE_DIFF)
            ? "Trained fresh with diff stop mode"
            : "Trained fresh for fixed rounds";
    }

    if (g_verbose) {
        std::cout << "\n=== Full deviations strategy ===\n";
        std::cout << *fullDeviations;
        std::cout << "===============================\n";
    }
    logger.fileStream() << "\n=== Full deviations strategy ===\n";
    logger.fileStream() << *fullDeviations;
    logger.fileStream() << "===============================\n";
    graphs.push_back(measureEvCountGraph(
        c, "Full deviations", *fullDeviations, requestedCount, evalRounds));
    results.push_back(evaluateCase(
        "Full deviations", fullNote, c, *fullDeviations, requestedCount,
        graphs.back().flatStatistics.secondMoment > 0.0
            ? 1.0 / graphs.back().flatStatistics.secondMoment : 0.0));

    printResultsTable(results);

    json graphsJson = graphsToJson(graphs);
    const StrategyGraphArtifact& histogramGraph = graphs.front();
    json histJson = histogramToJson(histogramGraph);
    std::ofstream(logger.pathFor("ev_count_graph.json")) << std::setw(2) << graphsJson << "\n";
    std::ofstream(logger.pathFor("ev_count_graph.svg")) << graphsToSvg(graphs);
    std::ofstream(logger.pathFor("count_histograms.json")) << std::setw(2) << histJson << "\n";
    std::ofstream(logger.pathFor("count_histograms.svg")) << histogramToSvg(histogramGraph);
    std::vector<ConditionalSecondMomentCurve> secondMomentCurves;
    secondMomentCurves.reserve(graphs.size());
    for (const auto& graph : graphs)
        secondMomentCurves.push_back(conditionalSecondMomentCurve(graph));
    std::ofstream(logger.pathFor("second_moment_count_graph.json"))
        << std::setw(2)
        << conditionalSecondMomentCurvesToJson(secondMomentCurves) << "\n";
    std::ofstream(logger.pathFor("second_moment_count_graph.svg"))
        << conditionalSecondMomentCurvesToSvg(
               "Conditional second moment by strategy", secondMomentCurves);
    std::vector<KellyGrowthCurve> kellyCurves;
    for (const auto& result : results) {
        if (!result.kellyCurve.points.empty()) kellyCurves.push_back(result.kellyCurve);
    }
    std::ofstream(logger.pathFor("kelly_fraction_graph.json"))
        << std::setw(2) << kellyGrowthCurvesToJson(kellyCurves) << "\n";
    std::ofstream(logger.pathFor("kelly_fraction_graph.svg"))
        << kellyGrowthCurvesToSvg("Kelly growth by strategy", kellyCurves);

    std::cout << "\nSaved artifacts:\n";
    std::cout << "  " << logger.pathFor("run.log").string() << "\n";
    std::cout << "  " << logger.pathFor("ev_count_graph.json").string() << "\n";
    std::cout << "  " << logger.pathFor("ev_count_graph.svg").string() << "\n";
    std::cout << "  " << logger.pathFor("count_histograms.json").string() << "\n";
    std::cout << "  " << logger.pathFor("count_histograms.svg").string() << "\n";
    std::cout << "  " << logger.pathFor("second_moment_count_graph.json").string() << "\n";
    std::cout << "  " << logger.pathFor("second_moment_count_graph.svg").string() << "\n";
    std::cout << "  " << logger.pathFor("kelly_fraction_graph.json").string() << "\n";
    std::cout << "  " << logger.pathFor("kelly_fraction_graph.svg").string() << "\n";
}

void printHelp(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n\n";
    std::cout << "Compare a Double Down Madness count under two play policies:\n";
    std::cout << "  1. Known version-specific basic strategy.\n";
    std::cout << "  2. Full deviations learned for the count or loaded from an alternating checkpoint.\n\n";

    std::cout << "EVALUATION:\n";
    std::cout << "  --eval-rounds <N>      Spread 1:10 evaluation rounds (default: --num-rounds)\n";
    std::cout << "  --num-rounds <N>       Training rounds for fresh full deviations (default: 1000000000)\n";
    std::cout << "  --num-threads <N>      Threads (default: 10)\n";
    std::cout << "  --penetration <val>    Shoe penetration % (default: 75.0)\n";
    std::cout << "  --kelly-measurements <N>  Experiments per Kelly fraction, 1,000,000 rounds each (default: 10)\n\n";
    std::cout << "  --kelly-fraction-min <v>   Override sweep minimum\n";
    std::cout << "  --kelly-fraction-max <v>   Override sweep maximum\n";
    std::cout << "                              Default: nearest step to 1/E[X^2] +/- 0.25, clamped at 0\n";
    std::cout << "  --kelly-fraction-step <v>  Kelly multiplier sweep step (default: 0.05)\n\n";

    std::cout << "COUNT SPECIFICATION (pick one):\n";
    std::cout << "  --count <name>         none, hilo, ko, hiopt1, hiopt2, omega2, zen, halves\n";
    std::cout << "                         Default: hilo\n";
    std::cout << "  --count-weights <csv>  Explicit 13 weights\n";
    std::cout << "  --count-ols <dir>      Load OLS weights from checkpoints/checkpoints_ols/<dir>/data.json\n";
    std::cout << "  --factor <val>         Override E[game] factor\n";
    std::cout << "  --bias <val>           Override E[game] bias\n";
    std::cout << "  --count-resolution <v> (default: 1.0)\n";
    std::cout << "  --min-count <N>        (default: -5)\n";
    std::cout << "  --max-count <N>        (default: 5)\n\n";

    std::cout << "FULL DEVIATIONS:\n";
    std::cout << "  --deviations-checkpoint <dir>  Load full deviations from checkpoints/double-down-madness-alternating-checkpoints/<dir>/\n";
    std::cout << "  --deviations-agent <Pk>        Policy name within that checkpoint, e.g. P1\n";
    std::cout << "  Without explicit count flags, loading Pk also loads its matching Wk.\n";
    std::cout << "  If no checkpoint is given, full deviations are trained fresh using the RL flags below.\n\n";

    std::cout << "RL TRAINING FLAGS (fresh full deviations only):\n";
    std::cout << "  --stop-mode <rounds|diff>   (default: rounds)\n";
    std::cout << "  --sample-rounds <N>         Diff-mode sample interval (default: 100000000)\n";
    std::cout << "  --diff-threshold <val>      Diff stop threshold (default: 0.001)\n";
    std::cout << "  --exploration <mode>        epsilon or boltzmann (default: epsilon)\n";
    std::cout << "  --epsilon-start/min/decay   (defaults: 1.0 / 0.1 / 0.99999)\n";
    std::cout << "  --temp-start/min/decay      (defaults: 1.0 / 0.1 / 0.99999)\n";
    std::cout << "  --alpha-start/min/decay     (defaults: 0.01 / 0.0001 / 100)\n\n";

    std::cout << "OUTPUT:\n";
    std::cout << "  --verbose                   Print the final full-deviations strategy table\n\n";
    std::cout << "  Logs and graph artifacts are saved under checkpoints/DoubleDownMadnessCompareCountStrategies/<run-name>/\n";
    std::cout << "  Files include run.log, EV/count, unit-wager conditional second-moment,\n";
    std::cout << "  histogram, and kelly_fraction_graph artifacts.\n\n";

    std::cout << "GAME CONFIG:\n";
    std::cout << "  --decks <list>          Deck counts (default: [6])\n";
    std::cout << "  --ss17 <list>           Dealer stands on soft 17 (default: [false]/H17)\n";
    std::cout << "  --version <list>        DDM versions 1-3 (default: [1])\n";
}

} // namespace

int runDoubleDownMadnessCompareCountStrategies(int argc, char** argv) {
    g_command_line = commandLineFromArgs(argc, argv);
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") { printHelp(argv[0]); return 0; }
        else if ((arg == "--game" || arg == "--table-type") && i+1<argc) { ++i; }

        else if (arg == "--num-rounds"       && i+1<argc) g_num_rounds = std::stoull(argv[++i]);
        else if (arg == "--eval-rounds"      && i+1<argc) g_eval_rounds = std::stoull(argv[++i]);
        else if (arg == "--num-threads"      && i+1<argc) g_num_threads = std::stoi(argv[++i]);
        else if (arg == "--penetration"      && i+1<argc) g_penetration = std::stod(argv[++i]);
        else if (arg == "--kelly-measurements" && i+1<argc) g_kelly_measurements = std::stoi(argv[++i]);
        else if (arg == "--kelly-fraction-min" && i+1<argc) g_kelly_fraction_min = std::stod(argv[++i]);
        else if (arg == "--kelly-fraction-max" && i+1<argc) g_kelly_fraction_max = std::stod(argv[++i]);
        else if (arg == "--kelly-fraction-step" && i+1<argc) g_kelly_fraction_step = std::stod(argv[++i]);

        else if (arg == "--count"            && i+1<argc) g_count_name = argv[++i];
        else if (arg == "--count-weights"    && i+1<argc) g_count_weights_str = argv[++i];
        else if (arg == "--count-ols"        && i+1<argc) g_count_ols = argv[++i];
        else if (arg == "--factor"           && i+1<argc) g_factor = std::stod(argv[++i]);
        else if (arg == "--bias"             && i+1<argc) g_bias = std::stod(argv[++i]);
        else if (arg == "--count-resolution" && i+1<argc) g_count_resolution = std::stod(argv[++i]);
        else if (arg == "--min-count"        && i+1<argc) g_min_count = std::stoi(argv[++i]);
        else if (arg == "--max-count"        && i+1<argc) g_max_count = std::stoi(argv[++i]);

        else if (arg == "--deviations-checkpoint" && i+1<argc) g_deviations_checkpoint = argv[++i];
        else if (arg == "--deviations-agent"      && i+1<argc) g_deviations_agent = argv[++i];

        else if (arg == "--stop-mode"        && i+1<argc) {
            std::string mode = argv[++i];
            g_training_stop_mode = (mode == "diff") ? TrainingStopMode::TABLE_DIFF
                                                    : TrainingStopMode::FIXED_ROUNDS;
        }
        else if (arg == "--sample-rounds"    && i+1<argc) g_sample_rounds = std::stoull(argv[++i]);
        else if (arg == "--diff-threshold"   && i+1<argc) g_diff_threshold = std::stod(argv[++i]);
        else if (arg == "--exploration"      && i+1<argc) {
            std::string mode = argv[++i];
            g_agent.explorationMode = (mode == "boltzmann")
                ? ExplorationMode::BOLTZMANN
                : ExplorationMode::EPSILON_GREEDY;
        }
        else if (arg == "--epsilon-start"    && i+1<argc) g_agent.epsilonStart = std::stod(argv[++i]);
        else if (arg == "--epsilon-min"      && i+1<argc) g_agent.epsilonMin = std::stod(argv[++i]);
        else if (arg == "--epsilon-decay"    && i+1<argc) g_agent.epsilonDecay = std::stod(argv[++i]);
        else if (arg == "--temp-start"       && i+1<argc) g_agent.tempStart = std::stod(argv[++i]);
        else if (arg == "--temp-min"         && i+1<argc) g_agent.tempMin = std::stod(argv[++i]);
        else if (arg == "--temp-decay"       && i+1<argc) g_agent.tempDecay = std::stod(argv[++i]);
        else if (arg == "--alpha-start"      && i+1<argc) g_agent.alphaStart = std::stod(argv[++i]);
        else if (arg == "--alpha-min"        && i+1<argc) g_agent.alphaMin = std::stod(argv[++i]);
        else if (arg == "--alpha-decay"      && i+1<argc) g_agent.alphaDecaySteps = std::stod(argv[++i]);

        else if (arg == "--verbose") g_verbose = true;

        else if (arg == "--decks" && i+1<argc) g_deck_sizes         = parseList<int>(argv[++i]);
        else if (arg == "--ss17"  && i+1<argc) g_stand_soft17       = parseList<bool>(argv[++i]);
        else if (arg == "--version" && i+1<argc) g_versions         = parseList<int>(argv[++i]);
        else {
            std::cerr << "Unknown argument: " << arg << "\nUse --help.\n";
            return 1;
        }
    }

    if (!g_deviations_agent.empty() && g_deviations_checkpoint.empty()) {
        std::cerr << "Error: --deviations-agent requires --deviations-checkpoint.\n";
        return 1;
    }
    if (!g_deviations_checkpoint.empty() && g_deviations_agent.empty()) {
        std::cerr << "Error: --deviations-checkpoint requires --deviations-agent Pk.\n";
        return 1;
    }
    if (g_sample_rounds == 0 && g_training_stop_mode == TrainingStopMode::TABLE_DIFF) {
        std::cerr << "Error: --sample-rounds must be >= 1 in diff mode.\n";
        return 1;
    }
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

    try {
        const auto cases = generateMadnessCases();
        for (const auto& c : cases)
            runCase(c);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
