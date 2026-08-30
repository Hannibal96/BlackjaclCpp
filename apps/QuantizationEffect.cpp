#include "GameAppDispatcher.h"
#include "GameTraits.h"
#include "Game/BettingStrategy.h"
#include "Game/Player.h"
#include "RL/BasicStrategy.h"
#include "Utils/CountQuantization.h"
#include "Utils/RunLogger.h"
#include "Utils/SimulationAnalysis.h"
#include "Utils/Utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

template <typename Game>
struct QuantizationEffectApp {
    using Rules = typename Game::Rules;
    using Case = typename Game::Case;

    static constexpr uint64_t kDefaultKellyMeasurementRounds = 1'000'000ULL;
    static constexpr double kDefaultKellyMinimum = 0.75;
    static constexpr double kDefaultKellyMaximum = 1.25;
    static constexpr double kDefaultKellyStep = 0.05;
    static constexpr double kBalanceTolerance = 1e-9;

    struct CountConfig {
        CountingSystem system;
        double resolution = 1.0;
        int minCount = -5;
        int maxCount = 5;
    };

    struct KellyEvaluationResult {
        double growthMean = 1.0;
        double growthStddev = 0.0;
    };

    struct Evaluation {
        double quantum = 0.0;
        CountConfig count;
        bool sourceWasBalanced = false;
        bool balancePreserved = false;
        double weightSum = 0.0;
        double floatingPointWeightSum = 0.0;
        double spreadThreshold = 0.0;
        EdgeStatistics spread;
        KellyGrowthCurve kellyCurve;
    };

    static inline std::string g_checkpoint_input;
    static inline std::string g_count_label;
    static inline std::string g_policy_label;
    static inline std::string g_output_dir;
    static inline std::string g_command_line;
    static inline std::vector<double> g_quanta = {0.0, 0.01, 0.05, 0.1, 0.5, 1.0};
    static inline std::optional<uint64_t> g_eval_rounds_override;
    static inline std::optional<uint64_t> g_kelly_rounds_override;
    static inline std::optional<int> g_kelly_measurements_override;
    static inline std::optional<int> g_num_threads_override;
    static inline std::optional<double> g_kelly_minimum_override;
    static inline std::optional<double> g_kelly_maximum_override;
    static inline std::optional<double> g_kelly_step_override;
    static inline std::optional<uint64_t> g_seed;

    static inline uint64_t g_eval_rounds = 0;
    static inline uint64_t g_kelly_rounds = kDefaultKellyMeasurementRounds;
    static inline int g_kelly_measurements = 10;
    static inline int g_num_threads = 10;
    static inline double g_kelly_minimum = kDefaultKellyMinimum;
    static inline double g_kelly_maximum = kDefaultKellyMaximum;
    static inline double g_kelly_step = kDefaultKellyStep;
    static inline double g_penetration = 75.0;

    static std::string currentTimestamp() {
        const time_t now = time(nullptr);
        char buffer[20];
        strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", localtime(&now));
        return buffer;
    }

    static std::string formatNumber(double value) {
        if (value == 0.0) return "0";
        std::ostringstream output;
        output << std::setprecision(12) << value;
        return output.str();
    }

    static std::vector<double> parseQuanta(const std::string& text) {
        std::string cleaned = text;
        if (!cleaned.empty() && cleaned.front() == '[') cleaned.erase(cleaned.begin());
        if (!cleaned.empty() && cleaned.back() == ']') cleaned.pop_back();

        std::vector<double> values;
        std::istringstream input(cleaned);
        std::string token;
        while (std::getline(input, token, ',')) {
            const size_t first = token.find_first_not_of(" \t");
            const size_t last = token.find_last_not_of(" \t");
            if (first == std::string::npos)
                throw std::invalid_argument("Quantization list contains an empty value");
            const double value = std::stod(token.substr(first, last - first + 1));
            if (!std::isfinite(value) || value < 0.0)
                throw std::invalid_argument("Quantization values must be finite and non-negative");
            values.push_back(value);
        }
        if (values.empty())
            throw std::invalid_argument("Quantization list cannot be empty");

        std::vector<double> unique;
        for (double value : values) {
            if (std::find(unique.begin(), unique.end(), value) == unique.end())
                unique.push_back(value);
        }
        return unique;
    }

    static bool validLabel(const std::string& label, char prefix) {
        return label.size() >= 2 && label.front() == prefix &&
            std::all_of(label.begin() + 1, label.end(), [](char character) {
                return character >= '0' && character <= '9';
            });
    }

    static int labelIndex(const std::string& label, char prefix) {
        if (!validLabel(label, prefix))
            throw std::invalid_argument(
                std::string("Expected a label such as ") + prefix + "3, got '" + label + "'");
        return std::stoi(label.substr(1));
    }

    static std::filesystem::path resolveCheckpointPath(const std::string& input) {
        namespace fs = std::filesystem;
        const fs::path requested(input);
        std::vector<fs::path> candidates;
        if (requested.is_absolute()) {
            candidates.push_back(requested);
        } else {
            candidates.push_back(requested);
            candidates.push_back(fs::path(PROJECT_ROOT) / requested);
            candidates.push_back(
                fs::path(PROJECT_ROOT) / "checkpoints" /
                (std::string(Game::kCheckpointDirPrefix) + "alternating-checkpoints") /
                requested);
        }

        for (const auto& candidate : candidates) {
            if (fs::is_directory(candidate) && fs::is_regular_file(candidate / "meta.json"))
                return fs::weakly_canonical(candidate);
        }
        throw std::runtime_error(
            "Cannot find alternating checkpoint folder or meta.json for '" + input + "'");
    }

    static json readJson(const std::filesystem::path& path) {
        std::ifstream input(path);
        if (!input.is_open())
            throw std::runtime_error("Cannot open " + path.string());
        json value;
        input >> value;
        return value;
    }

    static std::pair<std::string, std::string> resolveLabels(
            const std::filesystem::path& checkpoint) {
        namespace fs = std::filesystem;
        if (g_count_label.empty()) {
            const std::regex weightPattern(R"(^W([1-9][0-9]*)\.json$)");
            int latest = -1;
            for (const auto& entry : fs::directory_iterator(checkpoint)) {
                if (!entry.is_regular_file()) continue;
                std::smatch match;
                const std::string filename = entry.path().filename().string();
                if (!std::regex_match(filename, match, weightPattern)) continue;
                const int countIndex = std::stoi(match[1].str());
                const fs::path policyPath =
                    checkpoint / ("P" + std::to_string(countIndex - 1) + "_strategy.json");
                if (fs::is_regular_file(policyPath)) latest = std::max(latest, countIndex);
            }
            if (latest < 1)
                throw std::runtime_error(
                    "Checkpoint contains no complete Wk + P(k-1) pair");
            g_count_label = "W" + std::to_string(latest);
        }

        const int countIndex = labelIndex(g_count_label, 'W');
        if (countIndex < 1)
            throw std::invalid_argument("W0 is implicit and is not a learned count artifact");
        if (g_policy_label.empty())
            g_policy_label = "P" + std::to_string(countIndex - 1);
        (void)labelIndex(g_policy_label, 'P');

        const fs::path countPath = checkpoint / (g_count_label + ".json");
        const fs::path policyPath = checkpoint / (g_policy_label + "_strategy.json");
        if (!fs::is_regular_file(countPath))
            throw std::runtime_error("Missing count artifact " + countPath.string());
        if (!fs::is_regular_file(policyPath))
            throw std::runtime_error("Missing policy artifact " + policyPath.string());
        return {g_count_label, g_policy_label};
    }

    static CountConfig loadCount(const std::filesystem::path& checkpoint,
                                 const std::string& countLabel,
                                 std::string& precisionSource) {
        const json artifact = readJson(checkpoint / (countLabel + ".json"));
        const json& stored = artifact.at("count_config");

        CountConfig count;
        count.system.factor = stored.value("factor", 1.0);
        count.system.bias = stored.value("bias", 0.0);
        count.system.continuousBettingCount =
            stored.value("continuous_betting_count", false);
        count.resolution = stored.value("resolution", 1.0);
        count.minCount = stored.value("min_count", -5);
        count.maxCount = stored.value("max_count", 5);

        const bool canReconstruct =
            artifact.contains("raw_solution") && artifact.at("raw_solution").is_array() &&
            artifact.at("raw_solution").size() == 14 &&
            artifact.contains("normalization_scale") &&
            artifact.at("normalization_scale").is_number();
        if (canReconstruct) {
            const double scale = artifact.at("normalization_scale").get<double>();
            if (!std::isfinite(scale) || std::abs(scale) < 1e-15)
                throw std::runtime_error("Invalid normalization_scale in " + countLabel + ".json");
            for (size_t i = 0; i < count.system.weights.size(); ++i) {
                count.system.weights[i] =
                    artifact.at("raw_solution").at(i).get<double>() * scale;
            }
            const double tenValueAverage =
                (count.system.weights[8] + count.system.weights[9] +
                 count.system.weights[10] + count.system.weights[11]) / 4.0;
            count.system.weights[8] = count.system.weights[9] =
                count.system.weights[10] = count.system.weights[11] = tenValueAverage;
            precisionSource = "raw_solution x normalization_scale";
        } else {
            if (!stored.contains("weights") || stored.at("weights").size() != 13)
                throw std::runtime_error("Count artifact requires exactly 13 weights");
            for (size_t i = 0; i < count.system.weights.size(); ++i)
                count.system.weights[i] = stored.at("weights").at(i).get<double>();
            precisionSource = "rounded count_config fallback";
            std::cout << "WARNING: " << countLabel
                      << " cannot be reconstructed at full precision; using saved count_config weights.\n";
        }
        return count;
    }

    static std::unique_ptr<BasicStrategy> loadPolicy(
            const std::filesystem::path& checkpoint,
            const std::string& policyLabel) {
        auto policy = std::make_unique<BasicStrategy>();
        const std::filesystem::path path = checkpoint / (policyLabel + "_strategy.json");
        if (!policy->loadFromFile(path.string()))
            throw std::runtime_error("Cannot load policy " + path.string());
        return policy;
    }

    static uint64_t mixSeed(uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    static std::optional<uint64_t> commonSeed(uint64_t stream) {
        if (!g_seed) return std::nullopt;
        return mixSeed(*g_seed + stream);
    }

    static double sumWeights(const std::array<double, 13>& weights) {
        double sum = 0.0;
        for (double weight : weights) sum += weight;
        return sum;
    }

    static double spreadThreshold(const CountConfig& count) {
        if (std::abs(count.system.factor) < 1e-12)
            return std::numeric_limits<double>::infinity();
        const double threshold = -count.system.bias / count.system.factor;
        return count.system.continuousBettingCount
            ? std::nextafter(threshold, std::numeric_limits<double>::infinity())
            : threshold;
    }

    static void configurePlayer(Player& player,
                                const Case& gameCase,
                                const BasicStrategy& policy,
                                const CountConfig& count) {
        player.setNumDecks(gameCase.deckSize);
        player.setCountSystem(count.system);
        player.setCountResolution(count.resolution);
        const auto [minimum, maximum] = policy.getCountRange();
        player.setCountRange(minimum, maximum);
    }

    static EdgeStatistics evaluateSpread(const Case& gameCase,
                                         const BasicStrategy& policy,
                                         const CountConfig& count) {
        auto* player = new Player(0.0, policy.clone());
        configurePlayer(*player, gameCase, policy, count);
        player->setBettingStrategy(std::make_unique<SpreadBetting>(
            std::vector<std::pair<double, double>>{{spreadThreshold(count), 10.0}}));
        player->enableRoundStats();

        const Rules rules = Game::buildRules(gameCase, g_penetration, 1.0, 10.0);
        std::vector<Player*> results = runParallelSimulation(
            rules, {player}, g_eval_rounds, g_num_threads, commonSeed(0x535052454144ULL));
        delete player;
        if (results.empty())
            throw std::runtime_error("Spread evaluation failed");
        const EdgeStatistics statistics = edgeStatisticsFromPlayer(*results.front());
        delete results.front();
        return statistics;
    }

    static KellyEvaluationResult evaluateKelly(
            const Case& gameCase,
            const BasicStrategy& policy,
            const CountConfig& count,
            double multiplier,
            size_t multiplierIndex) {
        const Rules rules = Game::buildRules(
            gameCase, g_penetration, 0.0, std::numeric_limits<double>::max());
        double sum = 0.0;
        double sumSquares = 0.0;

        for (int measurement = 0; measurement < g_kelly_measurements; ++measurement) {
            auto* player = new Player(1.0, policy.clone());
            configurePlayer(*player, gameCase, policy, count);
            player->setBettingStrategy(std::make_unique<KellyBetting>(multiplier));
            player->setEnforceBankrollActionLimits(true);

            const uint64_t stream = 0x4b454c4c59000000ULL
                + static_cast<uint64_t>(multiplierIndex) *
                    static_cast<uint64_t>(g_kelly_measurements)
                + static_cast<uint64_t>(measurement);
            std::vector<Player*> results = runParallelSimulation(
                rules, {player}, g_kelly_rounds, g_num_threads, commonSeed(stream));
            delete player;
            if (results.empty())
                throw std::runtime_error("Kelly evaluation failed");

            Player* result = results.front();
            const double averageLogFinal = result->getLogMoney();
            const double roundsPerThread =
                static_cast<double>(g_kelly_rounds) / static_cast<double>(g_num_threads);
            const double growth = std::isfinite(averageLogFinal) && roundsPerThread > 0.0
                ? std::exp(averageLogFinal / roundsPerThread)
                : 0.0;
            sum += growth;
            sumSquares += growth * growth;
            delete result;
        }

        KellyEvaluationResult result;
        const double measurements = static_cast<double>(g_kelly_measurements);
        result.growthMean = sum / measurements;
        if (g_kelly_measurements > 1) {
            const double sampleVariance =
                (sumSquares - sum * sum / measurements) / (measurements - 1.0);
            result.growthStddev = std::sqrt(std::max(0.0, sampleVariance));
        }
        return result;
    }

    static KellyGrowthCurve evaluateKellyCurve(const Case& gameCase,
                                               const BasicStrategy& policy,
                                               const CountConfig& count,
                                               double quantum) {
        KellyGrowthCurve curve;
        curve.label = "quantum=" + formatNumber(quantum);
        curve.predictedOptimalFraction = 1.0;
        const auto multipliers = makeKellyFractionGrid(
            g_kelly_minimum, g_kelly_maximum, g_kelly_step);
        for (size_t index = 0; index < multipliers.size(); ++index) {
            const double multiplier = multipliers[index];
            std::cout << "  Kelly multiplier " << formatNumber(multiplier)
                      << " (" << g_kelly_measurements << " measurements)\n";
            const KellyEvaluationResult result =
                evaluateKelly(gameCase, policy, count, multiplier, index);
            curve.points.push_back(
                {multiplier, result.growthMean, result.growthStddev});
        }
        return curve;
    }

    static const KellyGrowthPoint* pointAtOne(const KellyGrowthCurve& curve) {
        for (const auto& point : curve.points) {
            if (std::abs(point.fraction - 1.0) <= g_kelly_step * 1e-9)
                return &point;
        }
        return nullptr;
    }

    static void printWeights(double quantum, const Evaluation& evaluation) {
        static constexpr std::array<const char*, 13> ranks = {
            "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"
        };
        std::cout << "\n=== Quantum " << formatNumber(quantum);
        if (quantum == 0.0) std::cout << " (exact fresh reference)";
        std::cout << " ===\n";
        std::cout << std::left << std::setw(8) << "Rank" << std::right << std::setw(18)
                  << "Weight" << "\n";
        for (size_t i = 0; i < ranks.size(); ++i) {
            std::cout << std::left << std::setw(8) << ranks[i] << std::right
                      << std::setw(18) << std::fixed << std::setprecision(10)
                      << evaluation.count.system.weights[i] << "\n";
        }
        std::cout << "Sum(weights): " << std::setprecision(15) << evaluation.weightSum;
        if (evaluation.sourceWasBalanced)
            std::cout << "  (zero-sum constraint "
                      << (evaluation.balancePreserved ? "preserved" : "FAILED") << ")";
        if (evaluation.balancePreserved && evaluation.floatingPointWeightSum != 0.0)
            std::cout << "  floating residual=" << evaluation.floatingPointWeightSum;
        std::cout << "\nSpread threshold: " << evaluation.spreadThreshold << "\n";
    }

    static json weightsToJson(const std::array<double, 13>& weights) {
        json result = json::array();
        for (double weight : weights) result.push_back(weight);
        return result;
    }

    static json curveToJson(const KellyGrowthCurve& curve) {
        json result;
        result["search_minimum"] = g_kelly_minimum;
        result["search_maximum"] = g_kelly_maximum;
        result["search_step"] = g_kelly_step;
        result["points"] = json::array();
        for (const auto& point : curve.points) {
            result["points"].push_back({
                {"multiplier", point.fraction},
                {"growth_mean", point.growthMean},
                {"growth_stddev", point.growthStddev}
            });
        }
        if (const auto* atOne = pointAtOne(curve)) {
            result["at_multiplier_1"] = {
                {"growth_mean", atOne->growthMean},
                {"growth_stddev", atOne->growthStddev}
            };
        } else {
            result["at_multiplier_1"] = nullptr;
        }
        if (const auto* optimum = curve.optimalPoint()) {
            result["empirical_optimum"] = {
                {"multiplier", optimum->fraction},
                {"growth_mean", optimum->growthMean},
                {"growth_stddev", optimum->growthStddev}
            };
        }
        return result;
    }

    static json resultsToJson(const std::filesystem::path& checkpoint,
                              const json& sourceMeta,
                              const CountConfig& sourceCount,
                              const std::string& precisionSource,
                              const std::vector<Evaluation>& evaluations) {
        json root;
        root["app"] = "QuantizationEffect";
        root["command"] = g_command_line;
        root["source_checkpoint"] = checkpoint.string();
        root["source_count"] = g_count_label;
        root["source_policy"] = g_policy_label;
        root["weight_precision_source"] = precisionSource;
        root["source_meta"] = sourceMeta;
        root["configuration"] = {
            {"eval_rounds", g_eval_rounds},
            {"num_threads", g_num_threads},
            {"kelly_measurements", g_kelly_measurements},
            {"kelly_rounds_per_measurement", g_kelly_rounds},
            {"kelly_multiplier_minimum", g_kelly_minimum},
            {"kelly_multiplier_maximum", g_kelly_maximum},
            {"kelly_multiplier_step", g_kelly_step},
            {"seed", g_seed ? json(*g_seed) : json(nullptr)},
            {"common_random_seeds_across_quantization_levels", g_seed.has_value()}
        };
        root["source_count_config"] = {
            {"weights", weightsToJson(sourceCount.system.weights)},
            {"weight_sum", sumWeights(sourceCount.system.weights)},
            {"factor", sourceCount.system.factor},
            {"bias", sourceCount.system.bias},
            {"continuous_betting_count", sourceCount.system.continuousBettingCount},
            {"resolution", sourceCount.resolution},
            {"min_count", sourceCount.minCount},
            {"max_count", sourceCount.maxCount}
        };
        root["results"] = json::array();

        const Evaluation& reference = evaluations.front();
        const KellyGrowthPoint* referenceAtOne = pointAtOne(reference.kellyCurve);
        const KellyGrowthPoint* referenceOptimum = reference.kellyCurve.optimalPoint();
        for (const auto& evaluation : evaluations) {
            json row;
            row["quantum"] = evaluation.quantum;
            row["weights"] = weightsToJson(evaluation.count.system.weights);
            row["weight_sum"] = evaluation.weightSum;
            row["floating_point_weight_sum"] = evaluation.floatingPointWeightSum;
            row["source_was_balanced"] = evaluation.sourceWasBalanced;
            row["balance_preserved"] = evaluation.balancePreserved;
            row["spread_threshold"] = evaluation.spreadThreshold;
            const double standardError = evaluation.spread.samples > 0
                ? evaluation.spread.stddev /
                    std::sqrt(static_cast<double>(evaluation.spread.samples))
                : 0.0;
            row["spread"] = {
                {"rounds", evaluation.spread.samples},
                {"edge_per_round", evaluation.spread.mean},
                {"outcome_stddev", evaluation.spread.stddev},
                {"outcome_second_moment", evaluation.spread.secondMoment},
                {"edge_standard_error", standardError},
                {"edge_delta_from_quantum_0", evaluation.spread.mean - reference.spread.mean}
            };
            row["kelly"] = curveToJson(evaluation.kellyCurve);
            if (const auto* atOne = pointAtOne(evaluation.kellyCurve)) {
                row["kelly"]["at_multiplier_1"]["growth_delta_from_quantum_0"] =
                    referenceAtOne ? atOne->growthMean - referenceAtOne->growthMean : 0.0;
            }
            if (const auto* optimum = evaluation.kellyCurve.optimalPoint()) {
                row["kelly"]["empirical_optimum"]["growth_delta_from_quantum_0"] =
                    referenceOptimum ? optimum->growthMean - referenceOptimum->growthMean : 0.0;
            }
            root["results"].push_back(std::move(row));
        }
        return root;
    }

    static void saveCsv(const std::filesystem::path& path,
                        const std::vector<Evaluation>& evaluations) {
        std::ofstream output(path);
        if (!output.is_open())
            throw std::runtime_error("Cannot write " + path.string());
        output << "quantum,weight_sum,floating_point_weight_sum,spread_threshold,spread_edge,spread_edge_delta,"
                  "spread_stddev,spread_second_moment,spread_standard_error,"
                  "kelly_growth_at_1,kelly_stddev_at_1,optimal_kelly_multiplier,"
                  "optimal_kelly_growth,optimal_kelly_stddev\n";
        output << std::setprecision(17);
        const Evaluation& reference = evaluations.front();
        for (const auto& evaluation : evaluations) {
            const auto* atOne = pointAtOne(evaluation.kellyCurve);
            const auto* optimum = evaluation.kellyCurve.optimalPoint();
            const double standardError = evaluation.spread.samples > 0
                ? evaluation.spread.stddev /
                    std::sqrt(static_cast<double>(evaluation.spread.samples))
                : 0.0;
            output << evaluation.quantum << ',' << evaluation.weightSum << ','
                   << evaluation.floatingPointWeightSum << ','
                   << evaluation.spreadThreshold << ',' << evaluation.spread.mean << ','
                   << evaluation.spread.mean - reference.spread.mean << ','
                   << evaluation.spread.stddev << ',' << evaluation.spread.secondMoment << ','
                   << standardError << ',';
            if (atOne) output << atOne->growthMean << ',' << atOne->growthStddev;
            else output << ',';
            output << ',';
            if (optimum) {
                output << optimum->fraction << ',' << optimum->growthMean << ','
                       << optimum->growthStddev;
            } else {
                output << ",,";
            }
            output << '\n';
        }
    }

    static std::string resultsToSvg(const std::vector<Evaluation>& evaluations) {
        constexpr double width = 1200.0;
        constexpr double left = 105.0;
        constexpr double right = 55.0;
        constexpr double plotWidth = width - left - right;
        constexpr double panelHeight = 245.0;
        constexpr double topPanelY = 70.0;
        constexpr double bottomPanelY = 430.0;

        std::vector<double> spreadValues;
        std::vector<double> kellyAtOneValues;
        std::vector<double> kellyOptimumValues;
        for (const auto& evaluation : evaluations) {
            spreadValues.push_back(evaluation.spread.mean);
            const auto* atOne = pointAtOne(evaluation.kellyCurve);
            const auto* optimum = evaluation.kellyCurve.optimalPoint();
            kellyAtOneValues.push_back(atOne ? atOne->growthMean : 1.0);
            kellyOptimumValues.push_back(optimum ? optimum->growthMean : 1.0);
        }

        auto rangeFor = [](const std::vector<double>& first,
                           const std::vector<double>& second = {}) {
            double minimum = *std::min_element(first.begin(), first.end());
            double maximum = *std::max_element(first.begin(), first.end());
            if (!second.empty()) {
                minimum = std::min(minimum, *std::min_element(second.begin(), second.end()));
                maximum = std::max(maximum, *std::max_element(second.begin(), second.end()));
            }
            const double span = maximum - minimum;
            const double padding = span > 0.0 ? span * 0.15 : std::max(1e-8, std::abs(maximum) * 0.05);
            return std::pair<double, double>{minimum - padding, maximum + padding};
        };
        const auto spreadRange = rangeFor(spreadValues);
        const auto kellyRange = rangeFor(kellyAtOneValues, kellyOptimumValues);
        auto xAt = [&](size_t index) {
            return evaluations.size() == 1
                ? left + plotWidth / 2.0
                : left + static_cast<double>(index) * plotWidth /
                    static_cast<double>(evaluations.size() - 1);
        };
        auto yAt = [&](double value, double panelY, const std::pair<double, double>& range) {
            return panelY + panelHeight -
                (value - range.first) / (range.second - range.first) * panelHeight;
        };

        std::ostringstream svg;
        svg << std::fixed << std::setprecision(6);
        svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1200\" height=\"780\" viewBox=\"0 0 1200 780\">\n"
            << "<rect width=\"1200\" height=\"780\" fill=\"white\"/>\n"
            << "<text x=\"600\" y=\"30\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"20\">Count quantization effect</text>\n";

        auto drawAxes = [&](double panelY,
                            const std::pair<double, double>& range,
                            const std::string& title) {
            svg << "<line x1=\"" << left << "\" y1=\"" << panelY + panelHeight
                << "\" x2=\"" << left + plotWidth << "\" y2=\"" << panelY + panelHeight
                << "\" stroke=\"#333\"/>\n"
                << "<line x1=\"" << left << "\" y1=\"" << panelY
                << "\" x2=\"" << left << "\" y2=\"" << panelY + panelHeight
                << "\" stroke=\"#333\"/>\n"
                << "<text x=\"" << left << "\" y=\"" << panelY - 16
                << "\" font-family=\"sans-serif\" font-size=\"16\">" << title << "</text>\n";
            for (int tick = 0; tick <= 4; ++tick) {
                const double value = range.first + (range.second - range.first) * tick / 4.0;
                const double y = yAt(value, panelY, range);
                svg << "<line x1=\"" << left - 5 << "\" y1=\"" << y << "\" x2=\""
                    << left + plotWidth << "\" y2=\"" << y
                    << "\" stroke=\"#ddd\"/>\n"
                    << "<text x=\"" << left - 10 << "\" y=\"" << y + 4
                    << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"11\">"
                    << std::setprecision(8) << value << "</text>\n" << std::setprecision(6);
            }
        };
        drawAxes(topPanelY, spreadRange, "Spread 1:10 edge per round");
        drawAxes(bottomPanelY, kellyRange, "Kelly growth per round");

        auto drawSeries = [&](const std::vector<double>& values,
                              double panelY,
                              const std::pair<double, double>& range,
                              const char* color) {
            svg << "<polyline fill=\"none\" stroke=\"" << color
                << "\" stroke-width=\"2\" points=\"";
            for (size_t i = 0; i < values.size(); ++i)
                svg << xAt(i) << ',' << yAt(values[i], panelY, range) << ' ';
            svg << "\"/>\n";
            for (size_t i = 0; i < values.size(); ++i)
                svg << "<circle cx=\"" << xAt(i) << "\" cy=\""
                    << yAt(values[i], panelY, range)
                    << "\" r=\"4\" fill=\"" << color << "\"/>\n";
        };
        drawSeries(spreadValues, topPanelY, spreadRange, "#1f77b4");
        drawSeries(kellyAtOneValues, bottomPanelY, kellyRange, "#2ca02c");
        drawSeries(kellyOptimumValues, bottomPanelY, kellyRange, "#d62728");

        for (size_t i = 0; i < evaluations.size(); ++i) {
            svg << "<text x=\"" << xAt(i) << "\" y=\"" << bottomPanelY + panelHeight + 27
                << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">"
                << formatNumber(evaluations[i].quantum) << "</text>\n";
        }
        svg << "<text x=\"600\" y=\"755\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Quantization quantum</text>\n"
            << "<line x1=\"850\" y1=\"395\" x2=\"880\" y2=\"395\" stroke=\"#2ca02c\" stroke-width=\"2\"/>\n"
            << "<text x=\"888\" y=\"399\" font-family=\"sans-serif\" font-size=\"12\">Multiplier 1.0</text>\n"
            << "<line x1=\"1000\" y1=\"395\" x2=\"1030\" y2=\"395\" stroke=\"#d62728\" stroke-width=\"2\"/>\n"
            << "<text x=\"1038\" y=\"399\" font-family=\"sans-serif\" font-size=\"12\">Empirical optimum</text>\n"
            << "</svg>\n";
        return svg.str();
    }

    static void printSummary(const std::vector<Evaluation>& evaluations) {
        const Evaluation& reference = evaluations.front();
        std::cout << "\n=== Quantization comparison ===\n";
        std::cout << std::left << std::setw(12) << "Quantum"
                  << std::right << std::setw(16) << "Spread edge"
                  << std::setw(16) << "Spread delta"
                  << std::setw(16) << "Kelly @ 1"
                  << std::setw(16) << "Opt multiplier"
                  << std::setw(18) << "Optimal growth" << "\n";
        std::cout << std::string(94, '-') << "\n";
        for (const auto& evaluation : evaluations) {
            const auto* atOne = pointAtOne(evaluation.kellyCurve);
            const auto* optimum = evaluation.kellyCurve.optimalPoint();
            std::cout << std::left << std::setw(12) << formatNumber(evaluation.quantum)
                      << std::right << std::fixed << std::setprecision(9)
                      << std::setw(16) << evaluation.spread.mean
                      << std::setw(16) << evaluation.spread.mean - reference.spread.mean;
            if (atOne) std::cout << std::setw(16) << atOne->growthMean;
            else std::cout << std::setw(16) << "n/a";
            if (optimum) {
                std::cout << std::setw(16) << std::setprecision(2) << optimum->fraction
                          << std::setw(18) << std::setprecision(9) << optimum->growthMean;
            } else {
                std::cout << std::setw(16) << "n/a" << std::setw(18) << "n/a";
            }
            std::cout << "\n";
        }
    }

    static void loadRuntimeDefaults(const json& meta) {
        const json& game = meta.at("game");
        const json& algorithm = meta.at("algorithm_config");
        g_penetration = game.at("penetration").get<double>();
        g_eval_rounds = algorithm.value(
            "eval_rounds", algorithm.value("num_rounds_per_phase", uint64_t(1'000'000'000ULL)));
        g_num_threads = algorithm.value("num_threads", 10);
        g_kelly_measurements = algorithm.value("kelly_measurements", 10);
        g_kelly_rounds = kDefaultKellyMeasurementRounds;
        g_kelly_minimum = kDefaultKellyMinimum;
        g_kelly_maximum = kDefaultKellyMaximum;
        g_kelly_step = kDefaultKellyStep;

        if (g_eval_rounds_override) g_eval_rounds = *g_eval_rounds_override;
        if (g_num_threads_override) g_num_threads = *g_num_threads_override;
        if (g_kelly_measurements_override)
            g_kelly_measurements = *g_kelly_measurements_override;
        if (g_kelly_rounds_override) g_kelly_rounds = *g_kelly_rounds_override;
        if (g_kelly_minimum_override) g_kelly_minimum = *g_kelly_minimum_override;
        if (g_kelly_maximum_override) g_kelly_maximum = *g_kelly_maximum_override;
        if (g_kelly_step_override) g_kelly_step = *g_kelly_step_override;

        if (g_eval_rounds == 0 || g_kelly_rounds == 0)
            throw std::invalid_argument("Evaluation and Kelly rounds must be positive");
        if (g_num_threads < 1)
            throw std::invalid_argument("Number of threads must be at least 1");
        if (g_eval_rounds < static_cast<uint64_t>(g_num_threads) ||
            g_kelly_rounds < static_cast<uint64_t>(g_num_threads)) {
            throw std::invalid_argument("Round counts must be at least the number of threads");
        }
        if (g_kelly_measurements < 1)
            throw std::invalid_argument("Kelly measurements must be at least 1");
        (void)makeKellyFractionGrid(g_kelly_minimum, g_kelly_maximum, g_kelly_step);
    }

    static void runExperiment() {
        namespace fs = std::filesystem;
        const fs::path checkpoint = resolveCheckpointPath(g_checkpoint_input);
        const json meta = readJson(checkpoint / "meta.json");
        if (meta.value("algorithm", std::string{}).find("alternating_optimization") ==
            std::string::npos) {
            throw std::runtime_error("meta.json is not an alternating-optimization checkpoint");
        }
        const Case gameCase = Game::readMetaGameSection(meta.at("game"));
        loadRuntimeDefaults(meta);
        resolveLabels(checkpoint);

        std::string precisionSource;
        const CountConfig sourceCount = loadCount(checkpoint, g_count_label, precisionSource);
        const auto policy = loadPolicy(checkpoint, g_policy_label);

        // Quantum zero is always the reference and is run exactly once.
        auto zero = std::find(g_quanta.begin(), g_quanta.end(), 0.0);
        if (zero == g_quanta.end()) {
            g_quanta.insert(g_quanta.begin(), 0.0);
        } else if (zero != g_quanta.begin()) {
            std::rotate(g_quanta.begin(), zero, zero + 1);
        }

        const std::string runName = currentTimestamp() + "_" + g_count_label + "_" +
            g_policy_label + "_" + Game::checkpointFolderName(gameCase);
        fs::path outputPath = g_output_dir.empty()
            ? fs::path(PROJECT_ROOT) / "checkpoints" / "QuantizationEffect" / runName
            : fs::path(g_output_dir);
        if (outputPath.is_relative()) outputPath = fs::path(PROJECT_ROOT) / outputPath;
        RunLogger logger(outputPath.parent_path(), outputPath.filename().string());

        std::cout << "\n=== QuantizationEffect ===\n"
                  << "Command:          " << g_command_line << "\n"
                  << "Checkpoint:       " << checkpoint.string() << "\n"
                  << "Pair:             " << g_count_label << " + " << g_policy_label << "\n"
                  << "Weight source:    " << precisionSource << "\n"
                  << "Scenario:         " << Game::toString(gameCase) << "\n"
                  << "Penetration:      " << g_penetration << "%\n"
                  << "Eval rounds:      " << g_eval_rounds << "\n"
                  << "Threads:          " << g_num_threads << "\n"
                  << "Kelly search:     " << g_kelly_minimum << " to " << g_kelly_maximum
                  << " step " << g_kelly_step << "\n"
                  << "Kelly experiment: " << g_kelly_measurements << " x " << g_kelly_rounds
                  << " rounds per multiplier\n"
                  << "Seed:             "
                  << (g_seed ? std::to_string(*g_seed) + " (common across quanta)" : "random")
                  << "\nOutput:           " << logger.runDir().string() << "/\n"
                  << "==========================\n";

        std::vector<Evaluation> evaluations;
        evaluations.reserve(g_quanta.size());
        for (double quantum : g_quanta) {
            const CountQuantizationResult quantized =
                quantizeCountWeights(sourceCount.system.weights, quantum, kBalanceTolerance);
            Evaluation evaluation;
            evaluation.quantum = quantum;
            evaluation.count = sourceCount;
            evaluation.count.system.weights = quantized.weights;
            evaluation.sourceWasBalanced = quantized.sourceWasBalanced;
            evaluation.balancePreserved = quantized.balancePreserved;
            evaluation.floatingPointWeightSum = sumWeights(quantized.weights);
            evaluation.weightSum = quantized.balancePreserved
                ? 0.0
                : evaluation.floatingPointWeightSum;
            evaluation.spreadThreshold = spreadThreshold(evaluation.count);
            if (quantum == 0.0 && quantized.weights != sourceCount.system.weights)
                throw std::logic_error("Quantum zero did not preserve the exact source weights");
            if (quantized.sourceWasBalanced && !quantized.balancePreserved)
                throw std::logic_error("Zero-sum quantization constraint was not preserved");

            printWeights(quantum, evaluation);
            std::cout << "Running spread evaluation...\n";
            evaluation.spread = evaluateSpread(gameCase, *policy, evaluation.count);
            std::cout << "Spread edge: " << std::fixed << std::setprecision(10)
                      << evaluation.spread.mean << "  std(X): " << evaluation.spread.stddev
                      << "  E[X^2]: " << evaluation.spread.secondMoment << "\n";
            std::cout << "Running Kelly search...\n";
            evaluation.kellyCurve =
                evaluateKellyCurve(gameCase, *policy, evaluation.count, quantum);
            evaluations.push_back(std::move(evaluation));
        }

        printSummary(evaluations);
        const json outputJson = resultsToJson(
            checkpoint, meta, sourceCount, precisionSource, evaluations);
        std::ofstream(logger.pathFor("results.json"))
            << std::setw(2) << outputJson << '\n';
        saveCsv(logger.pathFor("results.csv"), evaluations);
        std::ofstream(logger.pathFor("quantization_effect.svg"))
            << resultsToSvg(evaluations);

        std::cout << "\nSaved artifacts:\n"
                  << "  " << logger.pathFor("run.log").string() << "\n"
                  << "  " << logger.pathFor("results.json").string() << "\n"
                  << "  " << logger.pathFor("results.csv").string() << "\n"
                  << "  " << logger.pathFor("quantization_effect.svg").string() << "\n";
    }

    static void printHelp(const char* program) {
        std::cout << "Usage: " << program << " --checkpoint <folder> [OPTIONS]\n\n"
                  << "Measure how quantizing a learned alternating-optimization count changes\n"
                  << "spread edge and Kelly growth. The app currently supports blackjack.\n\n"
                  << "CHECKPOINT:\n"
                  << "  --checkpoint <path>       Alternating checkpoint folder, path, or folder name\n"
                  << "  --count <Wk>              Learned count (default: latest complete pair)\n"
                  << "  --policy <Pk>             Policy (default: P(k-1) for selected Wk)\n\n"
                  << "QUANTIZATION:\n"
                  << "  --quanta <csv>            Quantization steps\n"
                  << "                            (default: 0,0.01,0.05,0.1,0.5,1.0)\n"
                  << "  Zero is always included once as an exact, freshly simulated reference.\n"
                  << "  Balanced counts retain sum(weights)=0 and a shared 10/J/Q/K tag.\n\n"
                  << "EVALUATION OVERRIDES:\n"
                  << "  --eval-rounds <N>         Spread rounds (default: checkpoint eval_rounds)\n"
                  << "  --num-threads <N>         Threads (default: checkpoint num_threads)\n"
                  << "  --kelly-measurements <N>  Experiments per multiplier\n"
                  << "                            (default: checkpoint kelly_measurements)\n"
                  << "  --kelly-rounds <N>        Rounds per Kelly experiment (default: 1000000)\n"
                  << "  --kelly-min <v>           Multiplier minimum (default: 0.75)\n"
                  << "  --kelly-max <v>           Multiplier maximum (default: 1.25)\n"
                  << "  --kelly-step <v>          Multiplier step (default: 0.05)\n"
                  << "  --seed <N>                Deterministic common seed across quanta\n\n"
                  << "OUTPUT:\n"
                  << "  --output-dir <path>       Exact output folder override\n"
                  << "  Default: checkpoints/QuantizationEffect/<timestamp_and_case>/\n"
                  << "  Artifacts: run.log, results.json, results.csv, quantization_effect.svg\n";
    }

    static int run(int argc, char** argv) {
        g_command_line = commandLineFromArgs(argc, argv);
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--help" || argument == "-h") {
                printHelp(argv[0]);
                return 0;
            } else if ((argument == "--game" || argument == "--table-type") && i + 1 < argc) {
                ++i;
            } else if (argument == "--checkpoint" && i + 1 < argc) {
                g_checkpoint_input = argv[++i];
            } else if (argument == "--count" && i + 1 < argc) {
                g_count_label = argv[++i];
            } else if (argument == "--policy" && i + 1 < argc) {
                g_policy_label = argv[++i];
            } else if (argument == "--quanta" && i + 1 < argc) {
                g_quanta = parseQuanta(argv[++i]);
            } else if (argument == "--eval-rounds" && i + 1 < argc) {
                g_eval_rounds_override = std::stoull(argv[++i]);
            } else if (argument == "--num-threads" && i + 1 < argc) {
                g_num_threads_override = std::stoi(argv[++i]);
            } else if (argument == "--kelly-measurements" && i + 1 < argc) {
                g_kelly_measurements_override = std::stoi(argv[++i]);
            } else if (argument == "--kelly-rounds" && i + 1 < argc) {
                g_kelly_rounds_override = std::stoull(argv[++i]);
            } else if (argument == "--kelly-min" && i + 1 < argc) {
                g_kelly_minimum_override = std::stod(argv[++i]);
            } else if (argument == "--kelly-max" && i + 1 < argc) {
                g_kelly_maximum_override = std::stod(argv[++i]);
            } else if (argument == "--kelly-step" && i + 1 < argc) {
                g_kelly_step_override = std::stod(argv[++i]);
            } else if (argument == "--seed" && i + 1 < argc) {
                g_seed = std::stoull(argv[++i]);
            } else if (argument == "--output-dir" && i + 1 < argc) {
                g_output_dir = argv[++i];
            } else if (!argument.empty() && argument.front() != '-' &&
                       g_checkpoint_input.empty()) {
                g_checkpoint_input = argument;
            } else {
                std::cerr << "Unknown or incomplete argument: " << argument << "\nUse --help.\n";
                return 1;
            }
        }

        if (g_checkpoint_input.empty()) {
            std::cerr << "Error: --checkpoint <folder> is required.\nUse --help.\n";
            return 1;
        }
        try {
            runExperiment();
        } catch (const std::exception& error) {
            std::cerr << "Error: " << error.what() << "\n";
            return 1;
        }
        return 0;
    }
};

int runBlackjackQuantizationEffect(int argc, char** argv) {
    return QuantizationEffectApp<BlackjackGame>::run(argc, argv);
}

int runUnsupportedDoubleDownMadnessQuantizationEffect(int, char**) {
    std::cerr << "Error: QuantizationEffect currently supports blackjack only; "
                 "Double Down Madness support is planned.\n";
    return 1;
}

int main(int argc, char** argv) {
    return dispatchGameApp(
        argc, argv, "QuantizationEffect",
        runBlackjackQuantizationEffect,
        runUnsupportedDoubleDownMadnessQuantizationEffect);
}
