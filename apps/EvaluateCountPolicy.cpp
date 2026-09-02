#include "AlternatingCheckpointLoader.h"
#include "GameAppDispatcher.h"
#include "GameTraits.h"
#include "Game/CountingMethods.h"
#include "RL/DecayingParameter.h"
#include "RL/QLearningStrategy.h"
#include "Utils/CountPolicyEvaluation.h"
#include "Utils/RunLogger.h"
#include "Utils/SimulationAnalysis.h"
#include "Utils/Utils.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

template <typename Game>
struct EvaluateCountPolicyApp {
    using Case = typename Game::Case;
    using Rules = typename Game::Rules;

    enum class TrainingStopMode { FIXED_ROUNDS, TABLE_DIFF };

    struct AgentConfig {
        ExplorationMode explorationMode = ExplorationMode::EPSILON_GREEDY;
        double epsilonStart = 1.0;
        double epsilonMinimum = 0.1;
        double epsilonDecay = 0.99999;
        double temperatureStart = 1.0;
        double temperatureMinimum = 0.1;
        double temperatureDecay = 0.99999;
        double alphaStart = 0.01;
        double alphaMinimum = 0.0001;
        double alphaDecaySteps = 100.0;
    };

    static inline std::string checkpointInput;
    static inline std::string checkpointCountLabel;
    static inline std::string policySpec;
    static inline std::string policyFile;
    static inline std::string countName;
    static inline std::string countWeightsText;
    static inline std::string countFile;
    static inline std::string outputDirectory;
    static inline std::string commandLine;

    static inline std::optional<double> factorOverride;
    static inline std::optional<double> biasOverride;
    static inline std::optional<CountNormalization> normalizationOverride;
    static inline std::optional<double> initialCountOverride;
    static inline std::optional<double> initialCountPerDeckOverride;
    static inline std::optional<double> resolutionOverride;
    static inline std::optional<int> minimumCountOverride;
    static inline std::optional<int> maximumCountOverride;
    static inline std::optional<bool> continuousBettingOverride;

    static inline std::optional<uint64_t> spreadRoundsOverride;
    static inline std::optional<uint64_t> kellyRoundsOverride;
    static inline std::optional<int> kellyMeasurementsOverride;
    static inline std::optional<int> threadsOverride;
    static inline std::optional<double> penetrationOverride;
    static inline std::optional<double> kellyMinimumOverride;
    static inline std::optional<double> kellyMaximumOverride;
    static inline std::optional<double> kellyStepOverride;
    static inline double maximumTotalWagerFraction = 1.0;
    static inline std::optional<uint64_t> seed;

    static inline uint64_t trainingRounds = 1'000'000'000ULL;
    static inline bool trainingRoundsExplicit = false;
    static inline TrainingStopMode trainingStopMode = TrainingStopMode::FIXED_ROUNDS;
    static inline uint64_t sampleRounds = 100'000'000ULL;
    static inline double differenceThreshold = 0.001;
    static inline AgentConfig agent;
    static inline bool verbose = false;

    static inline std::vector<int> deckSizes = {6};
    static inline std::vector<bool> standSoft17 = {Game::kDefaultStandSoft17};
    static inline bool standaloneRulesExplicit = false;

    static std::string timestamp() {
        const time_t now = time(nullptr);
        char buffer[20];
        strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", localtime(&now));
        return buffer;
    }

    static std::string formatNumber(double value) {
        std::ostringstream output;
        output << std::setprecision(12) << value;
        return output.str();
    }

    static std::array<double, 13> parseWeights(const std::string& text) {
        std::string cleaned = text;
        if (!cleaned.empty() && cleaned.front() == '[') cleaned.erase(cleaned.begin());
        if (!cleaned.empty() && cleaned.back() == ']') cleaned.pop_back();
        std::array<double, 13> weights{};
        std::istringstream input(cleaned);
        std::string token;
        size_t index = 0;
        while (std::getline(input, token, ',')) {
            if (index >= weights.size())
                throw std::invalid_argument("--count-weights requires exactly 13 values");
            weights[index++] = std::stod(token);
        }
        if (index != weights.size())
            throw std::invalid_argument("--count-weights requires exactly 13 values");
        for (double weight : weights) {
            if (!std::isfinite(weight))
                throw std::invalid_argument("Count weights must be finite");
        }
        return weights;
    }

    static CountPolicyConfig loadCountFile(const std::string& pathText) {
        const json root = AlternatingCheckpointLoader::readJson(pathText);
        if (root.contains("count_config"))
            return AlternatingCheckpointLoader::countFromJson(root.at("count_config"));
        if (root.contains("count"))
            return AlternatingCheckpointLoader::countFromJson(root.at("count"));
        return AlternatingCheckpointLoader::countFromJson(root);
    }

    static bool hasDirectCountSource() {
        return !countName.empty() || !countWeightsText.empty() || !countFile.empty();
    }

    static CountPolicyConfig resolveCount(
            const CountPolicyConfig* checkpointCount,
            std::string& sourceDescription) {
        const int sources = static_cast<int>(!countName.empty()) +
            static_cast<int>(!countWeightsText.empty()) +
            static_cast<int>(!countFile.empty());
        if (sources > 1) {
            throw std::invalid_argument(
                "Choose one of --count-name, --count-weights, or --count-file");
        }

        CountPolicyConfig count;
        if (!countFile.empty()) {
            count = loadCountFile(countFile);
            sourceDescription = "count file " + countFile;
        } else if (!countWeightsText.empty()) {
            count.system.weights = parseWeights(countWeightsText);
            count.system.factor = CountingMethods::kDefaultFactor;
            count.system.bias = CountingMethods::kDefaultBias;
            sourceDescription = "explicit rank tags";
        } else if (!countName.empty()) {
            const auto named = CountingMethods::fromName(countName);
            if (!named)
                throw std::invalid_argument("Unknown count name: " + countName);
            count.system = *named;
            sourceDescription = "named count " + countName;
        } else if (checkpointCount != nullptr) {
            count = *checkpointCount;
            sourceDescription = "alternating checkpoint " + checkpointCountLabel;
        } else {
            count.system = *CountingMethods::fromName("hilo");
            sourceDescription = "named count hilo (default)";
        }

        if (factorOverride) count.system.factor = *factorOverride;
        if (biasOverride) count.system.bias = *biasOverride;
        if (normalizationOverride) count.system.normalization = *normalizationOverride;
        if (initialCountOverride) count.system.initialCount = *initialCountOverride;
        if (initialCountPerDeckOverride)
            count.system.initialCountPerDeck = *initialCountPerDeckOverride;
        if (resolutionOverride) count.resolution = *resolutionOverride;
        if (minimumCountOverride) count.minCount = *minimumCountOverride;
        if (maximumCountOverride) count.maxCount = *maximumCountOverride;
        if (continuousBettingOverride)
            count.system.continuousBettingCount = *continuousBettingOverride;

        if (!std::isfinite(count.system.factor) || !std::isfinite(count.system.bias) ||
            !std::isfinite(count.system.initialCount) ||
            !std::isfinite(count.system.initialCountPerDeck) ||
            !std::isfinite(count.resolution) || count.resolution <= 0.0 ||
            count.minCount > count.maxCount) {
            throw std::invalid_argument("Invalid count transform or count range");
        }
        return count;
    }

    static bool isHiLo(const std::array<double, 13>& weights) {
        for (size_t index = 0; index < weights.size(); ++index) {
            if (std::abs(weights[index] - CountingMethods::HiLo[index]) > 1e-9)
                return false;
        }
        return true;
    }

    static std::unique_ptr<BasicStrategy> expandAcrossCounts(
            const BasicStrategy& base,
            int minimumCount,
            int maximumCount) {
        auto strategy = std::make_unique<BasicStrategy>();
        for (int count = minimumCount; count <= maximumCount; ++count) {
            for (int dealer = 2; dealer <= 11; ++dealer) {
                for (int hard = 4; hard <= 21; ++hard) {
                    strategy->setAction(
                        count, HandType::HARD, hard, dealer,
                        base.getActionFromTable(0, HandType::HARD, hard, dealer));
                }
                for (int soft = 13; soft <= 21; ++soft) {
                    strategy->setAction(
                        count, HandType::SOFT, soft, dealer,
                        base.getActionFromTable(0, HandType::SOFT, soft, dealer));
                }
                for (int pair = 2; pair <= 11; ++pair) {
                    strategy->setAction(
                        count, HandType::PAIR, pair, dealer,
                        base.getActionFromTable(0, HandType::PAIR, pair, dealer));
                }
            }
        }
        return strategy;
    }

    struct IndexedDeviation {
        HandType handType;
        int playerSum;
        int dealerCard;
        int index;
        ActionWithFallback action;
    };

    static void applyIllustrious18(BasicStrategy& strategy,
                                   int minimumCount,
                                   int maximumCount) {
        // Multi-deck S17 Hi-Lo reference indices. Insurance is omitted because
        // insurance is not a player decision in the current engine.
        static const std::array<IndexedDeviation, 17> deviations = {{
            {HandType::HARD, 16, 10,  0, ActionWithFallback(Action::STAND)},
            {HandType::HARD, 15, 10,  4, ActionWithFallback(Action::STAND)},
            {HandType::PAIR, 10,  5,  5, ActionWithFallback(Action::SPLIT, Action::STAND)},
            {HandType::PAIR, 10,  6,  4, ActionWithFallback(Action::SPLIT, Action::STAND)},
            {HandType::HARD, 10, 10,  4, ActionWithFallback(Action::DOUBLE_DOWN, Action::HIT)},
            {HandType::HARD, 12,  3,  2, ActionWithFallback(Action::STAND)},
            {HandType::HARD, 12,  2,  3, ActionWithFallback(Action::STAND)},
            {HandType::HARD, 11, 11,  1, ActionWithFallback(Action::DOUBLE_DOWN, Action::HIT)},
            {HandType::HARD,  9,  2,  1, ActionWithFallback(Action::DOUBLE_DOWN, Action::HIT)},
            {HandType::HARD, 10, 11,  4, ActionWithFallback(Action::DOUBLE_DOWN, Action::HIT)},
            {HandType::HARD,  9,  7,  3, ActionWithFallback(Action::DOUBLE_DOWN, Action::HIT)},
            {HandType::HARD, 16,  9,  5, ActionWithFallback(Action::STAND)},
            {HandType::HARD, 13,  2, -1, ActionWithFallback(Action::STAND)},
            {HandType::HARD, 12,  4,  0, ActionWithFallback(Action::STAND)},
            {HandType::HARD, 12,  5, -2, ActionWithFallback(Action::STAND)},
            {HandType::HARD, 12,  6, -1, ActionWithFallback(Action::STAND)},
            {HandType::HARD, 13,  3, -2, ActionWithFallback(Action::STAND)}
        }};

        for (const IndexedDeviation& deviation : deviations) {
            const ActionWithFallback below = strategy.getActionFromTable(
                0,
                deviation.handType,
                deviation.playerSum,
                deviation.dealerCard);
            for (int count = minimumCount; count <= maximumCount; ++count) {
                strategy.setAction(
                    count,
                    deviation.handType,
                    deviation.playerSum,
                    deviation.dealerCard,
                    count >= deviation.index ? deviation.action : below);
            }
        }
    }

    static std::unique_ptr<QLearningStrategy> makeQLearningStrategy() {
        auto alpha = std::make_unique<LinearDecayingParameter>(
            agent.alphaStart,
            agent.alphaMinimum,
            static_cast<int>(agent.alphaDecaySteps));
        std::unique_ptr<DecayingParameter> exploration;
        if (agent.explorationMode == ExplorationMode::EPSILON_GREEDY) {
            exploration = std::make_unique<EpsilonDecayingParameter>(
                agent.epsilonStart,
                agent.epsilonMinimum,
                agent.epsilonDecay);
        } else {
            exploration = std::make_unique<EpsilonDecayingParameter>(
                agent.temperatureStart,
                agent.temperatureMinimum,
                agent.temperatureDecay);
        }
        return std::make_unique<QLearningStrategy>(
            std::move(alpha), std::move(exploration), 1.0, agent.explorationMode);
    }

    static std::unique_ptr<BasicStrategy> trainFullDeviations(
            const Case& gameCase,
            double penetration,
            int numThreads,
            const CountPolicyConfig& count) {
        auto player = std::make_unique<Player>(0.0, makeQLearningStrategy());
        player->setNumDecks(gameCase.deckSize);
        player->setCountSystem(count.system);
        player->setCountResolution(count.resolution);
        player->setCountRange(count.minCount, count.maxCount);

        Rules rules = Game::buildRules(gameCase, penetration, 1.0, 1.0);
        QLearningStrategy::QTableSnapshot previousTable;
        uint64_t completedRounds = 0;
        uint64_t batchIndex = 0;
        const auto start = std::chrono::high_resolution_clock::now();

        while (true) {
            const uint64_t batch = trainingStopMode == TrainingStopMode::TABLE_DIFF
                ? sampleRounds
                : trainingRounds - completedRounds;
            if (batch == 0) break;

            std::vector<Player*> results = runParallelSimulation(
                rules,
                {player.get()},
                batch,
                numThreads,
                countPolicyStreamSeed(seed, 0x545241494e000000ULL + batchIndex));
            if (results.empty())
                throw std::runtime_error("Full-deviations training failed");
            player.reset(results.front());
            completedRounds += batch;
            ++batchIndex;

            const auto* q = dynamic_cast<const QLearningStrategy*>(player->getStrategy());
            if (q == nullptr)
                throw std::runtime_error("Full-deviations result is not QLearningStrategy");

            if (trainingStopMode == TrainingStopMode::TABLE_DIFF) {
                QLearningStrategy::QTableSnapshot current = q->snapshotQTable();
                const double difference = QLearningStrategy::averageAbsDifference(
                    current, previousTable);
                previousTable = std::move(current);
                const double seconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::high_resolution_clock::now() - start).count() / 1000.0;
                std::cout << "Training rounds=" << completedRounds
                          << " speed=" << std::fixed << std::setprecision(0)
                          << (seconds > 0.0 ? completedRounds / seconds : 0.0)
                          << " rounds/sec avg|delta Q|=" << std::setprecision(8)
                          << difference << " threshold=" << differenceThreshold << "\n";
                if (difference <= differenceThreshold) break;
            } else if (completedRounds >= trainingRounds) {
                break;
            }
        }

        const auto* q = dynamic_cast<const QLearningStrategy*>(player->getStrategy());
        if (q == nullptr)
            throw std::runtime_error("Full-deviations result is not QLearningStrategy");
        const auto learned = q->toBasicStrategy();
        const auto basic = Game::loadBasicStrategy(gameCase);
        auto complete = expandAcrossCounts(*basic, count.minCount, count.maxCount);
        complete->mergeFromStrategy(*learned);
        return complete;
    }

    static std::unique_ptr<BasicStrategy> resolvePolicy(
            const Case& gameCase,
            double penetration,
            int numThreads,
            const CountPolicyConfig& count,
            std::unique_ptr<BasicStrategy> checkpointPolicy,
            const std::string& checkpointPolicyLabel,
            std::string& sourceDescription) {
        if (!policyFile.empty()) {
            auto policy = std::make_unique<BasicStrategy>();
            if (!policy->loadFromFile(policyFile))
                throw std::runtime_error("Cannot load policy file " + policyFile);
            sourceDescription = "policy file " + policyFile;
            return policy;
        }

        std::string selected = policySpec;
        if (selected.empty()) selected = checkpointPolicy ? "checkpoint" : "basic";
        if (AlternatingCheckpointLoader::validLabel(selected, 'P')) {
            if (!checkpointPolicy || selected != checkpointPolicyLabel) {
                throw std::invalid_argument(
                    "Checkpoint policy " + selected + " was not loaded; pass it with --policy");
            }
            sourceDescription = "alternating checkpoint " + selected;
            return checkpointPolicy;
        }
        if (selected == "checkpoint") {
            if (!checkpointPolicy)
                throw std::invalid_argument("--policy checkpoint requires --checkpoint");
            sourceDescription = "alternating checkpoint " + checkpointPolicyLabel;
            return checkpointPolicy;
        }
        if (selected == "basic") {
            sourceDescription = "basic strategy";
            return Game::loadBasicStrategy(gameCase);
        }
        if (selected == "illustrious18" || selected == "i18" || selected == "illustrious") {
            if constexpr (!Game::kSupportsIllustrious18) {
                throw std::invalid_argument("Illustrious 18 is only available for blackjack");
            } else {
                if (!isHiLo(count.system.weights) ||
                    count.system.normalization != CountNormalization::TRUE_COUNT) {
                    throw std::invalid_argument(
                        "Illustrious 18 requires true-counted Hi-Lo rank tags");
                }
                const auto basic = Game::loadBasicStrategy(gameCase);
                auto policy = expandAcrossCounts(*basic, count.minCount, count.maxCount);
                applyIllustrious18(*policy, count.minCount, count.maxCount);
                sourceDescription = "standard Hi-Lo Illustrious 18 (insurance omitted)";
                return policy;
            }
        }
        if (selected == "full-deviations" || selected == "train-full-deviations") {
            const uint64_t firstBatch = trainingStopMode == TrainingStopMode::TABLE_DIFF
                ? sampleRounds
                : trainingRounds;
            if (firstBatch < static_cast<uint64_t>(numThreads)) {
                throw std::invalid_argument(
                    "Full-deviations training rounds must be at least the number of threads");
            }
            std::cout << "Training fresh full deviations for the selected count...\n";
            sourceDescription = "fresh full deviations trained for the selected count";
            return trainFullDeviations(gameCase, penetration, numThreads, count);
        }
        throw std::invalid_argument(
            "Unknown policy '" + selected +
            "'. Use basic, illustrious18, full-deviations, Pk, or --policy-file");
    }

    static json countToJson(const CountPolicyConfig& count) {
        json weights = json::array();
        for (double weight : count.system.weights) weights.push_back(weight);
        return {
            {"weights", weights},
            {"factor", count.system.factor},
            {"bias", count.system.bias},
            {"signal_formula", "bias + factor * selected_count"},
            {"continuous_betting_count", count.system.continuousBettingCount},
            {"count_normalization", countNormalizationToString(count.system.normalization)},
            {"initial_count", count.system.initialCount},
            {"initial_count_per_deck", count.system.initialCountPerDeck},
            {"resolution", count.resolution},
            {"min_count", count.minCount},
            {"max_count", count.maxCount}
        };
    }

    static json edgeToJson(const EdgeStatistics& edge) {
        const double standardError = edge.samples > 0
            ? edge.stddev / std::sqrt(static_cast<double>(edge.samples))
            : 0.0;
        return {
            {"rounds", edge.samples},
            {"edge_per_round", edge.mean},
            {"outcome_stddev", edge.stddev},
            {"outcome_second_moment", edge.secondMoment},
            {"edge_standard_error", standardError}
        };
    }

    static void writeSummaryCsv(const std::filesystem::path& path,
                                const CountPolicyEvaluationResult& result) {
        std::ofstream output(path);
        if (!output.is_open())
            throw std::runtime_error("Cannot write " + path.string());
        output << "kelly_multiplier,growth_mean,growth_stddev,exposure_rounds,"
                  "gross_exposure_mean,gross_exposure_max,absolute_return_mean,"
                  "absolute_return_max,mean_abs_taylor_error,max_abs_taylor_error\n";
        output << std::setprecision(17);
        for (const KellyGrowthPoint& point : result.kelly.points) {
            const KellyExposureStatistics& stats = point.exposure;
            const double valid = static_cast<double>(stats.validBankrollRounds);
            const uint64_t validLogs = stats.validBankrollRounds - stats.invalidLogRounds;
            output << point.fraction << ',' << point.growthMean << ','
                   << point.growthStddev << ',' << stats.rounds << ','
                   << (valid > 0.0 ? stats.grossExposureSum / valid : 0.0) << ','
                   << stats.grossExposureMaximum << ','
                   << (valid > 0.0 ? stats.absoluteReturnSum / valid : 0.0) << ','
                   << stats.absoluteReturnMaximum << ','
                   << (validLogs > 0
                       ? stats.absoluteTaylorErrorSum / static_cast<double>(validLogs)
                       : 0.0) << ','
                   << stats.absoluteTaylorErrorMaximum << '\n';
        }
    }

    static void printWeights(const CountPolicyConfig& count) {
        static constexpr std::array<const char*, 13> ranks = {
            "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"
        };
        std::cout << "\nCount weights:\n";
        for (size_t index = 0; index < ranks.size(); ++index) {
            std::cout << "  " << std::setw(2) << ranks[index] << "  "
                      << std::setprecision(12) << count.system.weights[index] << '\n';
        }
        std::cout << "Transform: running = " << count.system.initialCount
                  << " + (" << count.system.initialCountPerDeck << " * decks) + tags"
                  << "; selected="
                  << countNormalizationToString(count.system.normalization) << '\n'
                  << "Signal f(c) = " << count.system.bias << " + "
                  << count.system.factor << " * c\n";
    }

    static void saveArtifacts(const RunLogger& logger,
                              const json& root,
                              const CountPolicyEvaluationResult& result,
                              const KellyGrowthPoint& atOne) {
        std::ofstream(logger.pathFor("results.json"))
            << std::setw(2) << root << '\n';
        writeSummaryCsv(logger.pathFor("results.csv"), result);
        std::ofstream(logger.pathFor("kelly_growth.json"))
            << std::setw(2) << kellyGrowthCurvesToJson({result.kelly}) << '\n';
        std::ofstream(logger.pathFor("kelly_growth.svg"))
            << kellyGrowthCurvesToSvg("Count + policy Kelly growth", {result.kelly});
        std::ofstream(logger.pathFor("kelly_exposure_at_1.json"))
            << std::setw(2) << kellyExposureStatisticsToJson(atOne.exposure) << '\n';
        std::ofstream(logger.pathFor("kelly_exposure_at_1.csv"))
            << kellyExposureStatisticsToCsv(atOne.exposure);
        std::ofstream(logger.pathFor("kelly_exposure_at_1.svg"))
            << kellyExposureStatisticsToSvg(
                "Kelly exposure and |fX| at multiplier 1.0", atOne.exposure);
    }

    static int execute() {
        std::optional<AlternatingCheckpointSelection<Game>> checkpoint;
        if (!checkpointInput.empty()) {
            std::string requestedPolicy =
                AlternatingCheckpointLoader::validLabel(policySpec, 'P')
                    ? policySpec
                    : std::string{};
            checkpoint.emplace(AlternatingCheckpointLoader::load<Game>(
                checkpointInput, checkpointCountLabel, requestedPolicy));
            checkpointCountLabel = checkpoint->countLabel;
            if (standaloneRulesExplicit) {
                std::cout << "WARNING: game-rule flags are ignored because --checkpoint "
                             "supplies the complete rules.\n";
            }
        }

        Case gameCase;
        AlternatingEvaluationDefaults defaults;
        if (checkpoint) {
            gameCase = checkpoint->gameCase;
            defaults = checkpoint->defaults;
            if (!trainingRoundsExplicit) {
                trainingRounds = checkpoint->meta.at("algorithm_config").value(
                    "num_rounds_per_phase", trainingRounds);
            }
        } else {
            const std::vector<Case> cases = Game::generateCases(deckSizes, standSoft17);
            if (cases.size() != 1) {
                throw std::invalid_argument(
                    "EvaluateCountPolicy accepts one game scenario per invocation");
            }
            gameCase = cases.front();
        }

        if (penetrationOverride) defaults.penetration = *penetrationOverride;
        if (spreadRoundsOverride) defaults.spreadRounds = *spreadRoundsOverride;
        if (kellyRoundsOverride)
            defaults.kellyRoundsPerMeasurement = *kellyRoundsOverride;
        if (kellyMeasurementsOverride)
            defaults.kellyMeasurements = *kellyMeasurementsOverride;
        if (threadsOverride) defaults.numThreads = *threadsOverride;
        if (kellyMinimumOverride) defaults.kellyMinimum = *kellyMinimumOverride;
        if (kellyMaximumOverride) defaults.kellyMaximum = *kellyMaximumOverride;
        if (kellyStepOverride) defaults.kellyStep = *kellyStepOverride;
        if (defaults.numThreads < 1 || defaults.kellyMeasurements < 1 ||
            defaults.spreadRounds == 0 || defaults.kellyRoundsPerMeasurement == 0) {
            throw std::invalid_argument(
                "Threads, measurement count, and simulation round counts must be positive");
        }
        if (defaults.spreadRounds < static_cast<uint64_t>(defaults.numThreads) ||
            defaults.kellyRoundsPerMeasurement <
                static_cast<uint64_t>(defaults.numThreads)) {
            throw std::invalid_argument(
                "Spread and Kelly rounds must be at least the number of threads");
        }
        if (!std::isfinite(defaults.penetration) || defaults.penetration < 0.0 ||
            defaults.penetration > 100.0) {
            throw std::invalid_argument("Penetration must be between 0 and 100 percent");
        }

        std::vector<double> fractions = makeKellyFractionGrid(
            defaults.kellyMinimum, defaults.kellyMaximum, defaults.kellyStep);
        if (std::none_of(fractions.begin(), fractions.end(), [](double value) {
                return std::abs(value - 1.0) <= 1e-12;
            })) {
            fractions.push_back(1.0);
            std::sort(fractions.begin(), fractions.end());
        }

        std::string countSource;
        CountPolicyConfig count = resolveCount(
            checkpoint && !hasDirectCountSource() ? &checkpoint->count : nullptr,
            countSource);

        const std::string runName = timestamp() + "_" +
            Game::checkpointFolderName(gameCase);
        std::filesystem::path outputPath = outputDirectory.empty()
            ? std::filesystem::path(PROJECT_ROOT) / "checkpoints" /
                "EvaluateCountPolicy" / runName
            : std::filesystem::path(outputDirectory);
        if (outputPath.is_relative())
            outputPath = std::filesystem::path(PROJECT_ROOT) / outputPath;
        RunLogger logger(outputPath.parent_path(), outputPath.filename().string());

        std::string policySource;
        std::unique_ptr<BasicStrategy> policy = resolvePolicy(
            gameCase,
            defaults.penetration,
            defaults.numThreads,
            count,
            checkpoint ? std::move(checkpoint->policy) : nullptr,
            checkpoint ? checkpoint->policyLabel : std::string{},
            policySource);

        if (policySpec == "full-deviations" ||
            policySpec == "train-full-deviations") {
            const auto [minimum, maximum] = policy->getCountRange();
            if (minimum != count.minCount || maximum != count.maxCount) {
                throw std::runtime_error(
                    "Fresh full-deviations policy has an unexpected count range");
            }
        }

        if (policySpec == "full-deviations" ||
            policySpec == "train-full-deviations") {
            if (!policy->saveToJson(
                    logger.pathFor("full_deviations_strategy.json").string())) {
                throw std::runtime_error("Cannot save trained full-deviations policy");
            }
        }

        std::cout << "\n=== EvaluateCountPolicy ===\n"
                  << "Command:          " << commandLine << '\n'
                  << "Game:             blackjack\n"
                  << "Scenario:         " << Game::toString(gameCase) << '\n'
                  << "Penetration:      " << defaults.penetration << "%\n"
                  << "Count:            " << countSource << '\n'
                  << "Policy:           " << policySource << '\n'
                  << "Spread:           " << defaults.spreadRounds << " rounds, 1:10\n"
                  << "Kelly:            " << defaults.kellyMeasurements << " x "
                  << defaults.kellyRoundsPerMeasurement << " rounds per multiplier\n"
                  << "Kelly search:     " << defaults.kellyMinimum << " to "
                  << defaults.kellyMaximum << " step " << defaults.kellyStep << '\n'
                  << "Total wager cap:  " << maximumTotalWagerFraction
                  << " x round-start bankroll\n"
                  << "Threads:          " << defaults.numThreads << '\n'
                  << "Seed:             "
                  << (seed ? std::to_string(*seed) : std::string("random")) << '\n'
                  << "Output:           " << logger.runDir().string() << "/\n";
        printWeights(count);
        if (verbose) std::cout << "\nPolicy:\n" << *policy << '\n';

        CountPolicyEvaluationOptions options;
        options.spreadRounds = defaults.spreadRounds;
        options.kellyRoundsPerMeasurement = defaults.kellyRoundsPerMeasurement;
        options.kellyMeasurements = defaults.kellyMeasurements;
        options.numThreads = defaults.numThreads;
        options.kellyFractions = fractions;
        options.maximumTotalWagerFraction = maximumTotalWagerFraction;
        options.seed = seed;

        std::cout << "\nRunning spread and Kelly evaluations...\n";
        const Rules rules = Game::buildRules(
            gameCase,
            defaults.penetration,
            options.spreadMinimumBet,
            options.spreadMaximumBet);
        CountPolicyEvaluationResult result = evaluateCountPolicy(
            rules, *policy, count, options, policySource);
        const KellyGrowthPoint* atOne = kellyPointAtFraction(result.kelly, 1.0, 1e-9);
        if (atOne == nullptr)
            throw std::logic_error("Kelly multiplier 1.0 was not evaluated");

        json root;
        root["app"] = "EvaluateCountPolicy";
        root["game"] = "blackjack";
        root["command"] = commandLine;
        root["scenario"] = Game::toString(gameCase);
        root["rules"]["decks"] = gameCase.deckSize;
        root["rules"]["ss17"] = gameCase.standSoft17;
        root["rules"]["penetration"] = defaults.penetration;
        json gameRuleMeta;
        Game::writeMetaGameSection(gameRuleMeta, gameCase);
        root["rules"].update(gameRuleMeta.at("game"));
        root["count_source"] = countSource;
        root["policy_source"] = policySource;
        root["count"] = countToJson(count);
        root["configuration"] = {
            {"spread_rounds", options.spreadRounds},
            {"spread_minimum_bet", options.spreadMinimumBet},
            {"spread_maximum_bet", options.spreadMaximumBet},
            {"kelly_rounds_per_measurement", options.kellyRoundsPerMeasurement},
            {"kelly_measurements", options.kellyMeasurements},
            {"kelly_minimum", defaults.kellyMinimum},
            {"kelly_maximum", defaults.kellyMaximum},
            {"kelly_step", defaults.kellyStep},
            {"maximum_total_wager_fraction", options.maximumTotalWagerFraction},
            {"num_threads", options.numThreads},
            {"seed", seed ? json(*seed) : json(nullptr)}
        };
        root["spread"] = edgeToJson(result.spread);
        root["kelly"] = kellyGrowthCurvesToJson({result.kelly}).at("curves").at(0);
        root["kelly_at_multiplier_1"] = {
            {"growth_mean", atOne->growthMean},
            {"growth_stddev", atOne->growthStddev},
            {"exposure", kellyExposureStatisticsToJson(atOne->exposure)}
        };
        if (const KellyGrowthPoint* optimum = result.kelly.optimalPoint()) {
            root["kelly_empirical_optimum"] = {
                {"multiplier", optimum->fraction},
                {"growth_mean", optimum->growthMean},
                {"growth_stddev", optimum->growthStddev}
            };
        }
        if (checkpoint) {
            root["checkpoint"] = {
                {"path", checkpoint->root.string()},
                {"count", checkpoint->countLabel},
                {"policy", checkpoint->policyLabel},
                {"count_precision_source", checkpoint->countPrecisionSource}
            };
        }

        saveArtifacts(logger, root, result, *atOne);

        const double valid = static_cast<double>(atOne->exposure.validBankrollRounds);
        std::cout << "\n=== Result ===\n"
                  << "Spread edge:               " << std::setprecision(12)
                  << result.spread.mean << '\n'
                  << "Kelly growth @1:           " << atOne->growthMean << '\n'
                  << "Kelly exposure rounds @1: " << atOne->exposure.rounds << '\n'
                  << "Mean gross wager/bankroll: "
                  << (valid > 0.0 ? atOne->exposure.grossExposureSum / valid : 0.0) << '\n'
                  << "Max gross wager/bankroll:  "
                  << atOne->exposure.grossExposureMaximum << '\n'
                  << "Mean |fX|:                 "
                  << (valid > 0.0 ? atOne->exposure.absoluteReturnSum / valid : 0.0) << '\n'
                  << "Max |fX|:                  "
                  << atOne->exposure.absoluteReturnMaximum << "\n\n"
                  << "Saved results to " << logger.runDir().string() << "/\n";
        return 0;
    }

    static void printHelp(const char* program) {
        std::cout
            << "Usage: " << program << " --game blackjack [COUNT] [POLICY] [OPTIONS]\n\n"
            << "Evaluate one fixed counting system and one fixed playing policy using the\n"
            << "shared 1:10 spread and Kelly-growth simulations. Every Kelly multiplier\n"
            << "also collects gross wager/bankroll and realized |fX| distributions.\n\n"
            << "ALTERNATING CHECKPOINT SOURCE:\n"
            << "  --checkpoint <folder>       Load rules and runtime defaults from the folder\n"
            << "  --count <Wk>                Count artifact (default: last complete Wk)\n"
            << "  --policy <Pk>               Policy artifact (default: P(k-1))\n\n"
            << "DIRECT COUNT SOURCE (choose one; default: hilo):\n"
            << "  --count-name <name>         hilo, halves, ko, hiopt1, hiopt2, omega2, zen\n"
            << "  --count-weights <13 csv>    Explicit rank tags in 2,3,...,10,J,Q,K,A order\n"
            << "  --count-file <json>         Count config JSON, or object containing count_config\n"
            << "  --factor <a> --bias <b>     Betting signal f(c)=b+a*c\n"
            << "  --count-normalization <true|running>\n"
            << "  --initial-count <v>         Constant running-count offset\n"
            << "  --initial-count-per-deck <v>  Deck-scaled offset (Spanish 21: -4)\n"
            << "  --continuous-betting-count <bool>\n"
            << "  --count-resolution <v> --min-count <N> --max-count <N>\n\n"
            << "POLICY SOURCE:\n"
            << "  --policy basic              Rule-matched basic strategy\n"
            << "  --policy illustrious18      Basic + standard Hi-Lo I18 deviations\n"
            << "  --policy full-deviations    Train all deviations for the selected count\n"
            << "  --train-full-deviations     Alias for --policy full-deviations\n"
            << "  --policy-file <json>        Load a pre-calculated BasicStrategy table\n\n"
            << "EVALUATION:\n"
            << "  --eval-rounds <N>           Spread rounds (checkpoint default when supplied)\n"
            << "  --kelly-rounds <N>          Rounds per Kelly measurement (default: 1000000)\n"
            << "  --kelly-measurements <N>    Measurements per multiplier\n"
            << "  --kelly-min <v>             Default: 0.75\n"
            << "  --kelly-max <v>             Default: 1.25\n"
            << "  --kelly-step <v>            Default: 0.05; multiplier 1 is always included\n"
            << "  --max-total-wager-fraction <0..1>\n"
            << "                              Cap cumulative round wagers / starting bankroll\n"
            << "  --num-threads <N> --seed <N> --penetration <percent>\n\n"
            << "FULL-DEVIATIONS TRAINING:\n"
            << "  --num-rounds <N>            Fixed training rounds\n"
            << "  --stop-mode <rounds|diff> --sample-rounds <N> --diff-threshold <v>\n"
            << "  --exploration <epsilon|boltzmann>\n"
            << "  --epsilon-start/min/decay <v> --temp-start/min/decay <v>\n"
            << "  --alpha-start/min/decay <v>\n\n"
            << "BLACKJACK RULES (used without --checkpoint):\n"
            << "  --decks <N> --ss17 <bool>\n";
        Game::printGameHelp();
        std::cout
            << "\nOUTPUT:\n"
            << "  --output-dir <path>         Exact output folder\n"
            << "  --verbose                   Print the selected policy table\n"
            << "  Files include results.json/csv, Kelly growth JSON/SVG, and explicit\n"
            << "  kelly_exposure_at_1.json/csv/svg artifacts.\n";
    }

    static int run(int argc, char** argv) {
        commandLine = commandLineFromArgs(argc, argv);
        try {
            for (int index = 1; index < argc; ++index) {
                const std::string argument = argv[index];
                auto requireValue = [&](const char* option) -> std::string {
                    if (index + 1 >= argc)
                        throw std::invalid_argument(std::string(option) + " requires a value");
                    return argv[++index];
                };

                if (argument == "--help" || argument == "-h") {
                    printHelp(argv[0]);
                    return 0;
                } else if (argument == "--game" || argument == "--table-type") {
                    (void)requireValue(argument.c_str());
                } else if (argument == "--checkpoint") {
                    checkpointInput = requireValue("--checkpoint");
                } else if (argument == "--count") {
                    checkpointCountLabel = requireValue("--count");
                } else if (argument == "--policy") {
                    policySpec = requireValue("--policy");
                } else if (argument == "--train-full-deviations") {
                    if (!policySpec.empty())
                        throw std::invalid_argument(
                            "Choose either --policy or --train-full-deviations");
                    policySpec = "full-deviations";
                } else if (argument == "--policy-file") {
                    policyFile = requireValue("--policy-file");
                } else if (argument == "--count-name") {
                    countName = requireValue("--count-name");
                } else if (argument == "--count-weights") {
                    countWeightsText = requireValue("--count-weights");
                } else if (argument == "--count-file") {
                    countFile = requireValue("--count-file");
                } else if (argument == "--factor") {
                    factorOverride = std::stod(requireValue("--factor"));
                } else if (argument == "--bias") {
                    biasOverride = std::stod(requireValue("--bias"));
                } else if (argument == "--count-normalization") {
                    normalizationOverride = countNormalizationFromString(
                        requireValue("--count-normalization"));
                    if (!normalizationOverride)
                        throw std::invalid_argument("Unknown count normalization");
                } else if (argument == "--initial-count") {
                    initialCountOverride = std::stod(requireValue("--initial-count"));
                } else if (argument == "--initial-count-per-deck") {
                    initialCountPerDeckOverride =
                        std::stod(requireValue("--initial-count-per-deck"));
                } else if (argument == "--continuous-betting-count") {
                    continuousBettingOverride =
                        parseList<bool>(requireValue("--continuous-betting-count")).at(0);
                } else if (argument == "--count-resolution") {
                    resolutionOverride = std::stod(requireValue("--count-resolution"));
                } else if (argument == "--min-count") {
                    minimumCountOverride = std::stoi(requireValue("--min-count"));
                } else if (argument == "--max-count") {
                    maximumCountOverride = std::stoi(requireValue("--max-count"));
                } else if (argument == "--eval-rounds") {
                    spreadRoundsOverride = std::stoull(requireValue("--eval-rounds"));
                } else if (argument == "--kelly-rounds") {
                    kellyRoundsOverride = std::stoull(requireValue("--kelly-rounds"));
                } else if (argument == "--kelly-measurements") {
                    kellyMeasurementsOverride = std::stoi(requireValue("--kelly-measurements"));
                } else if (argument == "--num-threads") {
                    threadsOverride = std::stoi(requireValue("--num-threads"));
                } else if (argument == "--penetration") {
                    penetrationOverride = std::stod(requireValue("--penetration"));
                } else if (argument == "--kelly-min" ||
                           argument == "--kelly-fraction-min") {
                    kellyMinimumOverride = std::stod(requireValue(argument.c_str()));
                } else if (argument == "--kelly-max" ||
                           argument == "--kelly-fraction-max") {
                    kellyMaximumOverride = std::stod(requireValue(argument.c_str()));
                } else if (argument == "--kelly-step" ||
                           argument == "--kelly-fraction-step") {
                    kellyStepOverride = std::stod(requireValue(argument.c_str()));
                } else if (argument == "--max-total-wager-fraction" ||
                           argument == "--max-total-wager") {
                    maximumTotalWagerFraction =
                        std::stod(requireValue(argument.c_str()));
                } else if (argument == "--seed") {
                    seed = std::stoull(requireValue("--seed"));
                } else if (argument == "--output-dir") {
                    outputDirectory = requireValue("--output-dir");
                } else if (argument == "--num-rounds") {
                    trainingRounds = std::stoull(requireValue("--num-rounds"));
                    trainingRoundsExplicit = true;
                } else if (argument == "--stop-mode") {
                    const std::string mode = requireValue("--stop-mode");
                    if (mode == "rounds")
                        trainingStopMode = TrainingStopMode::FIXED_ROUNDS;
                    else if (mode == "diff")
                        trainingStopMode = TrainingStopMode::TABLE_DIFF;
                    else
                        throw std::invalid_argument("--stop-mode must be rounds or diff");
                } else if (argument == "--sample-rounds") {
                    sampleRounds = std::stoull(requireValue("--sample-rounds"));
                } else if (argument == "--diff-threshold") {
                    differenceThreshold = std::stod(requireValue("--diff-threshold"));
                } else if (argument == "--exploration") {
                    const std::string mode = requireValue("--exploration");
                    if (mode == "epsilon")
                        agent.explorationMode = ExplorationMode::EPSILON_GREEDY;
                    else if (mode == "boltzmann")
                        agent.explorationMode = ExplorationMode::BOLTZMANN;
                    else
                        throw std::invalid_argument("--exploration must be epsilon or boltzmann");
                } else if (argument == "--epsilon-start") {
                    agent.epsilonStart = std::stod(requireValue("--epsilon-start"));
                } else if (argument == "--epsilon-min") {
                    agent.epsilonMinimum = std::stod(requireValue("--epsilon-min"));
                } else if (argument == "--epsilon-decay") {
                    agent.epsilonDecay = std::stod(requireValue("--epsilon-decay"));
                } else if (argument == "--temp-start") {
                    agent.temperatureStart = std::stod(requireValue("--temp-start"));
                } else if (argument == "--temp-min") {
                    agent.temperatureMinimum = std::stod(requireValue("--temp-min"));
                } else if (argument == "--temp-decay") {
                    agent.temperatureDecay = std::stod(requireValue("--temp-decay"));
                } else if (argument == "--alpha-start") {
                    agent.alphaStart = std::stod(requireValue("--alpha-start"));
                } else if (argument == "--alpha-min") {
                    agent.alphaMinimum = std::stod(requireValue("--alpha-min"));
                } else if (argument == "--alpha-decay") {
                    agent.alphaDecaySteps = std::stod(requireValue("--alpha-decay"));
                } else if (argument == "--verbose") {
                    verbose = true;
                } else if (argument == "--decks") {
                    deckSizes = parseList<int>(requireValue("--decks"));
                    standaloneRulesExplicit = true;
                } else if (argument == "--ss17") {
                    standSoft17 = parseList<bool>(requireValue("--ss17"));
                    standaloneRulesExplicit = true;
                } else if (Game::parseArg(argument, index, argc, argv)) {
                    standaloneRulesExplicit = true;
                } else {
                    throw std::invalid_argument("Unknown argument: " + argument);
                }
            }

            if (!checkpointCountLabel.empty() && checkpointInput.empty())
                throw std::invalid_argument("--count Wk requires --checkpoint");
            if (!policyFile.empty() && !policySpec.empty())
                throw std::invalid_argument("Choose --policy or --policy-file, not both");
            if (AlternatingCheckpointLoader::validLabel(policySpec, 'P') &&
                checkpointInput.empty()) {
                throw std::invalid_argument("--policy Pk requires --checkpoint");
            }
            if (trainingRounds == 0 || sampleRounds == 0)
                throw std::invalid_argument("Training round counts must be positive");
            if (!std::isfinite(maximumTotalWagerFraction) ||
                maximumTotalWagerFraction < 0.0 ||
                maximumTotalWagerFraction > 1.0) {
                throw std::invalid_argument(
                    "--max-total-wager-fraction must be between 0 and 1");
            }
            return execute();
        } catch (const std::exception& error) {
            std::cerr << "Error: " << error.what() << "\nUse --help for usage.\n";
            return 1;
        }
    }
};

int runBlackjackEvaluateCountPolicy(int argc, char** argv) {
    return EvaluateCountPolicyApp<BlackjackGame>::run(argc, argv);
}

int runUnsupportedDoubleDownMadnessEvaluateCountPolicy(int, char**) {
    std::cerr << "Error: EvaluateCountPolicy currently supports blackjack only. "
                 "The --game dispatcher is intentionally in place for future games.\n";
    return 1;
}

int main(int argc, char** argv) {
    return dispatchGameApp(
        argc,
        argv,
        "EvaluateCountPolicy",
        runBlackjackEvaluateCountPolicy,
        runUnsupportedDoubleDownMadnessEvaluateCountPolicy);
}
