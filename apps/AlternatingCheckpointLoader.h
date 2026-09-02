#pragma once

#include "Game/CountingMethods.h"
#include "RL/BasicStrategy.h"
#include "Utils/CountPolicyEvaluation.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

struct AlternatingEvaluationDefaults {
    double penetration = 75.0;
    uint64_t spreadRounds = 1'000'000'000ULL;
    uint64_t kellyRoundsPerMeasurement = 1'000'000ULL;
    int kellyMeasurements = 10;
    int numThreads = 10;
    double kellyMinimum = 0.75;
    double kellyMaximum = 1.25;
    double kellyStep = 0.05;
};

template <typename Game>
struct AlternatingCheckpointSelection {
    using Case = typename Game::Case;

    std::filesystem::path root;
    nlohmann::json meta;
    Case gameCase;
    std::string countLabel;
    std::string policyLabel;
    std::string countPrecisionSource;
    CountPolicyConfig count;
    std::unique_ptr<BasicStrategy> policy;
    AlternatingEvaluationDefaults defaults;
};

namespace AlternatingCheckpointLoader {

inline nlohmann::json readJson(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input.is_open())
        throw std::runtime_error("Cannot open " + path.string());
    nlohmann::json value;
    input >> value;
    return value;
}

inline bool validLabel(const std::string& label, char prefix) {
    return label.size() >= 2 && label.front() == prefix &&
        std::all_of(label.begin() + 1, label.end(), [](char character) {
            return character >= '0' && character <= '9';
        });
}

inline int labelIndex(const std::string& label, char prefix) {
    if (!validLabel(label, prefix)) {
        throw std::invalid_argument(
            std::string("Expected a label such as ") + prefix + "3, got '" +
            label + "'");
    }
    return std::stoi(label.substr(1));
}

template <typename Game>
std::filesystem::path resolvePath(const std::string& input) {
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

    for (const fs::path& candidate : candidates) {
        if (fs::is_directory(candidate) &&
            fs::is_regular_file(candidate / "meta.json")) {
            return fs::weakly_canonical(candidate);
        }
    }
    throw std::runtime_error(
        "Cannot find alternating checkpoint folder or meta.json for '" + input + "'");
}

inline void resolveLabels(const std::filesystem::path& checkpoint,
                          std::string& countLabel,
                          std::string& policyLabel) {
    namespace fs = std::filesystem;
    if (countLabel.empty() && !policyLabel.empty()) {
        countLabel = "W" + std::to_string(labelIndex(policyLabel, 'P') + 1);
    }

    if (countLabel.empty()) {
        const std::regex weightPattern(R"(^W([1-9][0-9]*)\.json$)");
        int latest = -1;
        for (const fs::directory_entry& entry : fs::directory_iterator(checkpoint)) {
            if (!entry.is_regular_file()) continue;
            std::smatch match;
            const std::string filename = entry.path().filename().string();
            if (!std::regex_match(filename, match, weightPattern)) continue;
            const int countIndex = std::stoi(match[1].str());
            if (fs::is_regular_file(
                    checkpoint /
                    ("P" + std::to_string(countIndex - 1) + "_strategy.json"))) {
                latest = std::max(latest, countIndex);
            }
        }
        if (latest < 1)
            throw std::runtime_error("Checkpoint contains no complete Wk + P(k-1) pair");
        countLabel = "W" + std::to_string(latest);
    }

    const int countIndex = labelIndex(countLabel, 'W');
    if (policyLabel.empty())
        policyLabel = "P" + std::to_string(std::max(0, countIndex - 1));
    (void)labelIndex(policyLabel, 'P');

    const fs::path countPath = checkpoint / (countLabel + ".json");
    const fs::path policyPath = checkpoint / (policyLabel + "_strategy.json");
    if (countIndex > 0 && !fs::is_regular_file(countPath))
        throw std::runtime_error("Missing count artifact " + countPath.string());
    if (!fs::is_regular_file(policyPath))
        throw std::runtime_error("Missing policy artifact " + policyPath.string());
}

inline CountPolicyConfig countFromJson(const nlohmann::json& stored) {
    CountPolicyConfig count;
    if (!stored.contains("weights") || stored.at("weights").size() != 13)
        throw std::runtime_error("Count configuration requires exactly 13 weights");
    for (size_t index = 0; index < count.system.weights.size(); ++index)
        count.system.weights[index] = stored.at("weights").at(index).get<double>();
    count.system.factor = stored.value("factor", 1.0);
    count.system.bias = stored.value("bias", 0.0);
    count.system.continuousBettingCount =
        stored.value("continuous_betting_count", false);
    const auto normalization = countNormalizationFromString(
        stored.value("count_normalization", std::string("true_count")));
    if (!normalization)
        throw std::runtime_error("Invalid count_normalization in checkpoint count");
    count.system.normalization = *normalization;
    count.system.initialCount = stored.value("initial_count", 0.0);
    count.system.initialCountPerDeck =
        stored.value("initial_count_per_deck", 0.0);
    count.resolution = stored.value("resolution", 1.0);
    count.minCount = stored.value("min_count", -5);
    count.maxCount = stored.value("max_count", 5);
    return count;
}

inline CountPolicyConfig loadCount(const std::filesystem::path& checkpoint,
                                   const nlohmann::json& meta,
                                   const std::string& countLabel,
                                   std::string& precisionSource) {
    const int countIndex = labelIndex(countLabel, 'W');
    if (countIndex == 0) {
        precisionSource = "meta.algorithm_config.initial_count";
        return countFromJson(meta.at("algorithm_config").at("initial_count"));
    }

    const nlohmann::json artifact = readJson(checkpoint / (countLabel + ".json"));
    CountPolicyConfig count = countFromJson(artifact.at("count_config"));
    const bool canReconstruct =
        artifact.contains("raw_solution") && artifact.at("raw_solution").is_array() &&
        artifact.at("raw_solution").size() == 14 &&
        artifact.contains("normalization_scale") &&
        artifact.at("normalization_scale").is_number();
    if (!canReconstruct) {
        precisionSource = "saved count_config weights";
        return count;
    }

    const double scale = artifact.at("normalization_scale").get<double>();
    if (!std::isfinite(scale) || std::abs(scale) < 1e-15)
        throw std::runtime_error("Invalid normalization_scale in " + countLabel + ".json");
    for (size_t index = 0; index < count.system.weights.size(); ++index) {
        count.system.weights[index] =
            artifact.at("raw_solution").at(index).get<double>() * scale;
    }
    const double tenValueAverage =
        (count.system.weights[8] + count.system.weights[9] +
         count.system.weights[10] + count.system.weights[11]) / 4.0;
    count.system.weights[8] = tenValueAverage;
    count.system.weights[9] = tenValueAverage;
    count.system.weights[10] = tenValueAverage;
    count.system.weights[11] = tenValueAverage;
    precisionSource = "raw_solution x normalization_scale";
    return count;
}

inline AlternatingEvaluationDefaults evaluationDefaults(
        const nlohmann::json& meta) {
    AlternatingEvaluationDefaults defaults;
    const nlohmann::json& game = meta.at("game");
    const nlohmann::json& algorithm = meta.at("algorithm_config");
    defaults.penetration = game.value("penetration", 75.0);
    defaults.spreadRounds = algorithm.value(
        "eval_rounds",
        algorithm.value("num_rounds_per_phase", uint64_t(1'000'000'000ULL)));
    defaults.kellyRoundsPerMeasurement = algorithm.value(
        "kelly_rounds_per_measurement", uint64_t(1'000'000ULL));
    defaults.kellyMeasurements = algorithm.value("kelly_measurements", 10);
    defaults.numThreads = algorithm.value("num_threads", 10);
    // A narrow search around the directly fitted Kelly multiplier is the
    // evaluation default, independent of an older checkpoint's plotting range.
    defaults.kellyMinimum = 0.75;
    defaults.kellyMaximum = 1.25;
    defaults.kellyStep = 0.05;
    return defaults;
}

template <typename Game>
AlternatingCheckpointSelection<Game> load(
        const std::string& input,
        std::string countLabel = {},
        std::string policyLabel = {}) {
    AlternatingCheckpointSelection<Game> selection;
    selection.root = resolvePath<Game>(input);
    selection.meta = readJson(selection.root / "meta.json");
    if (selection.meta.value("algorithm", std::string{}).find(
            "alternating_optimization") == std::string::npos) {
        throw std::runtime_error(
            selection.root.string() + " is not an alternating-optimization checkpoint");
    }

    resolveLabels(selection.root, countLabel, policyLabel);
    selection.countLabel = std::move(countLabel);
    selection.policyLabel = std::move(policyLabel);
    selection.gameCase = Game::readMetaGameSection(selection.meta.at("game"));
    selection.defaults = evaluationDefaults(selection.meta);
    selection.count = loadCount(
        selection.root,
        selection.meta,
        selection.countLabel,
        selection.countPrecisionSource);

    selection.policy = std::make_unique<BasicStrategy>();
    const std::filesystem::path policyPath =
        selection.root / (selection.policyLabel + "_strategy.json");
    if (!selection.policy->loadFromFile(policyPath.string()))
        throw std::runtime_error("Cannot load policy " + policyPath.string());
    return selection;
}

} // namespace AlternatingCheckpointLoader
