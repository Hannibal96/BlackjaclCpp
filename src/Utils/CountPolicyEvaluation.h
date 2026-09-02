#pragma once

#include "Game/BettingStrategy.h"
#include "Game/Player.h"
#include "RL/BasicStrategy.h"
#include "SimulationAnalysis.h"
#include "Utils.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Shared configuration and simulation engine for evaluating one fixed count
// together with one fixed playing policy. Research workflows such as count
// quantization and strategy comparison should orchestrate this engine instead
// of maintaining their own copies of spread/Kelly simulation code.
struct CountPolicyConfig {
    CountingSystem system;
    double resolution = 1.0;
    int minCount = -5;
    int maxCount = 5;
};

struct CountPolicyEvaluationOptions {
    uint64_t spreadRounds = 1'000'000'000ULL;
    uint64_t kellyRoundsPerMeasurement = 1'000'000ULL;
    int kellyMeasurements = 10;
    int numThreads = 10;
    std::vector<double> kellyFractions = makeKellyFractionGrid(0.75, 1.25, 0.05);
    double spreadMinimumBet = 1.0;
    double spreadMaximumBet = 10.0;
    // Cumulative gross wagers in a Kelly round may not exceed this fraction
    // of bankroll immediately before the initial wager.
    double maximumTotalWagerFraction = 1.0;
    std::optional<uint64_t> seed;
};

struct CountPolicyEvaluationResult {
    EdgeStatistics spread;
    KellyGrowthCurve kelly;
};

double countPolicySpreadThreshold(const CountPolicyConfig& count);
uint64_t countPolicyMixSeed(uint64_t value);
std::optional<uint64_t> countPolicyStreamSeed(
    std::optional<uint64_t> baseSeed,
    uint64_t stream);
const KellyGrowthPoint* kellyPointAtFraction(
    const KellyGrowthCurve& curve,
    double fraction,
    double tolerance = 1e-12);

inline void configureCountPolicyPlayer(Player& player,
                                       int numDecks,
                                       const BasicStrategy& policy,
                                       const CountPolicyConfig& count) {
    player.setNumDecks(numDecks);
    player.setCountSystem(count.system);
    player.setCountResolution(count.resolution);

    // A loaded policy is authoritative about the keys it contains. Freshly
    // trained policies normally match count.minCount/maxCount, while a custom
    // policy file may legitimately cover a different range.
    const auto [policyMinimum, policyMaximum] = policy.getCountRange();
    player.setCountRange(policyMinimum, policyMaximum);
}

template <typename Rules>
CountPolicyEvaluationResult evaluateCountPolicy(
        Rules rules,
        const BasicStrategy& policy,
        const CountPolicyConfig& count,
        const CountPolicyEvaluationOptions& options,
        const std::string& label = "count + policy") {
    if (rules.numDecks < 1)
        throw std::invalid_argument("Count-policy evaluation requires at least one deck");
    if (options.spreadRounds == 0 || options.kellyRoundsPerMeasurement == 0)
        throw std::invalid_argument("Spread and Kelly round counts must be positive");
    if (options.numThreads < 1)
        throw std::invalid_argument("Number of threads must be at least one");
    if (options.spreadRounds < static_cast<uint64_t>(options.numThreads) ||
        options.kellyRoundsPerMeasurement < static_cast<uint64_t>(options.numThreads)) {
        throw std::invalid_argument("Round counts must be at least the number of threads");
    }
    if (options.kellyMeasurements < 1)
        throw std::invalid_argument("Kelly measurements must be at least one");
    if (options.kellyFractions.empty())
        throw std::invalid_argument("Kelly multiplier list cannot be empty");
    for (double fraction : options.kellyFractions) {
        if (!std::isfinite(fraction) || fraction < 0.0)
            throw std::invalid_argument("Kelly multipliers must be finite and non-negative");
    }
    if (!std::isfinite(options.spreadMinimumBet) ||
        !std::isfinite(options.spreadMaximumBet) ||
        options.spreadMinimumBet < 0.0 ||
        options.spreadMaximumBet < options.spreadMinimumBet) {
        throw std::invalid_argument("Spread bets require 0 <= minimum <= maximum");
    }
    if (!std::isfinite(options.maximumTotalWagerFraction) ||
        options.maximumTotalWagerFraction < 0.0 ||
        options.maximumTotalWagerFraction > 1.0) {
        throw std::invalid_argument(
            "Maximum total wager fraction must be between 0 and 1");
    }

    CountPolicyEvaluationResult evaluation;
    evaluation.kelly.label = label;
    evaluation.kelly.predictedOptimalFraction = 1.0;

    {
        Rules spreadRules = rules;
        spreadRules.minBet = options.spreadMinimumBet;
        spreadRules.maxBet = options.spreadMaximumBet;

        auto player = std::make_unique<Player>(0.0, policy.clone());
        configureCountPolicyPlayer(*player, rules.numDecks, policy, count);
        player->setBettingStrategy(std::make_unique<PositiveSignalSpreadBetting>(
            options.spreadMaximumBet,
            !count.system.continuousBettingCount));
        player->enableRoundStats();

        std::vector<Player*> results = runParallelSimulation(
            spreadRules,
            {player.get()},
            options.spreadRounds,
            options.numThreads,
            countPolicyStreamSeed(options.seed, 0x535052454144ULL));
        if (results.empty())
            throw std::runtime_error("Spread evaluation failed");
        std::unique_ptr<Player> result(results.front());
        evaluation.spread = edgeStatisticsFromPlayer(*result);
        if (evaluation.spread.samples != options.spreadRounds) {
            throw std::runtime_error(
                "Spread statistics are incomplete: expected " +
                std::to_string(options.spreadRounds) + " rounds, got " +
                std::to_string(evaluation.spread.samples));
        }
    }

    Rules kellyRules = rules;
    kellyRules.minBet = 0.0;
    kellyRules.maxBet = std::numeric_limits<double>::max();
    const uint64_t expectedExposureRounds =
        options.kellyRoundsPerMeasurement *
        static_cast<uint64_t>(options.kellyMeasurements);

    for (size_t fractionIndex = 0;
         fractionIndex < options.kellyFractions.size();
         ++fractionIndex) {
        const double fraction = options.kellyFractions[fractionIndex];
        double growthSum = 0.0;
        double growthSquaredSum = 0.0;
        KellyExposureStatistics exposure;

        for (int measurement = 0;
             measurement < options.kellyMeasurements;
             ++measurement) {
            auto player = std::make_unique<Player>(1.0, policy.clone());
            configureCountPolicyPlayer(*player, rules.numDecks, policy, count);
            player->setBettingStrategy(std::make_unique<KellyBetting>(fraction));
            player->setEnforceBankrollActionLimits(true);
            player->setMaximumTotalWagerFraction(
                options.maximumTotalWagerFraction);
            player->enableKellyExposureStats();

            const uint64_t stream = 0x4b454c4c59000000ULL +
                static_cast<uint64_t>(fractionIndex) *
                    static_cast<uint64_t>(options.kellyMeasurements) +
                static_cast<uint64_t>(measurement);
            std::vector<Player*> results = runParallelSimulation(
                kellyRules,
                {player.get()},
                options.kellyRoundsPerMeasurement,
                options.numThreads,
                countPolicyStreamSeed(options.seed, stream));
            if (results.empty())
                throw std::runtime_error("Kelly evaluation failed");
            std::unique_ptr<Player> result(results.front());

            // runParallelSimulation averages thread-local log bankrolls. The
            // matching average number of rounds therefore recovers the mean
            // log increment per simulated round, including remainder rounds.
            const double averageRoundsPerThread =
                static_cast<double>(options.kellyRoundsPerMeasurement) /
                static_cast<double>(options.numThreads);
            const double averageLogFinal = result->getLogMoney();
            const double growth = std::isfinite(averageLogFinal)
                ? std::exp(averageLogFinal / averageRoundsPerThread)
                : 0.0;
            growthSum += growth;
            growthSquaredSum += growth * growth;
            exposure += result->getKellyExposureStats();
        }

        if (exposure.rounds != expectedExposureRounds ||
            exposure.validBankrollRounds + exposure.invalidBankrollRounds !=
                expectedExposureRounds) {
            throw std::runtime_error(
                "Kelly exposure statistics are incomplete at multiplier " +
                std::to_string(fraction) + ": expected " +
                std::to_string(expectedExposureRounds) + " rounds, got " +
                std::to_string(exposure.rounds));
        }

        const double measurements =
            static_cast<double>(options.kellyMeasurements);
        KellyGrowthPoint point;
        point.fraction = fraction;
        point.growthMean = growthSum / measurements;
        if (options.kellyMeasurements > 1) {
            const double sampleVariance =
                (growthSquaredSum - growthSum * growthSum / measurements) /
                (measurements - 1.0);
            point.growthStddev = std::sqrt(std::max(0.0, sampleVariance));
        }
        point.exposure = std::move(exposure);
        evaluation.kelly.points.push_back(std::move(point));
    }

    return evaluation;
}
