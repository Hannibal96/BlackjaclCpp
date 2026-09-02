#pragma once

#include "Game/Player.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

struct EdgeStatistics {
    uint64_t samples = 0;
    double mean = 0.0;
    double stddev = 0.0;
    double secondMoment = 0.0;
};

struct KellyGrowthPoint {
    double fraction = 1.0;
    double growthMean = 1.0;
    double growthStddev = 0.0;
    KellyExposureStatistics exposure;
};

struct KellyGrowthCurve {
    std::string label;
    double predictedOptimalFraction = 0.0;
    std::vector<KellyGrowthPoint> points;

    const KellyGrowthPoint* optimalPoint() const;
};

struct KellyFractionRange {
    double minimum = 0.0;
    double maximum = 0.0;
};

struct ConditionalSecondMomentPoint {
    double count = 0.0;
    uint64_t samples = 0;
    double secondMoment = 0.0;
};

struct ConditionalSecondMomentCurve {
    std::string label;
    std::vector<ConditionalSecondMomentPoint> points;
};

using RegressionMatrix14 = std::array<std::array<double, 14>, 14>;
using RegressionVector14 = std::array<double, 14>;

EdgeStatistics edgeStatisticsFromPlayer(const Player& player);
std::vector<double> makeKellyFractionGrid(double minimum, double maximum, double step);
KellyFractionRange resolveKellyFractionRange(
    double predictedOptimalFraction,
    std::optional<double> minimumOverride = std::nullopt,
    std::optional<double> maximumOverride = std::nullopt,
    double step = 0.05,
    double radius = 0.25);
RegressionVector14 solveQuadraticKellyRegression(
    const RegressionMatrix14& weightedSecondMoment,
    const RegressionVector14& outcomeFeatureMoment);
RegressionVector14 solveQuadraticKellyRegressionWithFixedBias(
    const RegressionMatrix14& weightedSecondMoment,
    const RegressionVector14& outcomeFeatureMoment,
    double fixedBias);
double learnedCountNormalizationScale(
    const RegressionVector14& rawWeights,
    double targetTenValueTag = -1.0);
nlohmann::json kellyGrowthCurvesToJson(const std::vector<KellyGrowthCurve>& curves);
KellyGrowthCurve kellyGrowthCurveFromJson(const nlohmann::json& value);
nlohmann::json kellyExposureStatisticsToJson(
    const KellyExposureStatistics& statistics);
std::string kellyExposureStatisticsToCsv(
    const KellyExposureStatistics& statistics);
std::string kellyExposureStatisticsToSvg(
    const std::string& title,
    const KellyExposureStatistics& statistics);
std::string kellyGrowthCurvesToSvg(const std::string& title,
                                   const std::vector<KellyGrowthCurve>& curves);
nlohmann::json conditionalSecondMomentCurvesToJson(
    const std::vector<ConditionalSecondMomentCurve>& curves);
std::string conditionalSecondMomentCurvesToSvg(
    const std::string& title,
    const std::vector<ConditionalSecondMomentCurve>& curves);
