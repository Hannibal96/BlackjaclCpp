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
// Indices (0-12) among the 13 rank-weight regression variables whose column is
// identically zero in `A` -- that rank never appears with a nonzero removed
// count in any accumulated sample. This is a real, structural fact for some
// games (Spanish 21's 48-card shoe has no rank-TEN cards at all -- J/Q/K
// remain, but plain 10s are structurally absent, not just rarely dealt), not
// sampling noise. Such a dimension can't be fit: treating it as an ordinary
// free variable lets its arbitrary/underdetermined value corrupt every other
// weight once a sum-zero (or similar coupling) constraint ties them together.
// Every regression solver below excludes these dimensions instead.
std::vector<int> degenerateRegressionDimensions(const RegressionMatrix14& A);

RegressionVector14 solveQuadraticKellyRegression(
    const RegressionMatrix14& weightedSecondMoment,
    const RegressionVector14& outcomeFeatureMoment);
RegressionVector14 solveQuadraticKellyRegressionWithFixedBias(
    const RegressionMatrix14& weightedSecondMoment,
    const RegressionVector14& outcomeFeatureMoment,
    double fixedBias);
double learnedCountNormalizationScale(
    const RegressionVector14& rawWeights,
    double targetTenValueTag = -1.0,
    const std::vector<int>& degenerateDimensions = {});
nlohmann::json kellyGrowthCurvesToJson(const std::vector<KellyGrowthCurve>& curves);
KellyGrowthCurve kellyGrowthCurveFromJson(const nlohmann::json& value);
std::string kellyGrowthCurvesToSvg(const std::string& title,
                                   const std::vector<KellyGrowthCurve>& curves);
nlohmann::json conditionalSecondMomentCurvesToJson(
    const std::vector<ConditionalSecondMomentCurve>& curves);
std::string conditionalSecondMomentCurvesToSvg(
    const std::string& title,
    const std::vector<ConditionalSecondMomentCurve>& curves);
