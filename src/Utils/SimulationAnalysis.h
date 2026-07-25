#pragma once

#include "Game/Player.h"
#include <cstdint>
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

struct ConditionalSecondMomentPoint {
    double count = 0.0;
    uint64_t samples = 0;
    double secondMoment = 0.0;
};

struct ConditionalSecondMomentCurve {
    std::string label;
    std::vector<ConditionalSecondMomentPoint> points;
};

EdgeStatistics edgeStatisticsFromPlayer(const Player& player);
std::vector<double> makeKellyFractionGrid(double minimum, double maximum, double step);
nlohmann::json kellyGrowthCurvesToJson(const std::vector<KellyGrowthCurve>& curves);
KellyGrowthCurve kellyGrowthCurveFromJson(const nlohmann::json& value);
std::string kellyGrowthCurvesToSvg(const std::string& title,
                                   const std::vector<KellyGrowthCurve>& curves);
nlohmann::json conditionalSecondMomentCurvesToJson(
    const std::vector<ConditionalSecondMomentCurve>& curves);
std::string conditionalSecondMomentCurvesToSvg(
    const std::string& title,
    const std::vector<ConditionalSecondMomentCurve>& curves);
