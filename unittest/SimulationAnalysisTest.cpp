#include <gtest/gtest.h>

#include "Game/Player.h"
#include "Game/BlackjackRules.h"
#include "RL/RandomStrategy.h"
#include "Utils/SimulationAnalysis.h"
#include "Utils/Utils.h"

#include <cmath>
#include <memory>

namespace {

Player makePlayer() {
    return Player(0.0, std::make_unique<RandomStrategy>());
}

TEST(SimulationAnalysisTest, ComputesStreamingRoundMoments) {
    Player player = makePlayer();
    player.enableRoundStats();
    player.recordRoundOutcome(1.0);
    player.recordRoundOutcome(-1.0);
    player.recordRoundOutcome(2.0);

    const EdgeStatistics stats = edgeStatisticsFromPlayer(player);
    EXPECT_EQ(stats.samples, 3U);
    EXPECT_NEAR(stats.mean, 2.0 / 3.0, 1e-12);
    EXPECT_NEAR(stats.secondMoment, 2.0, 1e-12);
    EXPECT_NEAR(stats.stddev, std::sqrt(7.0 / 3.0), 1e-12);
}

TEST(SimulationAnalysisTest, CombinesAbsoluteMomentTotalsAcrossPlayers) {
    Player lhs = makePlayer();
    lhs.enableRoundStats();
    lhs.recordRoundOutcome(1.0);

    Player rhs = makePlayer();
    rhs.enableRoundStats();
    rhs.recordRoundOutcome(3.0);

    lhs += rhs;
    lhs *= 0.5;

    const EdgeStatistics stats = edgeStatisticsFromPlayer(lhs);
    EXPECT_EQ(stats.samples, 2U);
    EXPECT_NEAR(stats.mean, 2.0, 1e-12);
    EXPECT_NEAR(stats.secondMoment, 5.0, 1e-12);
    EXPECT_NEAR(stats.stddev, std::sqrt(2.0), 1e-12);
}

TEST(SimulationAnalysisTest, BuildsInclusiveDefaultKellyGrid) {
    const auto grid = makeKellyFractionGrid(0.65, 1.0, 0.05);
    ASSERT_EQ(grid.size(), 8U);
    EXPECT_NEAR(grid.front(), 0.65, 1e-12);
    EXPECT_NEAR(grid.back(), 1.0, 1e-12);
}

TEST(SimulationAnalysisTest, RejectsInvalidKellyGrid) {
    EXPECT_THROW(makeKellyFractionGrid(1.0, 0.5, 0.05), std::invalid_argument);
    EXPECT_THROW(makeKellyFractionGrid(0.5, 1.0, 0.0), std::invalid_argument);
}

TEST(SimulationAnalysisTest, SelectsLargestEmpiricalGrowth) {
    KellyGrowthCurve curve;
    curve.points = {{0.75, 1.00001, 0.0}, {0.80, 1.00003, 0.0}, {0.85, 1.00002, 0.0}};
    ASSERT_NE(curve.optimalPoint(), nullptr);
    EXPECT_DOUBLE_EQ(curve.optimalPoint()->fraction, 0.80);
}

TEST(SimulationAnalysisTest, SerializesKellyErrorBoundsAndRendersErrorBars) {
    KellyGrowthCurve curve;
    curve.label = "test";
    curve.points = {{0.65, 1.0, 0.0002}};

    const auto json = kellyGrowthCurvesToJson({curve});
    const auto& point = json.at("curves").at(0).at("points").at(0);
    EXPECT_EQ(json.at("error_bar"), "mean_plus_minus_sample_stddev");
    EXPECT_NEAR(point.at("error_lower").get<double>(), 0.9998, 1e-12);
    EXPECT_NEAR(point.at("error_upper").get<double>(), 1.0002, 1e-12);

    const std::string svg = kellyGrowthCurvesToSvg("test", {curve});
    EXPECT_NE(svg.find("Error bars: mean +/- sample stddev"), std::string::npos);
}

TEST(SimulationAnalysisTest, SerializesAndRendersConditionalSecondMoments) {
    ConditionalSecondMomentCurve curve;
    curve.label = "flat policy";
    curve.points = {
        {-0.25, 40, 1.15},
        {0.0, 0, 0.0},
        {0.25, 60, 1.35}
    };

    const auto json = conditionalSecondMomentCurvesToJson({curve});
    EXPECT_EQ(json.at("statistic"), "E[X^2 | count]");
    EXPECT_DOUBLE_EQ(
        json.at("curves").at(0).at("points").at(0)
            .at("conditional_second_moment").get<double>(),
        1.15);
    EXPECT_TRUE(
        json.at("curves").at(0).at("points").at(1)
            .at("conditional_second_moment").is_null());

    const std::string svg =
        conditionalSecondMomentCurvesToSvg("second moment", {curve});
    EXPECT_NE(svg.find("E[X^2 | count], unit initial wager"), std::string::npos);
    EXPECT_NE(svg.find("flat policy"), std::string::npos);
}

TEST(SimulationAnalysisTest, ParallelSimulationPreservesAllRoundSamples) {
    constexpr uint64_t rounds = 1000;
    constexpr int threads = 4;
    Player player(0.0, std::make_unique<RandomStrategy>(42));
    player.enableRoundStats();

    auto results = runParallelSimulation(BlackjackRules(), {&player}, rounds, threads);
    ASSERT_EQ(results.size(), 1U);
    const EdgeStatistics stats = edgeStatisticsFromPlayer(*results.front());
    EXPECT_EQ(stats.samples, rounds);
    EXPECT_NEAR(stats.mean,
                results.front()->getMoney() / (static_cast<double>(rounds) / threads),
                1e-12);
    delete results.front();
}

} // namespace
