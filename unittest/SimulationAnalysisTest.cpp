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

TEST(SimulationAnalysisTest, AccumulatesQuadraticKellySufficientStatistics) {
    Player player = makePlayer();
    player.enableRegression(RegressionObjective::QUADRATIC_KELLY);
    std::array<double, 13> features{};
    features[0] = 2.0;
    features[1] = -1.0;

    player.recordRound(features, 3.0);

    EXPECT_DOUBLE_EQ(player.getXtX()[0][0], 36.0);
    EXPECT_DOUBLE_EQ(player.getXtX()[0][1], -18.0);
    EXPECT_DOUBLE_EQ(player.getXtX()[0][13], 18.0);
    EXPECT_DOUBLE_EQ(player.getXtX()[13][13], 9.0);
    EXPECT_DOUBLE_EQ(player.getXty()[0], 6.0);
    EXPECT_DOUBLE_EQ(player.getXty()[1], -3.0);
    EXPECT_DOUBLE_EQ(player.getXty()[13], 3.0);
    EXPECT_EQ(player.getRegressionRounds(), 1U);
}

TEST(SimulationAnalysisTest, KeepsOrdinaryOlsAccumulationUnweighted) {
    Player player = makePlayer();
    player.enableRegression();
    std::array<double, 13> features{};
    features[0] = 2.0;

    player.recordRound(features, 3.0);

    EXPECT_DOUBLE_EQ(player.getXtX()[0][0], 4.0);
    EXPECT_DOUBLE_EQ(player.getXtX()[13][13], 1.0);
    EXPECT_DOUBLE_EQ(player.getXty()[0], 6.0);
}

TEST(SimulationAnalysisTest, CombinesQuadraticKellyStatisticsAcrossPlayers) {
    Player lhs = makePlayer();
    lhs.enableRegression(RegressionObjective::QUADRATIC_KELLY);
    Player rhs = makePlayer();
    rhs.enableRegression(RegressionObjective::QUADRATIC_KELLY);
    std::array<double, 13> features{};
    features[0] = 1.0;
    lhs.recordRound(features, 2.0);
    rhs.recordRound(features, -3.0);

    lhs += rhs;

    EXPECT_DOUBLE_EQ(lhs.getXtX()[0][0], 13.0);
    EXPECT_DOUBLE_EQ(lhs.getXty()[0], -1.0);
    EXPECT_EQ(lhs.getRegressionRounds(), 2U);
}

TEST(SimulationAnalysisTest, SolvesQuadraticKellyMomentEquationIncludingBias) {
    RegressionMatrix14 secondMoment{};
    RegressionVector14 outcomeMoment{};
    secondMoment[0][0] = 4.0;
    secondMoment[0][1] = 1.0;
    secondMoment[1][0] = 1.0;
    secondMoment[1][1] = 3.0;
    outcomeMoment[0] = 0.7;  // A * [0.2, -0.1]
    outcomeMoment[1] = -0.1;
    for (int i = 2; i < 13; ++i) {
        const double diagonal = static_cast<double>(i + 1);
        const double expectedWeight = 0.01 * static_cast<double>(i + 1);
        secondMoment[i][i] = diagonal;
        outcomeMoment[i] = diagonal * expectedWeight;
    }
    secondMoment[13][13] = 7.0;
    outcomeMoment[13] = 4.0;

    const auto solution =
        solveQuadraticKellyRegression(secondMoment, outcomeMoment);

    EXPECT_NEAR(solution[0], 0.2, 1e-12);
    EXPECT_NEAR(solution[1], -0.1, 1e-12);
    for (int i = 2; i < 13; ++i)
        EXPECT_NEAR(solution[i], 0.01 * static_cast<double>(i + 1), 1e-12);
    EXPECT_NEAR(solution[13], 4.0 / 7.0, 1e-12);
}

TEST(SimulationAnalysisTest, SolvesQuadraticKellyWithFixedZeroBias) {
    RegressionMatrix14 secondMoment{};
    RegressionVector14 outcomeMoment{};
    for (int i = 0; i < 14; ++i) secondMoment[i][i] = 1.0;
    secondMoment[0][13] = 0.5;
    secondMoment[13][0] = 0.5;
    outcomeMoment[0] = 0.25;
    outcomeMoment[13] = 0.75;

    const auto solution = solveQuadraticKellyRegressionWithFixedBias(
        secondMoment, outcomeMoment, 0.0);

    EXPECT_NEAR(solution[0], 0.25, 1e-12);
    EXPECT_DOUBLE_EQ(solution[13], 0.0);
}

TEST(SimulationAnalysisTest, UsesContinuousCountForDirectKellyBetSizing) {
    CountingSystem count;
    count.weights[0] = 1.0;
    count.factor = 1.0;
    count.continuousBettingCount = true;

    Player player(1.0, std::make_unique<RandomStrategy>());
    player.setNumDecks(1);
    player.setCountSystem(count);
    player.setCountResolution(1.0);
    player.setBettingStrategy(std::make_unique<KellyBetting>());
    std::array<int, 13> removed{};
    removed[0] = 1;

    EXPECT_NEAR(player.getBet(removed), 52.0 / 51.0, 1e-12);

    count.continuousBettingCount = false;
    player.setCountSystem(count);
    EXPECT_DOUBLE_EQ(player.getBet(removed), 1.0);
}

TEST(SimulationAnalysisTest, NormalizesTenValueTagsWithoutChangingSignal) {
    RegressionVector14 raw{};
    raw[0] = 0.004;
    raw[8] = -0.003;
    raw[9] = -0.004;
    raw[10] = -0.005;
    raw[11] = -0.004;
    raw[13] = -0.006;

    const double scale = learnedCountNormalizationScale(raw);
    const double normalizedTenAverage =
        (raw[8] * scale + raw[9] * scale + raw[10] * scale + raw[11] * scale) / 4.0;

    EXPECT_NEAR(normalizedTenAverage, -1.0, 1e-12);
    EXPECT_NEAR((raw[0] * scale) * (1.0 / scale), raw[0], 1e-12);
    EXPECT_DOUBLE_EQ(raw[13], -0.006);
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

TEST(SimulationAnalysisTest, SnapsDefaultKellyRangeCenterToStep) {
    const KellyFractionRange range = resolveKellyFractionRange(0.73);
    EXPECT_NEAR(range.minimum, 0.50, 1e-12);
    EXPECT_NEAR(range.maximum, 1.00, 1e-12);
}

TEST(SimulationAnalysisTest, ClampsDefaultKellyRangeAtZero) {
    const KellyFractionRange range = resolveKellyFractionRange(0.10);
    EXPECT_DOUBLE_EQ(range.minimum, 0.0);
    EXPECT_NEAR(range.maximum, 0.35, 1e-12);
}

TEST(SimulationAnalysisTest, AppliesKellyRangeEndpointOverridesIndependently) {
    const KellyFractionRange minimumOverride =
        resolveKellyFractionRange(0.74, 0.60, std::nullopt);
    EXPECT_DOUBLE_EQ(minimumOverride.minimum, 0.60);
    EXPECT_NEAR(minimumOverride.maximum, 1.00, 1e-12);

    const KellyFractionRange maximumOverride =
        resolveKellyFractionRange(0.74, std::nullopt, 0.90);
    EXPECT_NEAR(maximumOverride.minimum, 0.50, 1e-12);
    EXPECT_DOUBLE_EQ(maximumOverride.maximum, 0.90);
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
