#include "SimulationAnalysis.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

const KellyGrowthPoint* KellyGrowthCurve::optimalPoint() const {
    if (points.empty()) return nullptr;
    return &*std::max_element(points.begin(), points.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.growthMean < rhs.growthMean;
    });
}

EdgeStatistics edgeStatisticsFromPlayer(const Player& player) {
    EdgeStatistics result;
    result.samples = player.getRoundStatsCount();
    if (result.samples == 0) return result;

    const double n = static_cast<double>(result.samples);
    result.mean = player.getRoundRewardSum() / n;
    result.secondMoment = player.getRoundRewardSumSq() / n;
    if (result.samples > 1) {
        const double centered = player.getRoundRewardSumSq()
            - player.getRoundRewardSum() * player.getRoundRewardSum() / n;
        result.stddev = std::sqrt(std::max(0.0, centered / (n - 1.0)));
    }
    return result;
}

std::vector<double> makeKellyFractionGrid(double minimum, double maximum, double step) {
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || !std::isfinite(step) ||
        minimum < 0.0 || maximum < minimum || step <= 0.0) {
        throw std::invalid_argument(
            "Kelly fraction range requires 0 <= min <= max and step > 0");
    }

    std::vector<double> values;
    const double tolerance = step * 1e-9;
    for (uint64_t i = 0;; ++i) {
        const double value = minimum + static_cast<double>(i) * step;
        if (value > maximum + tolerance) break;
        values.push_back(std::min(value, maximum));
        if (i > 1'000'000)
            throw std::invalid_argument("Kelly fraction range contains too many points");
    }
    if (values.empty()) values.push_back(minimum);
    return values;
}

KellyFractionRange resolveKellyFractionRange(
        double predictedOptimalFraction,
        std::optional<double> minimumOverride,
        std::optional<double> maximumOverride,
        double step,
        double radius) {
    if (!std::isfinite(predictedOptimalFraction) || predictedOptimalFraction < 0.0 ||
        !std::isfinite(step) || step <= 0.0 ||
        !std::isfinite(radius) || radius < 0.0) {
        throw std::invalid_argument(
            "Kelly range center/radius must be non-negative and step must be positive");
    }

    const double snappedCenter =
        std::round(predictedOptimalFraction / step) * step;
    KellyFractionRange range;
    range.minimum = minimumOverride.value_or(
        std::max(0.0, snappedCenter - radius));
    range.maximum = maximumOverride.value_or(snappedCenter + radius);

    // Reuse grid validation so mixed dynamic/explicit endpoints fail consistently.
    (void)makeKellyFractionGrid(range.minimum, range.maximum, 1.0);
    return range;
}

std::vector<int> degenerateRegressionDimensions(const RegressionMatrix14& A) {
    std::vector<int> degenerate;
    for (int i = 0; i < 13; ++i) {
        if (std::abs(A[i][i]) < 1e-15) degenerate.push_back(i);
    }
    return degenerate;
}

RegressionVector14 solveQuadraticKellyRegression(
        const RegressionMatrix14& weightedSecondMoment,
        const RegressionVector14& outcomeFeatureMoment) {
    // Maximize the second-order approximation
    // E[log(1 + X w'c)] ~= w'E[Xc] - 0.5 w'E[X^2 cc']w,
    // with c augmented by a constant feature so the bias is learned as w[13].
    std::array<std::array<double, 15>, 14> augmented{};
    for (int i = 0; i < 14; ++i) {
        for (int j = 0; j < 14; ++j)
            augmented[i][j] = weightedSecondMoment[i][j];
        augmented[i][14] = outcomeFeatureMoment[i];
    }

    // Force degenerate ranks (see degenerateRegressionDimensions()) to exactly
    // 0 instead of leaving them as an unsolvable free variable. Their column is
    // already all zero elsewhere (that rank never varies), so this only
    // removes a dimension with no information -- it doesn't affect any other
    // weight, unlike a naive singular-pivot failure or an unconstrained fit.
    for (int idx : degenerateRegressionDimensions(weightedSecondMoment)) {
        for (int j = 0; j < 15; ++j) augmented[idx][j] = 0.0;
        augmented[idx][idx] = 1.0;
    }

    for (int col = 0; col < 14; ++col) {
        int pivot = col;
        for (int row = col + 1; row < 14; ++row) {
            if (std::abs(augmented[row][col]) > std::abs(augmented[pivot][col]))
                pivot = row;
        }
        if (pivot != col) std::swap(augmented[pivot], augmented[col]);

        const double diagonal = augmented[col][col];
        if (std::abs(diagonal) < 1e-15) {
            throw std::runtime_error(
                "Singular quadratic-Kelly moment matrix - not enough feature variation");
        }
        for (int j = col; j <= 14; ++j)
            augmented[col][j] /= diagonal;

        for (int row = 0; row < 14; ++row) {
            if (row == col) continue;
            const double factor = augmented[row][col];
            for (int j = col; j <= 14; ++j)
                augmented[row][j] -= factor * augmented[col][j];
        }
    }

    RegressionVector14 solution{};
    for (int i = 0; i < 14; ++i) solution[i] = augmented[i][14];
    return solution;
}

RegressionVector14 solveQuadraticKellyRegressionWithFixedBias(
        const RegressionMatrix14& weightedSecondMoment,
        const RegressionVector14& outcomeFeatureMoment,
        double fixedBias) {
    RegressionMatrix14 constrainedMoment = weightedSecondMoment;
    RegressionVector14 constrainedOutcome = outcomeFeatureMoment;

    // Substitute the fixed intercept into the 13 rank-feature equations, then
    // replace its own equation with beta = fixedBias.
    for (int i = 0; i < 13; ++i) {
        constrainedOutcome[i] -= weightedSecondMoment[i][13] * fixedBias;
        constrainedMoment[i][13] = 0.0;
        constrainedMoment[13][i] = 0.0;
    }
    constrainedMoment[13][13] = 1.0;
    constrainedOutcome[13] = fixedBias;
    return solveQuadraticKellyRegression(constrainedMoment, constrainedOutcome);
}

double learnedCountNormalizationScale(
        const RegressionVector14& rawWeights,
        double targetTenValueTag,
        const std::vector<int>& degenerateDimensions) {
    // Average only the ten-value ranks that actually exist in the shoe. A
    // degenerate one (e.g. Spanish 21's rank-TEN) was forced to exactly 0 by
    // the solver, not fitted -- including it here would drag the average (and
    // so the scale applied to every other weight) toward a meaningless value.
    double tenValueSum = 0.0;
    int tenValueCount = 0;
    for (int i = 8; i <= 11; ++i) {
        if (std::find(degenerateDimensions.begin(), degenerateDimensions.end(), i) !=
            degenerateDimensions.end()) {
            continue;
        }
        tenValueSum += rawWeights[i];
        ++tenValueCount;
    }
    if (tenValueCount == 0) {
        throw std::invalid_argument(
            "Cannot normalize learned count: no non-degenerate 10/J/Q/K rank to average");
    }
    const double tenValueAverage = tenValueSum / tenValueCount;
    if (!std::isfinite(targetTenValueTag) ||
        !std::isfinite(tenValueAverage) ||
        std::abs(tenValueAverage) < 1e-15) {
        throw std::invalid_argument(
            "Cannot normalize learned count with a zero or non-finite 10/J/Q/K average");
    }
    return targetTenValueTag / tenValueAverage;
}

nlohmann::json kellyGrowthCurvesToJson(const std::vector<KellyGrowthCurve>& curves) {
    nlohmann::json root;
    root["error_bar"] = "mean_plus_minus_sample_stddev";
    root["curves"] = nlohmann::json::array();
    for (const auto& curve : curves) {
        nlohmann::json row;
        row["label"] = curve.label;
        row["predicted_optimal_fraction_1_over_ex2"] = curve.predictedOptimalFraction;
        if (const auto* optimum = curve.optimalPoint()) {
            row["empirical_optimal_fraction"] = optimum->fraction;
            row["empirical_optimal_growth"] = optimum->growthMean;
        } else {
            row["empirical_optimal_fraction"] = nullptr;
            row["empirical_optimal_growth"] = nullptr;
        }
        row["points"] = nlohmann::json::array();
        for (const auto& point : curve.points) {
            row["points"].push_back({
                {"kelly_fraction", point.fraction},
                {"growth_mean", point.growthMean},
                {"growth_stddev", point.growthStddev},
                {"error_lower", point.growthMean - point.growthStddev},
                {"error_upper", point.growthMean + point.growthStddev}
            });
        }
        root["curves"].push_back(std::move(row));
    }
    return root;
}

KellyGrowthCurve kellyGrowthCurveFromJson(const nlohmann::json& value) {
    const auto& row = value.contains("curves") ? value.at("curves").at(0) : value;
    KellyGrowthCurve curve;
    curve.label = row.value("label", std::string{});
    curve.predictedOptimalFraction =
        row.value("predicted_optimal_fraction_1_over_ex2", 0.0);
    for (const auto& point : row.value("points", nlohmann::json::array())) {
        curve.points.push_back({
            point.value("kelly_fraction", 1.0),
            point.value("growth_mean", 1.0),
            point.value("growth_stddev", 0.0)
        });
    }
    return curve;
}

namespace {

std::string colorForCurve(size_t index) {
    static const char* colors[] = {
        "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"
    };
    return colors[index % (sizeof(colors) / sizeof(colors[0]))];
}

} // namespace

std::string kellyGrowthCurvesToSvg(const std::string& title,
                                   const std::vector<KellyGrowthCurve>& curves) {
    constexpr double width = 1200.0;
    constexpr double height = 720.0;
    constexpr double left = 105.0;
    constexpr double right = 300.0;
    constexpr double top = 48.0;
    constexpr double bottom = 78.0;
    const double plotWidth = width - left - right;
    const double plotHeight = height - top - bottom;

    double minX = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const auto& curve : curves) {
        for (const auto& point : curve.points) {
            minX = std::min(minX, point.fraction);
            maxX = std::max(maxX, point.fraction);
            minY = std::min(minY, point.growthMean - point.growthStddev);
            maxY = std::max(maxY, point.growthMean + point.growthStddev);
        }
    }
    if (!std::isfinite(minX) || !std::isfinite(maxX)) { minX = 0.65; maxX = 1.0; }
    if (std::abs(maxX - minX) < 1e-12) { minX -= 0.05; maxX += 0.05; }
    if (!std::isfinite(minY) || !std::isfinite(maxY)) { minY = 0.99999; maxY = 1.00001; }
    const double yPadding = std::max(1e-9, (maxY - minY) * 0.15);
    minY -= yPadding;
    maxY += yPadding;

    auto mapX = [&](double x) { return left + (x - minX) / (maxX - minX) * plotWidth; };
    auto mapY = [&](double y) { return top + (1.0 - (y - minY) / (maxY - minY)) * plotHeight; };

    std::ostringstream svg;
    svg << std::setprecision(12);
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    svg << "<text x=\"" << left << "\" y=\"30\" font-family=\"sans-serif\" font-size=\"20\">"
        << title << "</text>\n";

    for (int i = 0; i <= 10; ++i) {
        const double x = left + plotWidth * i / 10.0;
        const double y = top + plotHeight * i / 10.0;
        svg << "<line x1=\"" << x << "\" y1=\"" << top << "\" x2=\"" << x
            << "\" y2=\"" << (top + plotHeight) << "\" stroke=\"#c8c8c8\" stroke-width=\"1\"/>\n";
        svg << "<line x1=\"" << left << "\" y1=\"" << y << "\" x2=\"" << (left + plotWidth)
            << "\" y2=\"" << y << "\" stroke=\"#c8c8c8\" stroke-width=\"1\"/>\n";
        if (i % 2 == 0) {
            const double xValue = minX + (maxX - minX) * i / 10.0;
            const double yValue = maxY - (maxY - minY) * i / 10.0;
            svg << "<text x=\"" << x << "\" y=\"" << (top + plotHeight + 24)
                << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">"
                << std::fixed << std::setprecision(2) << xValue << "</text>\n";
            svg << "<text x=\"" << (left - 10) << "\" y=\"" << (y + 4)
                << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">"
                << std::setprecision(9) << yValue << "</text>\n";
        }
    }
    svg << std::defaultfloat << std::setprecision(12);
    svg << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plotWidth
        << "\" height=\"" << plotHeight << "\" fill=\"none\" stroke=\"#222\" stroke-width=\"1.5\"/>\n";

    for (size_t curveIndex = 0; curveIndex < curves.size(); ++curveIndex) {
        const auto& curve = curves[curveIndex];
        const std::string color = colorForCurve(curveIndex);
        std::ostringstream path;
        for (size_t i = 0; i < curve.points.size(); ++i) {
            path << (i == 0 ? "M " : " L ") << mapX(curve.points[i].fraction)
                 << " " << mapY(curve.points[i].growthMean);
        }
        if (!curve.points.empty())
            svg << "<path d=\"" << path.str() << "\" fill=\"none\" stroke=\"" << color
                << "\" stroke-width=\"3\"/>\n";
        for (const auto& point : curve.points) {
            const double x = mapX(point.fraction);
            const double upperY = mapY(point.growthMean + point.growthStddev);
            const double lowerY = mapY(point.growthMean - point.growthStddev);
            svg << "<line x1=\"" << x << "\" y1=\"" << upperY << "\" x2=\"" << x
                << "\" y2=\"" << lowerY << "\" stroke=\"" << color
                << "\" stroke-width=\"2\" opacity=\"0.85\"/>\n";
            svg << "<line x1=\"" << (x - 5.0) << "\" y1=\"" << upperY
                << "\" x2=\"" << (x + 5.0) << "\" y2=\"" << upperY
                << "\" stroke=\"" << color << "\" stroke-width=\"2\" opacity=\"0.85\"/>\n";
            svg << "<line x1=\"" << (x - 5.0) << "\" y1=\"" << lowerY
                << "\" x2=\"" << (x + 5.0) << "\" y2=\"" << lowerY
                << "\" stroke=\"" << color << "\" stroke-width=\"2\" opacity=\"0.85\"/>\n";
            svg << "<circle cx=\"" << mapX(point.fraction) << "\" cy=\"" << mapY(point.growthMean)
                << "\" r=\"4\" fill=\"" << color << "\"/>\n";
        }

        if (curve.predictedOptimalFraction >= minX && curve.predictedOptimalFraction <= maxX) {
            const double x = mapX(curve.predictedOptimalFraction);
            svg << "<line x1=\"" << x << "\" y1=\"" << top << "\" x2=\"" << x
                << "\" y2=\"" << (top + plotHeight) << "\" stroke=\"" << color
                << "\" stroke-width=\"1.5\" stroke-dasharray=\"7,5\" opacity=\"0.8\"/>\n";
        }

        const double legendY = 70.0 + 48.0 * curveIndex;
        svg << "<line x1=\"" << (width - right + 30) << "\" y1=\"" << legendY
            << "\" x2=\"" << (width - right + 72) << "\" y2=\"" << legendY
            << "\" stroke=\"" << color << "\" stroke-width=\"3\"/>\n";
        svg << "<text x=\"" << (width - right + 82) << "\" y=\"" << (legendY + 4)
            << "\" font-family=\"sans-serif\" font-size=\"12\">" << curve.label << "</text>\n";
        svg << "<text x=\"" << (width - right + 82) << "\" y=\"" << (legendY + 20)
            << "\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#555\">1/E[X^2]="
            << std::fixed << std::setprecision(4) << curve.predictedOptimalFraction << "</text>\n";
        svg << std::defaultfloat << std::setprecision(12);
    }

    svg << "<text x=\"" << (width - right + 30) << "\" y=\"" << (height - 35)
        << "\" font-family=\"sans-serif\" font-size=\"11\" fill=\"#555\">"
        << "Error bars: mean +/- sample stddev</text>\n";
    svg << "<text x=\"" << (left + plotWidth / 2.0) << "\" y=\"" << (height - 22)
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Kelly multiplier</text>\n";
    svg << "<text x=\"24\" y=\"" << (top + plotHeight / 2.0)
        << "\" transform=\"rotate(-90 24 " << (top + plotHeight / 2.0)
        << ")\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Expected growth per round</text>\n";
    svg << "</svg>\n";
    return svg.str();
}

nlohmann::json conditionalSecondMomentCurvesToJson(
    const std::vector<ConditionalSecondMomentCurve>& curves) {
    nlohmann::json root;
    root["statistic"] = "E[X^2 | count]";
    root["x_definition"] = "net profit from one complete round at unit initial wager";
    root["curves"] = nlohmann::json::array();
    for (const auto& curve : curves) {
        nlohmann::json row;
        row["label"] = curve.label;
        row["points"] = nlohmann::json::array();
        for (const auto& point : curve.points) {
            row["points"].push_back({
                {"count", point.count},
                {"n", point.samples},
                {"conditional_second_moment",
                 point.samples > 0 ? nlohmann::json(point.secondMoment)
                                   : nlohmann::json(nullptr)}
            });
        }
        root["curves"].push_back(std::move(row));
    }
    return root;
}

std::string conditionalSecondMomentCurvesToSvg(
    const std::string& title,
    const std::vector<ConditionalSecondMomentCurve>& curves) {
    constexpr double width = 1100.0;
    constexpr double height = 680.0;
    constexpr double left = 95.0;
    constexpr double right = 230.0;
    constexpr double top = 48.0;
    constexpr double bottom = 76.0;
    const double plotWidth = width - left - right;
    const double plotHeight = height - top - bottom;

    double minX = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const auto& curve : curves) {
        for (const auto& point : curve.points) {
            minX = std::min(minX, point.count);
            maxX = std::max(maxX, point.count);
            if (point.samples > 0) {
                minY = std::min(minY, point.secondMoment);
                maxY = std::max(maxY, point.secondMoment);
            }
        }
    }
    if (!std::isfinite(minX) || !std::isfinite(maxX)) {
        minX = -5.0;
        maxX = 5.0;
    }
    if (std::abs(maxX - minX) < 1e-12) {
        minX -= 0.5;
        maxX += 0.5;
    }
    if (!std::isfinite(minY) || !std::isfinite(maxY)) {
        minY = 0.0;
        maxY = 1.0;
    }
    if (std::abs(maxY - minY) < 1e-12) {
        const double padding = std::max(0.05, std::abs(minY) * 0.05);
        minY = std::max(0.0, minY - padding);
        maxY += padding;
    } else {
        const double padding = 0.1 * (maxY - minY);
        minY = std::max(0.0, minY - padding);
        maxY += padding;
    }

    auto mapX = [&](double x) {
        return left + (x - minX) / (maxX - minX) * plotWidth;
    };
    auto mapY = [&](double y) {
        return top + (1.0 - (y - minY) / (maxY - minY)) * plotHeight;
    };

    std::ostringstream svg;
    svg << std::setprecision(12);
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << " " << height << "\">\n";
    svg << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    svg << "<text x=\"" << left << "\" y=\"30\" font-family=\"sans-serif\" font-size=\"20\">"
        << title << "</text>\n";

    for (int i = 0; i <= 20; ++i) {
        const double x = left + plotWidth * i / 20.0;
        const double y = top + plotHeight * i / 20.0;
        const bool major = i % 2 == 0;
        svg << "<line x1=\"" << x << "\" y1=\"" << top << "\" x2=\"" << x
            << "\" y2=\"" << (top + plotHeight) << "\" stroke=\""
            << (major ? "#b8b8b8" : "#dddddd") << "\" stroke-width=\""
            << (major ? "1.5" : "1.1") << "\"/>\n";
        svg << "<line x1=\"" << left << "\" y1=\"" << y << "\" x2=\""
            << (left + plotWidth) << "\" y2=\"" << y << "\" stroke=\""
            << (major ? "#b8b8b8" : "#dddddd") << "\" stroke-width=\""
            << (major ? "1.5" : "1.1") << "\"/>\n";
        if (major) {
            const double xValue = minX + (maxX - minX) * i / 20.0;
            const double yValue = maxY - (maxY - minY) * i / 20.0;
            svg << "<text x=\"" << x << "\" y=\"" << (top + plotHeight + 23)
                << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"11\">"
                << std::fixed << std::setprecision(2) << xValue << "</text>\n";
            svg << "<text x=\"" << (left - 9.0) << "\" y=\"" << (y + 4.0)
                << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"11\">"
                << std::setprecision(4) << yValue << "</text>\n";
        }
    }
    svg << std::defaultfloat << std::setprecision(12);
    svg << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plotWidth
        << "\" height=\"" << plotHeight
        << "\" fill=\"none\" stroke=\"#222\" stroke-width=\"1.5\"/>\n";

    if (minX <= 0.0 && maxX >= 0.0) {
        const double zeroX = mapX(0.0);
        svg << "<line x1=\"" << zeroX << "\" y1=\"" << top << "\" x2=\"" << zeroX
            << "\" y2=\"" << (top + plotHeight)
            << "\" stroke=\"#555\" stroke-width=\"1.8\"/>\n";
    }

    for (size_t curveIndex = 0; curveIndex < curves.size(); ++curveIndex) {
        const auto& curve = curves[curveIndex];
        const std::string color = colorForCurve(curveIndex);
        std::ostringstream path;
        bool started = false;
        for (const auto& point : curve.points) {
            if (point.samples == 0) continue;
            path << (started ? " L " : "M ") << mapX(point.count)
                 << " " << mapY(point.secondMoment);
            started = true;
        }
        if (started) {
            svg << "<path d=\"" << path.str() << "\" fill=\"none\" stroke=\""
                << color << "\" stroke-width=\"2.6\"/>\n";
        }
        for (const auto& point : curve.points) {
            if (point.samples == 0) continue;
            svg << "<circle cx=\"" << mapX(point.count) << "\" cy=\""
                << mapY(point.secondMoment) << "\" r=\"2.5\" fill=\""
                << color << "\"/>\n";
        }

        const double legendY = 68.0 + 28.0 * curveIndex;
        svg << "<line x1=\"" << (width - right + 25.0) << "\" y1=\"" << legendY
            << "\" x2=\"" << (width - right + 65.0) << "\" y2=\"" << legendY
            << "\" stroke=\"" << color << "\" stroke-width=\"2.6\"/>\n";
        svg << "<text x=\"" << (width - right + 75.0) << "\" y=\""
            << (legendY + 4.0)
            << "\" font-family=\"sans-serif\" font-size=\"12\">"
            << curve.label << "</text>\n";
    }

    svg << "<text x=\"" << (left + plotWidth / 2.0) << "\" y=\"" << (height - 20.0)
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">Count</text>\n";
    svg << "<text x=\"22\" y=\"" << (top + plotHeight / 2.0)
        << "\" transform=\"rotate(-90 22 " << (top + plotHeight / 2.0)
        << ")\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"14\">"
        << "E[X^2 | count], unit initial wager</text>\n";
    svg << "</svg>\n";
    return svg.str();
}
