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
        double targetTenValueTag) {
    const double tenValueAverage =
        (rawWeights[8] + rawWeights[9] + rawWeights[10] + rawWeights[11]) / 4.0;
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
            nlohmann::json pointJson = {
                {"kelly_fraction", point.fraction},
                {"growth_mean", point.growthMean},
                {"growth_stddev", point.growthStddev},
                {"error_lower", point.growthMean - point.growthStddev},
                {"error_upper", point.growthMean + point.growthStddev}
            };
            if (point.exposure.rounds > 0)
                pointJson["exposure"] = kellyExposureStatisticsToJson(point.exposure);
            row["points"].push_back(std::move(pointJson));
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
            point.value("growth_stddev", 0.0),
            {}
        });
    }
    return curve;
}

nlohmann::json kellyExposureStatisticsToJson(
        const KellyExposureStatistics& statistics) {
    using json = nlohmann::json;
    json root;
    root["definitions"] = {
        {"bankroll", "bankroll immediately before the round's initial wager"},
        {"gross_exposure", "total gross wagers placed during the round divided by bankroll"},
        {"absolute_return", "absolute net round profit divided by bankroll; this is |fX|"},
        {"wager_multiple", "total gross wagers divided by the initial wager"},
        {"quadratic_log_increment", "r - 0.5*r^2, where r is net round profit divided by bankroll"},
        {"exact_log_increment", "log(1+r)"}
    };
    root["rounds"] = statistics.rounds;
    root["valid_bankroll_rounds"] = statistics.validBankrollRounds;
    root["invalid_bankroll_rounds"] = statistics.invalidBankrollRounds;
    root["zero_wager_rounds"] = statistics.zeroWagerRounds;
    root["invalid_log_rounds"] = statistics.invalidLogRounds;

    const double valid = static_cast<double>(statistics.validBankrollRounds);
    const uint64_t validLogRounds =
        statistics.validBankrollRounds >= statistics.invalidLogRounds
            ? statistics.validBankrollRounds - statistics.invalidLogRounds
            : 0;
    const double validLog = static_cast<double>(validLogRounds);
    root["gross_exposure_summary"] = {
        {"mean", valid > 0.0 ? json(statistics.grossExposureSum / valid) : json(nullptr)},
        {"maximum", statistics.grossExposureMaximum}
    };
    root["absolute_return_summary"] = {
        {"mean", valid > 0.0 ? json(statistics.absoluteReturnSum / valid) : json(nullptr)},
        {"maximum", statistics.absoluteReturnMaximum}
    };
    root["signed_return_summary"] = {
        {"mean", valid > 0.0 ? json(statistics.signedReturnSum / valid) : json(nullptr)},
        {"second_moment", valid > 0.0
            ? json(statistics.signedReturnSquaredSum / valid) : json(nullptr)}
    };
    root["linearization"] = {
        {"valid_log_rounds", validLogRounds},
        {"exact_mean_log_increment", validLog > 0.0
            ? json(statistics.exactLogIncrementSum / validLog) : json(nullptr)},
        {"quadratic_mean_log_increment", validLog > 0.0
            ? json(statistics.quadraticLogIncrementSum / validLog) : json(nullptr)},
        {"mean_signed_error_exact_minus_quadratic", validLog > 0.0
            ? json(statistics.taylorErrorSum / validLog) : json(nullptr)},
        {"mean_absolute_error", validLog > 0.0
            ? json(statistics.absoluteTaylorErrorSum / validLog) : json(nullptr)},
        {"maximum_absolute_error", statistics.absoluteTaylorErrorMaximum}
    };

    auto histogramJson = [&](const auto& histogram) {
        json bins = json::array();
        for (size_t i = 0; i < KellyExposureStatistics::kRegularHistogramBins; ++i) {
            if (histogram[i] == 0) continue;
            bins.push_back({
                {"lower_inclusive", i * KellyExposureStatistics::kHistogramBinWidth},
                {"upper_exclusive", (i + 1) * KellyExposureStatistics::kHistogramBinWidth},
                {"count", histogram[i]},
                {"probability", valid > 0.0
                    ? json(static_cast<double>(histogram[i]) / valid) : json(nullptr)}
            });
        }
        const uint64_t overflow = histogram.back();
        if (overflow > 0) {
            bins.push_back({
                {"lower_inclusive", KellyExposureStatistics::kHistogramMaximum},
                {"upper_exclusive", nullptr},
                {"count", overflow},
                {"probability", valid > 0.0
                    ? json(static_cast<double>(overflow) / valid) : json(nullptr)}
            });
        }
        return bins;
    };
    root["histogram_bin_width"] = KellyExposureStatistics::kHistogramBinWidth;
    root["gross_exposure_histogram"] =
        histogramJson(statistics.grossExposureHistogram);
    root["absolute_return_histogram"] =
        histogramJson(statistics.absoluteReturnHistogram);

    root["wager_multiple_histogram"] = json::array();
    for (size_t i = 0; i < statistics.wagerMultipleHistogram.size(); ++i) {
        const uint64_t count = statistics.wagerMultipleHistogram[i];
        if (count == 0) continue;
        root["wager_multiple_histogram"].push_back({
            {"multiple", i + 1 == statistics.wagerMultipleHistogram.size()
                ? json(nullptr) : json(i)},
            {"lower_bound", i + 1 == statistics.wagerMultipleHistogram.size()
                ? json(i) : json(nullptr)},
            {"count", count},
            {"probability", valid > 0.0
                ? json(static_cast<double>(count) / valid) : json(nullptr)}
        });
    }

    root["tail_probabilities"] = json::array();
    for (size_t i = 0; i < KellyExposureStatistics::kTailThresholds.size(); ++i) {
        root["tail_probabilities"].push_back({
            {"threshold", KellyExposureStatistics::kTailThresholds[i]},
            {"gross_exposure_count", statistics.grossExposureTailCounts[i]},
            {"gross_exposure_probability", valid > 0.0
                ? json(static_cast<double>(statistics.grossExposureTailCounts[i]) / valid)
                : json(nullptr)},
            {"absolute_return_count", statistics.absoluteReturnTailCounts[i]},
            {"absolute_return_probability", valid > 0.0
                ? json(static_cast<double>(statistics.absoluteReturnTailCounts[i]) / valid)
                : json(nullptr)}
        });
    }
    return root;
}

std::string kellyExposureStatisticsToCsv(
        const KellyExposureStatistics& statistics) {
    std::ostringstream output;
    output << "metric,lower_inclusive,upper_exclusive,count,probability\n";
    const double valid = static_cast<double>(statistics.validBankrollRounds);
    auto append = [&](const char* metric, const auto& histogram) {
        for (size_t i = 0; i < KellyExposureStatistics::kRegularHistogramBins; ++i) {
            if (histogram[i] == 0) continue;
            output << metric << ','
                   << i * KellyExposureStatistics::kHistogramBinWidth << ','
                   << (i + 1) * KellyExposureStatistics::kHistogramBinWidth << ','
                   << histogram[i] << ','
                   << (valid > 0.0 ? static_cast<double>(histogram[i]) / valid : 0.0)
                   << '\n';
        }
        if (histogram.back() > 0) {
            output << metric << ',' << KellyExposureStatistics::kHistogramMaximum
                   << ",," << histogram.back() << ','
                   << (valid > 0.0 ? static_cast<double>(histogram.back()) / valid : 0.0)
                   << '\n';
        }
    };
    output << std::setprecision(15);
    append("gross_exposure", statistics.grossExposureHistogram);
    append("absolute_return", statistics.absoluteReturnHistogram);
    return output.str();
}

std::string kellyExposureStatisticsToSvg(
        const std::string& title,
        const KellyExposureStatistics& statistics) {
    constexpr double width = 1200.0;
    constexpr double height = 720.0;
    constexpr double left = 125.0;
    constexpr double right = 45.0;
    constexpr double top = 92.0;
    constexpr double bottom = 92.0;
    const double plotWidth = width - left - right;
    const double plotHeight = height - top - bottom;

    const double observedMaximum = std::max(
        statistics.grossExposureMaximum,
        statistics.absoluteReturnMaximum);
    const size_t displayedBins = std::max<size_t>(1, std::min<size_t>(
        KellyExposureStatistics::kRegularHistogramBins,
        static_cast<size_t>(std::ceil(
            std::min(observedMaximum, KellyExposureStatistics::kHistogramMaximum) /
            KellyExposureStatistics::kHistogramBinWidth))));
    const double dataXMaximum = std::max(
        KellyExposureStatistics::kHistogramBinWidth,
        static_cast<double>(displayedBins) * KellyExposureStatistics::kHistogramBinWidth);
    const double xStep = [&]() {
        if (dataXMaximum <= 0.01) return 0.001;
        if (dataXMaximum <= 0.05) return 0.005;
        if (dataXMaximum <= 0.10) return 0.01;
        if (dataXMaximum <= 0.25) return 0.025;
        if (dataXMaximum <= 0.50) return 0.05;
        return 0.10;
    }();
    const double xMaximum = std::max(
        xStep, std::ceil(dataXMaximum / xStep - 1e-12) * xStep);
    const double valid = static_cast<double>(statistics.validBankrollRounds);

    uint64_t minimumPositiveCount = std::numeric_limits<uint64_t>::max();
    for (size_t i = 0; i < displayedBins; ++i) {
        const uint64_t gross = statistics.grossExposureHistogram[i];
        const uint64_t absoluteReturn = statistics.absoluteReturnHistogram[i];
        if (gross > 0) minimumPositiveCount = std::min(minimumPositiveCount, gross);
        if (absoluteReturn > 0)
            minimumPositiveCount = std::min(minimumPositiveCount, absoluteReturn);
    }
    const double minimumPositiveProbability =
        valid > 0.0 && minimumPositiveCount != std::numeric_limits<uint64_t>::max()
            ? static_cast<double>(minimumPositiveCount) / valid
            : 0.1;
    // Leave one full decade below the smallest observed nonzero bin so a
    // singleton observation still has a visible bar above the baseline.
    const int minimumExponent = std::min(
        -1,
        static_cast<int>(std::floor(std::log10(minimumPositiveProbability))) - 1);
    auto yForProbability = [&](double probability) {
        if (probability <= 0.0) return top + plotHeight;
        const double logProbability = std::clamp(
            std::log10(probability), static_cast<double>(minimumExponent), 0.0);
        return top + (-logProbability / -static_cast<double>(minimumExponent)) * plotHeight;
    };
    auto probabilityForCount = [&](uint64_t count) {
        return valid > 0.0 ? static_cast<double>(count) / valid : 0.0;
    };
    auto percentLabel = [](double fraction) {
        std::ostringstream label;
        const double percent = 100.0 * fraction;
        if (std::abs(percent - std::round(percent)) < 1e-10)
            label << std::fixed << std::setprecision(0) << percent;
        else if (percent >= 1.0)
            label << std::fixed << std::setprecision(1) << percent;
        else if (percent >= 0.01)
            label << std::fixed << std::setprecision(2) << percent;
        else
            label << std::scientific << std::setprecision(0) << percent;
        return label.str() + "%";
    };
    std::ostringstream svg;
    svg << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" << width
        << "\" height=\"" << height << "\" viewBox=\"0 0 " << width << ' ' << height
        << "\">\n<title>" << title << "</title>\n"
        << "<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n"
        << "<defs><clipPath id=\"exposure-plot\"><rect x=\"" << left
        << "\" y=\"" << top << "\" width=\"" << plotWidth
        << "\" height=\"" << plotHeight << "\"/></clipPath></defs>\n";
    svg << "<text x=\"" << left << "\" y=\"30\" font-family=\"sans-serif\" "
        << "font-size=\"21\" font-weight=\"600\">" << title << "</text>\n"
        << "<text x=\"" << left << "\" y=\"54\" font-family=\"sans-serif\" "
        << "font-size=\"13\" fill=\"#555\">Bin width: 0.1% of bankroll; "
        << "the first bin includes zero exposure</text>\n";

    // Horizontal logarithmic probability grid and labels.
    for (int exponent = 0; exponent >= minimumExponent; --exponent) {
        const double probability = std::pow(10.0, exponent);
        const double y = yForProbability(probability);
        svg << "<line x1=\"" << left << "\" y1=\"" << y << "\" x2=\""
            << (left + plotWidth) << "\" y2=\"" << y
            << "\" stroke=\"#d6dbe1\" stroke-width=\"1\"/>\n"
            << "<line x1=\"" << (left - 6) << "\" y1=\"" << y << "\" x2=\""
            << left << "\" y2=\"" << y << "\" stroke=\"#222\"/>\n"
            << "<text x=\"" << (left - 11) << "\" y=\"" << (y + 4)
            << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\">"
            << percentLabel(probability) << "</text>\n";
    }

    // Vertical bankroll-fraction grid and labels.
    for (size_t tick = 0;; ++tick) {
        const double value = static_cast<double>(tick) * xStep;
        if (value > xMaximum + xStep * 1e-9) break;
        const double x = left + value / xMaximum * plotWidth;
        svg << "<line x1=\"" << x << "\" y1=\"" << top << "\" x2=\"" << x
            << "\" y2=\"" << (top + plotHeight)
            << "\" stroke=\"#e2e6ea\" stroke-width=\"1\"/>\n"
            << "<line x1=\"" << x << "\" y1=\"" << (top + plotHeight)
            << "\" x2=\"" << x << "\" y2=\"" << (top + plotHeight + 6)
            << "\" stroke=\"#222\"/>\n"
            << "<text x=\"" << x << "\" y=\"" << (top + plotHeight + 23)
            << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"12\">"
            << percentLabel(value) << "</text>\n";
    }

    const double groupWidth =
        plotWidth * KellyExposureStatistics::kHistogramBinWidth / xMaximum;
    const double innerWidth = std::max(0.35, groupWidth * 0.44);
    const double baseline = top + plotHeight;
    svg << "<g clip-path=\"url(#exposure-plot)\">\n";
    for (size_t i = 0; i < displayedBins; ++i) {
        const double groupX = left + static_cast<double>(i) * groupWidth;
        const double grossProbability =
            probabilityForCount(statistics.grossExposureHistogram[i]);
        const double returnProbability =
            probabilityForCount(statistics.absoluteReturnHistogram[i]);
        if (grossProbability > 0.0) {
            const double y = yForProbability(grossProbability);
            svg << "<rect x=\"" << (groupX + groupWidth * 0.04) << "\" y=\"" << y
                << "\" width=\"" << innerWidth << "\" height=\"" << (baseline - y)
                << "\" fill=\"#2878b5\" opacity=\"0.82\"/>\n";
        }
        if (returnProbability > 0.0) {
            const double y = yForProbability(returnProbability);
            svg << "<rect x=\"" << (groupX + groupWidth * 0.52) << "\" y=\"" << y
                << "\" width=\"" << innerWidth << "\" height=\"" << (baseline - y)
                << "\" fill=\"#d1495b\" opacity=\"0.82\"/>\n";
        }
    }
    svg << "</g>\n";
    svg << "<rect x=\"" << left << "\" y=\"" << top << "\" width=\"" << plotWidth
        << "\" height=\"" << plotHeight
        << "\" fill=\"none\" stroke=\"#222\" stroke-width=\"1.3\"/>\n";
    svg << "<text x=\"" << (left + plotWidth / 2.0) << "\" y=\"" << (height - 25)
        << "\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"15\">"
        << "Fraction of round-start bankroll</text>\n"
        << "<text x=\"24\" y=\"" << (top + plotHeight / 2.0)
        << "\" transform=\"rotate(-90 24 " << (top + plotHeight / 2.0)
        << ")\" text-anchor=\"middle\" font-family=\"sans-serif\" font-size=\"15\">"
        << "Probability per bin (log scale)</text>\n";
    svg << "<rect x=\"" << (width - 355) << "\" y=\"22\" width=\"16\" height=\"12\" "
        << "fill=\"#2878b5\" opacity=\"0.82\"/>"
        << "<text x=\"" << (width - 332) << "\" y=\"33\" font-family=\"sans-serif\" "
        << "font-size=\"12\">Gross exposure</text>\n";
    svg << "<rect x=\"" << (width - 182) << "\" y=\"22\" width=\"16\" height=\"12\" "
        << "fill=\"#d1495b\" opacity=\"0.82\"/>"
        << "<text x=\"" << (width - 159) << "\" y=\"33\" font-family=\"sans-serif\" "
        << "font-size=\"12\">|fX|</text>\n";

    const uint64_t grossOverflow = statistics.grossExposureHistogram.back();
    const uint64_t returnOverflow = statistics.absoluteReturnHistogram.back();
    if (grossOverflow > 0 || returnOverflow > 0) {
        svg << "<text x=\"" << (left + plotWidth) << "\" y=\"" << (top + 17)
            << "\" text-anchor=\"end\" font-family=\"sans-serif\" font-size=\"12\" "
            << "fill=\"#7a1f1f\">Overflow at ≥100%: gross=" << grossOverflow
            << ", |fX|=" << returnOverflow << "</text>\n";
    }
    svg << "</svg>\n";
    return svg.str();
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
