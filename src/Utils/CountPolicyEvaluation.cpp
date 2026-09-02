#include "CountPolicyEvaluation.h"

#include <cmath>
#include <limits>

double countPolicySpreadThreshold(const CountPolicyConfig& count) {
    if (std::abs(count.system.factor) < 1e-12)
        return std::numeric_limits<double>::infinity();
    const double threshold = -count.system.bias / count.system.factor;
    return count.system.continuousBettingCount
        ? std::nextafter(threshold, std::numeric_limits<double>::infinity())
        : threshold;
}

uint64_t countPolicyMixSeed(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::optional<uint64_t> countPolicyStreamSeed(
        std::optional<uint64_t> baseSeed,
        uint64_t stream) {
    if (!baseSeed) return std::nullopt;
    return countPolicyMixSeed(*baseSeed + stream);
}

const KellyGrowthPoint* kellyPointAtFraction(
        const KellyGrowthCurve& curve,
        double fraction,
        double tolerance) {
    for (const KellyGrowthPoint& point : curve.points) {
        if (std::abs(point.fraction - fraction) <= tolerance)
            return &point;
    }
    return nullptr;
}
