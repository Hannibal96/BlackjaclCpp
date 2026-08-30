#include "Utils/CountQuantization.h"

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

bool close(double lhs, double rhs, double tolerance = 1e-12) {
    return std::abs(lhs - rhs) <= tolerance;
}

double sum(const std::array<double, 13>& weights) {
    double result = 0.0;
    for (double weight : weights) result += weight;
    return result;
}

} // namespace

int main() {
    const std::array<double, 13> learned = {
        0.6583743314422037,
        0.810831348789431,
        1.1374409378693022,
        1.4013387314297188,
        0.8982465532471269,
        0.5221690969752901,
        0.03067675015849421,
        -0.3107841301627139,
        -1.0000000000000002,
        -1.0000000000000002,
        -1.0000000000000002,
        -1.0000000000000002,
        -1.1482936197488522
    };

    const auto reference = quantizeCountWeights(learned, 0.0);
    require(reference.weights == learned, "quantum zero must be bit-for-bit exact");
    require(reference.sourceWasBalanced, "learned source should be detected as balanced");
    require(reference.balancePreserved, "quantum zero should preserve balance");

    const auto integer = quantizeCountWeights(learned, 1.0);
    const std::array<double, 13> expectedInteger = {
        1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1
    };
    require(integer.weights == expectedInteger,
            "constrained integer quantization should choose the minimum-error balanced tags");
    require(integer.balancePreserved && close(sum(integer.weights), 0.0),
            "integer quantization must preserve exact balance");

    const auto tenth = quantizeCountWeights(learned, 0.1);
    require(close(sum(tenth.weights), 0.0), "0.1 quantization should remain balanced");
    require(tenth.weights[8] == tenth.weights[9] &&
            tenth.weights[9] == tenth.weights[10] &&
            tenth.weights[10] == tenth.weights[11],
            "ten-valued ranks must retain a shared tag");

    std::array<double, 13> unbalanced{};
    unbalanced[0] = 0.6;
    const auto ordinary = quantizeCountWeights(unbalanced, 1.0);
    require(!ordinary.sourceWasBalanced, "unbalanced count should not be marked balanced");
    require(ordinary.weights[0] == 1.0,
            "unbalanced count should use ordinary nearest-lattice rounding");

    std::cout << "CountQuantizationTest passed\n";
    return 0;
}
