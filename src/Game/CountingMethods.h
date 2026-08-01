#pragma once
#include <array>
#include <string>
#include <optional>
#include <vector>
#include <algorithm>
#include <cctype>

// Index mapping: 0=2, 1=3, 2=4, 3=5, 4=6, 5=7, 6=8, 7=9, 8=10, 9=J, 10=Q, 11=K, 12=A

// A counting system describes the signal used for betting:
//   signal = bias + factor * trueCount
//   trueCount = dot(weights, removedCards) / remainingDecks
//
// For OLS-derived systems: set weights = w[0..12], factor = 1.0, bias = w[13].
//   The weights already encode expectancy directly, so no extra scaling is needed.
// For traditional systems (Hi-Lo etc.): the signal is estimated EV; factor converts
//   each true-count unit into an E[game] delta (e.g. 0.005 = 0.5% per count point),
//   and bias is the E[game] at neutral count (game-specific, typically ≈ -house_edge).
struct CountingSystem {
    std::array<double, 13> weights{};  // per-rank counting weights
    double factor = 1.0;               // signal change per unit of true count
    double bias   = 0.0;               // signal at neutral count
    bool continuousBettingCount = false; // true when w'c is a directly fitted wager fraction
};

namespace CountingMethods {

// No counting — always returns trueCount=0, E[game]=bias.
inline constexpr std::array<double, 13> None = {};

// Hi-Lo (Thorp): 2-6=+1, 7-9=0, 10/J/Q/K/A=-1  [balanced]
inline constexpr std::array<double, 13> HiLo =
    {1, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, -1};

// KO / Knock-Out (Vancura & Fuchs): like Hi-Lo but 7=+1  [unbalanced +4/deck]
inline constexpr std::array<double, 13> KO =
    {1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -1, -1, -1};

// Hi-Opt I (Humble): 3-6=+1, 10/J/Q/K=-1, 2/7/8/9/A=0  [balanced]
inline constexpr std::array<double, 13> HiOptI =
    {0, 1, 1, 1, 1, 0, 0, 0, -1, -1, -1, -1, 0};

// Hi-Opt II (Humble): 2/3/6/7=+1, 4/5=+2, 10/J/Q/K=-2, 8/9/A=0  [balanced]
inline constexpr std::array<double, 13> HiOptII =
    {1, 1, 2, 2, 1, 1, 0, 0, -2, -2, -2, -2, 0};

// Omega II (Carlson): 2/3/7=+1, 4/5/6=+2, 9=-1, 10/J/Q/K=-2, 8/A=0  [balanced]
inline constexpr std::array<double, 13> OmegaII =
    {1, 1, 2, 2, 2, 1, 0, -1, -2, -2, -2, -2, 0};

// Zen Count (Snyder): 2/3=+1, 4/5/6=+2, 7=+1, 8/9=0, 10/J/Q/K=-2, A=-1  [balanced]
inline constexpr std::array<double, 13> Zen =
    {1, 1, 2, 2, 2, 1, 0, 0, -2, -2, -2, -2, -1};

// Halves (Wong): 2=+½, 3/4/6=+1, 5=+1½, 7=+½, 8=0, 9=-½, 10/J/Q/K/A=-1  [balanced]
inline constexpr std::array<double, 13> Halves =
    {0.5, 1, 1, 1.5, 1, 0.5, 0, -0.5, -1, -1, -1, -1, -1};

// Canonical parameters for traditional systems:
//   factor ≈ +0.5% E[game] gain per true-count point
//   bias   ≈ -0.5% baseline E[game] at neutral count (≈ house edge)
inline constexpr double kDefaultFactor = 0.005;
inline constexpr double kDefaultBias   = -0.005;

inline CountingSystem make(const std::array<double, 13>& w,
                            double factor = kDefaultFactor,
                            double bias   = kDefaultBias) {
    return {w, factor, bias};
}

// Look up a named system (factor=kDefaultFactor, bias=kDefaultBias for traditional systems).
// Returns std::nullopt if the name is unrecognised.
inline std::optional<CountingSystem> fromName(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    if (name == "none")                             return CountingSystem{None,    kDefaultFactor, kDefaultBias};
    if (name == "hilo")                             return CountingSystem{HiLo,    kDefaultFactor, kDefaultBias};
    if (name == "ko")                               return CountingSystem{KO,      kDefaultFactor, kDefaultBias};
    if (name == "hiopt1" || name == "hi-opt-1")     return CountingSystem{HiOptI,  kDefaultFactor, kDefaultBias};
    if (name == "hiopt2" || name == "hi-opt-2")     return CountingSystem{HiOptII, kDefaultFactor, kDefaultBias};
    if (name == "omega2" || name == "omegaii")      return CountingSystem{OmegaII, kDefaultFactor, kDefaultBias};
    if (name == "zen")                              return CountingSystem{Zen,     kDefaultFactor, kDefaultBias};
    if (name == "halves")                           return CountingSystem{Halves,  kDefaultFactor, kDefaultBias};
    return std::nullopt;
}

inline std::vector<std::string> allNames() {
    return {"none", "hilo", "ko", "hiopt1", "hiopt2", "omega2", "zen", "halves"};
}

} // namespace CountingMethods
