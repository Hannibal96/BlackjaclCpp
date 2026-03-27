#pragma once
#include <cstdint>
#include <vector>

class Shoe;  // forward declaration — full definition only needed in GlobalRNG.cpp

// Cross-reproducible LCG RNG — identical to Python's SeededRNG.
//
// State is READ first, then advanced (matches Python semantics):
//   next_int(n)  = state % n   → advance
//   next_float() = state / 2^32 → advance
//   choice(v)    = v[next_int(v.size())]
//
// LCG:  state = (state * 1664525 + 1013904223) mod 2^32
//
// Python equivalent: src/Debug/DebugRNG.py :: SeededRNG

class GlobalRNG {
    uint32_t state;

    void advance() {
        state = state * 1664525u + 1013904223u;  // wraps naturally at 2^32
    }

public:
    explicit GlobalRNG(uint32_t seed = 42) : state(seed) {}

    void seed(uint32_t s) { state = s; }
    uint32_t getState() const { return state; }

    // Returns state % n, then advances — matches Python next_int(n)
    int nextInt(int n) {
        int val = static_cast<int>(state % static_cast<uint32_t>(n));
        advance();
        return val;
    }

    // Returns state / 2^32, then advances — matches Python next_float()
    double nextFloat() {
        double val = state / 4294967296.0;
        advance();
        return val;
    }

    // Returns v[nextInt(v.size())] — matches Python choice(lst)
    template<typename T>
    const T& choice(const std::vector<T>& v) {
        return v[nextInt(static_cast<int>(v.size()))];
    }

    // Fisher-Yates shuffle — implemented in GlobalRNG.cpp to avoid circular include
    void shuffleShoe(Shoe& shoe);
};

// Single global instance
inline GlobalRNG gRng(42);
