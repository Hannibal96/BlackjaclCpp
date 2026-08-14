# Debug Mode and Reproducible Simulations

This document explains how to enable fully deterministic, reproducible simulations for debugging and cross-validation with Python implementations.

## Overview

The codebase uses random number generators (RNGs) in several places:
1. **Shoe shuffling** - `std::mt19937` for card randomization
2. **DebugShoe** - Linear Congruential Generator (LCG) for deterministic card dealing
3. **RandomStrategy** - `std::mt19937` for random action selection
4. **QLearningStrategy** - `std::mt19937` for epsilon-greedy exploration

All of these can be seeded for reproducibility.

## Quick Start - Debug Mode

### Method 1: Using RandomSeedManager (Recommended)

```cpp
#include "Utils/RandomSeed.h"

// Enable debug mode with base seed
RandomSeedManager::enableDebugMode(42);

// Create components - they will automatically use deterministic seeds
Shoe shoe(6, 75.0);  // Uses seed derived from base seed
auto qlearning = std::make_unique<QLearningStrategy>(
    std::make_unique<EpsilonDecayingParameter>(0.1, 0.01, 0.9999),
    std::make_unique<LinearDecayingParameter>(0.5, 0.1, 1000),
    1.0  // gamma
);  // Will use RandomSeedManager if modified to check it

// Run deterministic simulation
// ...

// Disable debug mode
RandomSeedManager::disableDebugMode();
```

### Method 2: Using RAII Guard (Auto-cleanup)

```cpp
#include "Utils/RandomSeed.h"

{
    DebugModeGuard guard(42);  // Enable debug mode

    // All RNGs created here use deterministic seeds
    Shoe shoe(6, 75.0);
    // ...

}  // Debug mode automatically disabled when guard goes out of scope
```

### Method 3: Explicit Seeds (Component-level)

```cpp
// DebugShoe with explicit seed
DebugShoe shoe(1, 75.0, 42);

// Regular Shoe with explicit seed (NEW!)
Shoe shoe(6, 75.0, 42);

// RandomStrategy with seed
auto strategy = std::make_unique<RandomStrategy>(42);

// QLearningStrategy with seed (NEW!)
auto qlearning = std::make_unique<QLearningStrategy>(
    std::make_unique<EpsilonDecayingParameter>(0.1, 0.01, 0.9999),
    std::make_unique<LinearDecayingParameter>(0.5, 0.1, 1000),
    1.0,   // gamma
    42     // seed
);
```

## Python Alignment

Both C++ and Python use the Mersenne Twister 19937 algorithm, so setting the same seed produces identical random sequences.

### C++ Code

```cpp
#include <random>
#include <iostream>

int main() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < 5; ++i) {
        std::cout << dist(rng) << std::endl;
    }
}
```

### Python Code (NumPy)

```python
import numpy as np

rng = np.random.default_rng(42)
for i in range(5):
    print(rng.random())
```

### Python Code (random module)

```python
import random

random.seed(42)
for i in range(5):
    print(random.random())
```

**Note:** While the algorithms are the same, the exact sequence may differ between implementations due to:
- Different distribution implementations
- Different state initialization
- Version differences

For exact cross-validation, compare:
1. **Final results** (e.g., edge percentages) rather than individual random numbers
2. **DebugShoe sequences** - the LCG can be exactly replicated in Python

## DebugShoe and Python

The `DebugShoe` uses a simple Linear Congruential Generator (LCG) that can be exactly matched in Python:

### C++ DebugShoe

```cpp
DebugShoe shoe(1, 75.0, 42);  // 1 deck, 75% penetration, seed 42

for (int i = 0; i < 5; ++i) {
    Card card = shoe.dealCard();
    std::cout << "Card: " << card.rank << std::endl;
}
```

### Python Equivalent

```python
class DebugShoe:
    def __init__(self, num_decks, penetration, seed=42):
        self.state = seed & 0xFFFFFFFF  # 32-bit unsigned
        self.cards = self.initialize_cards(num_decks)
        self.initial_seed = seed

    def next_random(self):
        # LCG formula: state = (state * 1664525 + 1013904223) % 2^32
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.state

    def deal_card(self):
        n = len(self.cards)
        index = self.state % n
        card = self.cards[index]
        self.cards.pop(index)
        self.state = (self.state * 1664525 + 1013904223) & 0xFFFFFFFF
        return card
```

## Cross-Validation Example

To verify C++ and Python produce identical results:

### C++ Test

```cpp
DebugShoe shoe(1, 100.0, 12345);
auto strategy = std::make_unique<RandomStrategy>(12345);

BlackjackRules rules;
Player player(0.0, std::move(strategy));
BlackjackTable table(rules, {&player});

// Run 10000 hands
for (int i = 0; i < 10000; ++i) {
    table.round();
}

std::cout << "Final bankroll: " << player.getMoney() << std::endl;
```

### Python Test

```python
# Implement equivalent shoe, rules, strategy in Python
# with same seeds (12345)

# Run 10000 hands
# ...

print(f"Final bankroll: {player.money}")
```

If both implementations are correct, final bankroll should match within floating-point precision.

## Testing Reproducibility

The `ReproducibilityTest` executable demonstrates deterministic Q-learning training. It's a standalone program that uses:
- DebugShoe with explicit seed for deterministic card dealing
- QLearningStrategy with explicit seed for deterministic exploration
- Single-threaded execution for full reproducibility

### Running the Test

```bash
# Build and run
cmake --build build/debug
./build/debug/bin/ReproducibilityTest

# Verify reproducibility by running multiple times
./build/debug/bin/ReproducibilityTest > run1.txt
./build/debug/bin/ReproducibilityTest > run2.txt
diff run1.txt run2.txt  # Should be identical

# Or compare final results
./build/debug/bin/ReproducibilityTest | tail -10
```

## Best Practices

1. **Use DebugShoe for exact reproducibility** - The LCG is simpler to replicate across languages
2. **Use Shoe with seed for approximate reproducibility** - MT19937 is more random but harder to match exactly
3. **Use RandomSeedManager for regression runs** - Centralized seed management
4. **Document seeds in regressions** - Make it easy to reproduce failures
5. **Verify with Python** - Cross-validate critical algorithms

## Common Pitfalls

1. **Thread-local RNG state** - Each thread may have different state
2. **Parallel simulations** - May produce different results due to race conditions
3. **Floating-point differences** - Minor platform differences in FP arithmetic
4. **Library versions** - Different C++ standard library versions may produce different sequences

## Advanced: Parallel Reproducibility

For parallel simulations to be reproducible:

```cpp
// Create per-thread seeds deterministically
std::vector<unsigned int> threadSeeds;
for (int i = 0; i < numThreads; ++i) {
    threadSeeds.push_back(baseSeed + i * 1000);
}

// Each thread uses its own seed
auto workerThread = [&](int threadId) {
    unsigned int threadSeed = threadSeeds[threadId];

    // Create thread-local components with deterministic seed
    DebugShoe shoe(6, 75.0, threadSeed);
    auto strategy = std::make_unique<RandomStrategy>(threadSeed + 1);

    // ... run simulation
};
```

## Summary

- **DebugShoe**: Fully deterministic LCG, exact Python replication possible
- **Shoe with seed**: MT19937-based, approximate Python alignment
- **Strategy with seed**: Reproducible exploration/exploitation
- **RandomSeedManager**: Centralized seed management for complex simulations

For debugging, use DebugShoe + explicit seeds. For production with occasional debugging, use RandomSeedManager.
