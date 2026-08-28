#pragma once
#include "../Game/BlackjackRules.h"
#include "../Game/DoubleDownMadnessRules.h"
#include "../Game/SpanishRules.h"
#include "../Game/Player.h"
#include <ostream>
#include <string>
#include <vector>

// Simple utility function to run a blackjack simulation
// Parameters:
//   - rules: The blackjack rules to use
//   - players: Vector of players (their money will be updated in place)
//   - numRounds: Number of rounds to simulate
//   - showProgress: Whether to show progress bar (default true, set to false for multithreaded use)
// Returns:
//   - true if simulation completed successfully, false otherwise
bool runSimulation(const BlackjackRules& rules, std::vector<Player*>& players, uint64_t numRounds);

// Parallel simulation function
// Parameters:
//   - rules: The blackjack rules to use
//   - players: Vector of players to use as template (will be cloned for each thread)
//   - numRounds: Total number of rounds to simulate (distributed across threads)
//   - numThreads: Number of threads to use
// Returns:
//   - Vector of averaged players across all threads (caller must delete)
//   - Returns empty vector on error
std::vector<Player*> runParallelSimulation(const BlackjackRules& rules, 
                                               const std::vector<Player*>& players, 
                                               uint64_t numRounds, 
                                               int numThreads);

bool runSimulation(const DoubleDownMadnessRules& rules,
                   std::vector<Player*>& players,
                   uint64_t numRounds);

std::vector<Player*> runParallelSimulation(
    const DoubleDownMadnessRules& rules,
    const std::vector<Player*>& players,
    uint64_t numRounds,
    int numThreads);

bool runSimulation(const SpanishRules& rules,
                   std::vector<Player*>& players,
                   uint64_t numRounds);

std::vector<Player*> runParallelSimulation(
    const SpanishRules& rules,
    const std::vector<Player*>& players,
    uint64_t numRounds,
    int numThreads);

std::string commandLineFromArgs(int argc, char** argv);

// Single-threaded, verbose variants: same round loop as runSimulation, but each
// hand decision and outcome is printed to `out` (see Table::setVerbose). Intended
// for the DebugPlayerBehavior app — no metrics/training semantics beyond what
// the ordinary runSimulation already does via the players passed in.
bool runSimulationVerbose(const BlackjackRules& rules, std::vector<Player*>& players,
                          uint64_t numRounds, std::ostream& out);
bool runSimulationVerbose(const DoubleDownMadnessRules& rules, std::vector<Player*>& players,
                          uint64_t numRounds, std::ostream& out);
bool runSimulationVerbose(const SpanishRules& rules, std::vector<Player*>& players,
                          uint64_t numRounds, std::ostream& out);
