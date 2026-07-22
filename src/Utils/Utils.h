#pragma once
#include "../Game/BlackjackRules.h"
#include "../Game/Player.h"
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

std::string commandLineFromArgs(int argc, char** argv);
