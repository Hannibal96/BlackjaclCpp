#include "Utils.h"
#include "../Game/BlackjackTable.h"
#include "../Game/DoubleDownMadnessTable.h"
#include "RL/QLearningStrategy.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>
#include <vector>
#include <mutex>
#include <sstream>
#include <cctype>

// Global mutex for thread-safe console output
static std::mutex g_consoleMutex;

static std::string shellQuote(const std::string& arg) {
    if (arg.empty()) return "''";

    bool needsQuotes = false;
    for (char c : arg) {
        if (!(std::isalnum(static_cast<unsigned char>(c)) ||
              c == '_' || c == '-' || c == '.' || c == '/' || c == ':' || c == '=')) {
            needsQuotes = true;
            break;
        }
    }
    if (!needsQuotes) return arg;

    std::string quoted = "'";
    for (char c : arg) {
        if (c == '\'') quoted += "'\\''";
        else quoted += c;
    }
    quoted += "'";
    return quoted;
}

std::string commandLineFromArgs(int argc, char** argv) {
    std::ostringstream os;
    for (int i = 0; i < argc; ++i) {
        if (i > 0) os << ' ';
        os << shellQuote(argv[i] ? argv[i] : "");
    }
    return os.str();
}

bool runSimulation(const BlackjackRules& rules, std::vector<Player*>& players, uint64_t numRounds) {
    if (players.empty()) {
        std::lock_guard<std::mutex> lock(g_consoleMutex);
        std::cerr << "Error: No players provided" << std::endl;
        return false;
    }
    
    if (numRounds <= 0) {
        std::lock_guard<std::mutex> lock(g_consoleMutex);
        std::cerr << "Error: Number of rounds must be positive" << std::endl;
        return false;
    }
    
    try {
        // Create the table with the provided rules and players
        BlackjackTable table(rules, players);
        
        // Time tracking for progress bar
        auto startTime = std::chrono::high_resolution_clock::now();
        auto lastUpdateTime = startTime;
        int lastHandCount = 0;
        
        // Run the simulation with optional progress bar
        for (int i = 1; i <= numRounds; ++i) {
            table.round();
            // std::cout << table << std::endl;
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(g_consoleMutex);
        std::cerr << "Error during simulation: " << e.what() << std::endl;
        return false;
    }
}


std::vector<Player*> runParallelSimulation(const BlackjackRules& rules, 
                                                        const std::vector<Player*>& players, 
                                                        uint64_t numRounds, 
                                                        int numThreads) {
                                                            
    std::vector<Player*> resultPlayers;
    
    if (players.empty()) {
        std::cerr << "Error: No players provided" << std::endl;
        return resultPlayers;
    }
    
    if (numRounds <= 0) {
        std::cerr << "Error: Number of rounds must be positive" << std::endl;
        return resultPlayers;
    }
    
    if (numThreads <= 0) {
        std::cerr << "Error: Number of threads must be positive" << std::endl;
        return resultPlayers;
    }
    
    // Calculate rounds per thread
    uint64_t roundsPerThread = numRounds / numThreads;
    int remainingRounds = numRounds % numThreads;
    
    // Create thread data
    struct ThreadData {
        std::vector<Player*> players;
        uint64_t rounds;
    };
    
    std::vector<ThreadData> threadData(numThreads);
    
    // Clone players for each thread
    for (int t = 0; t < numThreads; ++t) {
        for (const auto* player : players) {
            threadData[t].players.push_back(player->clone());
        }
        threadData[t].rounds = roundsPerThread + (t < remainingRounds ? 1 : 0);
    }
    
    // Launch threads
    std::vector<std::future<bool>> futures;
    for (int t = 0; t < numThreads; ++t) {
        futures.push_back(std::async(
            std::launch::async,
            static_cast<bool (*)(const BlackjackRules&,
                                 std::vector<Player*>&,
                                 uint64_t)>(runSimulation),
            std::cref(rules),
            std::ref(threadData[t].players),
            threadData[t].rounds
        ));
    }
    
    // Track start time
    auto startTime = std::chrono::high_resolution_clock::now();
    
    for (auto& future : futures) {
        future.get();
    }
    
    // Calculate elapsed time
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    double elapsedSec = elapsedMs / 1000.0;
    
    // Create averaged result players
    // Start by cloning players from first thread
    for (size_t p = 0; p < players.size(); ++p) {
        resultPlayers.push_back(threadData[0].players[p]->clone());
    }
    
    // Accumulate players from remaining threads
    for (size_t t = 1; t < numThreads; ++t) {
        for (size_t p = 0; p < players.size(); ++p) {
            *resultPlayers[p] += *threadData[t].players[p];
        }
    }
    
    // Divide by number of threads to get average
    double factor = 1.0 / numThreads;
    for (size_t p = 0; p < players.size(); ++p) {
        *resultPlayers[p] *= factor;
    }
    
    // Clean up cloned players
    for (int t = 0; t < numThreads; ++t) {
        for (auto* player : threadData[t].players) {
            delete player;
        }
    }
    
    return resultPlayers;
}

bool runSimulation(const DoubleDownMadnessRules& rules,
                   std::vector<Player*>& players,
                   uint64_t numRounds) {
    if (players.empty() || numRounds == 0) return false;
    try {
        DoubleDownMadnessTable table(rules, players);
        for (uint64_t i = 0; i < numRounds; ++i)
            table.round();
        return true;
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(g_consoleMutex);
        std::cerr << "Error during Double Down Madness simulation: "
                  << e.what() << std::endl;
        return false;
    }
}

std::vector<Player*> runParallelSimulation(
        const DoubleDownMadnessRules& rules,
        const std::vector<Player*>& players,
        uint64_t numRounds,
        int numThreads) {
    std::vector<Player*> resultPlayers;
    if (players.empty() || numRounds == 0 || numThreads <= 0)
        return resultPlayers;

    struct ThreadData {
        std::vector<Player*> players;
        uint64_t rounds = 0;
    };
    std::vector<ThreadData> threadData(numThreads);
    const uint64_t roundsPerThread = numRounds / static_cast<uint64_t>(numThreads);
    const uint64_t remainingRounds = numRounds % static_cast<uint64_t>(numThreads);
    for (int t = 0; t < numThreads; ++t) {
        for (const auto* player : players)
            threadData[t].players.push_back(player->clone());
        threadData[t].rounds =
            roundsPerThread + (static_cast<uint64_t>(t) < remainingRounds ? 1 : 0);
    }

    std::vector<std::future<bool>> futures;
    futures.reserve(numThreads);
    for (int t = 0; t < numThreads; ++t) {
        futures.push_back(std::async(
            std::launch::async,
            static_cast<bool (*)(const DoubleDownMadnessRules&,
                                 std::vector<Player*>&,
                                 uint64_t)>(runSimulation),
            std::cref(rules),
            std::ref(threadData[t].players),
            threadData[t].rounds));
    }
    for (auto& future : futures)
        future.get();

    for (size_t p = 0; p < players.size(); ++p)
        resultPlayers.push_back(threadData[0].players[p]->clone());
    for (int t = 1; t < numThreads; ++t) {
        for (size_t p = 0; p < players.size(); ++p)
            *resultPlayers[p] += *threadData[t].players[p];
    }
    for (auto* player : resultPlayers)
        *player *= 1.0 / static_cast<double>(numThreads);

    for (auto& data : threadData) {
        for (auto* player : data.players)
            delete player;
    }
    return resultPlayers;
}
