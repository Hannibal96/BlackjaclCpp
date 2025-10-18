#include "Utils.h"
#include "../Game/BlackjackTable.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>
#include <vector>
#include <mutex>

// Global mutex for thread-safe console output
static std::mutex g_consoleMutex;

bool runSimulation(const BlackjackRules& rules, std::vector<Player*>& players, uint64_t numRounds, bool showProgress) {
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
            
            // Update progress every second or on completion (only if showProgress is true)
            if (showProgress) {
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto timeSinceUpdate = std::chrono::duration_cast<std::chrono::milliseconds>(
                    currentTime - lastUpdateTime).count();
                
                if (timeSinceUpdate >= 1000 || i == numRounds) {
                    auto totalElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        currentTime - startTime).count();
                    double secondsElapsed = totalElapsed / 1000.0;
                    
                    // Calculate hands per second
                    int handsSinceLastUpdate = i - lastHandCount;
                    double intervalSeconds = timeSinceUpdate / 1000.0;
                    double currentHandsPerSec = handsSinceLastUpdate / intervalSeconds;
                    double avgHandsPerSec = i / secondsElapsed;
                    
                    // Calculate progress percentage
                    double progress = (static_cast<double>(i) / numRounds) * 100.0;
                    
                    // Calculate ETA
                    double remainingHands = numRounds - i;
                    double etaSeconds = remainingHands / avgHandsPerSec;
                    int etaMinutes = static_cast<int>(etaSeconds / 60);
                    int etaSecs = static_cast<int>(etaSeconds) % 60;
                    
                    std::lock_guard<std::mutex> lock(g_consoleMutex);
                    std::cout << "\rHands: " << i << "/" << numRounds 
                              << " (" << std::fixed << std::setprecision(1) << progress << "%)"
                              << " | Speed: " << std::setprecision(0) << currentHandsPerSec << " hands/s"
                              << " | Avg: " << avgHandsPerSec << " hands/s"
                              << " | Elapsed: " << std::setprecision(1) << secondsElapsed << "s"
                              << " | ETA: " << etaMinutes << "m" << etaSecs << "s    ";
                    std::cout.flush();
                    
                    lastUpdateTime = currentTime;
                    lastHandCount = i;
                }
            }
        }
        
        if (showProgress) {
            std::lock_guard<std::mutex> lock(g_consoleMutex);
            std::cout << std::endl; // New line after progress completes
        }
        
        return true;
        
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lock(g_consoleMutex);
        std::cerr << "Error during simulation: " << e.what() << std::endl;
        return false;
    }
}


std::vector<std::vector<Player*>> runParallelSimulation(const BlackjackRules& rules, 
                                                        const std::vector<Player*>& players, 
                                                        uint64_t numRounds, 
                                                        int numThreads, 
                                                        bool showProgress) {
                                                            
    std::vector<std::vector<Player*>> resultPlayers;
    resultPlayers.resize(numThreads);
    
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
    
    // Launch threads with showProgress = false
    std::vector<std::future<bool>> futures;
    for (int t = 0; t < numThreads; ++t) {
        futures.push_back(std::async(
            std::launch::async,
            runSimulation,
            std::cref(rules),
            std::ref(threadData[t].players),
            threadData[t].rounds,
            showProgress  
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
    if(showProgress){
        std::cout.flush();
        std::cout << "All threads completed in " << std::fixed << std::setprecision(2) << elapsedSec << " seconds!\n";
        std::cout << "Average speed: " << std::setprecision(0) << (numRounds / elapsedSec) << " hands/sec\n\n";
    }
    
    for(size_t t = 0; t < numThreads ; ++t ){
        for (size_t p = 0; p < players.size(); ++p) {
            resultPlayers[t].push_back(threadData[t].players[p]->clone());
        }
    }
    
    // Clean up cloned players
    for (int t = 0; t < numThreads; ++t) {
        for (auto* player : threadData[t].players) {
            delete player;
        }
    }
    
    return resultPlayers;
}

