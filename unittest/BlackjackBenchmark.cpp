#include <gtest/gtest.h>
#include "../src/Game/BlackjackTable.h"
#include "../src/Game/Player.h"
#include "../src/Game/RandomStrategy.h"
#include "../src/Game/BasicStrategy.h"
#include "../src/Game/BlackjackRules.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>
#include <vector>
#include <memory>

// Worker function - runs blackjack rounds in a thread with BasicStrategy
void runRounds(const BlackjackRules& rules, int numRounds, std::shared_ptr<BasicStrategy> sharedStrategy) {
    auto setupStart = std::chrono::high_resolution_clock::now();
    
    // Clone the shared strategy (uses shared_ptr internally for lookup table)
    auto strategy = sharedStrategy->clone();
    Player player(10000.0, std::move(strategy));
    std::vector<Player*> players = {&player};
    
    // Create table
    BlackjackTable table(rules, players);
    
    auto setupEnd = std::chrono::high_resolution_clock::now();
    auto setupMs = std::chrono::duration_cast<std::chrono::milliseconds>(setupEnd - setupStart).count();
    
    // Run rounds
    auto roundsStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < numRounds; ++i) {
        table.round();
    }
    auto roundsEnd = std::chrono::high_resolution_clock::now();
    auto roundsMs = std::chrono::duration_cast<std::chrono::milliseconds>(roundsEnd - roundsStart).count();
    
    // Print timing breakdown (will be interleaved but gives us info)
    std::cout << "[Thread " << std::this_thread::get_id() << "] Setup: " << setupMs 
              << "ms, Rounds: " << roundsMs << "ms" << std::endl;
}

// Run blackjack simulation with specified threads
void runBlackjackWithThreads(int numThreads, int totalRounds, std::shared_ptr<BasicStrategy> sharedStrategy) {
    BlackjackRules rules;
    int roundsPerThread = totalRounds / numThreads;
    
    std::vector<std::future<void>> futures;
    
    for (int t = 0; t < numThreads; ++t) {
        futures.push_back(std::async(std::launch::async, runRounds, std::cref(rules), roundsPerThread, sharedStrategy));
    }
    
    // Wait for all threads
    for (auto& future : futures) {
        future.get();
    }
}

class BlackjackBenchmark : public ::testing::Test {
protected:
    const int TOTAL_ROUNDS = 100000000;  // 10M rounds - good balance between accuracy and speed
};

TEST_F(BlackjackBenchmark, BlackjackThreadScaling) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "BLACKJACK THREADING BENCHMARK (BasicStrategy)" << std::endl;
    std::cout << "Total rounds: " << TOTAL_ROUNDS << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Load BasicStrategy once
    auto basicStrategy = std::make_shared<BasicStrategy>();
    std::cout << "Loading strategy from 'data base.json'..." << std::endl;
    bool loaded = basicStrategy->loadFromJson("decks=4_ss17=True_das=True_surr=2-10_peek=False");
    ASSERT_TRUE(loaded) << "Failed to load strategy file 'data base.json'";
    std::cout << "Strategy loaded successfully!\n" << std::endl;
    
    // Store times for comparison
    double time1Thread = 0;
    
    // Test with 1 thread (baseline)
    {
        std::cout << "\n--- Testing with 1 thread ---" << std::endl;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        runBlackjackWithThreads(1, TOTAL_ROUNDS, basicStrategy);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        time1Thread = elapsedMs / 1000.0;
        
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << time1Thread << " seconds" << std::endl;
        std::cout << "  Speed: " << std::fixed << std::setprecision(0) << (TOTAL_ROUNDS / time1Thread) << " hands/sec" << std::endl;
    }
    
    // Test with different thread counts
    std::vector<int> threadCounts = {2, 4, 8, 10, 16, (int)std::thread::hardware_concurrency()};
    
    for (int threads : threadCounts) {
        std::cout << "\n--- Testing with " << threads << " threads ---" << std::endl;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        runBlackjackWithThreads(threads, TOTAL_ROUNDS, basicStrategy);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        double elapsedSec = elapsedMs / 1000.0;
        double speedup = time1Thread / elapsedSec;
        double efficiency = (speedup / threads) * 100.0;
        
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << elapsedSec << " seconds" << std::endl;
        std::cout << "  Speed: " << std::fixed << std::setprecision(0) << (TOTAL_ROUNDS / elapsedSec) << " hands/sec" << std::endl;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "  Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "If efficiency >70%: Blackjack code scales well" << std::endl;
    std::cout << "If efficiency <30%: There's a bottleneck in blackjack code" << std::endl;
    std::cout << "Note: BasicStrategy uses shared lookup table (no copying!)" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

