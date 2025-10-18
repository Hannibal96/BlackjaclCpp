#include <gtest/gtest.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <future>
#include <vector>
#include <cmath>

// Simple CPU-bound work function
double doWork(int iterations) {
    double result = 0.0;
    for (int i = 0; i < iterations; ++i) {
        result += std::sqrt(i) * std::sin(i) * std::cos(i);
    }
    return result;
}

// Run work with specified number of threads
void runWithThreads(int numThreads, int totalIterations) {
    int iterationsPerThread = totalIterations / numThreads;
    
    std::vector<std::future<double>> futures;
    
    for (int t = 0; t < numThreads; ++t) {
        futures.push_back(std::async(std::launch::async, doWork, iterationsPerThread));
    }
    
    // Wait for all threads
    double total = 0.0;
    for (auto& future : futures) {
        total += future.get();
    }
}

class PerformanceBenchmark : public ::testing::Test {
protected:
    const int TOTAL_ITERATIONS = 1000000000;
};

TEST_F(PerformanceBenchmark, SimpleThreadScaling) {
    std::cout << "\n========================================" << std::endl;
    std::cout << "SIMPLE THREADING BENCHMARK" << std::endl;
    std::cout << "Total iterations: " << TOTAL_ITERATIONS << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Store times for comparison
    double time1Thread = 0;
    
    // Test with 1 thread (baseline)
    {
        std::cout << "\n--- Testing with 1 thread ---" << std::endl;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        runWithThreads(1, TOTAL_ITERATIONS);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        time1Thread = elapsedMs / 1000.0;
        
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << time1Thread << " seconds" << std::endl;
        std::cout << "  Speed: " << std::fixed << std::setprecision(0) << (TOTAL_ITERATIONS / time1Thread) << " iter/sec" << std::endl;
    }
    
    // Test with different thread counts
    std::vector<int> threadCounts = {2, 4, 8, 10, 16};
    
    for (int threads : threadCounts) {
        std::cout << "\n--- Testing with " << threads << " threads ---" << std::endl;
        
        auto startTime = std::chrono::high_resolution_clock::now();
        runWithThreads(threads, TOTAL_ITERATIONS);
        auto endTime = std::chrono::high_resolution_clock::now();
        
        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        double elapsedSec = elapsedMs / 1000.0;
        double speedup = time1Thread / elapsedSec;
        double efficiency = (speedup / threads) * 100.0;
        
        std::cout << "  Time: " << std::fixed << std::setprecision(3) << elapsedSec << " seconds" << std::endl;
        std::cout << "  Speed: " << std::fixed << std::setprecision(0) << (TOTAL_ITERATIONS / elapsedSec) << " iter/sec" << std::endl;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
        std::cout << "  Efficiency: " << std::fixed << std::setprecision(1) << efficiency << "%" << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "If efficiency >70%: Threading works fine, problem is in blackjack code" << std::endl;
    std::cout << "If efficiency <20%: System/threading issue" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

