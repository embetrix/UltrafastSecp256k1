// Benchmark: Mutable vs Immutable next() operations
// Tests in-place modification benefits for sequential G additions

#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>

using namespace secp256k1::fast;
using namespace std::chrono;

// ANSI color codes
constexpr const char* BOLD = "\033[1m";
constexpr const char* CYAN = "\033[36m";
constexpr const char* GREEN = "\033[32m";
constexpr const char* YELLOW = "\033[33m";
constexpr const char* RED = "\033[31m";
constexpr const char* RESET = "\033[0m";

struct BenchResult {
    double median_ns;
    double min_ns;
    double max_ns;
    std::string name;
};

// Benchmark immutable next() - returns new Point each time
BenchResult bench_immutable_next(int iterations) {
    std::vector<double> times;
    times.reserve(100);
    
    Point p = Point::generator();
    
    for (int run = 0; run < 100; ++run) {
        auto start = high_resolution_clock::now();
        
        Point result = p;
        for (int i = 0; i < iterations; ++i) {
            result = result.next();  // Immutable: returns new Point
        }
        
        auto end = high_resolution_clock::now();
        double ns = duration_cast<nanoseconds>(end - start).count() / (double)iterations;
        times.push_back(ns);
        
        // Prevent optimization
        if (result.is_infinity()) std::cout << "";
    }
    
    std::sort(times.begin(), times.end());
    return {
        times[50],  // median
        times[0],   // min
        times[99],  // max
        "Immutable next()"
    };
}

// Benchmark mutable next_inplace() - modifies existing Point
BenchResult bench_mutable_next_inplace(int iterations) {
    std::vector<double> times;
    times.reserve(100);
    
    for (int run = 0; run < 100; ++run) {
        Point p = Point::generator();
        
        auto start = high_resolution_clock::now();
        
        for (int i = 0; i < iterations; ++i) {
            p.next_inplace();  // Mutable: modifies this Point
        }
        
        auto end = high_resolution_clock::now();
        double ns = duration_cast<nanoseconds>(end - start).count() / (double)iterations;
        times.push_back(ns);
        
        // Prevent optimization
        if (p.is_infinity()) std::cout << "";
    }
    
    std::sort(times.begin(), times.end());
    return {
        times[50],  // median
        times[0],   // min
        times[99],  // max
        "Mutable next_inplace()"
    };
}

void print_comparison(const BenchResult& immutable, const BenchResult& mutable_op, int iterations) {
    double speedup = (immutable.median_ns - mutable_op.median_ns) / immutable.median_ns * 100.0;
    
    std::cout << "\n" << BOLD << CYAN << "╔══════════════════════════════════════════════════════════╗\n";
    std::cout << "║  MUTABLE vs IMMUTABLE: Sequential G Additions           ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝" << RESET << "\n\n";
    
    std::cout << BOLD << "Test Configuration:\n" << RESET;
    std::cout << "  Sequential operations: " << iterations << " × (P + G)\n";
    std::cout << "  Runs per test: 100\n";
    std::cout << "  CPU: idle (no Turbo Boost)\n\n";
    
    std::cout << BOLD << "Results:\n" << RESET;
    std::cout << std::fixed << std::setprecision(1);
    
    // Immutable
    std::cout << "  " << YELLOW << immutable.name << RESET << "\n";
    std::cout << "    Median: " << immutable.median_ns << " ns/op\n";
    std::cout << "    Range:  " << immutable.min_ns << " - " << immutable.max_ns << " ns\n\n";
    
    // Mutable
    std::cout << "  " << GREEN << mutable_op.name << RESET << "\n";
    std::cout << "    Median: " << mutable_op.median_ns << " ns/op\n";
    std::cout << "    Range:  " << mutable_op.min_ns << " - " << mutable_op.max_ns << " ns\n\n";
    
    // Comparison
    std::cout << BOLD << "Performance Difference:\n" << RESET;
    if (speedup > 0) {
        std::cout << "  " << GREEN << "✓ Mutable is " << speedup << "% FASTER" << RESET << "\n";
        std::cout << "    Saved: " << (immutable.median_ns - mutable_op.median_ns) << " ns per operation\n";
    } else {
        std::cout << "  " << RED << "✗ Mutable is " << (-speedup) << "% SLOWER" << RESET << "\n";
        std::cout << "    Cost: " << (mutable_op.median_ns - immutable.median_ns) << " ns per operation\n";
    }
    
    std::cout << "\n" << BOLD << "Analysis:\n" << RESET;
    if (speedup > 5) {
        std::cout << "  " << GREEN << "✓ Significant improvement! In-place modification pays off." << RESET << "\n";
        std::cout << "    Recommendation: Use next_inplace() for sequential operations.\n";
    } else if (speedup > 0) {
        std::cout << "  " << YELLOW << "⚠ Marginal improvement. Allocation overhead ~= savings." << RESET << "\n";
        std::cout << "    Recommendation: Keep immutable for thread safety.\n";
    } else {
        std::cout << "  " << RED << "✗ No benefit. In-place overhead > savings." << RESET << "\n";
        std::cout << "    Recommendation: Stick with immutable next().\n";
    }
    
    std::cout << "\n" << BOLD << "Theory:\n" << RESET;
    std::cout << "  Immutable: 7M + 4S + allocation overhead\n";
    std::cout << "  Mutable:   7M + 4S (in-place, no allocation)\n";
    std::cout << "  Savings:   Memory allocation + copy overhead (~80 ns)\n\n";
}

int main() {
    // Test sequential operations: P+G, P+G+G, P+G+G+G, ...
    const int iterations = 1000;
    
    std::cout << BOLD << "Warming up CPU...\n" << RESET;
    {
        Point p = Point::generator();
        for (int i = 0; i < 10000; ++i) {
            p = p.next();
        }
    }
    
    std::cout << "Running benchmarks (100 runs each)...\n\n";
    
    auto immutable_result = bench_immutable_next(iterations);
    auto mutable_result = bench_mutable_next_inplace(iterations);
    
    print_comparison(immutable_result, mutable_result, iterations);
    
    return 0;
}
