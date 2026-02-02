// Benchmark: K*G with GLV vs Non-GLV
// Measures performance difference between GLV-optimized and regular scalar multiplication

#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/precompute.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <random>

using namespace secp256k1::fast;

struct BenchResult {
    double median_ns;
    double min_ns;
    double max_ns;
};

// Generate random scalars
std::vector<Scalar> generate_random_scalars(int count) {
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    
    std::vector<Scalar> scalars;
    scalars.reserve(count);
    
    for (int i = 0; i < count; ++i) {
        std::array<uint8_t, 32> bytes;
        for (size_t j = 0; j < 32; j += 8) {
            uint64_t val = dist(gen);
            std::memcpy(&bytes[j], &val, 8);
        }
        scalars.push_back(Scalar::from_bytes(bytes));
    }
    
    return scalars;
}

// K*G using GLV (precomputed tables)
BenchResult benchmark_kg_glv(const std::vector<Scalar>& scalars, int iterations) {
    std::vector<double> times;
    times.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (const auto& k : scalars) {
            Point result = scalar_mul_generator(k);
            volatile auto ptr = &result;
            (void)ptr;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::nano>(end - start).count();
        times.push_back(elapsed / scalars.size());
    }
    
    std::sort(times.begin(), times.end());
    
    return BenchResult{
        times[times.size() / 2],
        times.front(),
        times.back()
    };
}

// K*G using regular scalar multiplication (no GLV)
BenchResult benchmark_kg_noglv(const std::vector<Scalar>& scalars, int iterations) {
    std::vector<double> times;
    times.reserve(iterations);
    
    Point G = Point::generator();
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (const auto& k : scalars) {
            Point result = G.scalar_mul(k);
            volatile auto ptr = &result;
            (void)ptr;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::nano>(end - start).count();
        times.push_back(elapsed / scalars.size());
    }
    
    std::sort(times.begin(), times.end());
    
    return BenchResult{
        times[times.size() / 2],
        times.front(),
        times.back()
    };
}

void print_box_top() {
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
}

void print_box_mid() {
    std::cout << "╠════════════════════════════════════════════════════════════╣\n";
}

void print_box_bottom() {
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
}

void print_centered(const std::string& text) {
    int padding = (60 - text.length()) / 2;
    std::cout << "║" << std::string(padding, ' ') << text 
              << std::string(60 - padding - text.length(), ' ') << "║\n";
}

int main() {
    std::cout << "\n";
    print_box_top();
    print_centered("K*G Benchmark: GLV vs Non-GLV");
    print_box_bottom();
    std::cout << "\n";
    
    const int NUM_SCALARS = 20;
    const int ITERATIONS = 100;
    
    std::cout << "Generating " << NUM_SCALARS << " random scalars...\n";
    auto scalars = generate_random_scalars(NUM_SCALARS);
    
    std::cout << "Warmup...\n";
    Point G = Point::generator();
    for (int i = 0; i < 10; ++i) {
        for (const auto& k : scalars) {
            auto r1 = scalar_mul_generator(k);
            auto r2 = G.scalar_mul(k);
        }
    }
    
    std::cout << "\nRunning benchmarks (" << ITERATIONS << " iterations)...\n\n";
    
    auto glv_result = benchmark_kg_glv(scalars, ITERATIONS);
    auto noglv_result = benchmark_kg_noglv(scalars, ITERATIONS);
    
    print_box_top();
    print_centered("Results");
    print_box_bottom();
    std::cout << "\n";
    
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "K*G with GLV (precomputed tables)\n";
    std::cout << "  Median:  " << std::setw(8) << glv_result.median_ns << " ns";
    if (glv_result.median_ns < 1000) {
        std::cout << "\n";
    } else if (glv_result.median_ns < 1000000) {
        std::cout << " (" << (glv_result.median_ns / 1000.0) << " μs)\n";
    } else {
        std::cout << " (" << (glv_result.median_ns / 1000000.0) << " ms)\n";
    }
    std::cout << "  Range:   " << std::setw(8) << glv_result.min_ns << " - " 
              << glv_result.max_ns << " ns\n\n";
    
    std::cout << "K*G without GLV (regular scalar_mul)\n";
    std::cout << "  Median:  " << std::setw(8) << noglv_result.median_ns << " ns";
    if (noglv_result.median_ns < 1000) {
        std::cout << "\n";
    } else if (noglv_result.median_ns < 1000000) {
        std::cout << " (" << (noglv_result.median_ns / 1000.0) << " μs)\n";
    } else {
        std::cout << " (" << (noglv_result.median_ns / 1000000.0) << " ms)\n";
    }
    std::cout << "  Range:   " << std::setw(8) << noglv_result.min_ns << " - " 
              << noglv_result.max_ns << " ns\n\n";
    
    print_box_top();
    print_centered("Performance Comparison");
    print_box_bottom();
    std::cout << "\n";
    
    double speedup = noglv_result.median_ns / glv_result.median_ns;
    double time_saved = noglv_result.median_ns - glv_result.median_ns;
    
    std::cout << "GLV Speedup:     " << speedup << "x faster\n";
    std::cout << "Time Saved:      " << time_saved << " ns";
    if (time_saved < 1000) {
        std::cout << "\n";
    } else if (time_saved < 1000000) {
        std::cout << " (" << (time_saved / 1000.0) << " μs)\n";
    } else {
        std::cout << " (" << (time_saved / 1000000.0) << " ms)\n";
    }
    
    std::cout << "\nAnalysis:\n";
    std::cout << "  • GLV optimization uses 2D precomputed tables\n";
    std::cout << "  • Reduces scalar multiplication from 256 steps to ~128\n";
    std::cout << "  • Trade-off: memory (precomputed points) for speed\n";
    std::cout << "  • Ideal for generator multiplication (K*G)\n\n";
    
    return 0;
}
