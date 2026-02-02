/**
 * Benchmark: K*Q Scalar Multiplication with Lazy Reduction
 * 
 * Measures real-world performance of scalar multiplication with lazy reduction
 * enabled in jacobian_add_mixed (hot path for wNAF).
 */

#include <iostream>
#include <chrono>
#include <random>
#include <cstring>
#include <vector>
#include <iomanip>
#include "secp256k1/fast.hpp"

using namespace secp256k1::fast;

// Generate random scalars
std::vector<Scalar> generate_scalars(size_t count) {
    std::vector<Scalar> result;
    result.reserve(count);
    
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    
    for (size_t i = 0; i < count; ++i) {
        std::array<uint8_t, 32> bytes;
        for (size_t j = 0; j < 32; j += 8) {
            uint64_t val = dist(gen);
            std::memcpy(&bytes[j], &val, std::min(size_t(8), 32 - j));
        }
        result.push_back(Scalar::from_bytes(bytes));
    }
    
    return result;
}

// Benchmark K*G (generator multiplication)
double bench_k_times_generator(const std::vector<Scalar>& scalars, size_t iterations) {
    Point G = Point::generator();
    Point result = G;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        result = G.scalar_mul(scalars[i % scalars.size()]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile auto prevent_opt = result.x_raw().limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

// Benchmark K*Q (arbitrary point multiplication)
double bench_k_times_point(const std::vector<Scalar>& scalars, size_t iterations) {
    Point G = Point::generator();
    Point Q = G.scalar_mul(Scalar::from_uint64(12345)); // Arbitrary point
    Point result = Q;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        result = Q.scalar_mul(scalars[i % scalars.size()]);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile auto prevent_opt = result.x_raw().limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

// Benchmark point addition (used in wNAF loop)
double bench_point_add(size_t iterations) {
    Point G = Point::generator();
    Point P = G.scalar_mul(Scalar::from_uint64(7));
    Point Q = G.scalar_mul(Scalar::from_uint64(11));
    Point result = P;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        result = result.add(Q);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile auto prevent_opt = result.x_raw().limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

// Benchmark point doubling
double bench_point_double(size_t iterations) {
    Point G = Point::generator();
    Point P = G.scalar_mul(Scalar::from_uint64(7));
    Point result = P;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        result = result.dbl();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile auto prevent_opt = result.x_raw().limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

void print_header() {
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Scalar Multiplication Benchmark (Lazy Reduction)  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";
}

void print_result(const std::string& name, double ns_per_op) {
    std::cout << std::left << std::setw(30) << name << ": ";
    std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2);
    std::cout << ns_per_op << " ns";
    
    // Convert to microseconds if large
    if (ns_per_op > 1000.0) {
        std::cout << " (" << std::fixed << std::setprecision(2) << (ns_per_op / 1000.0) << " μs)";
    }
    std::cout << "\n";
}

int main() {
    // Validate arithmetic correctness before benchmarking
    std::cout << "Running arithmetic validation...\n";
    secp256k1::fast::Selftest(true);
    std::cout << "\n";
    
    SECP256K1_INIT();  // Run integrity check
    
    print_header();
    
    constexpr size_t NUM_SCALARS = 20;
    constexpr size_t SCALAR_MUL_ITERATIONS = 1000;
    constexpr size_t POINT_OP_ITERATIONS = 100000;
    constexpr size_t WARMUP = 500;  // Increased for Turbo Boost activation
    
    std::cout << "Generating " << NUM_SCALARS << " random scalars...\n";
    auto scalars = generate_scalars(NUM_SCALARS);
    
    std::cout << "Warmup...\n";
    bench_k_times_point(scalars, WARMUP);
    bench_point_add(WARMUP);
    
    std::cout << "\nRunning benchmarks...\n\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    
    // Point operations (building blocks)
    double add_time = bench_point_add(POINT_OP_ITERATIONS);
    print_result("Point Addition", add_time);
    
    double dbl_time = bench_point_double(POINT_OP_ITERATIONS);
    print_result("Point Doubling", dbl_time);
    
    std::cout << "\n";
    
    // Scalar multiplication (main operations)
    double kg_time = bench_k_times_generator(scalars, SCALAR_MUL_ITERATIONS);
    print_result("K*G (Generator)", kg_time);
    
    double kq_time = bench_k_times_point(scalars, SCALAR_MUL_ITERATIONS);
    print_result("K*Q (Arbitrary Point)", kq_time);
    
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    
    // Analysis
    std::cout << "Analysis:\n";
    
    // Estimate operations per K*Q
    // wNAF w=4: ~64 point operations (43 additions + 21 doublings on average)
    double estimated_ops = (43.0 * add_time + 21.0 * dbl_time);
    std::cout << "  • Estimated from point ops: " << std::fixed << std::setprecision(2);
    std::cout << (estimated_ops / 1000.0) << " μs\n";
    
    std::cout << "  • Actual K*Q: " << std::fixed << std::setprecision(2);
    std::cout << (kq_time / 1000.0) << " μs\n";
    
    double overhead = ((kq_time - estimated_ops) / kq_time) * 100.0;
    std::cout << "  • Overhead (wNAF, precompute): " << std::fixed << std::setprecision(1);
    std::cout << overhead << "%\n";
    
    std::cout << "\n";
    std::cout << "Note: This benchmark includes lazy reduction in jacobian_add_mixed.\n";
    std::cout << "      Compare with previous results to measure improvement.\n";
    std::cout << "\nPrevious baseline (eager reduction):\n";
    std::cout << "  • K*Q: ~18.34 μs (under Turbo Boost)\n";
    std::cout << "  • Expected with lazy: ~15-16 μs (12-18% improvement)\n";
    
    return 0;
}
