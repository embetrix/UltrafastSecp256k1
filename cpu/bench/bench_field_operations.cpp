/**
 * Benchmark for FieldElement operations
 * 
 * Measures performance of:
 * - Multiplication (mul)
 * - Squaring (square)
 * - Addition (add)
 * - Normalization (normalize)
 * 
 * Tests both with and without BMI2 assembly to quantify speedup.
 */

#include <iostream>
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>
#include "secp256k1/field.hpp"

using namespace secp256k1::fast;

// Generate random FieldElements for testing
std::vector<FieldElement> generate_random_fields(size_t count) {
    std::vector<FieldElement> result;
    result.reserve(count);
    
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    
    for (size_t i = 0; i < count; ++i) {
        std::array<uint64_t, 4> limbs;
        for (auto& limb : limbs) {
            limb = dist(gen);
        }
        // Reduce to valid field element
        result.push_back(FieldElement::from_limbs(limbs));
    }
    
    return result;
}

// Benchmark multiply operation
double bench_multiply(const std::vector<FieldElement>& elements, size_t iterations) {
    FieldElement result = elements[0];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        size_t idx = i % elements.size();
        result = result * elements[idx];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

// Benchmark square operation
double bench_square(const std::vector<FieldElement>& elements, size_t iterations) {
    FieldElement result = elements[0];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        result = result.square();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

// Benchmark addition operation
double bench_add(const std::vector<FieldElement>& elements, size_t iterations) {
    FieldElement result = elements[0];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        size_t idx = i % elements.size();
        result = result + elements[idx];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

// Benchmark subtraction operation
double bench_sub(const std::vector<FieldElement>& elements, size_t iterations) {
    FieldElement result = elements[0];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        size_t idx = i % elements.size();
        result = result - elements[idx];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    // Prevent optimization
    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

// Benchmark inverse operation
double bench_inverse(const std::vector<FieldElement>& elements, size_t iterations) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        size_t idx = i % elements.size();
        volatile auto result = elements[idx].inverse();
        (void)result;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    return static_cast<double>(duration.count()) / iterations;
}

void print_header() {
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║     FieldElement Operations Benchmark                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";
}

void print_result(const std::string& operation, double ns_per_op) {
    std::cout << std::left << std::setw(20) << operation << ": ";
    std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2) << ns_per_op;
    std::cout << " ns/op\n";
}

int main() {
    print_header();
    
    // Configuration
    constexpr size_t NUM_ELEMENTS = 100;
    constexpr size_t ITERATIONS = 100000;
    constexpr size_t WARMUP = 10000;
    
    std::cout << "Generating " << NUM_ELEMENTS << " random field elements...\n";
    auto elements = generate_random_fields(NUM_ELEMENTS);
    
    std::cout << "Warmup " << WARMUP << " iterations...\n";
    bench_multiply(elements, WARMUP);
    bench_square(elements, WARMUP);
    
    std::cout << "\nRunning benchmarks (" << ITERATIONS << " iterations each)...\n\n";
    
    // Benchmark each operation
    std::cout << "═══════════════════════════════════════════════════════\n";
    
    double mul_time = bench_multiply(elements, ITERATIONS);
    print_result("Multiply", mul_time);
    
    double sqr_time = bench_square(elements, ITERATIONS);
    print_result("Square", sqr_time);
    
    double add_time = bench_add(elements, ITERATIONS);
    print_result("Add", add_time);
    
    double sub_time = bench_sub(elements, ITERATIONS);
    print_result("Subtract", sub_time);
    
    double inv_time = bench_inverse(elements, 1000);  // Inverse is slow, use fewer iterations
    print_result("Inverse", inv_time);
    
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    
    // Analysis
    std::cout << "Analysis:\n";
    std::cout << "  • Square vs Multiply: ";
    double sqr_ratio = (mul_time - sqr_time) / mul_time * 100.0;
    std::cout << std::fixed << std::setprecision(1) << sqr_ratio << "% faster\n";
    
    std::cout << "  • Add cost: ";
    double add_ratio = add_time / mul_time * 100.0;
    std::cout << std::fixed << std::setprecision(1) << add_ratio << "% of multiply\n";
    
    std::cout << "  • Inverse cost: ";
    double inv_ratio = inv_time / mul_time;
    std::cout << std::fixed << std::setprecision(1) << inv_ratio << "× multiply\n";
    
    std::cout << "\nNote: This benchmark runs with BMI2 assembly if CPU supports it.\n";
    std::cout << "      Check field.cpp mul_impl/square_impl for runtime detection.\n";
    
    return 0;
}
