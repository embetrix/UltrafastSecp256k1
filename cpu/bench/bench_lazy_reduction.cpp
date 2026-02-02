/**
 * Benchmark: Lazy Reduction vs Eager Reduction
 * 
 * Measures the performance impact of lazy reduction in field operations.
 * Compares:
 * - Eager: reduce after every operation
 * - Lazy: defer reduction until final result
 */

#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include <random>
#include "secp256k1/field.hpp"

using namespace secp256k1::fast;

// Generate random field elements
std::vector<FieldElement> generate_fields(size_t count) {
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
        result.push_back(FieldElement::from_limbs(limbs));
    }
    
    return result;
}

// Benchmark: Chain of operations with EAGER reduction
// Simulates: (a*b) + (c*d) + (e*f) with immediate reductions
double bench_eager_chain(const std::vector<FieldElement>& elems, size_t iterations) {
    FieldElement result = elems[0];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        size_t idx = i % (elems.size() - 5);
        
        // Each operation reduces immediately
        FieldElement ab = elems[idx] * elems[idx + 1];      // mul + reduce
        FieldElement cd = elems[idx + 2] * elems[idx + 3];  // mul + reduce
        FieldElement ef = elems[idx + 4] * elems[idx + 5];  // mul + reduce
        
        result = ab + cd;                                    // add + reduce
        result = result + ef;                                // add + reduce
        
        // Total: 3 mul_reduce + 2 add_reduce = 5 reductions
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

// Benchmark: Chain of operations with LAZY reduction
// Defers reduction until final result
double bench_lazy_chain(const std::vector<FieldElement>& elems, size_t iterations) {
    FieldElement result = elems[0];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        size_t idx = i % (elems.size() - 5);
        
        // Lazy operations: no intermediate reductions
        auto ab_lazy = elems[idx].mul_lazy(elems[idx + 1]);     // mul, no reduce
        auto cd_lazy = elems[idx + 2].mul_lazy(elems[idx + 3]); // mul, no reduce
        auto ef_lazy = elems[idx + 4].mul_lazy(elems[idx + 5]); // mul, no reduce
        
        // Add lazy results
        auto sum1 = ab_lazy.add_lazy(cd_lazy.reduce());          // add, no reduce
        auto sum2 = sum1.add_lazy(ef_lazy.reduce());             // add, no reduce
        
        result = sum2.reduce();                                  // reduce once at end
        
        // Total: 1 final reduction vs 5 reductions in eager
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

// Benchmark: Single multiply (baseline)
double bench_single_mul(const std::vector<FieldElement>& elems, size_t iterations) {
    FieldElement result = elems[0];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < iterations; ++i) {
        size_t idx = i % (elems.size() - 1);
        result = elems[idx] * elems[idx + 1];
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    
    volatile auto prevent_opt = result.limbs()[0];
    (void)prevent_opt;
    
    return static_cast<double>(duration.count()) / iterations;
}

void print_header() {
    std::cout << "╔════════════════════════════════════════════════════════╗\n";
    std::cout << "║     Lazy vs Eager Reduction Benchmark                 ║\n";
    std::cout << "╚════════════════════════════════════════════════════════╝\n\n";
}

void print_result(const std::string& name, double ns_per_op) {
    std::cout << std::left << std::setw(30) << name << ": ";
    std::cout << std::right << std::setw(8) << std::fixed << std::setprecision(2);
    std::cout << ns_per_op << " ns/op\n";
}

int main() {
    print_header();
    
    constexpr size_t NUM_ELEMENTS = 100;
    constexpr size_t ITERATIONS = 50000;
    constexpr size_t WARMUP = 5000;
    
    std::cout << "Generating " << NUM_ELEMENTS << " random field elements...\n";
    auto elements = generate_fields(NUM_ELEMENTS);
    
    std::cout << "Warmup " << WARMUP << " iterations...\n";
    bench_eager_chain(elements, WARMUP);
    bench_lazy_chain(elements, WARMUP);
    
    std::cout << "\nRunning benchmarks (" << ITERATIONS << " iterations each)...\n\n";
    std::cout << "═══════════════════════════════════════════════════════\n";
    
    // Baseline: single multiply
    double single_mul_time = bench_single_mul(elements, ITERATIONS);
    print_result("Single Multiply (baseline)", single_mul_time);
    
    std::cout << "\n";
    
    // Chain: 3 muls + 2 adds with eager reduction
    double eager_time = bench_eager_chain(elements, ITERATIONS);
    print_result("Eager (5 reductions)", eager_time);
    
    // Chain: 3 muls + 2 adds with lazy reduction
    double lazy_time = bench_lazy_chain(elements, ITERATIONS);
    print_result("Lazy (1 reduction)", lazy_time);
    
    std::cout << "═══════════════════════════════════════════════════════\n\n";
    
    // Analysis
    std::cout << "Analysis:\n";
    
    double speedup = ((eager_time - lazy_time) / eager_time) * 100.0;
    std::cout << "  • Lazy Reduction Speedup: ";
    std::cout << std::fixed << std::setprecision(1) << speedup << "%\n";
    
    double reduction_cost = (eager_time - 3 * single_mul_time) / 5.0;
    std::cout << "  • Estimated reduction cost: ";
    std::cout << std::fixed << std::setprecision(2) << reduction_cost << " ns\n";
    
    double saved_reductions = eager_time - lazy_time;
    std::cout << "  • Time saved (4 reductions): ";
    std::cout << std::fixed << std::setprecision(2) << saved_reductions << " ns\n";
    
    std::cout << "\nConclusion:\n";
    if (speedup > 5.0) {
        std::cout << "  ✓ Lazy reduction provides significant benefit!\n";
        std::cout << "    Consider enabling in hot paths (jacobian_add_mixed)\n";
    } else if (speedup > 0.0) {
        std::cout << "  ⚠ Lazy reduction provides marginal benefit\n";
        std::cout << "    Overhead may outweigh savings in practice\n";
    } else {
        std::cout << "  ✗ Lazy reduction slower than eager\n";
        std::cout << "    Overhead exceeds savings - keep eager reduction\n";
    }
    
    return 0;
}
