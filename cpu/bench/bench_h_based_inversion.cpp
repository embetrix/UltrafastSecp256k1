// Benchmark H-based serial inversion vs standard Montgomery batch inversion
#include "secp256k1/fast.hpp"
#include "secp256k1/field_h_based.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

using namespace secp256k1::fast;
using namespace std::chrono;

// Generate random field elements for testing
std::vector<FieldElement> generate_random_field_elements(size_t count, uint64_t seed = 12345) {
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<uint64_t> dist;
    
    std::vector<FieldElement> elements;
    elements.reserve(count);
    
    for (size_t i = 0; i < count; i++) {
        // Generate non-zero field element
        FieldElement fe;
        do {
            fe = FieldElement::from_limbs({
                dist(rng), dist(rng), dist(rng), dist(rng)
            });
        } while (fe == FieldElement::zero());
        
        elements.push_back(fe);
    }
    
    return elements;
}

// Simulate Jacobian walk: compute H values (Z_next / Z_current ratios)
std::vector<FieldElement> simulate_jacobian_walk_h_values(const FieldElement& z0, size_t steps) {
    std::vector<FieldElement> h_values;
    h_values.reserve(steps);
    
    FieldElement z_current = z0;
    
    for (size_t i = 0; i < steps; i++) {
        // Simulate point addition: Z_next = Z_current * some_factor
        // For testing, use deterministic H values
        FieldElement h = FieldElement::from_uint64(2 + (i % 7));  // H ∈ {2, 3, 4, 5, 6, 7, 8}
        h_values.push_back(h);
        
        z_current *= h;
    }
    
    return h_values;
}

int main() {
    Selftest(true);  // ✅ Validate arithmetic
    
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  H-Based Serial Inversion vs Standard Montgomery Batch Inversion\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "\n";
    
    std::cout << "Testing batch inversion performance with different batch sizes...\n";
    std::cout << "GPU showed 7.75× speedup with H-based method. What about CPU?\n\n";
    
    // Test different batch sizes
    const std::vector<size_t> batch_sizes = {10, 32, 64, 128, 224, 256, 512, 1024};
    const int warmup_iters = 100;
    const int bench_iters = 1000;
    
    std::cout << std::left << std::setw(12) << "Batch Size"
              << std::setw(18) << "Montgomery (us)"
              << std::setw(18) << "H-Based (us)"
              << std::setw(15) << "Speedup"
              << std::setw(20) << "Winner\n";
    std::cout << std::string(80, '-') << "\n";
    
    for (size_t batch_size : batch_sizes) {
        // Generate test data: Z_0 and H values
        FieldElement z0 = FieldElement::from_uint64(0x123456789ABCDEF0ULL);
        std::vector<FieldElement> h_values_original = simulate_jacobian_walk_h_values(z0, batch_size);
        
        // Method 1: Standard Montgomery batch inversion (current baseline)
        // This is what CPU library currently uses: fe_batch_inverse
        std::vector<FieldElement> z_coords_montgomery;
        z_coords_montgomery.reserve(batch_size);
        
        // Build Z coordinates: Z_0, Z_0*H_0, Z_0*H_0*H_1, ...
        FieldElement z_current = z0;
        for (size_t i = 0; i < batch_size; i++) {
            z_coords_montgomery.push_back(z_current);
            z_current *= h_values_original[i];
        }
        
        // Warmup
        for (int iter = 0; iter < warmup_iters; iter++) {
            std::vector<FieldElement> z_copy = z_coords_montgomery;
            fe_batch_inverse(z_copy.data(), z_copy.size());
        }
        
        // Benchmark Montgomery batch inversion
        auto t0 = high_resolution_clock::now();
        for (int iter = 0; iter < bench_iters; iter++) {
            std::vector<FieldElement> z_copy = z_coords_montgomery;
            fe_batch_inverse(z_copy.data(), z_copy.size());
        }
        auto t1 = high_resolution_clock::now();
        double montgomery_us = duration_cast<nanoseconds>(t1 - t0).count() / (double)bench_iters / 1000.0;
        
        // Method 2: H-based serial inversion (GPU-inspired)
        std::vector<FieldElement> h_values_hbased = h_values_original;
        
        // Warmup
        for (int iter = 0; iter < warmup_iters; iter++) {
            std::vector<FieldElement> h_copy = h_values_original;
            fe_h_based_inversion(h_copy.data(), z0, h_copy.size());
        }
        
        // Benchmark H-based inversion
        t0 = high_resolution_clock::now();
        for (int iter = 0; iter < bench_iters; iter++) {
            std::vector<FieldElement> h_copy = h_values_original;
            fe_h_based_inversion(h_copy.data(), z0, h_copy.size());
        }
        t1 = high_resolution_clock::now();
        double hbased_us = duration_cast<nanoseconds>(t1 - t0).count() / (double)bench_iters / 1000.0;
        
        // Verify correctness: both methods should produce same Z^{-2} values
        std::vector<FieldElement> z_inv2_montgomery;
        std::vector<FieldElement> z_inv2_hbased;
        
        // Montgomery: invert Z coords, then square
        {
            std::vector<FieldElement> z_copy = z_coords_montgomery;
            fe_batch_inverse(z_copy.data(), z_copy.size());
            for (auto& z_inv : z_copy) {
                z_inv2_montgomery.push_back(z_inv.square());
            }
        }
        
        // H-based: produces Z^{-2} directly
        {
            std::vector<FieldElement> h_copy = h_values_original;
            fe_h_based_inversion(h_copy.data(), z0, h_copy.size());
            z_inv2_hbased = h_copy;  // H-based overwrites h_values with Z^{-2}
        }
        
        // Verify
        bool correct = true;
        for (size_t i = 0; i < batch_size; i++) {
            if (z_inv2_montgomery[i] != z_inv2_hbased[i]) {
                correct = false;
                break;
            }
        }
        
        // Results
        double speedup = montgomery_us / hbased_us;
        std::string winner = "[+] H-Based";
        if (speedup < 1.0) {
            winner = "[*] Montgomery";
            speedup = 1.0 / speedup;
        }
        
        if (!correct) {
            winner = "[!] MISMATCH!";
        }
        
        std::cout << std::left << std::setw(12) << batch_size
                  << std::setw(18) << std::fixed << std::setprecision(3) << montgomery_us
                  << std::setw(18) << hbased_us
                  << std::setw(15) << std::fixed << std::setprecision(2) << speedup << "×"
                  << std::setw(20) << winner << "\n";
    }
    
    std::cout << "\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "  Analysis\n";
    std::cout << "═══════════════════════════════════════════════════════════════════\n";
    std::cout << "\n";
    std::cout << "Montgomery Batch Inversion:\n";
    std::cout << "  • Cost: N multiplications + 1 inversion\n";
    std::cout << "  • Memory: O(N) temporary storage (prefix products)\n";
    std::cout << "  • Memory access: Random (prefix array)\n";
    std::cout << "  • Best for: Independent inversions\n\n";
    
    std::cout << "H-Based Serial Inversion:\n";
    std::cout << "  • Cost: 2N multiplications + 1 inversion + N squares\n";
    std::cout << "  • Memory: O(1) temporary storage\n";
    std::cout << "  • Memory access: Sequential (cache-friendly)\n";
    std::cout << "  • Best for: Fixed-step Jacobian walks\n";
    std::cout << "  • GPU speedup: 7.75× (measured on RTX 4090)\n\n";
    
    std::cout << "Why H-based can be faster despite more operations?\n";
    std::cout << "  1. Sequential memory access → better cache utilization\n";
    std::cout << "  2. No temporary array allocation\n";
    std::cout << "  3. Field multiplication (~18ns) << Field inversion (~350ns)\n";
    std::cout << "  4. Modern CPUs love predictable access patterns\n\n";
    
    std::cout << "Expected behavior:\n";
    std::cout << "  • Small batches (N < 64): Montgomery may win (less work)\n";
    std::cout << "  • Large batches (N > 128): H-based should win (cache matters)\n";
    std::cout << "  • GPU: H-based dominates at all sizes (memory bandwidth limited)\n";
    std::cout << "\n";
    
    return 0;
}
