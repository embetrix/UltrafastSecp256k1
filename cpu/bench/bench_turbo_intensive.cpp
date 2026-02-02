// Intensive benchmark to force Turbo Boost activation
// Runs continuous operations for 10 seconds to heat up CPU

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

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       Intensive Benchmark - Turbo Boost Activator         ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    
    std::cout << "Generating test data...\n";
    auto scalars = generate_random_scalars(100);
    Point Q = Point::generator();
    for (int i = 0; i < 50; ++i) {
        Q = Q.next();
    }
    
    std::cout << "\nPhase 1: Warming up CPU (5 seconds intense load)...\n";
    std::cout << "This will activate Turbo Boost if thermal headroom allows.\n\n";
    
    auto warmup_start = std::chrono::steady_clock::now();
    unsigned long long operations = 0;
    
    // Intense warmup for 5 seconds
    while (true) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - warmup_start).count();
        if (elapsed >= 5.0) break;
        
        for (int i = 0; i < 10; ++i) {
            volatile Point result = Q.scalar_mul(scalars[i % scalars.size()]);
            operations++;
        }
        
        if (operations % 100 == 0) {
            std::cout << "\r  Warmup: " << std::fixed << std::setprecision(1) 
                      << elapsed << "s / 5.0s  [" 
                      << std::string((int)(elapsed * 10), '=') 
                      << std::string(50 - (int)(elapsed * 10), ' ') << "]" << std::flush;
        }
    }
    
    std::cout << "\r  Warmup: 5.0s / 5.0s  [" << std::string(50, '=') << "] DONE!\n\n";
    std::cout << "Operations completed: " << operations << "\n";
    std::cout << "CPU should now be at Turbo Boost frequency!\n\n";
    
    // Now measure with hot CPU
    std::cout << "Phase 2: Benchmarking with Turbo Boost active...\n\n";
    
    std::vector<double> kg_times, kq_times, add_times, dbl_times;
    
    Point G = Point::generator();
    Point P1 = G;
    Point P2 = G.next();
    
    // Run quick measurements while CPU is hot
    const int QUICK_SAMPLES = 50;
    
    for (int i = 0; i < QUICK_SAMPLES; ++i) {
        // K*G
        auto start = std::chrono::high_resolution_clock::now();
        volatile Point r1 = scalar_mul_generator(scalars[i % scalars.size()]);
        auto end = std::chrono::high_resolution_clock::now();
        kg_times.push_back(std::chrono::duration<double, std::nano>(end - start).count());
        
        // K*Q
        start = std::chrono::high_resolution_clock::now();
        volatile Point r2 = Q.scalar_mul(scalars[i % scalars.size()]);
        end = std::chrono::high_resolution_clock::now();
        kq_times.push_back(std::chrono::duration<double, std::nano>(end - start).count());
        
        // Point Add
        start = std::chrono::high_resolution_clock::now();
        volatile Point r3 = P1.add(P2);
        end = std::chrono::high_resolution_clock::now();
        add_times.push_back(std::chrono::duration<double, std::nano>(end - start).count());
        
        // Point Double
        start = std::chrono::high_resolution_clock::now();
        volatile Point r4 = P1.dbl();
        end = std::chrono::high_resolution_clock::now();
        dbl_times.push_back(std::chrono::duration<double, std::nano>(end - start).count());
    }
    
    std::sort(kg_times.begin(), kg_times.end());
    std::sort(kq_times.begin(), kq_times.end());
    std::sort(add_times.begin(), add_times.end());
    std::sort(dbl_times.begin(), dbl_times.end());
    
    double kg_median = kg_times[kg_times.size() / 2];
    double kq_median = kq_times[kq_times.size() / 2];
    double add_median = add_times[add_times.size() / 2];
    double dbl_median = dbl_times[dbl_times.size() / 2];
    
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║              Results (Turbo Boost Active)                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "K*G (Generator):        " << std::setw(8) << kg_median << " ns";
    if (kg_median < 1000) {
        std::cout << "\n";
    } else if (kg_median < 1000000) {
        std::cout << " (" << (kg_median / 1000.0) << " μs)\n";
    }
    
    std::cout << "K*Q (Arbitrary Point):  " << std::setw(8) << kq_median << " ns";
    if (kq_median < 1000) {
        std::cout << "\n";
    } else if (kq_median < 1000000) {
        std::cout << " (" << (kq_median / 1000.0) << " μs)\n";
    }
    
    std::cout << "Point Addition:         " << std::setw(8) << add_median << " ns\n";
    std::cout << "Point Doubling:         " << std::setw(8) << dbl_median << " ns\n\n";
    
    std::cout << "Expected K*Q on Turbo Boost: ~18-20 μs\n";
    std::cout << "If result is ~150+ μs, CPU is still throttled.\n\n";
    
    return 0;
}
