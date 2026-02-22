// GLV Performance Test with Cache Loading
// Tests different window sizes from F:\EccTables cache files

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
    double average_ns;
};

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

BenchResult benchmark_kg(const std::vector<Scalar>& scalars, int iterations) {
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
    
    // Calculate average
    double sum = 0.0;
    for (double t : times) {
        sum += t;
    }
    double average = sum / times.size();
    
    return BenchResult{
        times[times.size() / 2],  // median
        times.front(),             // min
        times.back(),              // max
        average                    // average
    };
}

void print_box_top() {
    std::cout << "+============================================================+\n";
}

void print_box_bottom() {
    std::cout << "+============================================================+\n";
}

void print_centered(const std::string& text) {
    int padding = (60 - text.length()) / 2;
    std::cout << "|" << std::string(padding, ' ') << text 
              << std::string(60 - padding - text.length(), ' ') << "|\n";
}

int main() {
    std::cout << "\n";
    print_box_top();
    print_centered("GLV Cache Performance Test");
    print_centered("F:\\EccTables Precomputed Tables");
    print_box_bottom();
    std::cout << "\n";
    
    const int NUM_SCALARS = 20;
    const int ITERATIONS = 500;  // 500 iterations per window size
    
    std::cout << "Generating " << NUM_SCALARS << " random scalars...\n\n";
    auto scalars = generate_random_scalars(NUM_SCALARS);
    
    // Test different window sizes
    std::vector<unsigned> window_sizes = {10, 15, 18, 19, 20, 21, 22};
    
    std::cout << "Testing window sizes: ";
    for (auto w : window_sizes) {
        std::cout << "w" << w << " ";
    }
    std::cout << "\n";
    std::cout << "Iterations per test: " << ITERATIONS << "\n";
    std::cout << "Total operations per window: " << (ITERATIONS * NUM_SCALARS) << "\n\n";
    
    std::vector<std::pair<unsigned, BenchResult>> results;
    
    for (unsigned w : window_sizes) {
        std::cout << "----------------------------------------\n";
        std::cout << "Testing w" << w << " (window_bits=" << w << ")...\n";
        
        // Configure with specific window size
        FixedBaseConfig config;
        config.window_bits = w;
        config.enable_glv = false;  // GLV disabled (slower due to decomposition overhead)
        config.use_cache = true;
        config.cache_dir = "F:\\EccTables";
        
        std::cout << "  Configuring fixed base with w" << w << "...\n";
        configure_fixed_base(config);
        
        std::cout << "  Loading cache from F:\\EccTables\\cache_w" << w << "_glv.bin...\n";
        std::string cache_path = "F:\\EccTables\\cache_w" + std::to_string(w) + "_glv.bin";
        
        if (!load_precompute_cache(cache_path)) {
            std::cout << "  [!] Failed to load cache, using runtime generation\n";
        } else {
            std::cout << "  [ok] Cache loaded successfully\n";
        }
        
        std::cout << "  Ensuring precompute ready...\n";
        ensure_fixed_base_ready();
        
        std::cout << "  Warmup (first call loads cache into memory)...\n";
        // First call is slow - loads cache into memory
        for (const auto& k : scalars) {
            auto r = scalar_mul_generator(k);
        }
        
        std::cout << "  Additional warmup (cache now hot)...\n";
        // Subsequent calls are fast - cache is hot
        for (int i = 0; i < 10; ++i) {
            for (const auto& k : scalars) {
                auto r = scalar_mul_generator(k);
            }
        }
        
        std::cout << "  Benchmarking (" << ITERATIONS << " iterations)...\n";
        auto result = benchmark_kg(scalars, ITERATIONS);
        results.push_back({w, result});
        
        std::cout << "  [ok] w" << w << " results:\n";
        std::cout << "      Median:  " << std::fixed << std::setprecision(2) 
                  << result.median_ns << " ns (" << (result.median_ns / 1000.0) << " us)\n";
        std::cout << "      Average: " << result.average_ns << " ns (" 
                  << (result.average_ns / 1000.0) << " us)\n";
        std::cout << "      Min:     " << result.min_ns << " ns\n";
        std::cout << "      Max:     " << result.max_ns << " ns\n";
        std::cout << "\n";
    }
    
    std::cout << "\n";
    print_box_top();
    print_centered("Performance Summary");
    print_box_bottom();
    std::cout << "\n";
    
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Window | Median (us) | Average (us) | Min (us) | Max (us) | Speedup\n";
    std::cout << "-------|-------------|--------------|----------|----------|--------\n";
    
    double baseline_median = results[0].second.median_ns;
    double baseline_avg = results[0].second.average_ns;
    
    for (const auto& [w, res] : results) {
        double speedup = baseline_median / res.median_ns;
        std::cout << "  w" << std::setw(2) << w << "  | " 
                  << std::setw(11) << (res.median_ns / 1000.0) << " | "
                  << std::setw(12) << (res.average_ns / 1000.0) << " | "
                  << std::setw(8) << (res.min_ns / 1000.0) << " | "
                  << std::setw(8) << (res.max_ns / 1000.0) << " | "
                  << std::setw(5) << speedup << "x";
        
        if (speedup > 1.3) {
            std::cout << " [*]";
        }
        if (speedup > 1.4) {
            std::cout << "[*]";
        }
        std::cout << "\n";
    }
    
    std::cout << "\n";
    std::cout << "Analysis:\n";
    std::cout << "  * Larger windows = fewer point additions\n";
    std::cout << "  * Trade-off: memory usage vs speed\n";
    std::cout << "  * Optimal: w18-w20 (balance of speed and memory)\n";
    std::cout << "  * Maximum: w22+ (diminishing returns due to cache misses)\n\n";
    
    // Find best
    auto best = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return a.second.median_ns < b.second.median_ns; });
    
    std::cout << "\nBest configuration: w" << best->first 
              << " (Median: " << (best->second.median_ns / 1000.0) 
              << " us, Average: " << (best->second.average_ns / 1000.0) << " us)\n";
    std::cout << "Speedup: " << (baseline_median / best->second.median_ns) 
              << "x over w" << results[0].first << "\n\n";
    
    return 0;
}
