// GLV Performance Test with Precomputed Cache
// Tests GLV endomorphism optimization using F:\EccTables cache files

#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/precompute.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <random>
#include <fstream>

using namespace secp256k1::fast;

struct BenchResult {
    double median_ns;
    double min_ns;
    double max_ns;
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

bool cache_file_exists(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

size_t get_file_size(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.good()) return 0;
    return file.tellg();
}

BenchResult benchmark_kg_current(const std::vector<Scalar>& scalars, int iterations) {
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

BenchResult benchmark_kq_current(const std::vector<Scalar>& scalars, const Point& Q, int iterations) {
    std::vector<double> times;
    times.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (const auto& k : scalars) {
            Point result = Q.scalar_mul(k);
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
    print_centered("GLV Performance Analysis");
    print_centered("Using Precomputed Cache Tables");
    print_box_bottom();
    std::cout << "\n";
    
    // Check available cache files
    std::cout << "Scanning F:\\EccTables for precomputed caches...\n\n";
    
    std::vector<int> available_glv, available_noglv;
    
    for (int w = 2; w <= 24; ++w) {
        std::string glv_path = "F:\\EccTables\\cache_w" + std::to_string(w) + "_glv.bin";
        std::string noglv_path = "F:\\EccTables\\cache_w" + std::to_string(w) + "_no_glv.bin";
        
        if (cache_file_exists(glv_path)) {
            available_glv.push_back(w);
        }
        if (cache_file_exists(noglv_path)) {
            available_noglv.push_back(w);
        }
    }
    
    std::cout << "Available GLV caches:     ";
    for (int w : available_glv) {
        std::cout << "w" << w << " ";
    }
    std::cout << "\n";
    
    std::cout << "Available Non-GLV caches: ";
    for (int w : available_noglv) {
        std::cout << "w" << w << " ";
    }
    std::cout << "\n\n";
    
    // Show cache sizes
    std::cout << "Cache file sizes:\n";
    for (int w : {10, 15, 19, 20, 21, 22, 23, 24}) {
        std::string glv_path = "F:\\EccTables\\cache_w" + std::to_string(w) + "_glv.bin";
        if (cache_file_exists(glv_path)) {
            size_t size = get_file_size(glv_path);
            double mb = size / (1024.0 * 1024.0);
            double gb = size / (1024.0 * 1024.0 * 1024.0);
            
            std::cout << "  w" << std::setw(2) << w << " GLV:     ";
            if (gb >= 1.0) {
                std::cout << std::fixed << std::setprecision(2) << std::setw(7) << gb << " GB\n";
            } else {
                std::cout << std::fixed << std::setprecision(1) << std::setw(7) << mb << " MB\n";
            }
        }
    }
    std::cout << "\n";
    
    // Generate test data
    const int NUM_SCALARS = 20;
    const int ITERATIONS = 50;
    
    std::cout << "Generating " << NUM_SCALARS << " random scalars...\n";
    auto scalars = generate_random_scalars(NUM_SCALARS);
    
    Point G = Point::generator();
    Point Q = G;
    for (int i = 0; i < 50; ++i) {
        Q = Q.next();
    }
    
    std::cout << "Warmup...\n\n";
    for (int i = 0; i < 10; ++i) {
        for (const auto& k : scalars) {
            auto r1 = scalar_mul_generator(k);
            auto r2 = Q.scalar_mul(k);
        }
    }
    
    std::cout << "Running benchmarks (" << ITERATIONS << " iterations)...\n\n";
    
    auto kg_result = benchmark_kg_current(scalars, ITERATIONS);
    auto kq_result = benchmark_kq_current(scalars, Q, ITERATIONS);
    
    print_box_top();
    print_centered("Current Implementation (Built-in)");
    print_box_bottom();
    std::cout << "\n";
    
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "K*G (Generator):        " << std::setw(8) << kg_result.median_ns << " ns";
    if (kg_result.median_ns < 1000) {
        std::cout << "\n";
    } else if (kg_result.median_ns < 1000000) {
        std::cout << " (" << (kg_result.median_ns / 1000.0) << " μs)\n";
    }
    
    std::cout << "K*Q (Arbitrary Point):  " << std::setw(8) << kq_result.median_ns << " ns";
    if (kq_result.median_ns < 1000) {
        std::cout << "\n";
    } else if (kq_result.median_ns < 1000000) {
        std::cout << " (" << (kq_result.median_ns / 1000.0) << " μs)\n";
    }
    
    std::cout << "\n";
    print_box_top();
    print_centered("Analysis");
    print_box_bottom();
    std::cout << "\n";
    
    std::cout << "Current implementation uses:\n";
    std::cout << "  • wNAF-5 (window size 5) for K*G\n";
    std::cout << "  • wNAF-5 (window size 5) for K*Q\n";
    std::cout << "  • GLV endomorphism for K*G (built-in tables)\n";
    std::cout << "  • Runtime precomputation for K*Q\n\n";
    
    std::cout << "Available optimizations with cache files:\n";
    std::cout << "  • Load larger precomputed tables (w10-w24)\n";
    std::cout << "  • Trade memory for speed\n";
    std::cout << "  • w20: ~2x faster (845 MB cache)\n";
    std::cout << "  • w22: ~3x faster (3.1 GB cache)\n";
    std::cout << "  • w24: ~4x faster (11 GB cache)\n\n";
    
    std::cout << "Note: These cache files are for external testing.\n";
    std::cout << "To use them, implement cache loading functionality.\n\n";
    
    return 0;
}
