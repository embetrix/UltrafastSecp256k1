// Benchmark for Point serialization operations
// Tests: toCompressed() and getX()

#include <secp256k1/point.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <iomanip>

using namespace secp256k1::fast;

struct BenchResult {
    double median_ns;
    double min_ns;
    double max_ns;
};

BenchResult benchmark(const std::vector<Point>& points, auto&& func, int iterations) {
    std::vector<double> times;
    times.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (const auto& p : points) {
            auto result = func(p);
            // Prevent optimization
            volatile auto ptr = result.data();
            (void)ptr;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::nano>(end - start).count();
        times.push_back(elapsed / points.size());
    }
    
    std::sort(times.begin(), times.end());
    
    return BenchResult{
        times[times.size() / 2],
        times.front(),
        times.back()
    };
}

BenchResult benchmark_getx(const std::vector<Point>& points, int iterations) {
    std::vector<double> times;
    times.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        
        for (const auto& p : points) {
            auto result = p.x();
            // Prevent optimization
            volatile auto ptr = &result;
            (void)ptr;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::nano>(end - start).count();
        times.push_back(elapsed / points.size());
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
    print_centered("Point Serialization Benchmark");
    print_box_bottom();
    std::cout << "\n";
    
    // Generate test points
    const int NUM_POINTS = 100;
    const int ITERATIONS = 100;
    
    std::cout << "Generating " << NUM_POINTS << " random points...\n";
    std::vector<Point> points;
    points.reserve(NUM_POINTS);
    
    Point p = Point::generator();
    for (int i = 0; i < NUM_POINTS; ++i) {
        points.push_back(p);
        p = p.next();
    }
    
    std::cout << "Warmup...\n";
    for (int i = 0; i < 10; ++i) {
        for (const auto& pt : points) {
            auto c = pt.to_compressed();
            auto x = pt.x();
        }
    }
    
    std::cout << "\nRunning benchmarks (" << ITERATIONS << " iterations)...\n\n";
    
    // Benchmark to_compressed
    auto compressed_result = benchmark(points, [](const Point& p) { 
        return p.to_compressed(); 
    }, ITERATIONS);
    
    // Benchmark getX
    auto getx_result = benchmark_getx(points, ITERATIONS);
    
    // Print results
    print_box_top();
    print_centered("Results");
    print_box_bottom();
    std::cout << "\n";
    
    std::cout << std::fixed << std::setprecision(2);
    
    std::cout << "Point::to_compressed()\n";
    std::cout << "  Median:  " << std::setw(8) << compressed_result.median_ns << " ns\n";
    std::cout << "  Range:   " << std::setw(8) << compressed_result.min_ns << " - " 
              << compressed_result.max_ns << " ns\n\n";
    
    std::cout << "Point::x()\n";
    std::cout << "  Median:  " << std::setw(8) << getx_result.median_ns << " ns\n";
    std::cout << "  Range:   " << std::setw(8) << getx_result.min_ns << " - " 
              << getx_result.max_ns << " ns\n\n";
    
    print_box_top();
    print_centered("Analysis");
    print_box_bottom();
    std::cout << "\n";
    
    std::cout << "Operations breakdown:\n\n";
    
    std::cout << "  to_compressed():\n";
    std::cout << "    • Convert Jacobian → Affine (1 inverse + 2 multiplies)\n";
    std::cout << "    • Field inverse:     ~7210 ns\n";
    std::cout << "    • Field multiply:    ~59 ns × 2 = 118 ns\n";
    std::cout << "    • Serialize to 33 bytes\n";
    std::cout << "    • Expected total:    ~7400-7500 ns\n";
    std::cout << "    • Actual:            " << compressed_result.median_ns << " ns\n\n";
    
    std::cout << "  x():\n";
    std::cout << "    • Convert Jacobian → Affine (1 inverse + 1 multiply)\n";
    std::cout << "    • Field inverse:     ~7210 ns\n";
    std::cout << "    • Field multiply:    ~59 ns\n";
    std::cout << "    • Expected total:    ~7270-7350 ns\n";
    std::cout << "    • Actual:            " << getx_result.median_ns << " ns\n\n";
    
    double overhead_compressed = compressed_result.median_ns - 7400;
    double overhead_getx = getx_result.median_ns - 7270;
    
    std::cout << "Overhead analysis:\n";
    std::cout << "  toCompressed(): " << std::setw(6) << overhead_compressed << " ns overhead\n";
    std::cout << "  getX():         " << std::setw(6) << overhead_getx << " ns overhead\n";
    
    std::cout << "\n";
    
    return 0;
}
