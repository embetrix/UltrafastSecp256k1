#include <iostream>
#include <chrono>
#include <vector>
#include <iomanip>
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field_52.hpp"

using namespace secp256k1::fast;
using namespace std::chrono;

int main() {
    const int ITERS = 1000000;
    
    // Setup
    Point G = Point::generator();
    Scalar k = Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000002");
    std::vector<Scalar> scalars(ITERS/10);
    for(int i=0; i<ITERS/10; ++i) {
        scalars[i] = k;
        k = k + Scalar::one();
    }
    
    // 1. Point Addition (G + G)
    auto start = high_resolution_clock::now();
    Point sum = G;
    for(int i=0; i<ITERS; ++i) {
        sum = sum.add(G);
    }
    auto end = high_resolution_clock::now();
    double add_ns = duration_cast<nanoseconds>(end - start).count() / (double)ITERS;
    
    // 2. Point Doubling (2*G)
    start = high_resolution_clock::now();
    Point dbl = G;
    for(int i=0; i<ITERS; ++i) {
        dbl = dbl.dbl();
    }
    end = high_resolution_clock::now();
    double dbl_ns = duration_cast<nanoseconds>(end - start).count() / (double)ITERS;
    
    // 3. Scalar Multiplication (k * G)
    start = high_resolution_clock::now();
    Point mul = G;
    for(int i=0; i<ITERS/10; ++i) { // Fewer iters for mul
        mul = G.scalar_mul(scalars[i]);
    }
    end = high_resolution_clock::now();
    double mul_us = duration_cast<microseconds>(end - start).count() / (double)(ITERS/10);
    
    // 4. Field Multiplication
    FieldElement52 f1 = FieldElement52::from_fe(G.x());
    FieldElement52 f2 = FieldElement52::from_fe(G.y());
    start = high_resolution_clock::now();
    for(int i=0; i<ITERS*10; ++i) {
        f1 = f1 * f2;
        f2 = f2 + FieldElement52::one(); // Prevent optimization
    }
    end = high_resolution_clock::now();
    double fmul_ns = duration_cast<nanoseconds>(end - start).count() / (double)(ITERS*10);

    std::cout << "UltrafastSecp256k1 x86_64 Benchmark\n";
    std::cout << "-----------------------------------\n";
    std::cout << "Field Mul:      " << std::fixed << std::setprecision(1) << fmul_ns << " ns\n";
    std::cout << "Point Add:      " << add_ns << " ns\n";
    std::cout << "Point Double:   " << dbl_ns << " ns\n";
    std::cout << "Scalar Mul:     " << mul_us << " us\n";
    
    return 0;
}
