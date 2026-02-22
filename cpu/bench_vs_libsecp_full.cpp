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
        sum.add_inplace(G);
    }
    auto end = high_resolution_clock::now();
    double add_ns = duration_cast<nanoseconds>(end - start).count() / (double)ITERS;
    volatile uint64_t dummy_add = sum.x().limbs()[0];
    
    // 1b. Point Addition (sum + sum)
    start = high_resolution_clock::now();
    Point sum2 = G;
    for(int i=0; i<ITERS; ++i) {
        sum2.add_inplace(sum2);
    }
    end = high_resolution_clock::now();
    double add_full_ns = duration_cast<nanoseconds>(end - start).count() / (double)ITERS;
    volatile uint64_t dummy_add_full = sum2.x().limbs()[0];

    // 2. Point Doubling (2*G)
    start = high_resolution_clock::now();
    Point dbl = G;
    for(int i=0; i<ITERS; ++i) {
        dbl.dbl_inplace();
    }
    end = high_resolution_clock::now();
    double dbl_ns = duration_cast<nanoseconds>(end - start).count() / (double)ITERS;
    volatile uint64_t dummy_dbl = dbl.x().limbs()[0];
    
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

    }
    end = high_resolution_clock::now();
    double fmul_ns = duration_cast<nanoseconds>(end - start).count() / (double)(ITERS*10);
    volatile uint64_t dummy1 = f1.n[0]; // Prevent optimization

    // 5. Field Squaring
    f1 = FieldElement52::from_fe(G.x());
    start = high_resolution_clock::now();
    for(int i=0; i<ITERS*10; ++i) {
        f1 = f1.square();
    }
    end = high_resolution_clock::now();
    double fsqr_ns = duration_cast<nanoseconds>(end - start).count() / (double)(ITERS*10);
    volatile uint64_t dummy2 = f1.n[0]; // Prevent optimization

    // 6. Field Addition
    f1 = FieldElement52::from_fe(G.x());
    f2 = FieldElement52::from_fe(G.y());
    start = high_resolution_clock::now();
    for(int i=0; i<ITERS*10; ++i) {
        f1 = f1 + f2;
    }
    end = high_resolution_clock::now();
    double fadd_ns = duration_cast<nanoseconds>(end - start).count() / (double)(ITERS*10);
    volatile uint64_t dummy3 = f1.n[0]; // Prevent optimization

    std::cout << "UltrafastSecp256k1 x86_64 Benchmark\n";
    std::cout << "-----------------------------------\n";
    std::cout << "Field Mul:      " << std::fixed << std::setprecision(1) << fmul_ns << " ns\n";
    std::cout << "Field Sqr:      " << fsqr_ns << " ns\n";
    std::cout << "Field Add:      " << fadd_ns << " ns\n";
    std::cout << "Point Add(Mix): " << add_ns << " ns\n";
    std::cout << "Point Add(Full):" << add_full_ns << " ns\n";
    std::cout << "Point Double:   " << dbl_ns << " ns\n";
    std::cout << "Scalar Mul:     " << mul_us << " us\n";
    
    return 0;
}
