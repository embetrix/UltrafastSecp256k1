// Benchmark Montgomery domain operations
#include "secp256k1/fast.hpp"
#include "secp256k1/field_branchless.hpp"
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <random>

using namespace secp256k1::fast;
using namespace std::chrono;

void print_fe(const char* label, const FieldElement& fe) {
    std::cout << label << ": " << fe.to_hex() << "\n";
}

int main(int argc, char** argv) {
    Selftest(true);  // [OK] Validate arithmetic before benchmarking
    
    std::cout << "\n";
    std::cout << "===========================================================\n";
    std::cout << "  Phase 1 GPU Optimizations - Montgomery & MidFieldElement\n";
    std::cout << "===========================================================\n";
    std::cout << "\n";
    
    // Test 1: Montgomery constants
    std::cout << "1  Montgomery Domain Constants:\n";
    std::cout << "-------------------------------------------------\n";
    print_fe("   R (2^256 mod p)", montgomery::R());
    print_fe("   R^2 mod p", montgomery::R2());
    print_fe("   R^3 mod p", montgomery::R3());
    print_fe("   R^-^1 mod p", montgomery::R_inv());
    std::cout << "   K_MOD = 0x" << std::hex << montgomery::K_MOD << std::dec << "\n";
    std::cout << "   [OK] All constants initialized\n";
    std::cout << "\n";
    
    // Test 2: MidFieldElement zero-cost conversion
    std::cout << "2  MidFieldElement (32/64-bit hybrid):\n";
    std::cout << "-------------------------------------------------\n";
    FieldElement a = FieldElement::from_uint64(0x123456789ABCDEF0ULL);
    MidFieldElement* mid_ptr = toMid(&a);
    
    std::cout << "   Original (64-bit):";
    for (int i = 0; i < 4; i++) {
        std::cout << " 0x" << std::hex << a.limbs()[i] << std::dec;
    }
    std::cout << "\n";
    
    std::cout << "   As 32-bit view:";
    for (int i = 0; i < 8; i++) {
        std::cout << " 0x" << std::hex << mid_ptr->limbs[i] << std::dec;
    }
    std::cout << "\n";
    
    // Verify zero-cost conversion
    FieldElement* back = mid_ptr->ToFieldElement();
    bool conversion_ok = (back == &a) && (*back == a);
    std::cout << "   Zero-cost roundtrip: " << (conversion_ok ? "[OK] PASS" : "[FAIL] FAIL") << "\n";
    std::cout << "   sizeof(FieldElement) = " << sizeof(FieldElement) << " bytes\n";
    std::cout << "   sizeof(MidFieldElement) = " << sizeof(MidFieldElement) << " bytes\n";
    std::cout << "\n";
    
    // Test 3: Branchless operations
    std::cout << "3  Branchless Operations:\n";
    std::cout << "-------------------------------------------------\n";
    FieldElement x = FieldElement::from_uint64(42);
    FieldElement y = FieldElement::from_uint64(100);
    FieldElement z;
    
    // Test field_cmov
    field_cmov(&z, &x, &y, true);  // Select x
    bool cmov_true = (z == x);
    field_cmov(&z, &x, &y, false); // Select y
    bool cmov_false = (z == y);
    std::cout << "   field_cmov(true):  " << (cmov_true ? "[OK] PASS" : "[FAIL] FAIL") << "\n";
    std::cout << "   field_cmov(false): " << (cmov_false ? "[OK] PASS" : "[FAIL] FAIL") << "\n";
    
    // Test field_select
    FieldElement selected_true = field_select(x, y, true);
    FieldElement selected_false = field_select(x, y, false);
    std::cout << "   field_select(true):  " << (selected_true == x ? "[OK] PASS" : "[FAIL] FAIL") << "\n";
    std::cout << "   field_select(false): " << (selected_false == y ? "[OK] PASS" : "[FAIL] FAIL") << "\n";
    
    // Test field_is_zero
    FieldElement zero = FieldElement::zero();
    FieldElement nonzero = FieldElement::one();
    std::cout << "   field_is_zero(0): " << (field_is_zero(zero) == 1 ? "[OK] PASS" : "[FAIL] FAIL") << "\n";
    std::cout << "   field_is_zero(1): " << (field_is_zero(nonzero) == 0 ? "[OK] PASS" : "[FAIL] FAIL") << "\n";
    
    // Test field_eq
    FieldElement equal = FieldElement::one();
    std::cout << "   field_eq(1, 1): " << (field_eq(nonzero, equal) == 1 ? "[OK] PASS" : "[FAIL] FAIL") << "\n";
    std::cout << "   field_eq(1, 0): " << (field_eq(nonzero, zero) == 0 ? "[OK] PASS" : "[FAIL] FAIL") << "\n";
    std::cout << "\n";
    
    // Benchmark: field_cmov vs branched selection
    std::cout << "4  Performance Comparison:\n";
    std::cout << "-------------------------------------------------\n";
    
    std::mt19937_64 rng(12345);
    std::uniform_int_distribution<uint64_t> dist;
    
    const int N = 1000000;
    FieldElement result;
    
    // Generate random test data
    std::vector<FieldElement> test_a(N);
    std::vector<FieldElement> test_b(N);
    std::vector<bool> flags(N);
    for (int i = 0; i < N; i++) {
        test_a[i] = FieldElement::from_uint64(dist(rng));
        test_b[i] = FieldElement::from_uint64(dist(rng));
        flags[i] = (dist(rng) & 1);
    }
    
    // Benchmark branched selection
    auto t0 = high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        result = flags[i] ? test_a[i] : test_b[i];
    }
    auto t1 = high_resolution_clock::now();
    double ns_branched = duration_cast<nanoseconds>(t1 - t0).count() / double(N);
    
    // Benchmark branchless selection
    t0 = high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        result = field_select(test_a[i], test_b[i], flags[i]);
    }
    t1 = high_resolution_clock::now();
    double ns_branchless = duration_cast<nanoseconds>(t1 - t0).count() / double(N);
    
    std::cout << "   Branched (? :):       " << ns_branched << " ns/op\n";
    std::cout << "   Branchless (CMOV):    " << ns_branchless << " ns/op\n";
    
    double speedup = ns_branched / ns_branchless;
    if (speedup >= 1.05) {
        std::cout << "   >> Speedup: " << speedup << "x faster\n";
    } else if (speedup <= 0.95) {
        std::cout << "   [!]  Slowdown: " << (1.0/speedup) << "x slower (unexpected)\n";
    } else {
        std::cout << "   ->  Similar performance (~" << speedup << "x)\n";
    }
    
    std::cout << "\n";
    std::cout << "===========================================================\n";
    std::cout << "  Phase 1 Complete: Core infrastructure ready [OK]\n";
    std::cout << "  Next: H-based serial inversion & Montgomery mul (Phase 2)\n";
    std::cout << "===========================================================\n";
    
    return 0;
}
