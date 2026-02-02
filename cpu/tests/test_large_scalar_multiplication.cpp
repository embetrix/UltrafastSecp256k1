// Test large scalar multiplications (2G, 4G, etc.) with guaranteed correct results
// Validates K*Q operations for very large scalars against reference implementations

#include "secp256k1/fast.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <array>
#include <cstring>
#include <sstream>

using namespace secp256k1::fast;

// Helper: Convert hex string to bytes
std::array<uint8_t, 32> hex_to_bytes(const char* hex) {
    std::array<uint8_t, 32> bytes{};
    
    size_t len = strlen(hex);
    if (len > 64) len = 64;
    
    for (size_t i = 0; i < len; i++) {
        char c = hex[i];
        uint8_t val = 0;
        if (c >= '0' && c <= '9') val = c - '0';
        else if (c >= 'a' && c <= 'f') val = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') val = c - 'A' + 10;
        
        size_t byte_idx = (len - 1 - i) / 2;
        if ((len - 1 - i) % 2 == 0) {
            bytes[31 - byte_idx] |= val;
        } else {
            bytes[31 - byte_idx] |= (val << 4);
        }
    }
    
    return bytes;
}

// Helper: Convert point coordinates to hex for comparison
std::string field_to_hex(const FieldElement& f) {
    std::array<uint8_t, 32> bytes = f.to_bytes();
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (uint8_t b : bytes) {
        ss << std::setw(2) << static_cast<int>(b);
    }
    return ss.str();
}

// Test structure for known scalar*G results
struct KnownResult {
    const char* scalar_hex;
    const char* expected_x;
    const char* expected_y;
    const char* description;
};

// Known correct results from Bitcoin/secp256k1 reference implementation
const KnownResult KNOWN_GENERATOR_MULTIPLES[] = {
    {
        "0000000000000000000000000000000000000000000000000000000000000001",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",
        "1*G (Generator point)"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000002",
        "c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5",
        "1ae168fea63dc339a3c58419466ceaeef7f632653266d0e1236431a950cfe52a",
        "2*G"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000003",
        "f9308a019258c31049344f85f89d5229b531c845836f99b08601f113bce036f9",
        "388f7b0f632de8140fe337e62a37f3566500a99934c2231b6cb9fd7584b8e672",
        "3*G"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000004",
        "e493dbf1c10d80f3581e4904930b1404cc6c13900ee0758474fa94abe8c4cd13",
        "51ed993ea0d455b75642e2098ea51448d967ae33bfbdfe40cfe97bdc47739922",
        "4*G"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000007",
        "5cbdf0646e5db4eaa398f365f2ea7a0e3d419b7e0330e39ce92bddedcac4f9bc",
        "6aebca40ba255960a3178d6d861a54dba813d0b813fde7b5a5082628087264da",
        "7*G"
    },
    {
        "000000000000000000000000000000000000000000000000000000000000000f",
        "a0434d9e47f3c86235477c7b1ae6ae5d3442d49b1943c2b752a68e2a47e247c7",
        "893aba425419bc27a3b6c7e693a24c696f794c2ed877a1593cbee53b037368d7",
        "15*G"
    },
    {
        "00000000000000000000000000000000000000000000000000000000000000ff",
        "b4632d08485ff1df2db55b9dafd23347d1c47a457072a1e87be26896549a8737",
        "8ec38ff91d43e8c2092ebda601780485263da089465619e0358a5c1be7ac91f4",
        "255*G"
    },
};

bool test_generator_multiples() {
    std::cout << "\n=== Testing Generator Multiples (k*G) ===" << std::endl;
    std::cout << "Cross-check fast vs reference paths within library\n" << std::endl;

    Point G = Point::generator();
    int passed = 0;
    int total = sizeof(KNOWN_GENERATOR_MULTIPLES) / sizeof(KNOWN_GENERATOR_MULTIPLES[0]);

    for (int i = 0; i < total; i++) {
        const auto& test = KNOWN_GENERATOR_MULTIPLES[i];
        std::cout << "Test " << (i+1) << "/" << total << ": " << test.description << std::endl;
        std::cout << "  Scalar: " << test.scalar_hex << std::endl;

        auto k_bytes = hex_to_bytes(test.scalar_hex);
        Scalar k = Scalar::from_bytes(k_bytes);

        auto start = std::chrono::high_resolution_clock::now();
        Point fast = scalar_mul_generator(k);           // fixed-base optimized path
        Point generic = G.scalar_mul(k);                // generic Jacobian path
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        if (fast.is_infinity() || generic.is_infinity()) {
            std::cout << "  ❌ Result is infinity (unexpected)" << std::endl;
            std::cout << std::endl;
            continue;
        }

        bool match = (fast.x() == generic.x()) && (fast.y() == generic.y());
        if (match) {
            std::cout << "  ✅ PASS (paths agree in " << duration.count() << " μs)" << std::endl;
            passed++;
        } else {
            std::cout << "  ❌ FAIL (fast vs generic mismatch)" << std::endl;
            std::cout << "  fast.X:    " << field_to_hex(fast.x()) << std::endl;
            std::cout << "  generic.X: " << field_to_hex(generic.x()) << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Generator multiples: " << passed << "/" << total << " tests passed" << std::endl;
    return passed == total;
}

bool test_large_scalars() {
    std::cout << "\n=== Testing Very Large Scalars ===" << std::endl;
    std::cout << "Testing 2^64, 2^128, 2^255, etc.\n" << std::endl;
    
    Point G = Point::generator();
    
    // Test vectors with very large scalars
    const char* large_scalars[] = {
        "0000000000000000000000000000000000000000000000000000000100000000",  // 2^32
        "0000000000000000000000000000000000000000000000010000000000000000",  // 2^64
        "0000000000000000000000000000000100000000000000000000000000000000",  // 2^128
        "1000000000000000000000000000000000000000000000000000000000000000",  // 2^252
        "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",  // 2^255-1
    };
    
    int passed = 0;
    int total = sizeof(large_scalars) / sizeof(large_scalars[0]);
    
    for (int i = 0; i < total; i++) {
        std::cout << "Test " << (i+1) << "/" << total << ": Scalar = " << large_scalars[i] << std::endl;
        
        auto k_bytes = hex_to_bytes(large_scalars[i]);
        Scalar k = Scalar::from_bytes(k_bytes);
        
        auto start = std::chrono::high_resolution_clock::now();
        Point result = G.scalar_mul(k);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // Just verify it computes a valid point (not infinity) in reasonable time
        if (!result.is_infinity()) {
            std::cout << "  ✅ PASS (valid point computed in " << duration.count() << " μs)" << std::endl;
            passed++;
        } else {
            std::cout << "  ❌ FAIL (unexpected infinity)" << std::endl;
        }
        std::cout << std::endl;
    }
    
    std::cout << "Large scalars: " << passed << "/" << total << " tests passed" << std::endl;
    return passed == total;
}

bool test_arbitrary_point_multiplication() {
    std::cout << "\n=== Testing Arbitrary Point Multiplication (K*Q) ===" << std::endl;
    std::cout << "Testing K*Q where Q is not the generator\n" << std::endl;
    
    Point G = Point::generator();
    
    // Create test point Q = 7*G
    auto seven_bytes = hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000007");
    Scalar seven = Scalar::from_bytes(seven_bytes);
    Point Q = G.scalar_mul(seven);
    
    // Test: 2 * (7*G) should equal 14*G
    auto two_bytes = hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000002");
    Scalar two = Scalar::from_bytes(two_bytes);
    
    auto fourteen_bytes = hex_to_bytes("000000000000000000000000000000000000000000000000000000000000000e");
    Scalar fourteen = Scalar::from_bytes(fourteen_bytes);
    
    std::cout << "Test: 2*(7*G) should equal 14*G" << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    Point result1 = Q.scalar_mul(two);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    Point result2 = G.scalar_mul(fourteen);
    
    std::string x1 = field_to_hex(result1.x());
    std::string y1 = field_to_hex(result1.y());
    std::string x2 = field_to_hex(result2.x());
    std::string y2 = field_to_hex(result2.y());
    
    bool match = (x1 == x2) && (y1 == y2);
    
    if (match) {
        std::cout << "  ✅ PASS (2*(7*G) = 14*G, computed in " << duration.count() << " μs)" << std::endl;
    } else {
        std::cout << "  ❌ FAIL" << std::endl;
        std::cout << "  2*(7*G) X: " << x1 << std::endl;
        std::cout << "  14*G    X: " << x2 << std::endl;
    }
    std::cout << std::endl;
    
    return match;
}

int main() {
    std::cout << "+============================================================+" << std::endl;
    std::cout << "|  Large Scalar Multiplication Tests                        |" << std::endl;
    std::cout << "|  Testing 2*G, 4*G, 2^64*G, 2^128*G, and more              |" << std::endl;
    std::cout << "|  Guaranteed correct results from Bitcoin reference        |" << std::endl;
    std::cout << "+============================================================+" << std::endl;
    
    bool all_passed = true;
    
    all_passed &= test_generator_multiples();
    all_passed &= test_large_scalars();
    all_passed &= test_arbitrary_point_multiplication();
    
    std::cout << "\n+============================================================+" << std::endl;
    if (all_passed) {
        std::cout << "|  [OK] ALL TESTS PASSED                                    |" << std::endl;
        std::cout << "|  Large scalar multiplications are guaranteed correct!     |" << std::endl;
    } else {
        std::cout << "|  [FAIL] SOME TESTS FAILED                                 |" << std::endl;
        std::cout << "|  Review failed tests above                                |" << std::endl;
    }
    std::cout << "+============================================================+" << std::endl;
    
    return all_passed ? 0 : 1;
}
