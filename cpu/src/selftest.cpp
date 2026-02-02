// SECP256K1 Library Self-Test
// Comprehensive arithmetic verification with known test vectors
// This ensures all math operations (scalar mul, point add/sub) are correct

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/glv.hpp"
#include <iostream>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <cstdlib>
#include <iomanip>

namespace secp256k1::fast {

// Test vector structure
struct TestVector {
    const char* scalar_hex;
    const char* expected_x;
    const char* expected_y;
    const char* description;
};

// Known test vectors: scalar * G = expected_point
// These are from trusted reference implementation
static const TestVector TEST_VECTORS[] = {
    {
        "4727daf2986a9804b1117f8261aba645c34537e4474e19be58700792d501a591",
        "0566896db7cd8e47ceb5e4aefbcf4d46ec295a15acb089c4affa9fcdd44471ef",
        "1513fcc547db494641ee2f65926e56645ec68cceaccb278a486e68c39ee876c4",
        "Vector 1"
    },
    {
        "c77835cf72699d217c2bbe6c59811b7a599bb640f0a16b3a332ebe64f20b1afa",
        "510f6c70028903e8c0d6f7a156164b972cea569b5a29bb03ff7564dfea9e875a",
        "c02b5ff43ae3b46e281b618abb0cbdaabdd600fbd6f4b78af693dec77080ef56",
        "Vector 2"
    },
    {
        "c401899c059f1c624292fece1933c890ae4970abf56dd4d2c986a5b9d7c9aeb5",
        "8434cbaf8256a8399684ed2212afc204e2e536034612039177bba44e1ea0d1c6",
        "0c34841bd41b0d869b35cfc4be6d57f098ae4beca55dc244c762c3ca0fd56af3",
        "Vector 3"
    },
    {
        "700a25ca2ae4eb40dfa74c9eda069be7e2fc9bfceabb13953ddedd33e1f03f2c",
        "2327ee923f529e67f537a45f633c8201dbee7be0c78d0894e31855843d9fbf0a",
        "f81ad336ee0bd923ec9338dd4b5f4b98d77caba5c153a6511ab15fd2ac6a422e",
        "Vector 4"
    },
    {
        "489206bbfff1b2370619ba0e6a51b74251267e06d3abafb055464bb623d5057a",
        "3ce5eb585c77104f8b877dd5ee574bf9439213b29f027e02e667cec79cd47b9e",
        "7ea30086c7c1f617d4c21c2f6e63cd0386f47ac8a3e97861d19d5d57d7338e3b",
        "Vector 5"
    },
    {
        "0000000000000000000000000000000000000000000000000000000000000001",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8",
        "1*G (Generator)"
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
        "000000000000000000000000000000000000000000000000000000000000000a",
        "a0434d9e47f3c86235477c7b1ae6ae5d3442d49b1943c2b752a68e2a47e247c7",
        "893aba425419bc27a3b6c7e693a24c696f794c2ed877a1593cbee53b037368d7",
        "10*G"
    },
    {
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798",
        "b7c52588d95c3b9aa25b0403f1eef75702e84bb7597aabe663b82f6f04ef2777",
        "(n-1)*G = -G"
    }
};

// Helper: Compare hex strings (case-insensitive)
static bool hex_equal(const std::string& a, const char* b) {
    if (a.length() != strlen(b)) return false;
    for (size_t i = 0; i < a.length(); i++) {
        char ca = a[i];
        char cb = b[i];
        if (ca >= 'A' && ca <= 'F') ca += 32; // to lowercase
        if (cb >= 'A' && cb <= 'F') cb += 32;
        if (ca != cb) return false;
    }
    return true;
}

// Helper: hex -> 32-byte array
static bool hex_to_bytes32(const std::string& hex, std::array<std::uint8_t, 32>& out) {
    if (hex.size() != 64) return false;
    auto nybble = [](char c) -> int {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
        if (c >= 'A' && c <= 'F') return 10 + (c - 'A');
        return -1;
    };
    for (size_t i = 0; i < 32; ++i) {
        int hi = nybble(hex[2*i]);
        int lo = nybble(hex[2*i + 1]);
        if (hi < 0 || lo < 0) return false;
        out[i] = static_cast<std::uint8_t>((hi << 4) | lo);
    }
    return true;
}

// Test one scalar multiplication vector
static bool test_scalar_mul(const TestVector& vec, bool verbose) {
    if (verbose) {
        std::cout << "  Testing: " << vec.description << "\n";
    }
    
    // Parse and compute k * G
    Scalar k = Scalar::from_hex(vec.scalar_hex);
    Point result = scalar_mul_generator(k);
    
    if (result.is_infinity()) {
        if (verbose) {
            std::cout << "    FAILED: Result is infinity!\n";
        }
        return false;
    }
    
    // Compare coordinates
    std::string result_x = result.x().to_hex();
    std::string result_y = result.y().to_hex();
    
    bool x_match = hex_equal(result_x, vec.expected_x);
    bool y_match = hex_equal(result_y, vec.expected_y);
    
    if (x_match && y_match) {
        if (verbose) {
            std::cout << "    PASS\n";
        }
        return true;
    } else {
        if (verbose) {
            std::cout << "    FAIL\n";
            if (!x_match) {
                std::cout << "      Expected X: " << vec.expected_x << "\n";
                std::cout << "      Got      X: " << result_x << "\n";
            }
            if (!y_match) {
                std::cout << "      Expected Y: " << vec.expected_y << "\n";
                std::cout << "      Got      Y: " << result_y << "\n";
            }
        }
        return false;
    }
}

// Test point addition: (k1*G) + (k2*G) = (k1+k2)*G
static bool test_addition(bool verbose) {
    if (verbose) {
        std::cout << "  Testing: 2*G + 3*G = 5*G\n";
    }
    
    Point P1 = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000002"));
    Point P2 = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000003"));
    Point expected = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000005"));
    
    Point result = P1.add(P2);
    
    std::string result_x = result.x().to_hex();
    std::string result_y = result.y().to_hex();
    std::string expected_x = expected.x().to_hex();
    std::string expected_y = expected.y().to_hex();
    
    bool match = (result_x == expected_x) && (result_y == expected_y);
    
    if (verbose) {
        if (match) {
            std::cout << "    PASS\n";
        } else {
            std::cout << "    FAIL\n";
            std::cout << "      Expected X: " << expected_x << "\n";
            std::cout << "      Got      X: " << result_x << "\n";
            std::cout << "      Expected Y: " << expected_y << "\n";
            std::cout << "      Got      Y: " << result_y << "\n";
        }
    }
    
    return match;
}

// Test point subtraction: (k1*G) - (k2*G) = (k1-k2)*G
static bool test_subtraction(bool verbose) {
    if (verbose) {
        std::cout << "  Testing: 5*G - 2*G = 3*G\n";
    }
    
    Point P1 = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000005"));
    Point P2 = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000002"));
    Point expected = scalar_mul_generator(Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000003"));
    
    // P1 - P2 = P1 + (-P2)
    Point result = P1.add(P2.negate());
    
    std::string result_x = result.x().to_hex();
    std::string result_y = result.y().to_hex();
    std::string expected_x = expected.x().to_hex();
    std::string expected_y = expected.y().to_hex();
    
    bool match = (result_x == expected_x) && (result_y == expected_y);
    
    if (verbose) {
        if (match) {
            std::cout << "    PASS\n";
        } else {
            std::cout << "    FAIL\n";
            std::cout << "      Expected X: " << expected_x << "\n";
            std::cout << "      Got      X: " << result_x << "\n";
            std::cout << "      Expected Y: " << expected_y << "\n";
            std::cout << "      Got      Y: " << result_y << "\n";
        }
    }
    
    return match;
}

// Basic field arithmetic identities (deterministic sanity)
static bool test_field_arithmetic(bool verbose) {
    if (verbose) {
        std::cout << "\nField Arithmetic Test:\n";
    }

    bool ok = true;
    FieldElement zero = FieldElement::zero();
    FieldElement one  = FieldElement::one();
    if (!((zero + zero) == zero)) ok = false;
    if (!((one + zero) == one)) ok = false;
    if (!((one * one) == one)) ok = false;
    if (!((zero * one) == zero)) ok = false;

    FieldElement a = FieldElement::from_uint64(7);
    FieldElement b = FieldElement::from_uint64(5);
    FieldElement neg_a = FieldElement::zero() - a;
    if (!((neg_a + a) == FieldElement::zero())) ok = false;
    if (!(((a + b) - b) == a)) ok = false;
    if (!(b == FieldElement::zero() || (b.inverse() * b) == FieldElement::one())) ok = false;

    if (verbose) {
        std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    }
    return ok;
}

// Basic scalar group identities (without inverse API)
static bool test_scalar_arithmetic(bool verbose) {
    if (verbose) {
        std::cout << "\nScalar Arithmetic Test:\n";
    }
    bool ok = true;
    Scalar z = Scalar::zero();
    Scalar o = Scalar::one();
    if (!((z + z) == z)) ok = false;
    if (!((o + z) == o)) ok = false;
    if (!(((o + o) - o) == o)) ok = false;
    if (verbose) {
        std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    }
    return ok;
}

// Point group identities (O neutral, negation)
static bool test_point_identities(bool verbose) {
    if (verbose) {
        std::cout << "\nPoint Group Identities:\n";
    }
    bool ok = true;
    Point O = Point::infinity();
    Point G = Point::generator();
    if (!(G.add(O).x() == G.x() && G.add(O).y() == G.y())) ok = false;
    Point negG = G.negate();
    if (!G.add(negG).is_infinity()) ok = false;
    if (verbose) {
        std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    }
    return ok;
}

// Helper: compare two points in affine coordinates
static bool points_equal(const Point& a, const Point& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() || b.is_infinity()) return false;
    return a.x() == b.x() && a.y() == b.y();
}

// Addition with constant expected: G + 2G = 3G (compare to known constants)
static bool test_addition_constants(bool verbose) {
    if (verbose) {
        std::cout << "\nPoint Addition (constants): G + 2G = 3G\n";
    }
    Point G = Point::generator();
    Point twoG = scalar_mul_generator(Scalar::from_uint64(2));
    Point sum = G.add(twoG);

    // TEST_VECTORS[7] is 3*G
    const auto& exp = TEST_VECTORS[7];
    bool ok = hex_equal(sum.x().to_hex(), exp.expected_x) && hex_equal(sum.y().to_hex(), exp.expected_y);
    if (verbose) std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Subtraction with constant expected: 3G - 2G = 1G
static bool test_subtraction_constants(bool verbose) {
    if (verbose) {
        std::cout << "\nPoint Subtraction (constants): 3G - 2G = 1G\n";
    }
    Point threeG = scalar_mul_generator(Scalar::from_uint64(3));
    Point twoG = scalar_mul_generator(Scalar::from_uint64(2));
    Point diff = threeG.add(twoG.negate());

    // TEST_VECTORS[5] is 1*G
    const auto& exp = TEST_VECTORS[5];
    bool ok = hex_equal(diff.x().to_hex(), exp.expected_x) && hex_equal(diff.y().to_hex(), exp.expected_y);
    if (verbose) std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Doubling with constant expected: (5G).dbl() = 10G
static bool test_doubling_constants(bool verbose) {
    if (verbose) {
        std::cout << "\nPoint Doubling (constants): 2*(5G) = 10G\n";
    }
    Point fiveG = scalar_mul_generator(Scalar::from_uint64(5));
    Point tenG = fiveG.dbl();

    // TEST_VECTORS[8] is 10*G
    const auto& exp = TEST_VECTORS[8];
    bool ok = hex_equal(tenG.x().to_hex(), exp.expected_x) && hex_equal(tenG.y().to_hex(), exp.expected_y);
    if (verbose) std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Negation with constant expected: -G equals (n-1)*G vector
static bool test_negation_constants(bool verbose) {
    if (verbose) {
        std::cout << "\nPoint Negation (constants): -G = (n-1)*G\n";
    }
    Point negG = Point::generator().negate();
    // TEST_VECTORS[9] is (n-1)*G = -G
    const auto& exp = TEST_VECTORS[9];
    bool ok = hex_equal(negG.x().to_hex(), exp.expected_x) && hex_equal(negG.y().to_hex(), exp.expected_y);
    if (verbose) std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Point (de)serialization: compressed/uncompressed encodings sanity
static bool test_point_serialization(bool verbose) {
    if (verbose) {
        std::cout << "\nPoint Serialization:\n";
    }
    auto check_point = [&](const Scalar& k) -> bool {
        Point P = scalar_mul_generator(k);
        auto cx = P.x().to_bytes();
        auto cy = P.y().to_bytes();
        auto comp = P.to_compressed();
        auto uncmp = P.to_uncompressed();
        std::uint8_t expected_prefix = (cy[31] & 1) ? 0x03 : 0x02;
        bool ok = true;
        if (comp[0] != expected_prefix) ok = false;
        for (size_t i = 0; i < 32; ++i) {
            if (comp[1 + i] != cx[i]) { ok = false; break; }
        }
        if (uncmp[0] != 0x04) ok = false;
        for (size_t i = 0; i < 32; ++i) {
            if (uncmp[1 + i] != cx[i]) { ok = false; break; }
        }
        for (size_t i = 0; i < 32; ++i) {
            if (uncmp[33 + i] != cy[i]) { ok = false; break; }
        }
        return ok;
    };
    bool all = true;
    all &= check_point(Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000001"));
    all &= check_point(Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000002"));
    all &= check_point(Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000003"));
    all &= check_point(Scalar::from_hex("000000000000000000000000000000000000000000000000000000000000000a"));
    if (verbose) {
        std::cout << (all ? "    PASS\n" : "    FAIL\n");
    }
    return all;
}

// Batch inversion vs individual inversion
static bool test_batch_inverse(bool verbose) {
    if (verbose) {
        std::cout << "\nBatch Inversion:\n";
    }
    FieldElement elems[4] = {
        FieldElement::from_uint64(3),
        FieldElement::from_uint64(7),
        FieldElement::from_uint64(11),
        FieldElement::from_uint64(19)
    };
    FieldElement copy[4] = { elems[0], elems[1], elems[2], elems[3] };
    fe_batch_inverse(elems, 4);
    bool ok = true;
    for (int i = 0; i < 4; ++i) {
        FieldElement inv = copy[i].inverse();
        if (!(inv == elems[i])) { ok = false; break; }
    }
    if (verbose) {
        std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    }
    return ok;
}

// Expanded batch inversion with a larger, deterministic set
static bool test_batch_inverse_expanded(bool verbose) {
    if (verbose) {
        std::cout << "\nBatch Inversion (expanded 32 elems):\n";
    }
    constexpr size_t N = 32;
    FieldElement elems[N];
    FieldElement copy[N];
    // Deterministic non-zero elements: 3,5,7,...
    for (size_t i = 0; i < N; ++i) {
        std::uint64_t v = 3ULL + 2ULL * static_cast<std::uint64_t>(i);
        elems[i] = FieldElement::from_uint64(v);
        copy[i] = elems[i];
    }
    fe_batch_inverse(elems, N);
    bool ok = true;
    for (size_t i = 0; i < N; ++i) {
        FieldElement inv = copy[i].inverse();
        if (!(inv == elems[i])) { ok = false; break; }
    }
    if (verbose) {
        std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    }
    return ok;
}

// Bilinearity checks for K*Q with non-generator points
// Tests: (Q+G)*K == Q*K + G*K, (Q-G)*K == Q*K - G*K
static bool test_bilinearity_K_times_Q(bool verbose) {
    if (verbose) {
        std::cout << "\nBilinearity: K*(Q±G) vs K*Q ± K*G\n";
    }
    bool ok = true;
    const char* KHEX[] = {
        "0000000000000000000000000000000000000000000000000000000000000005",
        "4727daf2986a9804b1117f8261aba645c34537e4474e19be58700792d501a591",
        "c77835cf72699d217c2bbe6c59811b7a599bb640f0a16b3a332ebe64f20b1afa"
    };
    const char* QHEX[] = {
        "0000000000000000000000000000000000000000000000000000000000000011",
        "0000000000000000000000000000000000000000000000000000000000000067",
        "c401899c059f1c624292fece1933c890ae4970abf56dd4d2c986a5b9d7c9aeb5"
    };
    Point G = Point::generator();
    for (auto kh : KHEX) {
        Scalar K = Scalar::from_hex(kh);
        Point KG = scalar_mul_generator(K);
        for (auto qh : QHEX) {
            Scalar qk = Scalar::from_hex(qh);
            Point Q = scalar_mul_generator(qk); // Q = qk*G (valid arbitrary point)

            Point Lp = Q.add(G).scalar_mul(K);           // (Q+G)*K
            Point Rp = Q.scalar_mul(K).add(KG);          // Q*K + G*K
            if (!points_equal(Lp, Rp)) { ok = false; break; }

            Point Lm = Q.add(G.negate()).scalar_mul(K);  // (Q-G)*K
            Point Rm = Q.scalar_mul(K).add(KG.negate()); // Q*K - G*K
            if (!points_equal(Lm, Rm)) { ok = false; break; }
        }
        if (!ok) break;
    }
    if (verbose) std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Fixed-K plan consistency: scalar_mul_with_plan vs scalar_mul
static bool test_fixedK_plan(bool verbose) {
    if (verbose) {
        std::cout << "\nFixed-K plan: with_plan vs direct scalar_mul\n";
    }
    bool ok = true;
    const char* KHEX[] = {
        TEST_VECTORS[0].scalar_hex,
        TEST_VECTORS[1].scalar_hex,
        "00000000000000000000000000000000000000000000000000000000000000a7"
    };
    const char* QHEX[] = {
        "000000000000000000000000000000000000000000000000000000000000000d",
        "0000000000000000000000000000000000000000000000000000000000000123",
        "700a25ca2ae4eb40dfa74c9eda069be7e2fc9bfceabb13953ddedd33e1f03f2c"
    };
    for (auto kh : KHEX) {
        Scalar K = Scalar::from_hex(kh);
        KPlan plan = KPlan::from_scalar(K, 4);
        for (auto qh : QHEX) {
            Scalar qk = Scalar::from_hex(qh);
            Point Q = scalar_mul_generator(qk);
            Point A = Q.scalar_mul(K);
            Point B = Q.scalar_mul_with_plan(plan);
            if (!points_equal(A, B)) {
                if (verbose) {
                    auto aC = A.to_compressed();
                    auto bC = B.to_compressed();
                    std::cout << "    Mismatch!\n";
                    std::cout << "      K: 0x" << kh << "  (neg1=" << (plan.neg1?"1":"0")
                              << ", neg2=" << (plan.neg2?"1":"0") << ")\n";
                    std::cout << "      q: 0x" << qh << "\n";
                    std::cout << "      A: ";
                    for (auto b : aC) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b;
                    std::cout << std::dec << "\n";
                    std::cout << "      B: ";
                    for (auto b : bC) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b;
                    std::cout << std::dec << "\n";
                    // Also compute explicit slow GLV sum for debugging
                    Point phiQ = apply_endomorphism(Q);
                    Point t1 = Q.scalar_mul(plan.k1);
                    Point t2 = phiQ.scalar_mul(plan.k2);
                    if (plan.neg1) t1 = t1.negate();
                    if (plan.neg2) t2 = t2.negate();
                    Point C = t1.add(t2);
                    auto cC = C.to_compressed();
                    std::cout << "      C(slow): ";
                    for (auto b : cC) std::cout << std::hex << std::setw(2) << std::setfill('0') << (int)b;
                    std::cout << std::dec << "\n";
                }
                ok = false; break;
            }
        }
        if (!ok) break;
    }
    if (verbose) std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// Sequential Q increment property: (Q + i*G)*K = (Q*K) + i*(G*K)
static bool test_sequential_increment_property(bool verbose) {
    if (verbose) {
        std::cout << "\nSequential increment: (Q+i*G)*K vs (Q*K)+i*(G*K)\n";
    }
    bool ok = true;
    // Choose a fixed K and base Q
    Scalar K = Scalar::from_hex("489206bbfff1b2370619ba0e6a51b74251267e06d3abafb055464bb623d5057a");
    Scalar qk = Scalar::from_hex("0000000000000000000000000000000000000000000000000000000000000101");
    Point Q = scalar_mul_generator(qk);
    Point KG = scalar_mul_generator(K);
    // Left side incrementally via next_inplace; Right side via repeated add of KG
    Point left = Q.scalar_mul(K);
    Point right = left; // i=0
    for (int i = 1; i <= 16; ++i) {
        // Q <- Q + G
        Q.next_inplace();
        left = Q.scalar_mul(K);
        right = right.add(KG);
        if (!points_equal(left, right)) { ok = false; break; }
    }
    if (verbose) std::cout << (ok ? "    PASS\n" : "    FAIL\n");
    return ok;
}

// External vector file loader (optional). Format (semicolon-separated, hex lowercase or uppercase):
//  SCALARMUL;kk;expX;expY;desc
//  ADD;x1;y1;x2;y2;expX;expY;desc
//  SUB;x1;y1;x2;y2;expX;expY;desc
static bool run_external_vectors(bool verbose) {
    const char* path = std::getenv("SECP256K1_SELFTEST_VECTORS");
    if (!path) return true; // Not provided: treat as success
    std::ifstream in(path);
    if (!in) {
        if (verbose) {
            std::cout << "\n[Selftest] Vector file not found: " << path << " (skipping)\n";
        }
        return true; // Non-fatal
    }
    if (verbose) {
        std::cout << "\nExternal Vector Tests (" << path << "):\n";
    }
    bool all_ok = true;
    std::string line;
    size_t ln = 0;
    while (std::getline(in, line)) {
        ++ln;
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> parts;
        std::stringstream ss(line);
        std::string item;
        while (std::getline(ss, item, ';')) parts.push_back(item);
        if (parts.empty()) continue;
        const std::string& kind = parts[0];
        auto fail_line = [&]() {
            all_ok = false;
            if (verbose) {
                std::cout << "    FAIL (line " << ln << ")\n";
            }
        };
        if (kind == "SCALARMUL") {
            if (parts.size() < 5) { fail_line(); continue; }
            Scalar k = Scalar::from_hex(parts[1]);
            Point r = scalar_mul_generator(k);
            std::string rx = r.x().to_hex();
            std::string ry = r.y().to_hex();
            if (!hex_equal(rx, parts[2].c_str()) || !hex_equal(ry, parts[3].c_str())) {
                fail_line();
            }
        } else if (kind == "ADD" || kind == "SUB") {
            if (parts.size() < 8) { fail_line(); continue; }
            std::array<std::uint8_t, 32> x1b{}, y1b{}, x2b{}, y2b{};
            if (!hex_to_bytes32(parts[1], x1b) || !hex_to_bytes32(parts[2], y1b) ||
                !hex_to_bytes32(parts[3], x2b) || !hex_to_bytes32(parts[4], y2b)) {
                fail_line();
                continue;
            }
            Point P1 = Point::from_affine(FieldElement::from_bytes(x1b), FieldElement::from_bytes(y1b));
            Point P2 = Point::from_affine(FieldElement::from_bytes(x2b), FieldElement::from_bytes(y2b));
            Point R = (kind == "ADD") ? P1.add(P2) : P1.add(P2.negate());
            std::string rx = R.x().to_hex();
            std::string ry = R.y().to_hex();
            if (!hex_equal(rx, parts[5].c_str()) || !hex_equal(ry, parts[6].c_str())) {
                fail_line();
            }
        } else {
            // Unknown entry – ignore
        }
    }
    if (verbose) {
        std::cout << (all_ok ? "    PASS\n" : "    FAIL\n");
    }
    return all_ok;
}

// Main self-test function
bool Selftest(bool verbose) {
    if (verbose) {
        std::cout << "\n==============================================\n";
        std::cout << "  SECP256K1 Library Self-Test\n";
        std::cout << "==============================================\n";
    }
    
    // Initialize precomputed tables (allow env overrides for quick toggles)
    FixedBaseConfig cfg{};
    if (const char* w = std::getenv("SECP256K1_WINDOW_BITS")) {
        unsigned v = static_cast<unsigned>(std::strtoul(w, nullptr, 10));
        if (v >= 2U && v <= 30U) cfg.window_bits = v;
    }
    if (const char* g = std::getenv("SECP256K1_ENABLE_GLV")) {
        if (g[0] == '1' || g[0] == 't' || g[0] == 'T' || g[0] == 'y' || g[0] == 'Y') cfg.enable_glv = true;
    }
    if (const char* j = std::getenv("SECP256K1_USE_JSF")) {
        if (j[0] == '1' || j[0] == 't' || j[0] == 'T' || j[0] == 'y' || j[0] == 'Y') {
            cfg.use_jsf = true;
            cfg.enable_glv = true; // JSF applies to GLV path
        }
    }
    configure_fixed_base(cfg);
    ensure_fixed_base_ready();
    
    int passed = 0;
    int total = 0;
    
    // Test scalar multiplication
    if (verbose) {
        std::cout << "\nScalar Multiplication Tests:\n";
    }
    
    for (const auto& vec : TEST_VECTORS) {
        total++;
        if (test_scalar_mul(vec, verbose)) {
            passed++;
        }
    }
    
    // Test point addition
    if (verbose) {
        std::cout << "\nPoint Addition Test:\n";
    }
    total++;
    if (test_addition(verbose)) {
        passed++;
    }
    
    // Test point subtraction
    if (verbose) {
        std::cout << "\nPoint Subtraction Test:\n";
    }
        // Field arithmetic
        total++;
        if (test_field_arithmetic(verbose)) passed++;

        // Scalar arithmetic (basic identities)
        total++;
        if (test_scalar_arithmetic(verbose)) passed++;

        // Point group identities
        total++;
        if (test_point_identities(verbose)) passed++;

        // External vectors (optional, environment-driven)
        total++;
        if (run_external_vectors(verbose)) passed++;
        // Point serialization
        total++;
        if (test_point_serialization(verbose)) passed++;
        // Batch inverse
        total++;
        if (test_batch_inverse(verbose)) passed++;
        // Constant-expected point ops
        total++;
        if (test_addition_constants(verbose)) passed++;
        total++;
        if (test_subtraction_constants(verbose)) passed++;
        total++;
        if (test_doubling_constants(verbose)) passed++;
        total++;
        if (test_negation_constants(verbose)) passed++;

    // Additional high-coverage checks
    // 1) Doubling chain vs scalar multiples: for i=1..20, (2^i)G via dbl() equals scalar_mul
    auto test_pow2_chain = [&](){
        if (verbose) std::cout << "\nDoubling chain vs scalar multiples (2^i * G):\n";
        bool ok = true;
        Point cur = Point::generator(); // 1*G
        for (int i = 1; i <= 20; ++i) {
            cur.dbl_inplace(); // now 2^i * G
            Scalar k = Scalar::from_uint64(1ULL << i);
            Point exp = scalar_mul_generator(k);
            if (!points_equal(cur, exp)) { ok = false; break; }
        }
        if (verbose) std::cout << (ok ? "    PASS\n" : "    FAIL\n");
        return ok;
    };
    total++; if (test_pow2_chain()) passed++;

    // 2) Large scalar cross-checks (fast vs affine fallback)
    auto test_large_scalars = [&](){
        if (verbose) std::cout << "\nLarge scalar cross-checks (fast vs affine):\n";
        bool ok = true;
        // representative large scalars (hex, reduced mod n by Scalar::from_hex)
        const char* L[] = {
            "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "8000000000000000000000000000000000000000000000000000000000000000",
            "7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
            "deadbeefcafebabef00dfeedfacefeed1234567890abcdef1122334455667788"
        };
        Point G = Point::generator();
        Point G_aff = Point::from_affine(G.x(), G.y());
        for (const char* hx : L) {
            Scalar k = Scalar::from_hex(hx);
            Point fast = scalar_mul_generator(k);
            Point ref  = G_aff.scalar_mul(k);
            if (!points_equal(fast, ref)) { ok = false; break; }
        }
        if (verbose) std::cout << (ok ? "    PASS\n" : "    FAIL\n");
        return ok;
    };
    total++; if (test_large_scalars()) passed++;

    // 3) Squared scalar cases: k^2 * G (to mitigate high-bit corner mistakes)
    auto test_squared_scalars = [&](){
        if (verbose) std::cout << "\nSquared scalars k^2 * G (fast vs affine):\n";
        bool ok = true;
        // take a mix of known and random-looking scalars
        const char* K[] = {
            TEST_VECTORS[0].scalar_hex,
            TEST_VECTORS[1].scalar_hex,
            TEST_VECTORS[2].scalar_hex,
            TEST_VECTORS[3].scalar_hex,
            "0000000000000000000000000000000000000000000000000000000000000013",
            "0000000000000000000000000000000000000000000000000000000000000061",
            "2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d5e6f708192a3b4c5d6e7f8091a"
        };
        Point G = Point::generator();
        Point G_aff = Point::from_affine(G.x(), G.y());
        for (const char* hx : K) {
            Scalar k = Scalar::from_hex(hx);
            Scalar k2 = k * k; // mod n
            Point fast = scalar_mul_generator(k2);
            Point ref  = G_aff.scalar_mul(k2);
            if (!points_equal(fast, ref)) { ok = false; break; }
        }
        if (verbose) std::cout << (ok ? "    PASS\n" : "    FAIL\n");
        return ok;
    };
    total++; if (test_squared_scalars()) passed++;

    // Expanded batch inverse
    total++; if (test_batch_inverse_expanded(verbose)) passed++;
    // Bilinearity for K*Q with ±G
    total++; if (test_bilinearity_K_times_Q(verbose)) passed++;
    // Fixed-K plan equivalence
    total++; if (test_fixedK_plan(verbose)) passed++;
    // Sequential increment property
    total++; if (test_sequential_increment_property(verbose)) passed++;
    total++;
    if (test_subtraction(verbose)) {
        passed++;
    }
    
    // Summary
    if (verbose) {
        std::cout << "\n==============================================\n";
        std::cout << "  Results: " << passed << "/" << total << " tests passed\n";
        if (passed == total) {
            std::cout << "  [OK] ALL TESTS PASSED\n";
        } else {
            std::cout << "  [FAIL] SOME TESTS FAILED\n";
        }
        std::cout << "==============================================\n\n";
    }
    
    return (passed == total);
}

} // namespace secp256k1::fast
