// ============================================================================
// Fuzz target: Scalar arithmetic
// ============================================================================
// Build: clang++ -fsanitize=fuzzer,address -O2 -std=c++20 \
//        -I cpu/include fuzz_scalar.cpp cpu/src/scalar.cpp \
//        -o fuzz_scalar
// Run:   ./fuzz_scalar -max_len=64 -runs=10000000
// ============================================================================

#include "secp256k1/scalar.hpp"
#include <cstdint>
#include <cstddef>
#include <array>

using secp256k1::fast::Scalar;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 64) return 0; // need two 32-byte scalars

    std::array<uint8_t, 32> buf_a{}, buf_b{};
    __builtin_memcpy(buf_a.data(), data, 32);
    __builtin_memcpy(buf_b.data(), data + 32, 32);

    auto a = Scalar::from_bytes(buf_a);
    auto b = Scalar::from_bytes(buf_b);
    auto zero = Scalar::zero();

    // ── Closure: add/sub round-trip ──────────────────────────────────────────
    auto c = a + b;
    auto d = c - b;
    if (d != a) __builtin_trap();

    // ── Closure: mul by 1 = identity ─────────────────────────────────────────
    auto one = Scalar::one();
    auto e = a * one;
    if (e != a) __builtin_trap();

    // ── Closure: a - a = 0 ───────────────────────────────────────────────────
    auto f = a - a;
    if (!f.is_zero()) __builtin_trap();

    // ── Closure: a + 0 = a ───────────────────────────────────────────────────
    auto g = a + zero;
    if (g != a) __builtin_trap();

    // ── Closure: (a * b) * 1 = a * b ─────────────────────────────────────────
    auto h = a * b;
    auto i = h * one;
    if (i != h) __builtin_trap();

    // ── Closure: distributive: a*(b+1) = a*b + a ─────────────────────────────
    auto b_plus_1 = b + one;
    auto lhs = a * b_plus_1;
    auto rhs = (a * b) + a;
    if (lhs != rhs) __builtin_trap();

    return 0;
}
