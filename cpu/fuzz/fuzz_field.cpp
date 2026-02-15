// ============================================================================
// Fuzz target: Field arithmetic
// ============================================================================
// Build: clang++ -fsanitize=fuzzer,address -O2 -std=c++20 \
//        -I cpu/include fuzz_field.cpp cpu/src/field.cpp cpu/src/field_asm.cpp \
//        -o fuzz_field
// Run:   ./fuzz_field -max_len=64 -runs=10000000
// ============================================================================

#include "secp256k1/field.hpp"
#include <cstdint>
#include <cstddef>
#include <array>

using secp256k1::fast::FieldElement;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 64) return 0; // need two 32-byte field elements

    std::array<uint8_t, 32> buf_a{}, buf_b{};
    __builtin_memcpy(buf_a.data(), data, 32);
    __builtin_memcpy(buf_b.data(), data + 32, 32);

    auto a = FieldElement::from_bytes(buf_a);
    auto b = FieldElement::from_bytes(buf_b);

    // ── Closure: add/sub round-trip ──────────────────────────────────────────
    auto c = a + b;
    auto d = c - b;
    auto a_rt = d.to_bytes();
    auto a_orig = a.to_bytes();
    if (a_rt != a_orig) __builtin_trap();

    // ── Closure: mul by 1 = identity ─────────────────────────────────────────
    auto one = FieldElement::one();
    auto e = a * one;
    if (e.to_bytes() != a_orig) __builtin_trap();

    // ── Closure: a * a == a.square() ─────────────────────────────────────────
    auto f = a * a;
    auto g = a.square();
    if (f.to_bytes() != g.to_bytes()) __builtin_trap();

    // ── Closure: a * a^-1 == 1 (if a != 0) ──────────────────────────────────
    if (a != FieldElement::zero()) {
        auto inv = a.inverse();
        auto prod = a * inv;
        if (prod.to_bytes() != one.to_bytes()) __builtin_trap();
    }

    return 0;
}
