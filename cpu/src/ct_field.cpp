// ============================================================================
// Constant-Time Field Arithmetic — Implementation
// ============================================================================
// All operations have data-independent execution traces.
// p = 2^256 - 2^32 - 977 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
//
// FAST-PATH ASM INTEGRATION (x86_64):
//   field_mul and field_sqr call the BMI2+ADX assembly directly
//   (field_mul_full_asm / field_sqr_full_asm) which already produce
//   fully normalized results in [0, p) via branchless cmovc.
//   This bypasses the operator* wrapper chain and eliminates
//   the redundant field_normalize pass — ~2.5× faster field_mul.
// ============================================================================

#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/ops.hpp"
#include <cstring>

// ─── Direct ASM declarations (bypass wrapper overhead) ───────────────────────
// These are the fused multiply+reduce functions from field_asm_x64_gas.S.
// They produce output in [0, p) using branchless normalization (cmovc).
// Signature: (const uint64_t a[4], const uint64_t b[4], uint64_t result[4])
//        sqr: (const uint64_t a[4], uint64_t result[4])
#if defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
extern "C" {
    void field_mul_full_asm(const std::uint64_t* a, const std::uint64_t* b,
                            std::uint64_t* result);
    void field_sqr_full_asm(const std::uint64_t* a, std::uint64_t* result);
}
#define SECP256K1_CT_HAS_DIRECT_ASM 1
#else
#define SECP256K1_CT_HAS_DIRECT_ASM 0
#endif

namespace secp256k1::ct {

// secp256k1 prime p in 4 x 64-bit limbs (little-endian)
static constexpr std::uint64_t P[4] = {
    0xFFFFFFFEFFFFFC2FULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL
};

// 2^256 mod p = 2^32 + 977 = 0x1000003D1
static constexpr std::uint64_t MOD_K = 0x1000003D1ULL;

// ─── Internal helpers ────────────────────────────────────────────────────────

// CT 256-bit addition with carry out. Returns carry (0 or 1).
static inline std::uint64_t add256(std::uint64_t r[4],
                                    const std::uint64_t a[4],
                                    const std::uint64_t b[4]) noexcept {
    std::uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        // r[i] = a[i] + b[i] + carry
        std::uint64_t sum_lo = a[i] + b[i];
        std::uint64_t c1 = static_cast<std::uint64_t>(sum_lo < a[i]);
        std::uint64_t sum = sum_lo + carry;
        std::uint64_t c2 = static_cast<std::uint64_t>(sum < sum_lo);
        r[i] = sum;
        carry = c1 + c2;
    }
    return carry;
}

// CT 256-bit subtraction with borrow out. Returns borrow (0 or 1).
static inline std::uint64_t sub256(std::uint64_t r[4],
                                    const std::uint64_t a[4],
                                    const std::uint64_t b[4]) noexcept {
    std::uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        std::uint64_t diff = a[i] - b[i];
        std::uint64_t b1 = static_cast<std::uint64_t>(a[i] < b[i]);
        std::uint64_t result = diff - borrow;
        std::uint64_t b2 = static_cast<std::uint64_t>(diff < borrow);
        r[i] = result;
        borrow = b1 + b2;
    }
    return borrow;
}

// CT normalize: reduce to [0, p) without branches
// If value >= p, subtract p. Uses cmov.
static inline void ct_reduce_once(std::uint64_t r[4]) noexcept {
    std::uint64_t tmp[4];
    std::uint64_t borrow = sub256(tmp, r, P);
    // If borrow == 0, r >= p → use tmp (reduced). Else keep r.
    // mask = 0xFFF...F if no borrow (r >= p), else 0
    std::uint64_t mask = is_zero_mask(borrow);
    cmov256(r, tmp, mask);
}

// ─── Public API ──────────────────────────────────────────────────────────────

FieldElement field_normalize(const FieldElement& a) noexcept {
    std::uint64_t r[4];
    std::memcpy(r, &a.limbs()[0], 32);
    ct_reduce_once(r);
    return FieldElement::from_limbs_raw({r[0], r[1], r[2], r[3]});
}

FieldElement field_add(const FieldElement& a, const FieldElement& b) noexcept {
    std::uint64_t r[4];
    std::uint64_t carry = add256(r, a.limbs().data(), b.limbs().data());

    // If carry OR r >= p, subtract p
    // First subtract p unconditionally
    std::uint64_t tmp[4];
    std::uint64_t borrow = sub256(tmp, r, P);

    // If carry, we definitely need to subtract p (result overflowed 256 bits)
    // If no carry and no borrow, we still use tmp (r was >= p)
    // If no carry and borrow, keep r (r was < p)
    // Combined: use_tmp if (carry OR (borrow == 0))
    // mask = all-ones if we should use tmp
    std::uint64_t no_borrow = is_zero_mask(borrow);
    std::uint64_t has_carry = is_nonzero_mask(carry);
    std::uint64_t mask = no_borrow | has_carry;
    cmov256(r, tmp, mask);

    return FieldElement::from_limbs_raw({r[0], r[1], r[2], r[3]});
}

FieldElement field_sub(const FieldElement& a, const FieldElement& b) noexcept {
    std::uint64_t r[4];
    std::uint64_t borrow = sub256(r, a.limbs().data(), b.limbs().data());

    // If borrow, add p back: r += p
    std::uint64_t tmp[4];
    add256(tmp, r, P);

    // mask = all-ones if borrow occurred
    std::uint64_t mask = is_nonzero_mask(borrow);
    cmov256(r, tmp, mask);

    return FieldElement::from_limbs_raw({r[0], r[1], r[2], r[3]});
}

FieldElement field_mul(const FieldElement& a, const FieldElement& b) noexcept {
#if SECP256K1_CT_HAS_DIRECT_ASM
    // Direct ASM: field_mul_full_asm does 4×4 schoolbook multiplication
    // + secp256k1 reduction + branchless normalization (cmovc) in one call.
    // Output is always in [0, p) — no extra normalize needed.
    // Bypasses operator* wrapper chain: saves ~15ns call/memcpy overhead
    // + ~20ns redundant field_normalize = ~35ns savings per mul.
    std::uint64_t result[4];
    field_mul_full_asm(a.limbs().data(), b.limbs().data(), result);
    return FieldElement::from_limbs_raw({result[0], result[1], result[2], result[3]});
#else
    // Fallback: use operator* (platform-specific mul) + CT normalize.
    // operator* always produces a value in [0, p) but the reduce step
    // may use data-dependent loops, so we CT-normalize afterward.
    FieldElement r = a * b;
    return field_normalize(r);
#endif
}

FieldElement field_sqr(const FieldElement& a) noexcept {
#if SECP256K1_CT_HAS_DIRECT_ASM
    // Direct ASM: fused squaring + reduction + branchless normalization.
    // Output always in [0, p). No extra normalize needed.
    std::uint64_t result[4];
    field_sqr_full_asm(a.limbs().data(), result);
    return FieldElement::from_limbs_raw({result[0], result[1], result[2], result[3]});
#else
    FieldElement r = a.square();
    return field_normalize(r);
#endif
}

FieldElement field_neg(const FieldElement& a) noexcept {
    // -a mod p = p - a (if a != 0), 0 (if a == 0)
    // CT: always compute p - a, then cmov to 0 if a was zero
    std::uint64_t r[4];
    sub256(r, P, a.limbs().data());

    std::uint64_t zero_mask = field_is_zero(a);
    // If a == 0, set r to 0
    std::uint64_t z[4] = {0, 0, 0, 0};
    cmov256(r, z, zero_mask);

    return FieldElement::from_limbs_raw({r[0], r[1], r[2], r[3]});
}

FieldElement field_inv(const FieldElement& a) noexcept {
    // Fermat's little theorem: a^(p-2) mod p
    // Using addition chain optimized for secp256k1:
    // p-2 = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2D
    //
    // This is a fixed sequence of squarings and multiplications.
    // Always the same number of operations regardless of input.
    //
    // Optimal addition chain:
    // 1. Compute a² ... a^(2^k) powers via repeated squaring
    // 2. Multiply specific powers together
    // Total: 255 squarings + 14 multiplications (fixed)

    FieldElement x = a;

    // x2 = a^(2^2 - 1) = a^3
    FieldElement x2 = field_sqr(x);
    x2 = field_mul(x2, x);

    // x3 = a^(2^3 - 1) = a^7
    FieldElement x3 = field_sqr(x2);
    x3 = field_mul(x3, x);

    // x6 = a^(2^6 - 1)
    FieldElement x6 = x3;
    for (int i = 0; i < 3; ++i) x6 = field_sqr(x6);
    x6 = field_mul(x6, x3);

    // x9 = a^(2^9 - 1)
    FieldElement x9 = x6;
    for (int i = 0; i < 3; ++i) x9 = field_sqr(x9);
    x9 = field_mul(x9, x3);

    // x11 = a^(2^11 - 1)
    FieldElement x11 = x9;
    for (int i = 0; i < 2; ++i) x11 = field_sqr(x11);
    x11 = field_mul(x11, x2);

    // x22 = a^(2^22 - 1)
    FieldElement x22 = x11;
    for (int i = 0; i < 11; ++i) x22 = field_sqr(x22);
    x22 = field_mul(x22, x11);

    // x44 = a^(2^44 - 1)
    FieldElement x44 = x22;
    for (int i = 0; i < 22; ++i) x44 = field_sqr(x44);
    x44 = field_mul(x44, x22);

    // x88 = a^(2^88 - 1)
    FieldElement x88 = x44;
    for (int i = 0; i < 44; ++i) x88 = field_sqr(x88);
    x88 = field_mul(x88, x44);

    // x176 = a^(2^176 - 1)
    FieldElement x176 = x88;
    for (int i = 0; i < 88; ++i) x176 = field_sqr(x176);
    x176 = field_mul(x176, x88);

    // x220 = a^(2^220 - 1)
    FieldElement x220 = x176;
    for (int i = 0; i < 44; ++i) x220 = field_sqr(x220);
    x220 = field_mul(x220, x44);

    // x223 = a^(2^223 - 1)
    FieldElement x223 = x220;
    for (int i = 0; i < 3; ++i) x223 = field_sqr(x223);
    x223 = field_mul(x223, x3);

    // Final: t = x223^(2^23) * x22^(2^1) * ...
    // p-2 = 2^256 - 2^32 - 977 - 2
    //      = 2^256 - 0x1000003D3
    // Exponent in binary from bit 255 down:
    //   bits 255..33: all ones (223 ones = x223 covers bits 255..33)
    //   bit 32: 0
    //   bits 31..0: FFFFFC2D = ...11111100 00101101
    //
    // After x223 (covers bits 255..33):
    //   Square 23 times → x223 * 2^23, then multiply by x22
    //   Square 5 times → then multiply by a
    //   Square 3 times → then multiply by x2
    //   Square 2 times → then multiply by a

    FieldElement t = x223;

    // Square 23 times
    for (int i = 0; i < 23; ++i) t = field_sqr(t);
    t = field_mul(t, x22);

    // Square 5 times
    for (int i = 0; i < 5; ++i) t = field_sqr(t);
    t = field_mul(t, x);

    // Square 3 times
    for (int i = 0; i < 3; ++i) t = field_sqr(t);
    t = field_mul(t, x2);

    // Square 2 times
    for (int i = 0; i < 2; ++i) t = field_sqr(t);
    t = field_mul(t, x);

    return t;
}

// ─── Conditional Operations ──────────────────────────────────────────────────

void field_cmov(FieldElement* r, const FieldElement& a,
                std::uint64_t mask) noexcept {
    // Avoid const_cast UB: use field_select + assignment
    *r = field_select(a, *r, mask);
}

void field_cswap(FieldElement* a, FieldElement* b,
                 std::uint64_t mask) noexcept {
    FieldElement old_a = *a;
    FieldElement old_b = *b;
    *a = field_select(old_b, old_a, mask);
    *b = field_select(old_a, old_b, mask);
}

FieldElement field_select(const FieldElement& a, const FieldElement& b,
                          std::uint64_t mask) noexcept {
    const auto& al = a.limbs();
    const auto& bl = b.limbs();
    return FieldElement::from_limbs_raw({
        ct_select(al[0], bl[0], mask),
        ct_select(al[1], bl[1], mask),
        ct_select(al[2], bl[2], mask),
        ct_select(al[3], bl[3], mask)
    });
}

FieldElement field_cneg(const FieldElement& a, std::uint64_t mask) noexcept {
    FieldElement neg = field_neg(a);
    return field_select(neg, a, mask);
}

// ─── Comparison ──────────────────────────────────────────────────────────────

std::uint64_t field_is_zero(const FieldElement& a) noexcept {
    const auto& l = a.limbs();
    std::uint64_t z = l[0] | l[1] | l[2] | l[3];
    return is_zero_mask(z);
}

std::uint64_t field_eq(const FieldElement& a, const FieldElement& b) noexcept {
    const auto& al = a.limbs();
    const auto& bl = b.limbs();
    std::uint64_t diff = (al[0] ^ bl[0]) | (al[1] ^ bl[1]) |
                         (al[2] ^ bl[2]) | (al[3] ^ bl[3]);
    return is_zero_mask(diff);
}

} // namespace secp256k1::ct
