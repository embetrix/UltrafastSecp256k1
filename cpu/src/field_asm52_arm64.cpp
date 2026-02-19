// ═══════════════════════════════════════════════════════════════════════════
// 5×52 Field Arithmetic — ARM64 (AArch64) Inline Assembly
// ═══════════════════════════════════════════════════════════════════════════
//
// Optimized field multiplication and squaring using ARM64 MUL/UMULH
// instructions for 64×64→128-bit products.
//
// ARM64 has 31 GPRs, so register pressure is not an issue.
// The approach uses MUL for low half, UMULH for high half, and
// ADDS/ADC pairs for 128-bit accumulation.
//
// Required: AArch64 (ARMv8-A or later)
// ═══════════════════════════════════════════════════════════════════════════

#if defined(__aarch64__) || defined(_M_ARM64)

#include "secp256k1/field_52.hpp"
#include <cstdint>

namespace secp256k1::fast {

// Constants
static constexpr uint64_t FE52_M  = 0xFFFFFFFFFFFFFULL;
static constexpr uint64_t FE52_R  = 0x1000003D10ULL;
static constexpr uint64_t FE52_R4 = 0x1000003D1ULL;     // R >> 4
static constexpr uint64_t FE52_R12= 0x1000003D10000ULL;  // R << 12

// ── Inline assembly helper: 128-bit multiply-accumulate ──────────────────
// (d_hi:d_lo) += a * b
#define MULACCUM128(d_lo, d_hi, a_reg, b_reg, t_lo, t_hi)       \
    __asm__ volatile(                                             \
        "mul  %[tl], %[ar], %[br]  \n\t"                        \
        "umulh %[th], %[ar], %[br] \n\t"                        \
        "adds %[dl], %[dl], %[tl]  \n\t"                        \
        "adc  %[dh], %[dh], %[th]  \n\t"                        \
        : [dl] "+r"(d_lo), [dh] "+r"(d_hi),                     \
          [tl] "=&r"(t_lo), [th] "=&r"(t_hi)                    \
        : [ar] "r"(a_reg), [br] "r"(b_reg)                      \
        : "cc"                                                   \
    )

// (d_hi:d_lo) = a * b  (initial product, no accumulate)
#define MULPROD128(d_lo, d_hi, a_reg, b_reg)                     \
    __asm__ volatile(                                             \
        "mul  %[dl], %[ar], %[br]  \n\t"                        \
        "umulh %[dh], %[ar], %[br] \n\t"                        \
        : [dl] "=r"(d_lo), [dh] "=r"(d_hi)                      \
        : [ar] "r"(a_reg), [br] "r"(b_reg)                      \
    )

extern "C"
void fe52_mul_inner_arm64(uint64_t* __restrict r,
                          const uint64_t* __restrict a,
                          const uint64_t* __restrict b) noexcept {
    const uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    uint64_t d_lo, d_hi, c_lo, c_hi;
    uint64_t t3, t4, tx, u0;
    uint64_t tmp_lo, tmp_hi;
    const uint64_t M = FE52_M;
    const uint64_t R = FE52_R;

    // ── Step 1: d = a0*b3 + a1*b2 + a2*b1 + a3*b0 ──────────────────
    MULPROD128(d_lo, d_hi, a0, b[3]);
    MULACCUM128(d_lo, d_hi, a1, b[2], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a2, b[1], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a3, b[0], tmp_lo, tmp_hi);

    // c = a4 * b4
    MULPROD128(c_lo, c_hi, a4, b[4]);

    // d += R * c_lo
    MULACCUM128(d_lo, d_hi, R, c_lo, tmp_lo, tmp_hi);

    // c >>= 64 → c = c_hi
    c_lo = c_hi;

    // t3 = d_lo & M;  d >>= 52
    t3 = d_lo & M;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;

    // ── Step 2: d += a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 ────────
    MULACCUM128(d_lo, d_hi, a0, b[4], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a1, b[3], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a2, b[2], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a3, b[1], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a4, b[0], tmp_lo, tmp_hi);

    // d += (R<<12) * c_remaining
    MULACCUM128(d_lo, d_hi, FE52_R12, c_lo, tmp_lo, tmp_hi);

    // t4 = d & M;  d >>= 52;  tx = t4>>48;  t4 &= M>>4
    t4 = d_lo & M;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;
    tx = t4 >> 48;
    t4 &= (M >> 4);

    // ── Step 3: col0 + col5 ────────────────────────────────────────
    MULACCUM128(d_lo, d_hi, a1, b[4], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a2, b[3], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a3, b[2], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a4, b[1], tmp_lo, tmp_hi);

    u0 = d_lo & M;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;
    u0 = (u0 << 4) | tx;

    MULPROD128(c_lo, c_hi, a0, b[0]);
    MULACCUM128(c_lo, c_hi, u0, FE52_R4, tmp_lo, tmp_hi);

    r[0] = c_lo & M;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi = c_hi >> 52;

    // ── Step 4: col1 + col6 ────────────────────────────────────────
    MULACCUM128(c_lo, c_hi, a0, b[1], tmp_lo, tmp_hi);
    MULACCUM128(c_lo, c_hi, a1, b[0], tmp_lo, tmp_hi);

    MULACCUM128(d_lo, d_hi, a2, b[4], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a3, b[3], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a4, b[2], tmp_lo, tmp_hi);

    u0 = d_lo & M;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;
    MULACCUM128(c_lo, c_hi, u0, R, tmp_lo, tmp_hi);

    r[1] = c_lo & M;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi = c_hi >> 52;

    // ── Step 5: col2 + col7 ────────────────────────────────────────
    MULACCUM128(c_lo, c_hi, a0, b[2], tmp_lo, tmp_hi);
    MULACCUM128(c_lo, c_hi, a1, b[1], tmp_lo, tmp_hi);
    MULACCUM128(c_lo, c_hi, a2, b[0], tmp_lo, tmp_hi);

    MULACCUM128(d_lo, d_hi, a3, b[4], tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a4, b[3], tmp_lo, tmp_hi);

    // c += R * (uint64)d  — full 64-bit d_lo!
    MULACCUM128(c_lo, c_hi, R, d_lo, tmp_lo, tmp_hi);

    // d >>= 64
    d_lo = d_hi;
    d_hi = 0;

    r[2] = c_lo & M;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi = c_hi >> 52;

    // ── Step 6: Finalize ───────────────────────────────────────────
    MULACCUM128(c_lo, c_hi, FE52_R12, d_lo, tmp_lo, tmp_hi);

    // c += t3
    __asm__ volatile("adds %[cl], %[cl], %[t]\n\t"
                     "adc  %[ch], %[ch], xzr\n\t"
                     : [cl] "+r"(c_lo), [ch] "+r"(c_hi) : [t] "r"(t3) : "cc");

    r[3] = c_lo & M;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi = c_hi >> 52;

    c_lo += t4;
    r[4] = c_lo;
}

extern "C"
void fe52_sqr_inner_arm64(uint64_t* __restrict r,
                          const uint64_t* __restrict a) noexcept {
    const uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const uint64_t a0_2 = a0 * 2, a1_2 = a1 * 2, a2_2 = a2 * 2, a3_2 = a3 * 2;
    uint64_t d_lo, d_hi, c_lo, c_hi;
    uint64_t t3, t4, tx, u0;
    uint64_t tmp_lo, tmp_hi;
    const uint64_t M = FE52_M;
    const uint64_t R = FE52_R;

    // ── Step 1 ──
    MULPROD128(d_lo, d_hi, a0_2, a3);
    MULACCUM128(d_lo, d_hi, a1_2, a2, tmp_lo, tmp_hi);
    MULPROD128(c_lo, c_hi, a4, a4);
    MULACCUM128(d_lo, d_hi, R, c_lo, tmp_lo, tmp_hi);
    c_lo = c_hi;
    t3 = d_lo & M;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi >>= 52;

    // ── Step 2 ──
    MULACCUM128(d_lo, d_hi, a0_2, a4, tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a1_2, a3, tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a2, a2, tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, FE52_R12, c_lo, tmp_lo, tmp_hi);
    t4 = d_lo & M;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi >>= 52;
    tx = t4 >> 48;
    t4 &= (M >> 4);

    // ── Step 3 ──
    MULACCUM128(d_lo, d_hi, a1_2, a4, tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a2_2, a3, tmp_lo, tmp_hi);
    u0 = d_lo & M;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi >>= 52;
    u0 = (u0 << 4) | tx;
    MULPROD128(c_lo, c_hi, a0, a0);
    MULACCUM128(c_lo, c_hi, u0, FE52_R4, tmp_lo, tmp_hi);
    r[0] = c_lo & M;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi >>= 52;

    // ── Step 4 ──
    MULACCUM128(c_lo, c_hi, a0_2, a1, tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a2_2, a4, tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a3, a3, tmp_lo, tmp_hi);
    u0 = d_lo & M;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi >>= 52;
    MULACCUM128(c_lo, c_hi, u0, R, tmp_lo, tmp_hi);
    r[1] = c_lo & M;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi >>= 52;

    // ── Step 5 ──
    MULACCUM128(c_lo, c_hi, a0_2, a2, tmp_lo, tmp_hi);
    MULACCUM128(c_lo, c_hi, a1, a1, tmp_lo, tmp_hi);
    MULACCUM128(d_lo, d_hi, a3_2, a4, tmp_lo, tmp_hi);
    MULACCUM128(c_lo, c_hi, R, d_lo, tmp_lo, tmp_hi);
    d_lo = d_hi;
    d_hi = 0;
    r[2] = c_lo & M;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi >>= 52;

    // ── Step 6 ──
    MULACCUM128(c_lo, c_hi, FE52_R12, d_lo, tmp_lo, tmp_hi);
    __asm__ volatile("adds %[cl], %[cl], %[t]\n\t"
                     "adc  %[ch], %[ch], xzr\n\t"
                     : [cl] "+r"(c_lo), [ch] "+r"(c_hi) : [t] "r"(t3) : "cc");
    r[3] = c_lo & M;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi >>= 52;
    c_lo += t4;
    r[4] = c_lo;
}

#undef MULACCUM128
#undef MULPROD128

} // namespace secp256k1::fast

#endif // __aarch64__ || _M_ARM64
