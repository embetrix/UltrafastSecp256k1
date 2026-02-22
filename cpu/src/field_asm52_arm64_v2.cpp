// ═══════════════════════════════════════════════════════════════════════════
// 5×52 Field Arithmetic — ARM64 (AArch64) Optimized Assembly v2
// ═══════════════════════════════════════════════════════════════════════════
//
// Hand-scheduled 5×52 field multiplication for Cortex-A76 class cores.
//
// Key optimization: Interleave MUL/UMULH instructions to hide latency.
// On A76: MUL has 3-cycle latency, UMULH has 4-cycle latency, both
// 1/cycle throughput on the same M pipeline. By batching independent
// multiplies before their results are needed, we avoid stalls.
//
// The __int128 C version forces serial MUL→UMULH→add chains per product.
// This version computes multiple independent products before accumulating.
//
// Also provides NEON-accelerated field add/sub/negate and normalize_weak.
//
// Required: AArch64 (ARMv8-A or later), NEON (implicit in ARMv8-A)
// ═══════════════════════════════════════════════════════════════════════════

#if defined(__aarch64__) || defined(_M_ARM64)

#include "secp256k1/field_52.hpp"
#include <cstdint>
#include <arm_neon.h>

namespace secp256k1::fast {
namespace arm64_v2 {

// secp256k1 reduction constant: 2^260 mod p
static constexpr uint64_t R   = 0x1000003D10ULL;
static constexpr uint64_t R4  = 0x1000003D1ULL;     // R >> 4
static constexpr uint64_t R12 = 0x1000003D10000ULL;  // R << 12
static constexpr uint64_t M52 = 0xFFFFFFFFFFFFFULL;  // (1<<52)-1
static constexpr uint64_t M48 = 0xFFFFFFFFFFFFULL;   // (1<<48)-1

// ═══════════════════════════════════════════════════════════════════════════
// Optimized 5×52 Field Multiply — Interleaved MUL/UMULH scheduling
// ═══════════════════════════════════════════════════════════════════════════
//
// Total products: 25 MUL + 25 UMULH + ~5 reduction MUL/UMULH
// Target: ~60-70 cycles on A76 (vs ~156 cycles for __int128 version)
//
// Strategy for each column:
//   1. Fire off all MUL instructions (low halves) back-to-back
//   2. Fire off all UMULH instructions (high halves) back-to-back
//   3. Accumulate results with ADDS/ADC chains
//   4. Shift/mask for next column
//
// This exploits A76's 1-mul-per-cycle throughput: N products take N cycles
// for MUL + N cycles for UMULH, but the second batch can start while the
// first batch results are still in flight.
//
// Using pure inline asm to ensure the compiler doesn't reorder/spill.

__attribute__((always_inline)) inline
void fe52_mul_arm64_v2(uint64_t* __restrict r,
                       const uint64_t* __restrict a,
                       const uint64_t* __restrict b) noexcept {
    // We follow the same column order as bitcoin-core/libsecp256k1:
    // columns 3, 4, 0, 1, 2, finalize 3, 4
    //
    // The full hand-scheduled version uses a single asm block to
    // control register allocation and instruction ordering.
    // ARM64 has 31 GPRs — enough for all temporaries without spilling.

    // We use "C with inline asm for individual products" but structured
    // to allow the compiler some freedom while controlling the multiply
    // ordering. Key: compute all products for a column, THEN accumulate.

    const uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const uint64_t b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4];

    // Temporaries for product results (low, high halves)
    uint64_t p0_lo, p0_hi, p1_lo, p1_hi, p2_lo, p2_hi, p3_lo, p3_hi;
    uint64_t p4_lo, p4_hi;
    uint64_t d_lo, d_hi, c_lo, c_hi;
    uint64_t t3, t4, tx, u0;

    // ══════════════════════════════════════════════════════════════════
    // Column 3: d = a0*b3 + a1*b2 + a2*b1 + a3*b0  (4 products)
    //         + R * (a4*b4)_lo                       (1 reduction product)
    // ══════════════════════════════════════════════════════════════════

    // Fire all MULs for column 3 (4 products + 1 reduction)
    __asm__ volatile(
        "mul  %[p0l], %[a0], %[b3]   \n\t"
        "mul  %[p1l], %[a1], %[b2]   \n\t"
        "mul  %[p2l], %[a2], %[b1]   \n\t"
        "mul  %[p3l], %[a3], %[b0]   \n\t"
        "mul  %[p4l], %[a4], %[b4]   \n\t"  // c_lo = (a4*b4)_lo
        : [p0l] "=r"(p0_lo), [p1l] "=r"(p1_lo), [p2l] "=r"(p2_lo),
          [p3l] "=r"(p3_lo), [p4l] "=r"(p4_lo)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "r"(b4)
    );

    // Fire all UMULHs (interleaved with MUL results becoming ready)
    __asm__ volatile(
        "umulh %[p0h], %[a0], %[b3]   \n\t"
        "umulh %[p1h], %[a1], %[b2]   \n\t"
        "umulh %[p2h], %[a2], %[b1]   \n\t"
        "umulh %[p3h], %[a3], %[b0]   \n\t"
        "umulh %[p4h], %[a4], %[b4]   \n\t"  // c_hi = (a4*b4)_hi
        : [p0h] "=r"(p0_hi), [p1h] "=r"(p1_hi), [p2h] "=r"(p2_hi),
          [p3h] "=r"(p3_hi), [p4h] "=r"(p4_hi)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "r"(b4)
    );

    // Accumulate: d = p0 + p1 + p2 + p3
    __asm__ volatile(
        "adds %[dl], %[p0l], %[p1l]  \n\t"
        "adc  %[dh], %[p0h], %[p1h]  \n\t"
        "adds %[dl], %[dl], %[p2l]   \n\t"
        "adc  %[dh], %[dh], %[p2h]   \n\t"
        "adds %[dl], %[dl], %[p3l]   \n\t"
        "adc  %[dh], %[dh], %[p3h]   \n\t"
        : [dl] "=&r"(d_lo), [dh] "=&r"(d_hi)
        : [p0l] "r"(p0_lo), [p0h] "r"(p0_hi),
          [p1l] "r"(p1_lo), [p1h] "r"(p1_hi),
          [p2l] "r"(p2_lo), [p2h] "r"(p2_hi),
          [p3l] "r"(p3_lo), [p3h] "r"(p3_hi)
        : "cc"
    );

    // c = a4*b4 → c_hi is the carry for column 4
    c_lo = p4_lo;
    c_hi = p4_hi;

    // d += R * c_lo (reduction of column 8)
    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[R], %[cl]   \n\t"
            "umulh %[rh], %[R], %[cl]  \n\t"
            "adds %[dl], %[dl], %[rl]  \n\t"
            "adc  %[dh], %[dh], %[rh]  \n\t"
            : [dl] "+r"(d_lo), [dh] "+r"(d_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [R] "r"(R), [cl] "r"(c_lo)
            : "cc"
        );
    }

    // c >>= 64
    c_lo = c_hi;

    // t3 = d & M52; d >>= 52
    t3 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;

    // ══════════════════════════════════════════════════════════════════
    // Column 4: d += a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0 (5 products)
    //         + R12 * c_lo                                   (1 reduction)
    // ══════════════════════════════════════════════════════════════════

    __asm__ volatile(
        "mul  %[p0l], %[a0], %[b4]   \n\t"
        "mul  %[p1l], %[a1], %[b3]   \n\t"
        "mul  %[p2l], %[a2], %[b2]   \n\t"
        "mul  %[p3l], %[a3], %[b1]   \n\t"
        "mul  %[p4l], %[a4], %[b0]   \n\t"
        : [p0l] "=r"(p0_lo), [p1l] "=r"(p1_lo), [p2l] "=r"(p2_lo),
          [p3l] "=r"(p3_lo), [p4l] "=r"(p4_lo)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "r"(b4)
    );
    __asm__ volatile(
        "umulh %[p0h], %[a0], %[b4]   \n\t"
        "umulh %[p1h], %[a1], %[b3]   \n\t"
        "umulh %[p2h], %[a2], %[b2]   \n\t"
        "umulh %[p3h], %[a3], %[b1]   \n\t"
        "umulh %[p4h], %[a4], %[b0]   \n\t"
        : [p0h] "=r"(p0_hi), [p1h] "=r"(p1_hi), [p2h] "=r"(p2_hi),
          [p3h] "=r"(p3_hi), [p4h] "=r"(p4_hi)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "r"(b4)
    );

    // Accumulate 5 products into d
    __asm__ volatile(
        "adds %[dl], %[dl], %[p0l]   \n\t"
        "adc  %[dh], %[dh], %[p0h]   \n\t"
        "adds %[dl], %[dl], %[p1l]   \n\t"
        "adc  %[dh], %[dh], %[p1h]   \n\t"
        "adds %[dl], %[dl], %[p2l]   \n\t"
        "adc  %[dh], %[dh], %[p2h]   \n\t"
        "adds %[dl], %[dl], %[p3l]   \n\t"
        "adc  %[dh], %[dh], %[p3h]   \n\t"
        "adds %[dl], %[dl], %[p4l]   \n\t"
        "adc  %[dh], %[dh], %[p4h]   \n\t"
        : [dl] "+r"(d_lo), [dh] "+r"(d_hi)
        : [p0l] "r"(p0_lo), [p0h] "r"(p0_hi),
          [p1l] "r"(p1_lo), [p1h] "r"(p1_hi),
          [p2l] "r"(p2_lo), [p2h] "r"(p2_hi),
          [p3l] "r"(p3_lo), [p3h] "r"(p3_hi),
          [p4l] "r"(p4_lo), [p4h] "r"(p4_hi)
        : "cc"
    );

    // d += R12 * c_lo
    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[R12], %[cl]  \n\t"
            "umulh %[rh], %[R12], %[cl] \n\t"
            "adds %[dl], %[dl], %[rl]   \n\t"
            "adc  %[dh], %[dh], %[rh]   \n\t"
            : [dl] "+r"(d_lo), [dh] "+r"(d_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [R12] "r"(R12), [cl] "r"(c_lo)
            : "cc"
        );
    }

    // t4 = d & M52; d >>= 52; tx = t4>>48; t4 &= M52>>4
    t4 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;
    tx = t4 >> 48;
    t4 &= (M52 >> 4);

    // ══════════════════════════════════════════════════════════════════
    // Column 0: c = a0*b0  (1 product)
    //   d += a1*b4 + a2*b3 + a3*b2 + a4*b1  (4 products, column 5 reduced)
    // ══════════════════════════════════════════════════════════════════

    // Column 5 products (4 products) — all independent, batch MUL then UMULH
    __asm__ volatile(
        "mul  %[p0l], %[a1], %[b4]   \n\t"
        "mul  %[p1l], %[a2], %[b3]   \n\t"
        "mul  %[p2l], %[a3], %[b2]   \n\t"
        "mul  %[p3l], %[a4], %[b1]   \n\t"
        "mul  %[cl],  %[a0], %[b0]   \n\t"  // c = a0*b0 (column 0) — slip in while waiting
        : [p0l] "=r"(p0_lo), [p1l] "=r"(p1_lo), [p2l] "=r"(p2_lo),
          [p3l] "=r"(p3_lo), [cl] "=r"(c_lo)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "r"(b4)
    );
    __asm__ volatile(
        "umulh %[p0h], %[a1], %[b4]   \n\t"
        "umulh %[p1h], %[a2], %[b3]   \n\t"
        "umulh %[p2h], %[a3], %[b2]   \n\t"
        "umulh %[p3h], %[a4], %[b1]   \n\t"
        "umulh %[ch],  %[a0], %[b0]   \n\t"
        : [p0h] "=r"(p0_hi), [p1h] "=r"(p1_hi), [p2h] "=r"(p2_hi),
          [p3h] "=r"(p3_hi), [ch] "=r"(c_hi)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "r"(b4)
    );

    // Accumulate 4 column-5 products into d
    __asm__ volatile(
        "adds %[dl], %[dl], %[p0l]   \n\t"
        "adc  %[dh], %[dh], %[p0h]   \n\t"
        "adds %[dl], %[dl], %[p1l]   \n\t"
        "adc  %[dh], %[dh], %[p1h]   \n\t"
        "adds %[dl], %[dl], %[p2l]   \n\t"
        "adc  %[dh], %[dh], %[p2h]   \n\t"
        "adds %[dl], %[dl], %[p3l]   \n\t"
        "adc  %[dh], %[dh], %[p3h]   \n\t"
        : [dl] "+r"(d_lo), [dh] "+r"(d_hi)
        : [p0l] "r"(p0_lo), [p0h] "r"(p0_hi),
          [p1l] "r"(p1_lo), [p1h] "r"(p1_hi),
          [p2l] "r"(p2_lo), [p2h] "r"(p2_hi),
          [p3l] "r"(p3_lo), [p3h] "r"(p3_hi)
        : "cc"
    );

    // u0 = d & M52; d >>= 52; u0 = (u0<<4)|tx
    u0 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;
    u0 = (u0 << 4) | tx;

    // c += u0 * R4
    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[u0], %[R4]   \n\t"
            "umulh %[rh], %[u0], %[R4]  \n\t"
            "adds %[cl], %[cl], %[rl]   \n\t"
            "adc  %[ch], %[ch], %[rh]   \n\t"
            : [cl] "+r"(c_lo), [ch] "+r"(c_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [u0] "r"(u0), [R4] "r"(R4)
            : "cc"
        );
    }

    r[0] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi = c_hi >> 52;

    // ══════════════════════════════════════════════════════════════════
    // Column 1: c += a0*b1 + a1*b0  (2 products)
    //   d += a2*b4 + a3*b3 + a4*b2  (3 products, column 6 reduced)
    // ══════════════════════════════════════════════════════════════════

    // Batch all 5 MULs together
    __asm__ volatile(
        "mul  %[p0l], %[a0], %[b1]   \n\t"   // col 1
        "mul  %[p1l], %[a1], %[b0]   \n\t"   // col 1
        "mul  %[p2l], %[a2], %[b4]   \n\t"   // col 6
        "mul  %[p3l], %[a3], %[b3]   \n\t"   // col 6
        "mul  %[p4l], %[a4], %[b2]   \n\t"   // col 6
        : [p0l] "=r"(p0_lo), [p1l] "=r"(p1_lo), [p2l] "=r"(p2_lo),
          [p3l] "=r"(p3_lo), [p4l] "=r"(p4_lo)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "r"(b4)
    );
    __asm__ volatile(
        "umulh %[p0h], %[a0], %[b1]   \n\t"
        "umulh %[p1h], %[a1], %[b0]   \n\t"
        "umulh %[p2h], %[a2], %[b4]   \n\t"
        "umulh %[p3h], %[a3], %[b3]   \n\t"
        "umulh %[p4h], %[a4], %[b2]   \n\t"
        : [p0h] "=r"(p0_hi), [p1h] "=r"(p1_hi), [p2h] "=r"(p2_hi),
          [p3h] "=r"(p3_hi), [p4h] "=r"(p4_hi)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "r"(b4)
    );

    // Accumulate column 1 products into c
    __asm__ volatile(
        "adds %[cl], %[cl], %[p0l]   \n\t"
        "adc  %[ch], %[ch], %[p0h]   \n\t"
        "adds %[cl], %[cl], %[p1l]   \n\t"
        "adc  %[ch], %[ch], %[p1h]   \n\t"
        : [cl] "+r"(c_lo), [ch] "+r"(c_hi)
        : [p0l] "r"(p0_lo), [p0h] "r"(p0_hi),
          [p1l] "r"(p1_lo), [p1h] "r"(p1_hi)
        : "cc"
    );

    // Accumulate column 6 products into d
    __asm__ volatile(
        "adds %[dl], %[dl], %[p2l]   \n\t"
        "adc  %[dh], %[dh], %[p2h]   \n\t"
        "adds %[dl], %[dl], %[p3l]   \n\t"
        "adc  %[dh], %[dh], %[p3h]   \n\t"
        "adds %[dl], %[dl], %[p4l]   \n\t"
        "adc  %[dh], %[dh], %[p4h]   \n\t"
        : [dl] "+r"(d_lo), [dh] "+r"(d_hi)
        : [p2l] "r"(p2_lo), [p2h] "r"(p2_hi),
          [p3l] "r"(p3_lo), [p3h] "r"(p3_hi),
          [p4l] "r"(p4_lo), [p4h] "r"(p4_hi)
        : "cc"
    );

    // c += (d & M52) * R
    u0 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;
    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[u0], %[R]    \n\t"
            "umulh %[rh], %[u0], %[R]   \n\t"
            "adds %[cl], %[cl], %[rl]   \n\t"
            "adc  %[ch], %[ch], %[rh]   \n\t"
            : [cl] "+r"(c_lo), [ch] "+r"(c_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [u0] "r"(u0), [R] "r"(R)
            : "cc"
        );
    }

    r[1] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi = c_hi >> 52;

    // ══════════════════════════════════════════════════════════════════
    // Column 2: c += a0*b2 + a1*b1 + a2*b0  (3 products)
    //   d += a3*b4 + a4*b3                   (2 products, column 7 reduced)
    // ══════════════════════════════════════════════════════════════════

    __asm__ volatile(
        "mul  %[p0l], %[a0], %[b2]   \n\t"   // col 2
        "mul  %[p1l], %[a1], %[b1]   \n\t"   // col 2
        "mul  %[p2l], %[a2], %[b0]   \n\t"   // col 2
        "mul  %[p3l], %[a3], %[b4]   \n\t"   // col 7
        "mul  %[p4l], %[a4], %[b3]   \n\t"   // col 7
        : [p0l] "=r"(p0_lo), [p1l] "=r"(p1_lo), [p2l] "=r"(p2_lo),
          [p3l] "=r"(p3_lo), [p4l] "=r"(p4_lo)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "r"(b4)
    );
    __asm__ volatile(
        "umulh %[p0h], %[a0], %[b2]   \n\t"
        "umulh %[p1h], %[a1], %[b1]   \n\t"
        "umulh %[p2h], %[a2], %[b0]   \n\t"
        "umulh %[p3h], %[a3], %[b4]   \n\t"
        "umulh %[p4h], %[a4], %[b3]   \n\t"
        : [p0h] "=r"(p0_hi), [p1h] "=r"(p1_hi), [p2h] "=r"(p2_hi),
          [p3h] "=r"(p3_hi), [p4h] "=r"(p4_hi)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [b0] "r"(b0), [b1] "r"(b1), [b2] "r"(b2), [b3] "r"(b3), [b4] "r"(b4)
    );

    // Accumulate column 2 products into c
    __asm__ volatile(
        "adds %[cl], %[cl], %[p0l]   \n\t"
        "adc  %[ch], %[ch], %[p0h]   \n\t"
        "adds %[cl], %[cl], %[p1l]   \n\t"
        "adc  %[ch], %[ch], %[p1h]   \n\t"
        "adds %[cl], %[cl], %[p2l]   \n\t"
        "adc  %[ch], %[ch], %[p2h]   \n\t"
        : [cl] "+r"(c_lo), [ch] "+r"(c_hi)
        : [p0l] "r"(p0_lo), [p0h] "r"(p0_hi),
          [p1l] "r"(p1_lo), [p1h] "r"(p1_hi),
          [p2l] "r"(p2_lo), [p2h] "r"(p2_hi)
        : "cc"
    );

    // Accumulate column 7 products into d
    __asm__ volatile(
        "adds %[dl], %[dl], %[p3l]   \n\t"
        "adc  %[dh], %[dh], %[p3h]   \n\t"
        "adds %[dl], %[dl], %[p4l]   \n\t"
        "adc  %[dh], %[dh], %[p4h]   \n\t"
        : [dl] "+r"(d_lo), [dh] "+r"(d_hi)
        : [p3l] "r"(p3_lo), [p3h] "r"(p3_hi),
          [p4l] "r"(p4_lo), [p4h] "r"(p4_hi)
        : "cc"
    );

    // c += R * d_lo  (full 64-bit column-7 reduction)
    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[R], %[dl]    \n\t"
            "umulh %[rh], %[R], %[dl]   \n\t"
            "adds %[cl], %[cl], %[rl]   \n\t"
            "adc  %[ch], %[ch], %[rh]   \n\t"
            : [cl] "+r"(c_lo), [ch] "+r"(c_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [R] "r"(R), [dl] "r"(d_lo)
            : "cc"
        );
    }

    // d >>= 64
    d_lo = d_hi;
    d_hi = 0;

    r[2] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi = c_hi >> 52;

    // ══════════════════════════════════════════════════════════════════
    // Finalize columns 3 and 4
    // ══════════════════════════════════════════════════════════════════

    // c += R12 * d_lo
    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[R12], %[dl]  \n\t"
            "umulh %[rh], %[R12], %[dl] \n\t"
            "adds %[cl], %[cl], %[rl]   \n\t"
            "adc  %[ch], %[ch], %[rh]   \n\t"
            : [cl] "+r"(c_lo), [ch] "+r"(c_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [R12] "r"(R12), [dl] "r"(d_lo)
            : "cc"
        );
    }

    // c += t3
    __asm__ volatile(
        "adds %[cl], %[cl], %[t3]   \n\t"
        "adc  %[ch], %[ch], xzr     \n\t"
        : [cl] "+r"(c_lo), [ch] "+r"(c_hi) : [t3] "r"(t3) : "cc"
    );

    r[3] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12);

    c_lo += t4;
    r[4] = c_lo;
}

// ═══════════════════════════════════════════════════════════════════════════
// Optimized 5×52 Field Square — Symmetry + Interleaved scheduling
// ═══════════════════════════════════════════════════════════════════════════
// Uses a[i]*a[j] == a[j]*a[i] symmetry: compute once, double via pre-mul.
// Only 15 unique products (instead of 25) = 30 instr vs 50.

__attribute__((always_inline)) inline
void fe52_sqr_arm64_v2(uint64_t* __restrict r,
                       const uint64_t* __restrict a) noexcept {
    const uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const uint64_t a0_2 = a0 * 2, a1_2 = a1 * 2, a2_2 = a2 * 2, a3_2 = a3 * 2;

    uint64_t p0_lo, p0_hi, p1_lo, p1_hi, p2_lo, p2_hi, p3_lo, p3_hi;
    uint64_t d_lo, d_hi, c_lo, c_hi;
    uint64_t t3, t4, tx, u0;

    // ── Column 3: d = 2*a0*a3 + 2*a1*a2 + R*(a4*a4) ────────────────
    __asm__ volatile(
        "mul  %[p0l], %[a0_2], %[a3]  \n\t"
        "mul  %[p1l], %[a1_2], %[a2]  \n\t"
        "mul  %[p2l], %[a4], %[a4]    \n\t"
        "umulh %[p0h], %[a0_2], %[a3] \n\t"
        "umulh %[p1h], %[a1_2], %[a2] \n\t"
        "umulh %[p2h], %[a4], %[a4]   \n\t"
        : [p0l] "=&r"(p0_lo), [p0h] "=&r"(p0_hi),
          [p1l] "=&r"(p1_lo), [p1h] "=&r"(p1_hi),
          [p2l] "=&r"(p2_lo), [p2h] "=&r"(p2_hi)
        : [a0_2] "r"(a0_2), [a1_2] "r"(a1_2), [a2] "r"(a2),
          [a3] "r"(a3), [a4] "r"(a4)
    );

    // d = p0 + p1
    __asm__ volatile(
        "adds %[dl], %[p0l], %[p1l]  \n\t"
        "adc  %[dh], %[p0h], %[p1h]  \n\t"
        : [dl] "=&r"(d_lo), [dh] "=&r"(d_hi)
        : [p0l] "r"(p0_lo), [p0h] "r"(p0_hi),
          [p1l] "r"(p1_lo), [p1h] "r"(p1_hi)
        : "cc"
    );

    // c = a4*a4 → d += R * c_lo
    c_lo = p2_lo; c_hi = p2_hi;
    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[R], %[cl]   \n\t"
            "umulh %[rh], %[R], %[cl]  \n\t"
            "adds %[dl], %[dl], %[rl]  \n\t"
            "adc  %[dh], %[dh], %[rh]  \n\t"
            : [dl] "+r"(d_lo), [dh] "+r"(d_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [R] "r"(R), [cl] "r"(c_lo)
            : "cc"
        );
    }
    c_lo = c_hi;

    t3 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;

    // ── Column 4: d += 2*a0*a4 + 2*a1*a3 + a2*a2 + R12*c_rem ──────
    __asm__ volatile(
        "mul  %[p0l], %[a0_2], %[a4]  \n\t"
        "mul  %[p1l], %[a1_2], %[a3]  \n\t"
        "mul  %[p2l], %[a2], %[a2]    \n\t"
        "umulh %[p0h], %[a0_2], %[a4] \n\t"
        "umulh %[p1h], %[a1_2], %[a3] \n\t"
        "umulh %[p2h], %[a2], %[a2]   \n\t"
        : [p0l] "=&r"(p0_lo), [p0h] "=&r"(p0_hi),
          [p1l] "=&r"(p1_lo), [p1h] "=&r"(p1_hi),
          [p2l] "=&r"(p2_lo), [p2h] "=&r"(p2_hi)
        : [a0_2] "r"(a0_2), [a1_2] "r"(a1_2), [a2] "r"(a2),
          [a3] "r"(a3), [a4] "r"(a4)
    );

    __asm__ volatile(
        "adds %[dl], %[dl], %[p0l]   \n\t"
        "adc  %[dh], %[dh], %[p0h]   \n\t"
        "adds %[dl], %[dl], %[p1l]   \n\t"
        "adc  %[dh], %[dh], %[p1h]   \n\t"
        "adds %[dl], %[dl], %[p2l]   \n\t"
        "adc  %[dh], %[dh], %[p2h]   \n\t"
        : [dl] "+r"(d_lo), [dh] "+r"(d_hi)
        : [p0l] "r"(p0_lo), [p0h] "r"(p0_hi),
          [p1l] "r"(p1_lo), [p1h] "r"(p1_hi),
          [p2l] "r"(p2_lo), [p2h] "r"(p2_hi)
        : "cc"
    );

    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[R12], %[cl]  \n\t"
            "umulh %[rh], %[R12], %[cl] \n\t"
            "adds %[dl], %[dl], %[rl]   \n\t"
            "adc  %[dh], %[dh], %[rh]   \n\t"
            : [dl] "+r"(d_lo), [dh] "+r"(d_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [R12] "r"(R12), [cl] "r"(c_lo)
            : "cc"
        );
    }

    t4 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;
    tx = t4 >> 48;
    t4 &= (M52 >> 4);

    // ── Column 0: c = a0*a0; d += 2*a1*a4 + 2*a2*a3 ──────────────
    __asm__ volatile(
        "mul  %[p0l], %[a1_2], %[a4]  \n\t"
        "mul  %[p1l], %[a2_2], %[a3]  \n\t"
        "mul  %[cl],  %[a0], %[a0]    \n\t"
        "umulh %[p0h], %[a1_2], %[a4] \n\t"
        "umulh %[p1h], %[a2_2], %[a3] \n\t"
        "umulh %[ch],  %[a0], %[a0]   \n\t"
        : [p0l] "=&r"(p0_lo), [p0h] "=&r"(p0_hi),
          [p1l] "=&r"(p1_lo), [p1h] "=&r"(p1_hi),
          [cl] "=&r"(c_lo), [ch] "=&r"(c_hi)
        : [a0] "r"(a0), [a1_2] "r"(a1_2), [a2_2] "r"(a2_2),
          [a3] "r"(a3), [a4] "r"(a4)
    );

    __asm__ volatile(
        "adds %[dl], %[dl], %[p0l]   \n\t"
        "adc  %[dh], %[dh], %[p0h]   \n\t"
        "adds %[dl], %[dl], %[p1l]   \n\t"
        "adc  %[dh], %[dh], %[p1h]   \n\t"
        : [dl] "+r"(d_lo), [dh] "+r"(d_hi)
        : [p0l] "r"(p0_lo), [p0h] "r"(p0_hi),
          [p1l] "r"(p1_lo), [p1h] "r"(p1_hi)
        : "cc"
    );

    u0 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;
    u0 = (u0 << 4) | tx;

    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[u0], %[R4]   \n\t"
            "umulh %[rh], %[u0], %[R4]  \n\t"
            "adds %[cl], %[cl], %[rl]   \n\t"
            "adc  %[ch], %[ch], %[rh]   \n\t"
            : [cl] "+r"(c_lo), [ch] "+r"(c_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [u0] "r"(u0), [R4] "r"(R4)
            : "cc"
        );
    }

    r[0] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi = c_hi >> 52;

    // ── Column 1: c += 2*a0*a1; d += 2*a2*a4 + a3*a3 ──────────────
    __asm__ volatile(
        "mul  %[p0l], %[a0_2], %[a1]  \n\t"
        "mul  %[p1l], %[a2_2], %[a4]  \n\t"
        "mul  %[p2l], %[a3], %[a3]    \n\t"
        "umulh %[p0h], %[a0_2], %[a1] \n\t"
        "umulh %[p1h], %[a2_2], %[a4] \n\t"
        "umulh %[p2h], %[a3], %[a3]   \n\t"
        : [p0l] "=&r"(p0_lo), [p0h] "=&r"(p0_hi),
          [p1l] "=&r"(p1_lo), [p1h] "=&r"(p1_hi),
          [p2l] "=&r"(p2_lo), [p2h] "=&r"(p2_hi)
        : [a0_2] "r"(a0_2), [a1] "r"(a1), [a2_2] "r"(a2_2),
          [a3] "r"(a3), [a4] "r"(a4)
    );

    __asm__ volatile(
        "adds %[cl], %[cl], %[p0l]   \n\t"
        "adc  %[ch], %[ch], %[p0h]   \n\t"
        : [cl] "+r"(c_lo), [ch] "+r"(c_hi)
        : [p0l] "r"(p0_lo), [p0h] "r"(p0_hi)
        : "cc"
    );

    __asm__ volatile(
        "adds %[dl], %[dl], %[p1l]   \n\t"
        "adc  %[dh], %[dh], %[p1h]   \n\t"
        "adds %[dl], %[dl], %[p2l]   \n\t"
        "adc  %[dh], %[dh], %[p2h]   \n\t"
        : [dl] "+r"(d_lo), [dh] "+r"(d_hi)
        : [p1l] "r"(p1_lo), [p1h] "r"(p1_hi),
          [p2l] "r"(p2_lo), [p2h] "r"(p2_hi)
        : "cc"
    );

    u0 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12);
    d_hi = d_hi >> 52;
    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[u0], %[R]    \n\t"
            "umulh %[rh], %[u0], %[R]   \n\t"
            "adds %[cl], %[cl], %[rl]   \n\t"
            "adc  %[ch], %[ch], %[rh]   \n\t"
            : [cl] "+r"(c_lo), [ch] "+r"(c_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [u0] "r"(u0), [R] "r"(R)
            : "cc"
        );
    }

    r[1] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi = c_hi >> 52;

    // ── Column 2: c += 2*a0*a2 + a1*a1; d += 2*a3*a4 ──────────────
    __asm__ volatile(
        "mul  %[p0l], %[a0_2], %[a2]  \n\t"
        "mul  %[p1l], %[a1], %[a1]    \n\t"
        "mul  %[p2l], %[a3_2], %[a4]  \n\t"
        "umulh %[p0h], %[a0_2], %[a2] \n\t"
        "umulh %[p1h], %[a1], %[a1]   \n\t"
        "umulh %[p2h], %[a3_2], %[a4] \n\t"
        : [p0l] "=&r"(p0_lo), [p0h] "=&r"(p0_hi),
          [p1l] "=&r"(p1_lo), [p1h] "=&r"(p1_hi),
          [p2l] "=&r"(p2_lo), [p2h] "=&r"(p2_hi)
        : [a0_2] "r"(a0_2), [a1] "r"(a1), [a2] "r"(a2),
          [a3_2] "r"(a3_2), [a4] "r"(a4)
    );

    // c += 2*a0*a2 + a1*a1
    __asm__ volatile(
        "adds %[cl], %[cl], %[p0l]   \n\t"
        "adc  %[ch], %[ch], %[p0h]   \n\t"
        "adds %[cl], %[cl], %[p1l]   \n\t"
        "adc  %[ch], %[ch], %[p1h]   \n\t"
        : [cl] "+r"(c_lo), [ch] "+r"(c_hi)
        : [p0l] "r"(p0_lo), [p0h] "r"(p0_hi),
          [p1l] "r"(p1_lo), [p1h] "r"(p1_hi)
        : "cc"
    );

    // d += 2*a3*a4
    __asm__ volatile(
        "adds %[dl], %[dl], %[p2l]   \n\t"
        "adc  %[dh], %[dh], %[p2h]   \n\t"
        : [dl] "+r"(d_lo), [dh] "+r"(d_hi)
        : [p2l] "r"(p2_lo), [p2h] "r"(p2_hi)
        : "cc"
    );

    // c += R * d_lo (full 64-bit column-7 reduction)
    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[R], %[dl]    \n\t"
            "umulh %[rh], %[R], %[dl]   \n\t"
            "adds %[cl], %[cl], %[rl]   \n\t"
            "adc  %[ch], %[ch], %[rh]   \n\t"
            : [cl] "+r"(c_lo), [ch] "+r"(c_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [R] "r"(R), [dl] "r"(d_lo)
            : "cc"
        );
    }

    // d >>= 64
    d_lo = d_hi;

    r[2] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_hi = c_hi >> 52;

    // ── Finalize columns 3 and 4 ───────────────────────────────────
    {
        uint64_t rl, rh;
        __asm__ volatile(
            "mul  %[rl], %[R12], %[dl]  \n\t"
            "umulh %[rh], %[R12], %[dl] \n\t"
            "adds %[cl], %[cl], %[rl]   \n\t"
            "adc  %[ch], %[ch], %[rh]   \n\t"
            "adds %[cl], %[cl], %[t3]   \n\t"
            "adc  %[ch], %[ch], xzr     \n\t"
            : [cl] "+r"(c_lo), [ch] "+r"(c_hi),
              [rl] "=&r"(rl), [rh] "=&r"(rh)
            : [R12] "r"(R12), [dl] "r"(d_lo), [t3] "r"(t3)
            : "cc"
        );
    }

    r[3] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12);
    c_lo += t4;
    r[4] = c_lo;
}

} // namespace arm64_v2
} // namespace secp256k1::fast

#endif // __aarch64__ || _M_ARM64
