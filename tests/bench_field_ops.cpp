// ============================================================================
// Field Operations Microbenchmark
// ============================================================================
// Measures cycle-level performance of:
//   - fe52_mul_inner  (5×52 field multiplication)
//   - fe52_sqr_inner  (5×52 field squaring)
//   - jac52_double_inplace  (point doubling, 2M+5S)
//   - jac52_add_mixed_inplace  (point mixed add, 7M+4S)
//   - Full verify (dual_scalar_mul_gen_point)
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <array>

#include "secp256k1/field_52.hpp"
#include "secp256k1/field_52_impl.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using namespace secp256k1::fast;
using Clock = std::chrono::high_resolution_clock;



// Prevent dead-code elimination
static volatile uint64_t g_sink = 0;
__attribute__((noinline)) void consume(const void* p, size_t n) {
    uint64_t acc = 0;
    for (size_t i = 0; i < n / 8; i++) acc ^= reinterpret_cast<const uint64_t*>(p)[i];
    g_sink = acc;
}

// ── rdtsc for cycle counting ────────────────────────────────────────────
static inline uint64_t rdtsc() {
    uint32_t lo, hi;
    __asm__ volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}
static inline void cpuid_fence() {
    uint32_t a, b, c, d;
    __asm__ volatile("cpuid" : "=a"(a), "=b"(b), "=c"(c), "=d"(d) : "a"(0) : "memory");
}

template<typename F>
double bench_ns(F&& fn, int warmup = 1000, int iters = 50000) {
    for (int i = 0; i < warmup; i++) fn();
    auto t0 = Clock::now();
    for (int i = 0; i < iters; i++) fn();
    auto t1 = Clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
}

template<typename F>
uint64_t bench_cycles(F&& fn, int warmup = 2000, int iters = 100000) {
    for (int i = 0; i < warmup; i++) fn();
    cpuid_fence();
    uint64_t start = rdtsc();
    for (int i = 0; i < iters; i++) fn();
    cpuid_fence();
    uint64_t end = rdtsc();
    return (end - start) / iters;
}

int main() {
    printf("═══════════════════════════════════════════════════════════════\n");
    printf("  Field Operations Microbenchmark (5×52 representation)\n");
    printf("═══════════════════════════════════════════════════════════════\n\n");

    // ── Setup test data ──────────────────────────────────────────────
    // Use known values that won't trigger special cases
    FieldElement52 fe_a, fe_b, fe_c;
    fe_a.n[0] = 0xA1B2C3D4E5F67ULL; fe_a.n[1] = 0x1234567890ABCULL;
    fe_a.n[2] = 0xFEDCBA9876543ULL; fe_a.n[3] = 0x0123456789AULL;
    fe_a.n[4] = 0xFFFFFFFFFFULL;

    fe_b.n[0] = 0x9876543210FEDULL; fe_b.n[1] = 0xABCDEF012345CULL;
    fe_b.n[2] = 0x1112223334445ULL; fe_b.n[3] = 0xDEADBEEFCAFEULL;
    fe_b.n[4] = 0xAAAAAAAAAAAAULL;

    // ── 1. Field Multiplication ──────────────────────────────────────
    {
        uint64_t r[5];
        uint64_t cyc = bench_cycles([&]{ fe52_mul_inner(r, fe_a.n, fe_b.n); });
        double ns = bench_ns([&]{ fe52_mul_inner(r, fe_a.n, fe_b.n); });
        printf("fe52_mul_inner:    %3lu cycles  %6.1f ns\n", cyc, ns);
        consume(r, sizeof(r));
    }

    // ── 2. Field Squaring ────────────────────────────────────────────
    {
        uint64_t r[5];
        uint64_t cyc = bench_cycles([&]{ fe52_sqr_inner(r, fe_a.n); });
        double ns = bench_ns([&]{ fe52_sqr_inner(r, fe_a.n); });
        printf("fe52_sqr_inner:    %3lu cycles  %6.1f ns\n", cyc, ns);
        consume(r, sizeof(r));
    }

    // ── 3. Chained multiplication (realistic: mul result feeds next mul) ─
    {
        FieldElement52 tmp = fe_a;
        double ns = bench_ns([&]{
            tmp = tmp * fe_b;
        });
        uint64_t cyc = bench_cycles([&]{
            tmp = tmp * fe_b;
        });
        printf("fe52_mul chained:  %3lu cycles  %6.1f ns\n", cyc, ns);
        consume(tmp.n, sizeof(tmp.n));
    }

    // ── 4. Chained squaring ──────────────────────────────────────────
    {
        FieldElement52 tmp = fe_a;
        double ns = bench_ns([&]{
            tmp = tmp.square();
        });
        uint64_t cyc = bench_cycles([&]{
            tmp = tmp.square();
        });
        printf("fe52_sqr chained:  %3lu cycles  %6.1f ns\n", cyc, ns);
        consume(tmp.n, sizeof(tmp.n));
    }

    // ── 5. Field inverse (255 sqr + 14 mul) ──────────────────────────
    {
        FieldElement52 tmp = fe_a;
        double ns = bench_ns([&]{
            tmp = fe_a.inverse();
        }, 100, 5000);
        printf("fe52_inverse:                  %6.0f ns\n", ns);
        consume(tmp.n, sizeof(tmp.n));
    }

    printf("\n");



    // ── 6. Point Doubling (2M + 5S + adds) ──────────────────────────
    {
        Point G = Point::generator();
        auto sk = Scalar::from_bytes({0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
                                      0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10,
                                      0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
                                      0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,0x01});
        Point P = G.scalar_mul(sk);

        double ns = bench_ns([&]{
            P.dbl_inplace();
        });
        uint64_t cyc = bench_cycles([&]{
            P.dbl_inplace();
        });
        printf("Point::dbl_inplace:  %3lu cycles  %6.1f ns\n", cyc, ns);
        consume(&P, sizeof(P));
    }

    // ── 7. Point Mixed Add (7M + 4S + adds) ─────────────────────────
    // We need to call the low-level jac52_add_mixed_inplace
    // but it's static. Measure via scalar_mul which uses it.
    // Instead, use Point::add_inplace which calls jac52_add_inplace (12M+5S)
    {
        Point G = Point::generator();
        auto sk1 = Scalar::from_bytes({0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x02});
        auto sk2 = Scalar::from_bytes({0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
                                       0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x03});
        Point P1 = G.scalar_mul(sk1);
        Point P2 = G.scalar_mul(sk2);

        double ns = bench_ns([&]{
            Point tmp = P1;
            tmp.add_inplace(P2);
            consume(&tmp, sizeof(tmp));
        }, 500, 20000);
        printf("Point::add_inplace:            %6.1f ns  (full Jac+Jac, 12M+5S)\n", ns);
    }

    printf("\n");

    // ── 8. Full verify operation (dual_scalar_mul_gen_point) ─────────
    {
        Point G = Point::generator();
        auto sk = Scalar::from_bytes({0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE,
                                      0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
                                      0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10,
                                      0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88});
        Point P = G.scalar_mul(sk);

        auto a = Scalar::from_bytes({0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,0x00,0x11,
                                     0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,
                                     0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
                                     0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x10});
        auto b = Scalar::from_bytes({0x10,0x0F,0x0E,0x0D,0x0C,0x0B,0x0A,0x09,
                                     0x08,0x07,0x06,0x05,0x04,0x03,0x02,0x01,
                                     0x99,0x88,0x77,0x66,0x55,0x44,0x33,0x22,
                                     0x11,0x00,0xFF,0xEE,0xDD,0xCC,0xBB,0xAA});

        double ns = bench_ns([&]{
            auto R = Point::dual_scalar_mul_gen_point(a, b, P);
            consume(&R, sizeof(R));
        }, 500, 10000);
        printf("dual_scalar_mul_gen_point:     %6.0f ns  (verify kernel: a*G + b*P)\n", ns);
    }

    // ── 9. Breakdown: measure G-table init vs main loop ──────────────
    // The first call initializes static tables; measure separately
    {
        Point G = Point::generator();
        auto sk = Scalar::from_bytes({0xFF,0xFE,0xFD,0xFC,0xFB,0xFA,0xF9,0xF8,
                                      0xF7,0xF6,0xF5,0xF4,0xF3,0xF2,0xF1,0xF0,
                                      0xEF,0xEE,0xED,0xEC,0xEB,0xEA,0xE9,0xE8,
                                      0xE7,0xE6,0xE5,0xE4,0xE3,0xE2,0xE1,0xE0});
        Point P = G.scalar_mul(sk);

        // 10 different scalars to spread across cache
        Scalar scalars_a[10], scalars_b[10];
        for (int i = 0; i < 10; i++) {
            std::array<uint8_t, 32> bytes{};
            bytes[0] = static_cast<uint8_t>(i + 1);
            bytes[31] = static_cast<uint8_t>(i + 0x42);
            scalars_a[i] = Scalar::from_bytes(bytes);
            bytes[0] = static_cast<uint8_t>(i + 0x80);
            bytes[31] = static_cast<uint8_t>(i + 0xA3);
            scalars_b[i] = Scalar::from_bytes(bytes);
        }

        // Warmup
        for (int i = 0; i < 100; i++) {
            auto R = Point::dual_scalar_mul_gen_point(scalars_a[i%10], scalars_b[i%10], P);
            consume(&R, sizeof(R));
        }

        // Tight measurement with varying scalars
        constexpr int N = 10000;
        auto t0 = Clock::now();
        for (int i = 0; i < N; i++) {
            auto R = Point::dual_scalar_mul_gen_point(scalars_a[i%10], scalars_b[i%10], P);
            consume(&R, sizeof(R));
        }
        auto t1 = Clock::now();
        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
        printf("dual_scalar_mul (varied):      %6.0f ns\n", ns);
    }

    printf("\n═══════════════════════════════════════════════════════════════\n");
    
    // ── Detailed comparison: mul_assign vs operator* ─────────────────
    printf("\n── Mul variants breakdown ───────────────────────────────────\n");
    {
        // Test 1: mul_assign (in-place with temp buffer, no copy)
        FieldElement52 tmp1 = fe_a;
        double ns1 = bench_ns([&]{
            tmp1.mul_assign(fe_b);
        });
        printf("  mul_assign (in-place):           %6.1f ns\n", ns1);
        consume(tmp1.n, sizeof(tmp1.n));
        
        // Test 2: operator* (returns by value)
        FieldElement52 tmp2 = fe_a;
        double ns2 = bench_ns([&]{
            tmp2 = tmp2 * fe_b;
        });
        printf("  operator* (return-by-value):     %6.1f ns\n", ns2);
        consume(tmp2.n, sizeof(tmp2.n));
        
        // Test 3: square_inplace
        FieldElement52 tmp3 = fe_a;
        double ns3 = bench_ns([&]{
            tmp3.square_inplace();
        });
        printf("  square_inplace:                  %6.1f ns\n", ns3);
        consume(tmp3.n, sizeof(tmp3.n));
        
        // Test 4: Direct fe52_mul_inner with temp buf (same as mul_assign pattern)
        FieldElement52 abuf = fe_a;
        double ns4 = bench_ns([&]{
            uint64_t tmp[5];
            fe52_mul_inner(tmp, abuf.n, fe_b.n);
            abuf.n[0] = tmp[0]; abuf.n[1] = tmp[1]; abuf.n[2] = tmp[2];
            abuf.n[3] = tmp[3]; abuf.n[4] = tmp[4];
        });
        printf("  C mul_inner + manual copy:       %6.1f ns\n", ns4);
        consume(abuf.n, sizeof(abuf.n));
    }

    printf("\n");
    printf("═══════════════════════════════════════════════════════════════\n");

    return 0;
}
