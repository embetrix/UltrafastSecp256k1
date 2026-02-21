// ============================================================================
// Constant-Time (CT) Layer Benchmarks
// ============================================================================
// Benchmarks for side-channel-resistant operations.
// Compare fast:: vs ct:: to quantify the protection cost.
// ============================================================================

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/selftest.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"

#include <chrono>
#include <cstdio>
#include <cstdint>

using namespace secp256k1::fast;
namespace ct = secp256k1::ct;

// ── timing helper ────────────────────────────────────────────────────────────

template <typename Func>
static double bench_us(Func&& f, int iters) {
    // warmup
    for (int i = 0; i < 10; ++i) f();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) f();
    auto t1 = std::chrono::high_resolution_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    return us / iters;
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    printf("================================================================\n");
    printf("  CT Layer Benchmark  (fast:: vs ct::)\n");
    printf("================================================================\n\n");

    // Fixed test data — all 256-bit, representative of real workloads
    auto G  = Point::generator();
    auto k  = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");
    auto k2 = Scalar::from_hex(
        "7c076ff316692a3d7eb3c3bb0f8b1488cf72e1afcd929e29307032997a838a3d");
    auto P  = G.scalar_mul(k);

    auto fe_a = FieldElement::from_hex(
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    auto fe_b = FieldElement::from_hex(
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8");

    // Pool of random 256-bit scalars to prevent branch-predictor warming
    // and ensure benchmarks reflect real-world workloads with varying inputs.
    constexpr int POOL = 32;
    Scalar scalar_pool[POOL];
    Point  point_pool[POOL];
    {
        auto base = Scalar::from_hex(
            "b5037ebecae0da656179c623f6cb73641db2aa0fabe888ffb78466fa18470379");
        auto step = Scalar::from_hex(
            "9e3779b97f4a7c15f39cc0605cedc8341082276bf3a27251f86c6a11d0c18e95");
        for (int i = 0; i < POOL; ++i) {
            scalar_pool[i] = base;
            point_pool[i]  = G.scalar_mul(base);
            base += step;
        }
    }

    constexpr int N_SCALAR_MUL = 50;
    constexpr int N_POINT_OPS  = 5000;
    constexpr int N_FIELD_OPS  = 50000;
    constexpr int N_SCALAR_OPS = 50000;

    printf("  Pool size: %d random 256-bit scalars\n\n", POOL);

    // ── Field operations ─────────────────────────────────────────────────────

    printf("--- Field Arithmetic ---\n");

    double fast_field_mul = bench_us([&]() {
        volatile auto r = fe_a * fe_b;
    }, N_FIELD_OPS);

    double ct_field_mul = bench_us([&]() {
        auto r = ct::field_mul(fe_a, fe_b);
        (void)r;
    }, N_FIELD_OPS);

    printf("  field_mul    fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n",
           fast_field_mul, ct_field_mul, ct_field_mul / fast_field_mul);

    double fast_field_sq = bench_us([&]() {
        volatile auto r = fe_a.square();
    }, N_FIELD_OPS);

    double ct_field_sq = bench_us([&]() {
        auto r = ct::field_sqr(fe_a);
        (void)r;
    }, N_FIELD_OPS);

    printf("  field_square fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n",
           fast_field_sq, ct_field_sq, ct_field_sq / fast_field_sq);

    double fast_field_inv = bench_us([&]() {
        volatile auto r = fe_a.inverse();
    }, N_FIELD_OPS / 10);

    double ct_field_inv = bench_us([&]() {
        auto r = ct::field_inv(fe_a);
        (void)r;
    }, N_FIELD_OPS / 10);

    printf("  field_inv    fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n\n",
           fast_field_inv, ct_field_inv, ct_field_inv / fast_field_inv);

    // ── Scalar operations ────────────────────────────────────────────────────

    printf("--- Scalar Arithmetic ---\n");

    double fast_scalar_add = bench_us([&]() {
        volatile auto r = k + k2;
    }, N_SCALAR_OPS);

    double ct_scalar_add = bench_us([&]() {
        auto r = ct::scalar_add(k, k2);
        (void)r;
    }, N_SCALAR_OPS);

    printf("  scalar_add   fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n",
           fast_scalar_add, ct_scalar_add, ct_scalar_add / fast_scalar_add);

    double fast_scalar_sub = bench_us([&]() {
        volatile auto r = k - k2;
    }, N_SCALAR_OPS);

    double ct_scalar_sub = bench_us([&]() {
        auto r = ct::scalar_sub(k, k2);
        (void)r;
    }, N_SCALAR_OPS);

    printf("  scalar_sub   fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n\n",
           fast_scalar_sub, ct_scalar_sub, ct_scalar_sub / fast_scalar_sub);

    // ── Point operations ─────────────────────────────────────────────────────

    printf("--- Point Operations ---\n");

    auto ct_p = ct::CTJacobianPoint::from_point(P);
    auto ct_g = ct::CTJacobianPoint::from_point(G);

    double fast_point_add = bench_us([&]() {
        volatile auto r = P.add(G);
    }, N_POINT_OPS);

    double ct_point_add = bench_us([&]() {
        auto r = ct::point_add_complete(ct_p, ct_g);
        (void)r;
    }, N_POINT_OPS);

    printf("  point_add    fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n",
           fast_point_add, ct_point_add, ct_point_add / fast_point_add);

    double fast_point_dbl = bench_us([&]() {
        volatile auto r = P.dbl();
    }, N_POINT_OPS);

    double ct_point_dbl = bench_us([&]() {
        auto r = ct::point_dbl(ct_p);
        (void)r;
    }, N_POINT_OPS);

    printf("  point_dbl    fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n\n",
           fast_point_dbl, ct_point_dbl, ct_point_dbl / fast_point_dbl);

    // ── Scalar multiplication ────────────────────────────────────────────────

    printf("--- Scalar Multiplication (k * P) ---\n");

    int idx_fast_mul = 0;
    double fast_mul = bench_us([&]() {
        volatile auto r = point_pool[idx_fast_mul % POOL].scalar_mul(scalar_pool[idx_fast_mul % POOL]);
        ++idx_fast_mul;
    }, N_SCALAR_MUL);

    int idx_ct_mul = 0;
    double ct_mul = bench_us([&]() {
        auto r = ct::scalar_mul(point_pool[idx_ct_mul % POOL], scalar_pool[idx_ct_mul % POOL]);
        (void)r;
        ++idx_ct_mul;
    }, N_SCALAR_MUL);

    printf("  scalar_mul   fast: %8.1f us   ct: %8.1f us   ratio: %.2fx\n\n",
           fast_mul, ct_mul, ct_mul / fast_mul);

    // ── Generator multiplication ─────────────────────────────────────────────

    printf("--- Generator Multiplication (k * G) ---\n");

    int idx_fast_gen = 0;
    double fast_gen = bench_us([&]() {
        volatile auto r = G.scalar_mul(scalar_pool[idx_fast_gen % POOL]);
        ++idx_fast_gen;
    }, N_SCALAR_MUL);

    int idx_ct_gen = 0;
    double ct_gen = bench_us([&]() {
        auto r = ct::generator_mul(scalar_pool[idx_ct_gen % POOL]);
        (void)r;
        ++idx_ct_gen;
    }, N_SCALAR_MUL);

    printf("  generator_mul fast: %7.1f us   ct: %8.1f us   ratio: %.2fx\n\n",
           fast_gen, ct_gen, ct_gen / fast_gen);

    printf("================================================================\n");
    printf("  Lower ratio = smaller CT overhead (1.0x = same speed)\n");
    printf("================================================================\n");

    return 0;
}
