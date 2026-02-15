// ============================================================================
// Multi-Scalar Multiplication: Strauss / Shamir's trick
// ============================================================================

#include "secp256k1/multiscalar.hpp"
#include <algorithm>
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// ── Window Width Selection ───────────────────────────────────────────────────

unsigned strauss_optimal_window(std::size_t n) {
    if (n <= 1)   return 4;
    if (n <= 8)   return 4;
    if (n <= 32)  return 4;
    if (n <= 128) return 5;
    return 6;
}

// ── Shamir's Trick (2-point) ─────────────────────────────────────────────────
// R = a*P + b*Q via joint double-and-add scanning bits from MSB to LSB.
// Pre-computes: table[0] = infinity, table[1] = P, table[2] = Q, table[3] = P+Q

Point shamir_trick(const Scalar& a, const Point& P,
                   const Scalar& b, const Point& Q) {
    // Handle trivial cases
    if (a.is_zero() && b.is_zero()) return Point::infinity();
    if (a.is_zero()) return Q.scalar_mul(b);
    if (b.is_zero()) return P.scalar_mul(a);

    // Pre-compute P+Q
    Point PQ = P.add(Q);

    // Joint double-and-add: scan bits from MSB to LSB
    Point R = Point::infinity();

    // Find highest set bit across both scalars
    int top_bit = 255;
    while (top_bit >= 0 && a.bit(top_bit) == 0 && b.bit(top_bit) == 0) {
        --top_bit;
    }

    for (int i = top_bit; i >= 0; --i) {
        R = R.dbl();

        uint8_t a_bit = a.bit(i);
        uint8_t b_bit = b.bit(i);
        uint8_t idx = (a_bit << 1) | b_bit;

        // Branchless-style: select from table based on idx
        // idx=0: skip, idx=1: +Q, idx=2: +P, idx=3: +P+Q
        if (idx == 1) {
            R = R.add(Q);
        } else if (idx == 2) {
            R = R.add(P);
        } else if (idx == 3) {
            R = R.add(PQ);
        }
    }

    return R;
}

// ── Strauss Multi-Scalar Multiplication ──────────────────────────────────────
// Interleaved wNAF: pre-compute odd multiples of each point, then scan
// all wNAFs simultaneously from MSB to LSB.

Point multi_scalar_mul(const Scalar* scalars,
                       const Point* points,
                       std::size_t n) {
    if (n == 0) return Point::infinity();
    if (n == 1) return points[0].scalar_mul(scalars[0]);
    if (n == 2) return shamir_trick(scalars[0], points[0], scalars[1], points[1]);

    unsigned w = strauss_optimal_window(n);
    std::size_t table_size = static_cast<std::size_t>(1) << (w - 1); // 2^(w-1) entries per point

    // Step 1: Compute wNAF for each scalar
    std::vector<std::vector<int8_t>> wnafs(n);
    int max_len = 0;
    for (std::size_t i = 0; i < n; ++i) {
        wnafs[i] = scalars[i].to_wnaf(w);
        if (static_cast<int>(wnafs[i].size()) > max_len) {
            max_len = static_cast<int>(wnafs[i].size());
        }
    }

    // Step 2: Pre-compute odd multiples: table[i][j] = (2j+1) * points[i]
    // table[i][0] = P, table[i][1] = 3P, table[i][2] = 5P, ...
    std::vector<std::vector<Point>> tables(n);
    for (std::size_t i = 0; i < n; ++i) {
        tables[i].resize(table_size);
        tables[i][0] = points[i];
        if (table_size > 1) {
            Point P2 = points[i].dbl();
            for (std::size_t j = 1; j < table_size; ++j) {
                tables[i][j] = tables[i][j - 1].add(P2);
            }
        }
    }

    // Step 3: Interleaved scan from MSB to LSB
    Point R = Point::infinity();

    for (int bit = max_len - 1; bit >= 0; --bit) {
        R = R.dbl();

        for (std::size_t i = 0; i < n; ++i) {
            if (bit >= static_cast<int>(wnafs[i].size())) continue;
            int8_t digit = wnafs[i][bit];
            if (digit == 0) continue;

            std::size_t idx;
            if (digit > 0) {
                idx = static_cast<std::size_t>((digit - 1) / 2);
                R = R.add(tables[i][idx]);
            } else {
                idx = static_cast<std::size_t>((-digit - 1) / 2);
                R = R.add(tables[i][idx].negate());
            }
        }
    }

    return R;
}

// Convenience: vector version
Point multi_scalar_mul(const std::vector<Scalar>& scalars,
                       const std::vector<Point>& points) {
    std::size_t n = std::min(scalars.size(), points.size());
    if (n == 0) return Point::infinity();
    return multi_scalar_mul(scalars.data(), points.data(), n);
}

} // namespace secp256k1
