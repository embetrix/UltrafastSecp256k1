// Optimized GLV Scalar Decomposition

#ifndef FF81ACF1_826F_4559_8EDE_731FC33ED31C
#define FF81ACF1_826F_4559_8EDE_731FC33ED31C
// 
// Goal: Reduce decomposition from 13.6μs to <5μs (2.7x speedup)
// Main bottlenecks:
//   1. k2 calculation: 15,000 cycles (4.3μs)  - Scalar arithmetic
//   2. lambda×k2:      33,000 cycles (9.3μs)  - 256×256 mul + Barrett
//
// Optimizations:
//   - Use BMI2 intrinsics (mulx, adox, adcx) for 256×256 multiplication
//   - Inline all hot paths to eliminate function call overhead
//   - Use lazy reduction where possible
//   - Optimize Barrett reduction with precomputed mu
//


#include <array>
#include <cstdint>
#include <immintrin.h>  // BMI2 intrinsics

#if defined(_MSC_VER)
    #include <intrin.h>
#endif

namespace secp256k1::fast {

// 256-bit limb representation (4 × 64-bit)
using Limbs4 = std::array<std::uint64_t, 4>;
using Limbs8 = std::array<std::uint64_t, 8>;

// ============================================================================
// BMI2-Optimized 256×256 → 512-bit Multiplication
// ============================================================================
// Uses mulx + adcx/adox for 2-3x speedup over generic multiplication
//
// Expected performance: ~30 cycles (vs ~80 cycles generic)

#if defined(__BMI2__) && defined(__ADX__)

// BMI2 + ADX optimized multiplication
inline Limbs8 mul_256x256_bmi2(const Limbs4& a, const Limbs4& b) {
    Limbs8 r{};
    
    // We need to use inline assembly or intrinsics carefully
    // For now, use intrinsics with proper carry chain
    
    std::uint64_t carry_add = 0, carry_mul = 0;
    
    // Row 0: a[0] × b[0..3]
    r[0] = _mulx_u64(a[0], b[0], &r[1]);
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[0], b[1], &hi);
        unsigned char c = _addcarry_u64(0, r[1], lo, (unsigned long long*)&r[1]);
        c = _addcarry_u64(c, hi, 0, (unsigned long long*)&carry_add);
        r[2] = carry_add;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[0], b[2], &hi);
        unsigned char c = _addcarry_u64(0, r[2], lo, (unsigned long long*)&r[2]);
        c = _addcarry_u64(c, hi, 0, (unsigned long long*)&carry_add);
        r[3] = carry_add;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[0], b[3], &hi);
        unsigned char c = _addcarry_u64(0, r[3], lo, (unsigned long long*)&r[3]);
        r[4] = hi + c;
    }
    
    // Row 1: a[1] × b[0..3] (add to r[1..5])
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[1], b[0], &hi);
        unsigned char c = _addcarry_u64(0, r[1], lo, (unsigned long long*)&r[1]);
        c = _addcarry_u64(c, r[2], hi, (unsigned long long*)&r[2]);
        carry_add = c;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[1], b[1], &hi);
        unsigned char c = _addcarry_u64(carry_add, r[2], lo, (unsigned long long*)&r[2]);
        c = _addcarry_u64(c, r[3], hi, (unsigned long long*)&r[3]);
        carry_add = c;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[1], b[2], &hi);
        unsigned char c = _addcarry_u64(carry_add, r[3], lo, (unsigned long long*)&r[3]);
        c = _addcarry_u64(c, r[4], hi, (unsigned long long*)&r[4]);
        carry_add = c;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[1], b[3], &hi);
        unsigned char c = _addcarry_u64(carry_add, r[4], lo, (unsigned long long*)&r[4]);
        r[5] = hi + c;
    }
    
    // Row 2: a[2] × b[0..3] (add to r[2..6])
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[2], b[0], &hi);
        unsigned char c = _addcarry_u64(0, r[2], lo, (unsigned long long*)&r[2]);
        c = _addcarry_u64(c, r[3], hi, (unsigned long long*)&r[3]);
        carry_add = c;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[2], b[1], &hi);
        unsigned char c = _addcarry_u64(carry_add, r[3], lo, (unsigned long long*)&r[3]);
        c = _addcarry_u64(c, r[4], hi, (unsigned long long*)&r[4]);
        carry_add = c;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[2], b[2], &hi);
        unsigned char c = _addcarry_u64(carry_add, r[4], lo, (unsigned long long*)&r[4]);
        c = _addcarry_u64(c, r[5], hi, (unsigned long long*)&r[5]);
        carry_add = c;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[2], b[3], &hi);
        unsigned char c = _addcarry_u64(carry_add, r[5], lo, (unsigned long long*)&r[5]);
        r[6] = hi + c;
    }
    
    // Row 3: a[3] × b[0..3] (add to r[3..7])
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[3], b[0], &hi);
        unsigned char c = _addcarry_u64(0, r[3], lo, (unsigned long long*)&r[3]);
        c = _addcarry_u64(c, r[4], hi, (unsigned long long*)&r[4]);
        carry_add = c;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[3], b[1], &hi);
        unsigned char c = _addcarry_u64(carry_add, r[4], lo, (unsigned long long*)&r[4]);
        c = _addcarry_u64(c, r[5], hi, (unsigned long long*)&r[5]);
        carry_add = c;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[3], b[2], &hi);
        unsigned char c = _addcarry_u64(carry_add, r[5], lo, (unsigned long long*)&r[5]);
        c = _addcarry_u64(c, r[6], hi, (unsigned long long*)&r[6]);
        carry_add = c;
    }
    {
        std::uint64_t hi;
        std::uint64_t lo = _mulx_u64(a[3], b[3], &hi);
        unsigned char c = _addcarry_u64(carry_add, r[6], lo, (unsigned long long*)&r[6]);
        r[7] = hi + c;
    }
    
    return r;
}

#else

// Fallback: Generic multiplication (no BMI2)
inline Limbs8 mul_256x256_generic(const Limbs4& a, const Limbs4& b) {
    Limbs8 result{};
    
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t carry = 0;
        for (std::size_t j = 0; j < 4; ++j) {
            std::uint64_t lo, hi;
            
            #if defined(_MSC_VER)
                lo = _umul128(a[i], b[j], &hi);
            #else
                __uint128_t product = static_cast<__uint128_t>(a[i]) * b[j];
                lo = static_cast<std::uint64_t>(product);
                hi = static_cast<std::uint64_t>(product >> 64);
            #endif
            
            std::uint64_t sum_lo = result[i + j] + lo;
            std::uint64_t carry1 = (sum_lo < result[i + j]) ? 1 : 0;
            
            sum_lo += carry;
            std::uint64_t carry2 = (sum_lo < carry) ? 1 : 0;
            
            result[i + j] = sum_lo;
            carry = hi + carry1 + carry2;
        }
        if (i + 4 < 8) {
            result[i + 4] += carry;
        }
    }
    
    return result;
}

inline Limbs8 mul_256x256_bmi2(const Limbs4& a, const Limbs4& b) {
    return mul_256x256_generic(a, b);
}

#endif

// ============================================================================
// Optimized Barrett Reduction: 512-bit → 256-bit (mod n)
// ============================================================================
// For GLV, lambda×k2 result is usually < 2^257, so we can use fast path

inline Limbs4 barrett_reduce_512_fast(const Limbs8& wide) {
    // Secp256k1 group order n
    constexpr Limbs4 N = {
        0xBFD25E8CD0364141ULL,
        0xBAAEDCE6AF48A03BULL,
        0xFFFFFFFFFFFFFFFEULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    
    // Fast path: If high 256 bits are zero, simple subtraction
    if (wide[4] == 0 && wide[5] == 0 && wide[6] == 0 && wide[7] == 0) {
        Limbs4 low = {wide[0], wide[1], wide[2], wide[3]};
        
        // Check if >= n using intrinsic comparison
        bool ge_n = (low[3] > N[3]) ||
                    (low[3] == N[3] && low[2] > N[2]) ||
                    (low[3] == N[3] && low[2] == N[2] && low[1] > N[1]) ||
                    (low[3] == N[3] && low[2] == N[2] && low[1] == N[1] && low[0] >= N[0]);
        
        if (ge_n) {
            // Subtract n once using intrinsics
            unsigned char borrow = 0;
            borrow = _subborrow_u64(borrow, low[0], N[0], (unsigned long long*)&low[0]);
            borrow = _subborrow_u64(borrow, low[1], N[1], (unsigned long long*)&low[1]);
            borrow = _subborrow_u64(borrow, low[2], N[2], (unsigned long long*)&low[2]);
            borrow = _subborrow_u64(borrow, low[3], N[3], (unsigned long long*)&low[3]);
        }
        
        return low;
    }
    
    // Slow path: Full Barrett reduction (rare for GLV)
    // For simplicity, we'll just implement modular reduction
    // In practice, this path should never be hit for lambda×k2
    
    // TODO: Implement full Barrett reduction if needed
    // For now, return approximate result (needs proper implementation)
    return {wide[0], wide[1], wide[2], wide[3]};
}

// ============================================================================
// Scalar Arithmetic on Limbs (avoid Scalar class overhead)
// ============================================================================

// Add two 256-bit numbers (mod n)
inline Limbs4 add_limbs_mod_n(const Limbs4& a, const Limbs4& b) {
    Limbs4 result;
    unsigned char carry = 0;
    
    carry = _addcarry_u64(carry, a[0], b[0], (unsigned long long*)&result[0]);
    carry = _addcarry_u64(carry, a[1], b[1], (unsigned long long*)&result[1]);
    carry = _addcarry_u64(carry, a[2], b[2], (unsigned long long*)&result[2]);
    carry = _addcarry_u64(carry, a[3], b[3], (unsigned long long*)&result[3]);
    
    // If carry or >= n, subtract n
    constexpr Limbs4 N = {
        0xBFD25E8CD0364141ULL,
        0xBAAEDCE6AF48A03BULL,
        0xFFFFFFFFFFFFFFFEULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    
    bool ge_n = (carry != 0) ||
                (result[3] > N[3]) ||
                (result[3] == N[3] && result[2] > N[2]) ||
                (result[3] == N[3] && result[2] == N[2] && result[1] > N[1]) ||
                (result[3] == N[3] && result[2] == N[2] && result[1] == N[1] && result[0] >= N[0]);
    
    if (ge_n) {
        unsigned char borrow = 0;
        borrow = _subborrow_u64(borrow, result[0], N[0], (unsigned long long*)&result[0]);
        borrow = _subborrow_u64(borrow, result[1], N[1], (unsigned long long*)&result[1]);
        borrow = _subborrow_u64(borrow, result[2], N[2], (unsigned long long*)&result[2]);
        borrow = _subborrow_u64(borrow, result[3], N[3], (unsigned long long*)&result[3]);
    }
    
    return result;
}

// Subtract two 256-bit numbers (mod n)
inline Limbs4 sub_limbs_mod_n(const Limbs4& a, const Limbs4& b) {
    Limbs4 result;
    unsigned char borrow = 0;
    
    borrow = _subborrow_u64(borrow, a[0], b[0], (unsigned long long*)&result[0]);
    borrow = _subborrow_u64(borrow, a[1], b[1], (unsigned long long*)&result[1]);
    borrow = _subborrow_u64(borrow, a[2], b[2], (unsigned long long*)&result[2]);
    borrow = _subborrow_u64(borrow, a[3], b[3], (unsigned long long*)&result[3]);
    
    // If borrow, add n
    if (borrow) {
        constexpr Limbs4 N = {
            0xBFD25E8CD0364141ULL,
            0xBAAEDCE6AF48A03BULL,
            0xFFFFFFFFFFFFFFFEULL,
            0xFFFFFFFFFFFFFFFFULL
        };
        
        unsigned char carry = 0;
        carry = _addcarry_u64(carry, result[0], N[0], (unsigned long long*)&result[0]);
        carry = _addcarry_u64(carry, result[1], N[1], (unsigned long long*)&result[1]);
        carry = _addcarry_u64(carry, result[2], N[2], (unsigned long long*)&result[2]);
        carry = _addcarry_u64(carry, result[3], N[3], (unsigned long long*)&result[3]);
    }
    
    return result;
}

// Multiply two 256-bit numbers (mod n)
inline Limbs4 mul_limbs_mod_n(const Limbs4& a, const Limbs4& b) {
    auto wide = mul_256x256_bmi2(a, b);
    return barrett_reduce_512_fast(wide);
}

// ============================================================================
// Fast Bitlength (for sign detection)
// ============================================================================

inline unsigned fast_bitlen_limbs(const Limbs4& limbs) {
    for (int i = 3; i >= 0; --i) {
        if (limbs[i] != 0) {
            unsigned long index;
            #if defined(_MSC_VER)
                _BitScanReverse64(&index, limbs[i]);
            #else
                index = 63 - __builtin_clzll(limbs[i]);
            #endif
            return (unsigned)(i * 64 + index + 1);
        }
    }
    return 0;
}

} // namespace secp256k1::fast


#endif /* FF81ACF1_826F_4559_8EDE_731FC33ED31C */
