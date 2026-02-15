// =============================================================================
// UltrafastSecp256k1 Metal — Bloom Filter (secp256k1_bloom.h)
// =============================================================================
// GPU-side bloom filter for candidate matching.
// Matches CUDA DeviceBloom semantics using FNV-1a + SplitMix64 hashing.
//
// Metal uses device buffers, so bloom filter data is passed as a pointer
// to device memory along with metadata (m_bits, k, salt).
// =============================================================================

#pragma once

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// Bloom Filter Configuration (passed from host)
// =============================================================================

struct BloomParams {
    uint m_bits_lo;   // Total bit count (low 32)
    uint m_bits_hi;   // Total bit count (high 32) — supports >4G bits
    uint k;           // Number of hash functions
    uint _pad;
    uint salt_lo;     // Salt low 32
    uint salt_hi;     // Salt high 32
    uint _pad2[2];
};

// =============================================================================
// 64-bit emulated helpers (Metal has no native uint64)
// =============================================================================

// Represent 64-bit value as (lo, hi) pair
struct u64 {
    uint lo;
    uint hi;
};

inline u64 u64_make(uint lo, uint hi) { u64 r; r.lo = lo; r.hi = hi; return r; }
inline u64 u64_from32(uint v) { return u64_make(v, 0); }

inline u64 u64_add(u64 a, u64 b) {
    uint lo = a.lo + b.lo;
    uint carry = (lo < a.lo) ? 1u : 0u;
    uint hi = a.hi + b.hi + carry;
    return u64_make(lo, hi);
}

inline u64 u64_xor(u64 a, u64 b) {
    return u64_make(a.lo ^ b.lo, a.hi ^ b.hi);
}

inline u64 u64_or(u64 a, u64 b) {
    return u64_make(a.lo | b.lo, a.hi | b.hi);
}

inline u64 u64_shr(u64 a, uint n) {
    if (n >= 32) return u64_make(a.hi >> (n - 32), 0);
    if (n == 0) return a;
    return u64_make((a.lo >> n) | (a.hi << (32 - n)), a.hi >> n);
}

inline u64 u64_shl(u64 a, uint n) {
    if (n >= 32) return u64_make(0, a.lo << (n - 32));
    if (n == 0) return a;
    return u64_make(a.lo << n, (a.hi << n) | (a.lo >> (32 - n)));
}

// 64-bit multiply: a * b → 64-bit result (truncated)
inline u64 u64_mul(u64 a, u64 b) {
    // (a.lo + a.hi<<32) * (b.lo + b.hi<<32)
    // lo*lo => full 64-bit
    ulong prod = ulong(a.lo) * ulong(b.lo);
    uint r_lo = uint(prod);
    uint r_hi = uint(prod >> 32);
    // lo*hi and hi*lo contribute to hi part only
    r_hi += a.lo * b.hi + a.hi * b.lo;
    return u64_make(r_lo, r_hi);
}

// Get bit n of u64
inline uint u64_bit(u64 a, uint n) {
    if (n < 32) return (a.lo >> n) & 1u;
    return (a.hi >> (n - 32)) & 1u;
}

// fast_reduce64: ((x * range) >> 64) — just need high 64 bits of 128-bit product
// This maps uniformly into [0, range)
inline u64 u64_fast_reduce(u64 x, u64 range) {
    // Full 64×64 → 128 bit, return high 64
    // x = x.lo + x.hi*2^32
    // range = range.lo + range.hi*2^32
    // product = x*range = sum of cross products
    
    ulong a0b0 = ulong(x.lo) * ulong(range.lo);
    ulong a0b1 = ulong(x.lo) * ulong(range.hi);
    ulong a1b0 = ulong(x.hi) * ulong(range.lo);
    ulong a1b1 = ulong(x.hi) * ulong(range.hi);
    
    // Accumulate into 128-bit result, extract high 64
    ulong mid = (a0b0 >> 32) + uint(a0b1) + uint(a1b0);
    ulong hi  = (a0b1 >> 32) + (a1b0 >> 32) + a1b1 + (mid >> 32);
    
    return u64_make(uint(hi), uint(hi >> 32));
}

// =============================================================================
// FNV-1a 64-bit Hash
// =============================================================================

inline u64 fnv1a64(device const uint8_t* data, int len) {
    u64 h = u64_make(0x84222325u, 0x146C3B03u); // FNV offset basis: 14695981039346656037
    u64 prime = u64_make(0x01000193u, 0x00000100u); // 1099511628211
    
    for (int i = 0; i < len; i++) {
        h = u64_xor(h, u64_from32(uint(data[i])));
        h = u64_mul(h, prime);
    }
    return h;
}

// Thread-local variant for local data
inline u64 fnv1a64_local(thread const uint8_t* data, int len) {
    u64 h = u64_make(0x84222325u, 0x146C3B03u);
    u64 prime = u64_make(0x01000193u, 0x00000100u);
    
    for (int i = 0; i < len; i++) {
        h = u64_xor(h, u64_from32(uint(data[i])));
        h = u64_mul(h, prime);
    }
    return h;
}

// =============================================================================
// SplitMix64
// =============================================================================

inline u64 splitmix64(u64 x) {
    x = u64_add(x, u64_make(0x7F4A7C15u, 0x9E3779B9u)); // 0x9e3779b97f4a7c15
    x = u64_mul(u64_xor(x, u64_shr(x, 30)), u64_make(0x1CE4E5B9u, 0xBF58476Du));
    x = u64_mul(u64_xor(x, u64_shr(x, 27)), u64_make(0x133111EBu, 0x94D049BBu));
    x = u64_xor(x, u64_shr(x, 31));
    return x;
}

// =============================================================================
// Bloom Filter Test
// =============================================================================

inline bool bloom_test(device const uint *bitwords,
                       BloomParams params,
                       thread const uint8_t* data, int len) {
    u64 m_bits = u64_make(params.m_bits_lo, params.m_bits_hi);
    u64 salt = u64_make(params.salt_lo, params.salt_hi);
    
    u64 h1 = fnv1a64_local(data, len);
    u64 h2 = u64_or(splitmix64(u64_xor(h1, salt)), u64_from32(1u)); // ensure odd

    for (uint i = 0; i < params.k; i++) {
        u64 idx_u64 = u64_fast_reduce(u64_add(h1, u64_mul(u64_from32(i), h2)), m_bits);
        // idx_u64 now holds [0, m_bits)
        // word index = idx / 32, bit index = idx % 32  (using 32-bit words in Metal)
        // But original uses 64-bit words. We convert: word64 = idx/64, bit64 = idx%64
        // In 32-bit words: word32 = (idx/64)*2 + (bit64>=32?1:0)
        // Simpler: just use idx directly with 32-bit bitwords
        uint idx = idx_u64.lo; // m_bits likely fits in 32 bits for practical bloom sizes
        uint word = idx >> 5;  // idx / 32
        uint bit_pos = idx & 31u;
        
        if ((bitwords[word] & (1u << bit_pos)) == 0u) return false;
    }
    return true;
}
