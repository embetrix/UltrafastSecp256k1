// Shim header to allow including bitcoin-core modinv64 directly
// Provides the int128 API and utility macros needed by modinv64_impl.h

#ifndef SECP256K1_MODINV_SHIM_H
#define SECP256K1_MODINV_SHIM_H

#include <cstdint>
#include <cstdlib>

// Suppress VERIFY_CHECK in non-debug mode
#ifndef VERIFY
// no-op
#endif

#define SECP256K1_WIDEMUL_INT128 1

// Use native __int128
typedef __int128 secp256k1_int128;
typedef unsigned __int128 secp256k1_uint128;

#define SECP256K1_INLINE inline
#define VERIFY_CHECK(x) ((void)0)

// int128 native implementation (matching bitcoin-core/secp256k1/src/int128_native_impl.h)
static inline void secp256k1_i128_mul(secp256k1_int128 *r, int64_t a, int64_t b) {
    *r = (secp256k1_int128)a * b;
}
static inline void secp256k1_i128_accum_mul(secp256k1_int128 *r, int64_t a, int64_t b) {
    *r += (secp256k1_int128)a * b;
}
static inline void secp256k1_i128_det(secp256k1_int128 *r, int64_t a, int64_t b, int64_t c, int64_t d) {
    *r = (secp256k1_int128)a * d - (secp256k1_int128)b * c;
}
static inline void secp256k1_i128_rshift(secp256k1_int128 *r, unsigned int n) {
    *r >>= n;
}
static inline uint64_t secp256k1_i128_to_u64(const secp256k1_int128 *a) {
    return (uint64_t)*a;
}
static inline int64_t secp256k1_i128_to_i64(const secp256k1_int128 *a) {
    return (int64_t)(uint64_t)*a;
}
static inline void secp256k1_i128_from_i64(secp256k1_int128 *r, int64_t a) {
    *r = a;
}
static inline int secp256k1_i128_eq_var(const secp256k1_int128 *a, const secp256k1_int128 *b) {
    return *a == *b;
}
static inline int secp256k1_i128_check_pow2(const secp256k1_int128 *r, unsigned int n, int sign) {
    secp256k1_int128 v = sign == 1 ? (secp256k1_int128)1 << n : -((secp256k1_int128)1 << n);
    return *r == v;
}

// ctz64
static inline int secp256k1_ctz64_var(uint64_t x) {
    return __builtin_ctzll(x);
}

// Dummy util.h replacement
// (modinv64.h includes util.h, we provide what's needed above)

#endif // SECP256K1_MODINV_SHIM_H
