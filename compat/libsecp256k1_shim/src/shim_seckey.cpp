// ============================================================================
// shim_seckey.cpp — Secret key verification and tweaking
// ============================================================================
#include "secp256k1.h"

#include <cstring>
#include <array>

#include "secp256k1/scalar.hpp"

using namespace secp256k1::fast;

extern "C" {

int secp256k1_ec_seckey_verify(
    const secp256k1_context *ctx, const unsigned char *seckey)
{
    (void)ctx;
    if (!seckey) return 0;

    try {
        std::array<uint8_t, 32> kb{};
        std::memcpy(kb.data(), seckey, 32);
        auto k = Scalar::from_bytes(kb);
        return k.is_zero() ? 0 : 1;
    } catch (...) { return 0; }
}

int secp256k1_ec_seckey_negate(
    const secp256k1_context *ctx, unsigned char *seckey)
{
    (void)ctx;
    if (!seckey) return 0;

    try {
        std::array<uint8_t, 32> kb{};
        std::memcpy(kb.data(), seckey, 32);
        auto k = Scalar::from_bytes(kb);
        if (k.is_zero()) return 0;
        auto neg = k.negate();
        auto out = neg.to_bytes();
        std::memcpy(seckey, out.data(), 32);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_ec_seckey_tweak_add(
    const secp256k1_context *ctx, unsigned char *seckey,
    const unsigned char *tweak32)
{
    (void)ctx;
    if (!seckey || !tweak32) return 0;

    try {
        std::array<uint8_t, 32> kb{}, tb{};
        std::memcpy(kb.data(), seckey, 32);
        std::memcpy(tb.data(), tweak32, 32);
        auto k = Scalar::from_bytes(kb);
        auto t = Scalar::from_bytes(tb);
        if (k.is_zero()) return 0;
        auto result = k + t;
        if (result.is_zero()) return 0;
        auto out = result.to_bytes();
        std::memcpy(seckey, out.data(), 32);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_ec_seckey_tweak_mul(
    const secp256k1_context *ctx, unsigned char *seckey,
    const unsigned char *tweak32)
{
    (void)ctx;
    if (!seckey || !tweak32) return 0;

    try {
        std::array<uint8_t, 32> kb{}, tb{};
        std::memcpy(kb.data(), seckey, 32);
        std::memcpy(tb.data(), tweak32, 32);
        auto k = Scalar::from_bytes(kb);
        auto t = Scalar::from_bytes(tb);
        if (k.is_zero() || t.is_zero()) return 0;
        auto result = k * t;
        if (result.is_zero()) return 0;
        auto out = result.to_bytes();
        std::memcpy(seckey, out.data(), 32);
        return 1;
    } catch (...) { return 0; }
}

} // extern "C"
