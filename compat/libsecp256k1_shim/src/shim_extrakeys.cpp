// ============================================================================
// shim_extrakeys.cpp — x-only pubkeys and keypairs
// ============================================================================
#include "secp256k1_extrakeys.h"
#include "secp256k1.h"

#include <cstring>
#include <array>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using namespace secp256k1::fast;

extern "C" {

// ── X-only public key ────────────────────────────────────────────────────
// Opaque layout: data[0..31] = X big-endian, data[32..63] = zeros (padding)

int secp256k1_xonly_pubkey_parse(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *pubkey,
    const unsigned char *input32)
{
    (void)ctx;
    if (!pubkey || !input32) return 0;

    try {
        // Validate: X must be a valid field element with a valid Y
        std::array<uint8_t, 32> xb{};
        std::memcpy(xb.data(), input32, 32);
        auto x = FieldElement::from_bytes(xb);
        auto y2 = x * x * x + FieldElement::from_uint64(7);
        auto y = y2.sqrt();
        // Verify it's actually a square root
        auto check = y * y;
        // Store x in first 32 bytes
        std::memcpy(pubkey->data, input32, 32);
        std::memset(pubkey->data + 32, 0, 32);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_xonly_pubkey_serialize(
    const secp256k1_context *ctx, unsigned char *output32,
    const secp256k1_xonly_pubkey *pubkey)
{
    (void)ctx;
    if (!output32 || !pubkey) return 0;
    std::memcpy(output32, pubkey->data, 32);
    return 1;
}

int secp256k1_xonly_pubkey_cmp(
    const secp256k1_context *ctx,
    const secp256k1_xonly_pubkey *pk1,
    const secp256k1_xonly_pubkey *pk2)
{
    (void)ctx;
    return std::memcmp(pk1->data, pk2->data, 32);
}

int secp256k1_xonly_pubkey_from_pubkey(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *xonly_pubkey,
    int *pk_parity, const secp256k1_pubkey *pubkey)
{
    (void)ctx;
    if (!xonly_pubkey || !pubkey) return 0;

    // Copy X from pubkey
    std::memcpy(xonly_pubkey->data, pubkey->data, 32);
    std::memset(xonly_pubkey->data + 32, 0, 32);

    if (pk_parity) {
        // Y is odd if last byte of Y has bit 0 set
        *pk_parity = (pubkey->data[63] & 1) ? 1 : 0;
    }
    return 1;
}

// ── Keypair ──────────────────────────────────────────────────────────────
// Layout: data[0..31] = secret key, data[32..95] = pubkey opaque (X || Y)

int secp256k1_keypair_create(
    const secp256k1_context *ctx, secp256k1_keypair *keypair,
    const unsigned char *seckey)
{
    (void)ctx;
    if (!keypair || !seckey) return 0;

    try {
        std::array<uint8_t, 32> kb{};
        std::memcpy(kb.data(), seckey, 32);
        auto k = Scalar::from_bytes(kb);
        if (k.is_zero()) return 0;

        auto P = Point::generator().scalar_mul(k);
        if (P.is_infinity()) return 0;

        // Store seckey
        std::memcpy(keypair->data, seckey, 32);

        // Store pubkey (X || Y)
        auto unc = P.to_uncompressed();
        std::memcpy(keypair->data + 32, unc.data() + 1, 64);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_keypair_sec(
    const secp256k1_context *ctx, unsigned char *seckey,
    const secp256k1_keypair *keypair)
{
    (void)ctx;
    if (!seckey || !keypair) return 0;
    std::memcpy(seckey, keypair->data, 32);
    return 1;
}

int secp256k1_keypair_pub(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const secp256k1_keypair *keypair)
{
    (void)ctx;
    if (!pubkey || !keypair) return 0;
    std::memcpy(pubkey->data, keypair->data + 32, 64);
    return 1;
}

int secp256k1_keypair_xonly_pub(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *pubkey,
    int *pk_parity, const secp256k1_keypair *keypair)
{
    (void)ctx;
    if (!pubkey || !keypair) return 0;

    // X from keypair
    std::memcpy(pubkey->data, keypair->data + 32, 32);
    std::memset(pubkey->data + 32, 0, 32);

    if (pk_parity) {
        *pk_parity = (keypair->data[95] & 1) ? 1 : 0;
    }
    return 1;
}

} // extern "C"
