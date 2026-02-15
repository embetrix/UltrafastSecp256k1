#ifndef SECP256K1_SCHNORR_HPP
#define SECP256K1_SCHNORR_HPP
#pragma once

// ============================================================================
// Schnorr Signatures (BIP-340) for secp256k1
// ============================================================================
// Implements BIP-340 Schnorr signatures:
//   - X-only public keys (32 bytes)
//   - 64-byte signatures (R.x || s)
//   - Tagged hashing per BIP-340 spec
//
// Reference: https://github.com/bitcoin/bips/blob/master/bip-0340.mediawiki
// ============================================================================

#include <array>
#include <cstdint>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1 {

// ── Schnorr Signature ────────────────────────────────────────────────────────

struct SchnorrSignature {
    std::array<std::uint8_t, 32> r;  // R.x (x-coordinate of nonce point)
    fast::Scalar s;                   // scalar s

    // 64-byte compact encoding: r (32) || s (32)
    std::array<std::uint8_t, 64> to_bytes() const;
    static SchnorrSignature from_bytes(const std::array<std::uint8_t, 64>& data);
};

// ── BIP-340 Operations ───────────────────────────────────────────────────────

// Sign a 32-byte message using BIP-340.
// private_key: 32-byte secret key (scalar)
// msg: 32-byte message (typically a hash)
// aux_rand: 32 bytes of auxiliary randomness (can be zeros for deterministic)
SchnorrSignature schnorr_sign(const fast::Scalar& private_key,
                              const std::array<std::uint8_t, 32>& msg,
                              const std::array<std::uint8_t, 32>& aux_rand);

// Verify a BIP-340 Schnorr signature.
// pubkey_x: 32-byte x-only public key
// msg: 32-byte message
// sig: 64-byte signature
bool schnorr_verify(const std::array<std::uint8_t, 32>& pubkey_x,
                    const std::array<std::uint8_t, 32>& msg,
                    const SchnorrSignature& sig);

// ── Tagged Hashing (BIP-340) ─────────────────────────────────────────────────

// H_tag(msg) = SHA256(SHA256(tag) || SHA256(tag) || msg)
std::array<std::uint8_t, 32> tagged_hash(const char* tag,
                                          const void* data, std::size_t len);

// X-only public key from private key (BIP-340: negate if Y is odd)
std::array<std::uint8_t, 32> schnorr_pubkey(const fast::Scalar& private_key);

} // namespace secp256k1

#endif // SECP256K1_SCHNORR_HPP
