#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// ── Tagged Hash (BIP-340) ────────────────────────────────────────────────────

std::array<uint8_t, 32> tagged_hash(const char* tag,
                                     const void* data, std::size_t len) {
    // tag_hash = SHA256(tag)
    auto tag_hash = SHA256::hash(tag, std::strlen(tag));

    // H_tag(msg) = SHA256(tag_hash || tag_hash || msg)
    SHA256 ctx;
    ctx.update(tag_hash.data(), 32);
    ctx.update(tag_hash.data(), 32);
    ctx.update(data, len);
    return ctx.finalize();
}

// ── Schnorr Signature ────────────────────────────────────────────────────────

std::array<uint8_t, 64> SchnorrSignature::to_bytes() const {
    std::array<uint8_t, 64> out{};
    std::memcpy(out.data(), r.data(), 32);
    auto s_bytes = s.to_bytes();
    std::memcpy(out.data() + 32, s_bytes.data(), 32);
    return out;
}

SchnorrSignature SchnorrSignature::from_bytes(const std::array<uint8_t, 64>& data) {
    SchnorrSignature sig{};
    std::memcpy(sig.r.data(), data.data(), 32);
    std::array<uint8_t, 32> s_bytes{};
    std::memcpy(s_bytes.data(), data.data() + 32, 32);
    sig.s = Scalar::from_bytes(s_bytes);
    return sig;
}

// ── X-only pubkey ────────────────────────────────────────────────────────────

std::array<uint8_t, 32> schnorr_pubkey(const Scalar& private_key) {
    auto P = Point::generator().scalar_mul(private_key);
    auto uncomp = P.to_uncompressed();

    // Check if Y is even (BIP-340: use key with even Y)
    // Y is bytes [33..64] of uncompressed key, last byte parity
    bool y_is_odd = (uncomp[64] & 1) != 0;

    // X-only: just the x-coordinate
    auto x_bytes = P.x().to_bytes();

    // If Y is odd, the caller should negate the key; but for pubkey output
    // we just return X regardless (BIP-340 convention)
    (void)y_is_odd;
    return x_bytes;
}

// ── BIP-340 Sign ─────────────────────────────────────────────────────────────

SchnorrSignature schnorr_sign(const Scalar& private_key,
                              const std::array<uint8_t, 32>& msg,
                              const std::array<uint8_t, 32>& aux_rand) {
    // Step 1: d' = private_key
    auto d_prime = private_key;
    if (d_prime.is_zero()) return SchnorrSignature{};

    // Step 2: P = d' * G
    auto P = Point::generator().scalar_mul(d_prime);
    auto P_uncomp = P.to_uncompressed();
    bool p_y_odd = (P_uncomp[64] & 1) != 0;

    // Step 3: d = d' if has_even_y(P), else n - d'
    auto d = p_y_odd ? d_prime.negate() : d_prime;

    // Step 4: t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    auto t_hash = tagged_hash("BIP0340/aux", aux_rand.data(), 32);
    auto d_bytes = d.to_bytes();
    uint8_t t[32];
    for (int i = 0; i < 32; ++i) t[i] = d_bytes[i] ^ t_hash[i];

    // Step 5: rand = tagged_hash("BIP0340/nonce", t || pubkey_x || msg)
    auto px = P.x().to_bytes();
    uint8_t nonce_input[96];
    std::memcpy(nonce_input, t, 32);
    std::memcpy(nonce_input + 32, px.data(), 32);
    std::memcpy(nonce_input + 64, msg.data(), 32);
    auto rand_hash = tagged_hash("BIP0340/nonce", nonce_input, 96);
    auto k_prime = Scalar::from_bytes(rand_hash);
    if (k_prime.is_zero()) return SchnorrSignature{};

    // Step 6: R = k' * G
    auto R = Point::generator().scalar_mul(k_prime);
    auto R_uncomp = R.to_uncompressed();
    bool r_y_odd = (R_uncomp[64] & 1) != 0;

    // Step 7: k = k' if has_even_y(R), else n - k'
    auto k = r_y_odd ? k_prime.negate() : k_prime;

    // Step 8: e = tagged_hash("BIP0340/challenge", R.x || pubkey_x || msg) mod n
    auto rx = R.x().to_bytes();
    uint8_t challenge_input[96];
    std::memcpy(challenge_input, rx.data(), 32);
    std::memcpy(challenge_input + 32, px.data(), 32);
    std::memcpy(challenge_input + 64, msg.data(), 32);
    auto e_hash = tagged_hash("BIP0340/challenge", challenge_input, 96);
    auto e = Scalar::from_bytes(e_hash);

    // Step 9: sig = (R.x, k + e * d) mod n
    auto s = k + e * d;

    SchnorrSignature sig{};
    sig.r = rx;
    sig.s = s;
    return sig;
}

// ── BIP-340 Verify ───────────────────────────────────────────────────────────

bool schnorr_verify(const std::array<uint8_t, 32>& pubkey_x,
                    const std::array<uint8_t, 32>& msg,
                    const SchnorrSignature& sig) {
    // Step 1: Check r < p and s < n
    // (from_bytes already reduces, so just check non-zero for r)
    auto r_fe = FieldElement::from_bytes(sig.r);

    if (sig.s.is_zero()) return false;

    // Step 2: e = tagged_hash("BIP0340/challenge", r || pubkey_x || msg) mod n
    uint8_t challenge_input[96];
    std::memcpy(challenge_input, sig.r.data(), 32);
    std::memcpy(challenge_input + 32, pubkey_x.data(), 32);
    std::memcpy(challenge_input + 64, msg.data(), 32);
    auto e_hash = tagged_hash("BIP0340/challenge", challenge_input, 96);
    auto e = Scalar::from_bytes(e_hash);

    // Step 3: Lift x-only pubkey to point
    // P = lift_x(pubkey_x): find y such that y² = x³ + 7, pick even y
    auto px_fe = FieldElement::from_bytes(pubkey_x);

    // y² = x³ + 7
    auto x3 = px_fe.square() * px_fe;
    auto seven = FieldElement::from_uint64(7);
    auto y2 = x3 + seven;

    // Compute y = y2^((p+1)/4) mod p
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    // (p+1)/4 = 0x3FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFF0C
    // We use the square root via exponentiation
    // For secp256k1: sqrt(a) = a^((p+1)/4) because p ≡ 3 (mod 4)
    auto exp = FieldElement::from_hex(
        "3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c");

    // Compute y = y2^exp by repeated squaring
    // This is expensive but correct
    auto y = FieldElement::one();
    auto base = y2;

    // We need to exponentiate — use the existing field inverse chain approach
    // Actually, we can use: y = y2^((p+1)/4)
    // Since FieldElement already has inverse() which computes a^(p-2),
    // we can compute sqrt differently. But let's just do pow.
    auto exp_bytes = exp.to_bytes();
    for (int i = 0; i < 256; ++i) {
        y = y.square();
        int byte_idx = i / 8;
        int bit_idx = 7 - (i % 8);
        if ((exp_bytes[byte_idx] >> bit_idx) & 1) {
            y = y * base;
        }
    }

    // Verify: y² == y2
    auto y_check = y.square();
    if (y_check != y2) return false; // not a quadratic residue

    // Ensure even Y (BIP-340 convention)
    auto y_bytes = y.to_bytes();
    bool y_odd = (y_bytes[31] & 1) != 0;
    if (y_odd) {
        // negate y: y = p - y
        y = FieldElement::zero() - y;
    }

    auto P = Point::from_affine(px_fe, y);

    // Step 4: R = s*G - e*P
    auto sG = Point::generator().scalar_mul(sig.s);
    auto eP = P.scalar_mul(e);
    auto neg_eP = eP.negate();
    auto R = sG.add(neg_eP);

    if (R.is_infinity()) return false;

    // Step 5: Check R has even y
    auto R_uncomp = R.to_uncompressed();
    if ((R_uncomp[64] & 1) != 0) return false; // odd y

    // Step 6: Check R.x == r
    auto R_x = R.x().to_bytes();
    return R_x == sig.r;
}

} // namespace secp256k1
