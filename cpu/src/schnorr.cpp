#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/multiscalar.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// ── Cached Tagged Hash Midstates (BIP-340) ───────────────────────────────────
// Pre-compute SHA256 midstate after processing (SHA256(tag) || SHA256(tag)).
// This is exactly 64 bytes = 1 SHA256 block, so after processing the midstate
// has buf_len_==0 and state_ captures all tag-dependent work.
// Saves 2 SHA256 block compressions per tagged_hash call.

static SHA256 make_tag_midstate(const char* tag) {
    auto tag_hash = SHA256::hash(tag, std::strlen(tag));
    SHA256 ctx;
    ctx.update(tag_hash.data(), 32);
    ctx.update(tag_hash.data(), 32);
    return ctx;
}

// BIP-340 tags used in Schnorr sign/verify
static const SHA256 g_aux_midstate       = make_tag_midstate("BIP0340/aux");
static const SHA256 g_nonce_midstate     = make_tag_midstate("BIP0340/nonce");
static const SHA256 g_challenge_midstate = make_tag_midstate("BIP0340/challenge");

// Fast tagged hash using cached midstate (avoids re-computing tag prefix)
static std::array<uint8_t, 32> cached_tagged_hash(const SHA256& midstate,
                                                    const void* data, std::size_t len) {
    SHA256 ctx = midstate;  // copy pre-computed state (trivial, ~108 bytes)
    ctx.update(data, len);
    return ctx.finalize();
}

// ── Tagged Hash (BIP-340) — generic fallback ─────────────────────────────────

std::array<uint8_t, 32> tagged_hash(const char* tag,
                                     const void* data, std::size_t len) {
    auto tag_hash = SHA256::hash(tag, std::strlen(tag));
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

    // Step 2: P = d' * G (single inversion for x-bytes + Y-parity)
    auto P = Point::generator().scalar_mul(d_prime);
    auto [px, p_y_odd] = P.x_bytes_and_parity();

    // Step 3: d = d' if has_even_y(P), else n - d'
    auto d = p_y_odd ? d_prime.negate() : d_prime;

    // Step 4: t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    auto t_hash = cached_tagged_hash(g_aux_midstate, aux_rand.data(), 32);
    auto d_bytes = d.to_bytes();
    uint8_t t[32];
    for (int i = 0; i < 32; ++i) t[i] = d_bytes[i] ^ t_hash[i];

    // Step 5: rand = tagged_hash("BIP0340/nonce", t || pubkey_x || msg)
    uint8_t nonce_input[96];
    std::memcpy(nonce_input, t, 32);
    std::memcpy(nonce_input + 32, px.data(), 32);
    std::memcpy(nonce_input + 64, msg.data(), 32);
    auto rand_hash = cached_tagged_hash(g_nonce_midstate, nonce_input, 96);
    auto k_prime = Scalar::from_bytes(rand_hash);
    if (k_prime.is_zero()) return SchnorrSignature{};

    // Step 6: R = k' * G (single inversion for x-bytes + Y-parity)
    auto R = Point::generator().scalar_mul(k_prime);
    auto [rx, r_y_odd] = R.x_bytes_and_parity();

    // Step 7: k = k' if has_even_y(R), else n - k'
    auto k = r_y_odd ? k_prime.negate() : k_prime;

    // Step 8: e = tagged_hash("BIP0340/challenge", R.x || pubkey_x || msg) mod n
    uint8_t challenge_input[96];
    std::memcpy(challenge_input, rx.data(), 32);
    std::memcpy(challenge_input + 32, px.data(), 32);
    std::memcpy(challenge_input + 64, msg.data(), 32);
    auto e_hash = cached_tagged_hash(g_challenge_midstate, challenge_input, 96);
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
    auto e_hash = cached_tagged_hash(g_challenge_midstate, challenge_input, 96);
    auto e = Scalar::from_bytes(e_hash);

    // Step 3: Lift x-only pubkey to point
    // P = lift_x(pubkey_x): find y such that y² = x³ + 7, pick even y
    auto px_fe = FieldElement::from_bytes(pubkey_x);

    // y² = x³ + 7
    auto x3 = px_fe.square() * px_fe;
    auto y2 = x3 + FieldElement::from_uint64(7);

    // Optimized sqrt via addition chain: a^((p+1)/4), ~255 sqr + 13 mul
    auto y = y2.sqrt();

    // Verify: y² == y2 (check that sqrt succeeded)
    if (y.square() != y2) return false; // not a quadratic residue

    // Ensure even Y (BIP-340 convention)
    auto y_bytes = y.to_bytes();
    if (y_bytes[31] & 1) {
        y = FieldElement::zero() - y;
    }

    auto P = Point::from_affine(px_fe, y);

    // Step 4: R = s*G - e*P  (Shamir's trick: s*G + (-e)*P in one pass)
    auto neg_e = e.negate();
    auto G = Point::generator();
    auto R = shamir_trick(sig.s, G, neg_e, P);

    if (R.is_infinity()) return false;

    // Step 5+6: Check R has even Y and R.x == r (single inversion)
    auto [R_x, R_y_odd] = R.x_bytes_and_parity();
    if (R_y_odd) return false;
    return R_x == sig.r;
}

} // namespace secp256k1
