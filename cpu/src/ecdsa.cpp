#include "secp256k1/ecdsa.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/config.hpp"  // SECP256K1_FAST_52BIT
#include "secp256k1/field_52.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// ── Half-order for low-S check ───────────────────────────────────────────────
// n/2 = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF5D576E7357A4501DDFE92F46681B20A0
static const Scalar HALF_ORDER = Scalar::from_hex(
    "7fffffffffffffffffffffffffffffff5d576e7357a4501ddfe92f46681b20a0");

// ── Signature Methods ────────────────────────────────────────────────────────

std::pair<std::array<std::uint8_t, 72>, std::size_t> ECDSASignature::to_der() const {
    auto r_bytes = r.to_bytes();
    auto s_bytes = s.to_bytes();

    // Find actual length (skip leading zeros, add 0x00 pad if high bit set)
    auto encode_int = [](const std::array<uint8_t, 32>& val,
                         uint8_t* out) -> size_t {
        size_t start = 0;
        while (start < 31 && val[start] == 0) ++start;
        bool need_pad = (val[start] & 0x80) != 0;
        size_t len = 32 - start + (need_pad ? 1 : 0);

        out[0] = 0x02; // INTEGER tag
        out[1] = static_cast<uint8_t>(len);
        size_t pos = 2;
        if (need_pad) out[pos++] = 0x00;
        std::memcpy(out + pos, val.data() + start, 32 - start);
        return 2 + len;
    };

    std::array<uint8_t, 72> buf{};
    uint8_t r_enc[35]{}, s_enc[35]{};
    size_t r_len = encode_int(r_bytes, r_enc);
    size_t s_len = encode_int(s_bytes, s_enc);

    buf[0] = 0x30; // SEQUENCE tag
    buf[1] = static_cast<uint8_t>(r_len + s_len);
    std::memcpy(buf.data() + 2, r_enc, r_len);
    std::memcpy(buf.data() + 2 + r_len, s_enc, s_len);

    return {buf, 2 + r_len + s_len};
}

std::array<std::uint8_t, 64> ECDSASignature::to_compact() const {
    std::array<uint8_t, 64> out{};
    auto rb = r.to_bytes();
    auto sb = s.to_bytes();
    std::memcpy(out.data(), rb.data(), 32);
    std::memcpy(out.data() + 32, sb.data(), 32);
    return out;
}

ECDSASignature ECDSASignature::from_compact(const std::array<std::uint8_t, 64>& data) {
    std::array<uint8_t, 32> rb{}, sb{};
    std::memcpy(rb.data(), data.data(), 32);
    std::memcpy(sb.data(), data.data() + 32, 32);
    return {Scalar::from_bytes(rb), Scalar::from_bytes(sb)};
}

ECDSASignature ECDSASignature::normalize() const {
    if (is_low_s()) return *this;
    return {r, s.negate()};
}

bool ECDSASignature::is_low_s() const {
    // s <= n/2 ?
    auto s_bytes = s.to_bytes();
    auto half_bytes = HALF_ORDER.to_bytes();
    for (size_t i = 0; i < 32; ++i) {
        if (s_bytes[i] < half_bytes[i]) return true;
        if (s_bytes[i] > half_bytes[i]) return false;
    }
    return true; // equal
}

// ── RFC 6979 Deterministic Nonce ─────────────────────────────────────────────
// Simplified RFC 6979 using HMAC-SHA256.

namespace {

// HMAC-SHA256
struct HMAC_SHA256 {
    static std::array<uint8_t, 32> compute(const uint8_t* key, size_t key_len,
                                            const uint8_t* msg, size_t msg_len) {
        uint8_t ipad[64]{}, opad[64]{};
        uint8_t k_buf[64]{};

        if (key_len > 64) {
            auto h = SHA256::hash(key, key_len);
            std::memcpy(k_buf, h.data(), 32);
        } else {
            std::memcpy(k_buf, key, key_len);
        }

        for (int i = 0; i < 64; ++i) {
            ipad[i] = k_buf[i] ^ 0x36;
            opad[i] = k_buf[i] ^ 0x5c;
        }

        // inner = SHA256(ipad || msg)
        SHA256 inner;
        inner.update(ipad, 64);
        inner.update(msg, msg_len);
        auto inner_hash = inner.finalize();

        // outer = SHA256(opad || inner)
        SHA256 outer;
        outer.update(opad, 64);
        outer.update(inner_hash.data(), 32);
        return outer.finalize();
    }
};

} // namespace

Scalar rfc6979_nonce(const Scalar& private_key,
                     const std::array<uint8_t, 32>& msg_hash) {
    // RFC 6979 Section 3.2
    auto x_bytes = private_key.to_bytes();

    // Step a: h1 = msg_hash (already hashed)
    // Step b: V = 0x01 * 32
    uint8_t V[32];
    std::memset(V, 0x01, 32);

    // Step c: K = 0x00 * 32
    uint8_t K[32];
    std::memset(K, 0x00, 32);

    // Step d: K = HMAC(K, V || 0x00 || x || h1)
    {
        uint8_t buf[97]; // 32 + 1 + 32 + 32
        std::memcpy(buf, V, 32);
        buf[32] = 0x00;
        std::memcpy(buf + 33, x_bytes.data(), 32);
        std::memcpy(buf + 65, msg_hash.data(), 32);
        auto h = HMAC_SHA256::compute(K, 32, buf, 97);
        std::memcpy(K, h.data(), 32);
    }

    // Step e: V = HMAC(K, V)
    {
        auto h = HMAC_SHA256::compute(K, 32, V, 32);
        std::memcpy(V, h.data(), 32);
    }

    // Step f: K = HMAC(K, V || 0x01 || x || h1)
    {
        uint8_t buf[97];
        std::memcpy(buf, V, 32);
        buf[32] = 0x01;
        std::memcpy(buf + 33, x_bytes.data(), 32);
        std::memcpy(buf + 65, msg_hash.data(), 32);
        auto h = HMAC_SHA256::compute(K, 32, buf, 97);
        std::memcpy(K, h.data(), 32);
    }

    // Step g: V = HMAC(K, V)
    {
        auto h = HMAC_SHA256::compute(K, 32, V, 32);
        std::memcpy(V, h.data(), 32);
    }

    // Step h: loop
    for (int attempt = 0; attempt < 100; ++attempt) {
        // V = HMAC(K, V)
        auto h = HMAC_SHA256::compute(K, 32, V, 32);
        std::memcpy(V, h.data(), 32);

        std::array<uint8_t, 32> t;
        std::memcpy(t.data(), V, 32);

        auto candidate = Scalar::from_bytes(t);
        if (!candidate.is_zero()) {
            // Check candidate < order (from_bytes already reduces mod n,
            // but we also need to ensure it wasn't zero before reduction)
            return candidate;
        }

        // K = HMAC(K, V || 0x00)
        uint8_t buf[33];
        std::memcpy(buf, V, 32);
        buf[32] = 0x00;
        auto k2 = HMAC_SHA256::compute(K, 32, buf, 33);
        std::memcpy(K, k2.data(), 32);

        // V = HMAC(K, V)
        auto v2 = HMAC_SHA256::compute(K, 32, V, 32);
        std::memcpy(V, v2.data(), 32);
    }

    return Scalar::zero(); // should never reach
}

// ── ECDSA Sign ───────────────────────────────────────────────────────────────

ECDSASignature ecdsa_sign(const std::array<uint8_t, 32>& msg_hash,
                          const Scalar& private_key) {
    if (private_key.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // z = message hash interpreted as scalar
    auto z = Scalar::from_bytes(msg_hash);

    // Generate deterministic nonce
    auto k = rfc6979_nonce(private_key, msg_hash);
    if (k.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // R = k * G
    auto R = Point::generator().scalar_mul(k);
    if (R.is_infinity()) return {Scalar::zero(), Scalar::zero()};

    // r = R.x mod n
    auto r_fe = R.x();
    auto r_bytes = r_fe.to_bytes();
    auto r = Scalar::from_bytes(r_bytes);
    if (r.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // s = k^{-1} * (z + r * d) mod n
    auto k_inv = k.inverse();
    auto s = k_inv * (z + r * private_key);
    if (s.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // Normalize to low-S (BIP-62)
    ECDSASignature sig{r, s};
    return sig.normalize();
}

// ── ECDSA Verify ─────────────────────────────────────────────────────────────

bool ecdsa_verify(const std::array<uint8_t, 32>& msg_hash,
                  const Point& public_key,
                  const ECDSASignature& sig) {
    // Check r, s in [1, n-1]
    if (sig.r.is_zero() || sig.s.is_zero()) return false;

    // z = message hash as scalar
    auto z = Scalar::from_bytes(msg_hash);

    // w = s^{-1} mod n
    auto w = sig.s.inverse();

    // u1 = z * w mod n
    auto u1 = z * w;

    // u2 = r * w mod n
    auto u2 = sig.r * w;

    // R' = u1 * G + u2 * Q  (4-stream GLV Strauss — single interleaved loop)
    auto R_prime = Point::dual_scalar_mul_gen_point(u1, u2, public_key);

    if (R_prime.is_infinity()) return false;

    // ── Fast Z²-based x-coordinate check (avoids field inverse) ──────
    // Check: R'.x/R'.z² mod n == sig.r
    // Equivalent: sig.r * R'.z² == R'.x (mod p)
    // This saves ~3μs by avoiding the field inversion in Point::x().
#if defined(SECP256K1_FAST_52BIT)
    using FE52 = fast::FieldElement52;

    // Direct Scalar→FE52: sig.r limbs are 4×64 LE, same layout as FieldElement.
    // Since sig.r < n < p, the raw limbs are a valid field element — no reduction needed.
    FE52 r52 = FE52::from_4x64_limbs(sig.r.limbs().data());
    FE52 z2 = R_prime.Z52().square();    // Z²  [1S] mag=1
    FE52 lhs = r52 * z2;                 // r·Z² [1M] mag=1
    lhs.normalize();

    FE52 rx = R_prime.X52();
    rx.normalize();

    if (lhs == rx) return true;

    // Rare case: x_R mod p ∈ [n, p), so x_R mod n = x_R - n = sig.r
    // → need to check (sig.r + n) · Z² == X.  Probability ~2^-128.
    // n = order, p - n ≈ 2^129.  sig.r < n, so sig.r + n < p iff sig.r < p - n.
    //
    // p - n as 4×64 LE limbs:
    // 0x000000000000000145512319_50b75fc4_402da173_2fc9bebf
    static constexpr std::uint64_t PMN_0 = 0x402da1732fc9bebfULL;
    static constexpr std::uint64_t PMN_1 = 0x14551231950b75fcULL;  // top nibble = 1
    // PMN limbs [2] = 0, [3] = 0  → sig.r < p-n iff sig.r fits in ~129 bits
    // Since sig.r < n ≈ 2^256, upper limbs are always >= PMN upper limbs (0).
    // Quick check: r[3]==0 && r[2]==0 → r < 2^128, definitely < p-n.
    // Otherwise compare lexicographically.
    const auto& rl = sig.r.limbs();
    bool r_less_than_pmn;
    if (rl[3] != 0 || rl[2] != 0) {
        r_less_than_pmn = false;  // r >= 2^128 > p-n
    } else if (rl[1] != PMN_1) {
        r_less_than_pmn = (rl[1] < PMN_1);
    } else {
        r_less_than_pmn = (rl[0] < PMN_0);
    }

    if (r_less_than_pmn) {
        // sig.r < p - n, so (sig.r + n) is a valid field element < p.
        // 256-bit addition: r + n (no modular reduction needed).
        static constexpr std::uint64_t N_LIMBS[4] = {
            0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
            0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL
        };
        alignas(32) std::uint64_t rn[4];
        unsigned __int128 acc = static_cast<unsigned __int128>(rl[0]) + N_LIMBS[0];
        rn[0] = static_cast<std::uint64_t>(acc);
        acc = static_cast<unsigned __int128>(rl[1]) + N_LIMBS[1] + static_cast<std::uint64_t>(acc >> 64);
        rn[1] = static_cast<std::uint64_t>(acc);
        acc = static_cast<unsigned __int128>(rl[2]) + N_LIMBS[2] + static_cast<std::uint64_t>(acc >> 64);
        rn[2] = static_cast<std::uint64_t>(acc);
        rn[3] = rl[3] + N_LIMBS[3] + static_cast<std::uint64_t>(acc >> 64);

        FE52 r2_52 = FE52::from_4x64_limbs(rn);
        FE52 lhs2 = r2_52 * z2;
        lhs2.normalize();
        if (lhs2 == rx) return true;
    }

    return false;
#else
    // Fallback: extract affine x via field inverse
    auto r_fe = FieldElement::from_bytes(sig.r.to_bytes());
    auto rx_fe = R_prime.x();
    if (r_fe == rx_fe) return true;

    // Rare case: check (sig.r + n) mod p
    static const Scalar N_SCALAR = Scalar::from_hex(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");
    auto r_plus_n_bytes = (sig.r + N_SCALAR).to_bytes();
    auto r2_fe = FieldElement::from_bytes(r_plus_n_bytes);
    return r2_fe == rx_fe;
#endif
}

} // namespace secp256k1
