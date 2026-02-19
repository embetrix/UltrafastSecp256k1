#ifndef SECP256K1_SHA256_HPP
#define SECP256K1_SHA256_HPP
#pragma once

// ============================================================================
// SHA-256 implementation for ECDSA / Schnorr
// ============================================================================
// Hardware-accelerated via SHA-NI when available (runtime dispatch).
// Falls back to portable C++ on non-x86 or CPUs without SHA extensions.
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>

namespace secp256k1 {

// Forward declare: implemented in hash_accel.cpp, dispatches to SHA-NI or scalar
namespace detail {
    void sha256_compress_dispatch(const std::uint8_t block[64],
                                  std::uint32_t state[8]) noexcept;
}

class SHA256 {
public:
    using digest_type = std::array<std::uint8_t, 32>;

    SHA256() noexcept { reset(); }

    void reset() noexcept {
        state_[0] = 0x6a09e667u; state_[1] = 0xbb67ae85u;
        state_[2] = 0x3c6ef372u; state_[3] = 0xa54ff53au;
        state_[4] = 0x510e527fu; state_[5] = 0x9b05688cu;
        state_[6] = 0x1f83d9abu; state_[7] = 0x5be0cd19u;
        total_ = 0;
        buf_len_ = 0;
    }

    void update(const void* data, std::size_t len) noexcept {
        auto ptr = static_cast<const std::uint8_t*>(data);
        total_ += len;

        if (buf_len_ > 0) {
            std::size_t fill = 64 - buf_len_;
            if (len < fill) {
                std::memcpy(buf_ + buf_len_, ptr, len);
                buf_len_ += len;
                return;
            }
            std::memcpy(buf_ + buf_len_, ptr, fill);
            detail::sha256_compress_dispatch(buf_, state_);
            ptr += fill;
            len -= fill;
            buf_len_ = 0;
        }

        while (len >= 64) {
            detail::sha256_compress_dispatch(ptr, state_);
            ptr += 64;
            len -= 64;
        }

        if (len > 0) {
            std::memcpy(buf_, ptr, len);
            buf_len_ = len;
        }
    }

    digest_type finalize() noexcept {
        std::uint64_t bits = total_ * 8;

        // Pad
        std::uint8_t pad = static_cast<std::uint8_t>(0x80);
        update(&pad, 1);
        std::uint8_t zero = 0;
        while (buf_len_ != 56) {
            update(&zero, 1);
        }

        // Append length (big-endian)
        std::uint8_t len_be[8];
        for (int i = 7; i >= 0; --i) {
            len_be[i] = static_cast<std::uint8_t>(bits);
            bits >>= 8;
        }
        update(len_be, 8);

        digest_type out{};
        for (int i = 0; i < 8; ++i) {
            out[i * 4 + 0] = static_cast<std::uint8_t>(state_[i] >> 24);
            out[i * 4 + 1] = static_cast<std::uint8_t>(state_[i] >> 16);
            out[i * 4 + 2] = static_cast<std::uint8_t>(state_[i] >> 8);
            out[i * 4 + 3] = static_cast<std::uint8_t>(state_[i]);
        }
        return out;
    }

    // One-shot convenience
    static digest_type hash(const void* data, std::size_t len) noexcept {
        SHA256 ctx;
        ctx.update(data, len);
        return ctx.finalize();
    }

    // Double-SHA256: SHA256(SHA256(data))
    static digest_type hash256(const void* data, std::size_t len) noexcept {
        auto h1 = hash(data, len);
        return hash(h1.data(), h1.size());
    }

private:
    std::uint32_t state_[8]{};
    std::uint8_t buf_[64]{};
    std::size_t buf_len_ = 0;
    std::uint64_t total_ = 0;
};

} // namespace secp256k1

#endif // SECP256K1_SHA256_HPP
