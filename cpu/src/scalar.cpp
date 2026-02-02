#include "secp256k1/scalar.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#if defined(_MSC_VER) && !defined(__clang__)
#include "rdtsc.h"
#endif

namespace secp256k1::fast {
namespace {

using limbs4 = std::array<std::uint64_t, 4>;

constexpr limbs4 ORDER{
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

constexpr limbs4 ONE{1ULL, 0ULL, 0ULL, 0ULL};

#if defined(_MSC_VER) && !defined(__clang__)

inline std::uint64_t add64(std::uint64_t a, std::uint64_t b, unsigned char& carry) {
    unsigned __int64 out;
    carry = _addcarry_u64(carry, a, b, &out);
    return out;
}

inline std::uint64_t sub64(std::uint64_t a, std::uint64_t b, unsigned char& borrow) {
    unsigned __int64 out;
    borrow = _subborrow_u64(borrow, a, b, &out);
    return out;
}

#else

inline std::uint64_t add64(std::uint64_t a, std::uint64_t b, unsigned char& carry) {
    unsigned __int128 sum = static_cast<unsigned __int128>(a) + b + carry;
    carry = static_cast<unsigned char>(sum >> 64);
    return static_cast<std::uint64_t>(sum);
}

inline std::uint64_t sub64(std::uint64_t a, std::uint64_t b, unsigned char& borrow) {
    unsigned __int128 diff = static_cast<unsigned __int128>(a) - b - borrow;
    borrow = static_cast<unsigned char>((diff >> 127) & 1);
    return static_cast<std::uint64_t>(diff);
}

#endif

[[nodiscard]] bool ge(const limbs4& a, const limbs4& b) {
    for (std::size_t i = 4; i-- > 0;) {
        if (a[i] > b[i]) {
            return true;
        }
        if (a[i] < b[i]) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] limbs4 sub_impl(const limbs4& a, const limbs4& b);

[[nodiscard]] limbs4 add_impl(const limbs4& a, const limbs4& b) {
    // Compute raw 256-bit sum with carry
    limbs4 sum{};
    unsigned char carry = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        sum[i] = add64(a[i], b[i], carry);
    }

    // Compute sum - ORDER without modular wrap to decide reduction
    limbs4 sum_minus_order{};
    unsigned char borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        sum_minus_order[i] = sub64(sum[i], ORDER[i], borrow);
    }

    // If carry from addition (sum >= 2^256) OR no borrow in subtraction (sum >= ORDER),
    // then result = sum - ORDER; otherwise result = sum.
    if (carry || (borrow == 0)) {
        return sum_minus_order;
    }
    return sum;
}

[[nodiscard]] limbs4 sub_impl(const limbs4& a, const limbs4& b) {
    limbs4 out{};
    unsigned char borrow = 0;
    for (std::size_t i = 0; i < 4; ++i) {
        out[i] = sub64(a[i], b[i], borrow);
    }
    if (borrow) {
        unsigned char carry = 0;
        for (std::size_t i = 0; i < 4; ++i) {
            out[i] = add64(out[i], ORDER[i], carry);
        }
    }
    return out;
}

} // namespace

Scalar::Scalar() = default;

Scalar::Scalar(const limbs_type& limbs, bool normalized) : limbs_(limbs) {
    if (!normalized && ge(limbs_, ORDER)) {
        limbs_ = sub_impl(limbs_, ORDER);
    }
}

Scalar Scalar::zero() {
    return Scalar();
}

Scalar Scalar::one() {
    return Scalar(ONE, true);
}

Scalar Scalar::from_uint64(std::uint64_t value) {
    limbs_type limbs{};
    limbs[0] = value;
    return Scalar(limbs, true);
}

Scalar Scalar::from_limbs(const limbs_type& limbs) {
    Scalar s;
    s.limbs_ = limbs;
    if (ge(s.limbs_, ORDER)) {
        s.limbs_ = sub_impl(s.limbs_, ORDER);
    }
    return s;
}

Scalar Scalar::from_bytes(const std::array<std::uint8_t, 32>& bytes) {
    limbs4 limbs{};
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t limb = 0;
        for (std::size_t j = 0; j < 8; ++j) {
            limb = (limb << 8) | bytes[i * 8 + j];
        }
        limbs[3 - i] = limb;
    }
    if (ge(limbs, ORDER)) {
        limbs = sub_impl(limbs, ORDER);
    }
    Scalar s;
    s.limbs_ = limbs;
    return s;
}

std::array<std::uint8_t, 32> Scalar::to_bytes() const {
    std::array<std::uint8_t, 32> out{};
    for (std::size_t i = 0; i < 4; ++i) {
        std::uint64_t limb = limbs_[3 - i];
        for (std::size_t j = 0; j < 8; ++j) {
            out[i * 8 + j] = static_cast<std::uint8_t>(limb >> (56 - 8 * j));
        }
    }
    return out;
}

std::string Scalar::to_hex() const {
    auto bytes = to_bytes();
    std::string hex;
    hex.reserve(64);
    static const char hex_chars[] = "0123456789abcdef";
    for (auto b : bytes) {
        hex += hex_chars[(b >> 4) & 0xF];
        hex += hex_chars[b & 0xF];
    }
    return hex;
}

Scalar Scalar::from_hex(const std::string& hex) {
    if (hex.length() != 64) {
        throw std::invalid_argument("Hex string must be exactly 64 characters (32 bytes)");
    }
    
    std::array<std::uint8_t, 32> bytes{};
    for (size_t i = 0; i < 32; i++) {
        char c1 = hex[i * 2];
        char c2 = hex[i * 2 + 1];
        
        auto hex_to_nibble = [](char c) -> uint8_t {
            if (c >= '0' && c <= '9') return c - '0';
            if (c >= 'a' && c <= 'f') return c - 'a' + 10;
            if (c >= 'A' && c <= 'F') return c - 'A' + 10;
            throw std::invalid_argument("Invalid hex character");
        };
        
        bytes[i] = (hex_to_nibble(c1) << 4) | hex_to_nibble(c2);
    }
    
    return from_bytes(bytes);
}

Scalar Scalar::operator+(const Scalar& rhs) const {
    return Scalar(add_impl(limbs_, rhs.limbs_), true);
}

Scalar Scalar::operator-(const Scalar& rhs) const {
    return Scalar(sub_impl(limbs_, rhs.limbs_), true);
}

Scalar Scalar::operator*(const Scalar& rhs) const {
    // Double-and-add with proper modular reduction at each step
    // This is slower but guaranteed correct
    Scalar result = Scalar::zero();
    Scalar base = *this;
    
    for (std::size_t bit = 0; bit < 256; ++bit) {
        if (rhs.bit(bit)) {
            result += base;
        }
        base += base;
    }
    
    return result;
}

Scalar& Scalar::operator+=(const Scalar& rhs) {
    limbs_ = add_impl(limbs_, rhs.limbs_);
    return *this;
}

Scalar& Scalar::operator-=(const Scalar& rhs) {
    limbs_ = sub_impl(limbs_, rhs.limbs_);
    return *this;
}

Scalar& Scalar::operator*=(const Scalar& rhs) {
    *this = *this * rhs;
    return *this;
}

bool Scalar::is_zero() const noexcept {
    for (auto limb : limbs_) {
        if (limb != 0) {
            return false;
        }
    }
    return true;
}

bool Scalar::operator==(const Scalar& rhs) const noexcept {
    return limbs_ == rhs.limbs_;
}

std::uint8_t Scalar::bit(std::size_t index) const {
    if (index >= 256) {
        return 0;
    }
    std::size_t limb_idx = index / 64;
    std::size_t bit_idx = index % 64;
    return static_cast<std::uint8_t>((limbs_[limb_idx] >> bit_idx) & 0x1u);
}

// Phase 5.6: NAF (Non-Adjacent Form) encoding
// Converts scalar to signed representation {-1, 0, 1}
// NAF property: no two adjacent non-zero digits
// This reduces the number of non-zero digits by ~33%
// Algorithm: scan from LSB, if odd → take ±1, adjust remaining
std::vector<int8_t> Scalar::to_naf() const {
    std::vector<int8_t> naf;
    naf.reserve(257);  // Maximum NAF length is n+1 for n-bit number
    
    // Work with a mutable copy
    Scalar k = *this;
    
    while (!k.is_zero()) {
        if (k.bit(0) == 1) {  // k is odd
            // Get lowest 2 bits to determine sign
            std::uint8_t low_bits = static_cast<std::uint8_t>(k.limbs_[0] & 0x3);
            int8_t digit;
            
            if (low_bits == 1 || low_bits == 2) {
                // k ≡ 1 or 2 (mod 4) → use +1
                digit = 1;
                k -= Scalar::one();
            } else {
                // k ≡ 3 (mod 4) → use -1 (equivalent to k-1 being even)
                digit = -1;
                k += Scalar::one();
            }
            naf.push_back(digit);
        } else {
            // k is even → digit is 0
            naf.push_back(0);
        }
        
        // Divide k by 2 (right shift)
        std::uint64_t carry = 0;
        for (int i = 3; i >= 0; --i) {
            std::uint64_t limb = k.limbs_[i];
            k.limbs_[i] = (limb >> 1) | (carry << 63);
            carry = limb & 1;
        }
    }
    
    // NAF can be one bit longer than the original number
    // but we're done when k becomes zero
    return naf;
}

// Phase 5.7: wNAF (width-w Non-Adjacent Form)
// Converts scalar to signed odd-digit representation
// Window width w → digits in range {±1, ±3, ±5, ..., ±(2^w - 1)}
// Property: At most one non-zero digit in any w consecutive positions
// This reduces precompute table size by ~50% (only odd multiples needed)
std::vector<int8_t> Scalar::to_wnaf(unsigned width) const {
    if (width < 2 || width > 8) {
        throw std::invalid_argument("wNAF width must be between 2 and 8");
    }
    
    std::vector<int8_t> wnaf;
    wnaf.reserve(257);  // Maximum length
    
    Scalar k = *this;
    const int window_size = 1 << width;          // 2^w
    const int window_mask = window_size - 1;      // 2^w - 1
    const int window_half = window_size >> 1;     // 2^(w-1)
    
    while (!k.is_zero()) {
        if (k.bit(0) == 1) {  // k is odd
            // Extract w bits
            int digit = static_cast<int>(k.limbs_[0] & window_mask);
            
            // If digit >= 2^(w-1), use negative representation
            if (digit >= window_half) {
                digit -= window_size;  // Make negative
                k += Scalar::from_uint64(static_cast<std::uint64_t>(-digit));
            } else {
                k -= Scalar::from_uint64(static_cast<std::uint64_t>(digit));
            }
            
            wnaf.push_back(static_cast<int8_t>(digit));
        } else {
            // k is even → digit is 0
            wnaf.push_back(0);
        }
        
        // Divide k by 2 (right shift)
        std::uint64_t carry = 0;
        for (int i = 3; i >= 0; --i) {
            std::uint64_t limb = k.limbs_[i];
            k.limbs_[i] = (limb >> 1) | (carry << 63);
            carry = limb & 1;
        }
    }
    
    return wnaf;
}

} // namespace secp256k1::fast
