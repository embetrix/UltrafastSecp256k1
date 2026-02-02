// GLV endomorphism implementation for secp256k1

#include "secp256k1/glv.hpp"
#include "secp256k1/field.hpp"
#include <cstring>

namespace secp256k1::fast {

// Helper: multiply 256-bit scalar by 128-bit value
// Returns the high 256 bits of the product (for division approximation)
static void scalar_mul_shift_256(const Scalar& k, const std::array<uint8_t, 16>& multiplier,
                                  std::array<uint8_t, 32>& result) {
    // Simple implementation: convert to wide integer and multiply
    // For production, this should use optimized 256x128 multiplication
    
    // Get k as bytes (big-endian)
    auto k_bytes = k.to_bytes();
    
    // Perform multiplication (simplified - treating as big-endian)
    uint64_t carry = 0;
    std::array<uint64_t, 8> wide_result{};
    
    // Convert multiplier to uint64_t array (big-endian)
    uint64_t mult_high = 0, mult_low = 0;
    for (int i = 0; i < 8; i++) {
        mult_high = (mult_high << 8) | multiplier[i];
        mult_low = (mult_low << 8) | multiplier[i + 8];
    }
    
    // Convert k to uint64_t array (big-endian)
    std::array<uint64_t, 4> k_limbs{};
    for (int i = 0; i < 4; i++) {
        k_limbs[i] = 0;
        for (int j = 0; j < 8; j++) {
            k_limbs[i] = (k_limbs[i] << 8) | k_bytes[i * 8 + j];
        }
    }
    
    // Multiply and accumulate
    for (int i = 0; i < 4; i++) {
        uint64_t prod_low_low = (k_limbs[i] & 0xFFFFFFFF) * (mult_low & 0xFFFFFFFF);
        uint64_t prod_low_high = (k_limbs[i] & 0xFFFFFFFF) * (mult_low >> 32);
        uint64_t prod_high_low = (k_limbs[i] >> 32) * (mult_low & 0xFFFFFFFF);
        uint64_t prod_high_high = (k_limbs[i] >> 32) * (mult_low >> 32);
        
        // Accumulate (simplified)
        wide_result[i + 2] += prod_low_low;
        wide_result[i + 1] += prod_low_high + prod_high_low;
        wide_result[i] += prod_high_high;
    }
    
    // Extract high 256 bits (shifted result)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 8; j++) {
            result[i * 8 + j] = (wide_result[i] >> (56 - j * 8)) & 0xFF;
        }
    }
}

GLVDecomposition glv_decompose(const Scalar& k) {
    using namespace glv_constants;
    
    GLVDecomposition result;
    
    // Step 1: Compute c1 = round(b2 * k / n) and c2 = round(-b1 * k / n)
    // These are approximations of k/λ using the lattice basis
    
    std::array<uint8_t, 32> c1_bytes{};
    std::array<uint8_t, 32> c2_bytes{};
    
    scalar_mul_shift_256(k, B2, c1_bytes);
    scalar_mul_shift_256(k, MINUS_B1, c2_bytes);
    
    // Step 2: Compute k1 = k - c1*a1 - c2*a2
    // Step 3: Compute k2 = -c1*b1 - c2*b2
    
    // For now, simplified implementation
    // TODO: Implement proper lattice reduction
    
    // Temporary: just split k into two halves for testing
    auto k_bytes = k.to_bytes();
    
    std::array<uint8_t, 32> k1_bytes{};
    std::array<uint8_t, 32> k2_bytes{};
    
    // k1 = lower 128 bits
    std::memset(k1_bytes.data(), 0, 16);
    std::memcpy(k1_bytes.data() + 16, k_bytes.data() + 16, 16);
    
    // k2 = upper 128 bits  
    std::memset(k2_bytes.data(), 0, 16);
    std::memcpy(k2_bytes.data() + 16, k_bytes.data(), 16);
    
    result.k1 = Scalar::from_bytes(k1_bytes);
    result.k2 = Scalar::from_bytes(k2_bytes);
    result.k1_neg = false;
    result.k2_neg = false;
    
    return result;
}

Point apply_endomorphism(const Point& P) {
    if (P.is_infinity()) {
        return P;
    }
    
    // φ(x, y) = (β·x, y)
    // β is a cube root of unity mod p
    
    // Create β as FieldElement
    auto beta_bytes = glv_constants::BETA;
    FieldElement beta = FieldElement::from_bytes(beta_bytes);
    
    // Multiply x by β (using * operator which calls mul())
    FieldElement new_x = P.x_raw() * beta;
    
    // y and z stay the same
    FieldElement new_y = P.y_raw();
    FieldElement new_z = P.z_raw();
    
    return Point::from_jacobian_coords(new_x, new_y, new_z, false);
}

bool verify_endomorphism(const Point& P) {
    if (P.is_infinity()) {
        return true;
    }
    
    // φ(φ(P)) + P should equal O (point at infinity)
    // Because φ³ = identity, so φ² + φ + 1 = 0
    // Therefore: φ²(P) = -P - φ(P)
    
    Point phi_P = apply_endomorphism(P);
    Point phi_phi_P = apply_endomorphism(phi_P);
    
    // φ²(P) + P should equal -φ(P)
    Point sum = phi_phi_P.add(P);
    Point neg_phi_P = phi_P.negate();
    
    // Compare coordinates (normalize to affine first)
    auto sum_x = sum.x().to_bytes();
    auto sum_y = sum.y().to_bytes();
    auto neg_x = neg_phi_P.x().to_bytes();
    auto neg_y = neg_phi_P.y().to_bytes();
    
    return (sum_x == neg_x) && (sum_y == neg_y);
}

} // namespace secp256k1::fast
