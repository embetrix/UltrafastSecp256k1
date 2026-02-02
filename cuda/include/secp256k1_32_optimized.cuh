#pragma once
#include <cuda_runtime.h>
#include <cstdint>

// ============================================================================
// Optimized 32-bit secp256k1 implementation
// Based on research from gecc (Montgomery CIOS) and best practices
// ============================================================================

namespace secp_opt {

struct FieldElement {
    uint32_t limbs[8];
};

struct Scalar {
    uint32_t limbs[8];
};

struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    bool infinity;
};

struct AffinePoint {
    FieldElement x;
    FieldElement y;
};

// Zero-cost conversion from 64-bit FieldElement (defined in secp256k1.cuh)
// Memory layout is IDENTICAL
__device__ __forceinline__ FieldElement* toOptField(secp256k1::cuda::FieldElement* fe) {
    return reinterpret_cast<FieldElement*>(fe);
}

__device__ __forceinline__ const FieldElement* toOptField(const secp256k1::cuda::FieldElement* fe) {
    return reinterpret_cast<const FieldElement*>(fe);
}

// P = 2^256 - 2^32 - 977
__constant__ static const uint32_t MODULUS[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ static const uint32_t ORDER[8] = {
    0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
    0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

__constant__ static const uint32_t GENERATOR_X[8] = {
    0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB, 
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};

__constant__ static const uint32_t GENERATOR_Y[8] = {
    0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448, 
    0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
};

__constant__ static const uint32_t FIELD_ONE[8] = {
    1, 0, 0, 0, 0, 0, 0, 0
};

// Inverse of -P mod 2^32 (for Montgomery)
static constexpr uint32_t FIELD_M_INV = 0xD2253531;

// ============================================================================
// Optimized multiplication using CIOS (Coarsely Integrated Operand Scanning)
// Simplified approach: multiply each limb independently with carry handling
// ============================================================================

// Simple schoolbook multiplication: a * bi -> result in acc[0..8]
__device__ __forceinline__ void mul_n(uint32_t* acc, const uint32_t* a, uint32_t bi) {
    // Clear accumulator
    #pragma unroll
    for(int i = 0; i < 9; i++) acc[i] = 0;
    
    // Schoolbook multiplication with carry
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        uint64_t prod = (uint64_t)a[i] * bi;
        uint64_t sum = acc[i] + (uint32_t)prod;
        acc[i] = (uint32_t)sum;
        
        uint32_t carry = (uint32_t)(prod >> 32) + (uint32_t)(sum >> 32);
        
        // Propagate carry
        int j = i + 1;
        while(carry && j < 9) {
            uint64_t tmp = (uint64_t)acc[j] + carry;
            acc[j] = (uint32_t)tmp;
            carry = (uint32_t)(tmp >> 32);
            j++;
        }
    }
}

// Cumulative multiply-add with carry chain (PTX optimized)
__device__ __forceinline__ void cmad_n(uint32_t* acc, const uint32_t* a, uint32_t bi) {
    asm("mad.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
        : "+r"(acc[0]), "+r"(acc[1])
        : "r"(a[0]), "r"(bi));
    
    #pragma unroll
    for (int j = 2; j < 8; j += 2) {
        asm("madc.lo.cc.u32 %0, %2, %3, %0; madc.hi.cc.u32 %1, %2, %3, %1;"
            : "+r"(acc[j]), "+r"(acc[j+1])
            : "r"(a[j]), "r"(bi));
    }
}

// Optimized 256×256 → 512 multiplication with reduced register pressure
__device__ __forceinline__ void mul_256_512_opt(const uint32_t* a, const uint32_t* b, uint32_t* r) {
    uint32_t acc[9];  // 8 limbs + 1 carry
    
    #pragma unroll
    for(int i = 0; i < 9; i++) acc[i] = 0;
    
    // Column-wise multiplication with carry propagation
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        if(i == 0) {
            mul_n(acc, a, b[0]);
        } else {
            cmad_n(acc, a, b[i]);
            asm("addc.u32 %0, %0, 0;" : "+r"(acc[8]));
        }
        
        r[i] = acc[0];
        
        // Shift accumulator
        #pragma unroll
        for(int j = 0; j < 8; j++) {
            acc[j] = acc[j+1];
        }
        acc[8] = 0;
    }
    
    // Store remaining high limbs
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        r[8 + i] = acc[i];
    }
}

// Optimized field reduction with multi-pass approach
__device__ __forceinline__ void field_reduce_opt(uint32_t* wide, FieldElement* r) {
    // Reduction: 2^256 ≡ 2^32 + 977 (mod P)
    
    // Multi-pass reduction with unrolled carries
    #pragma unroll
    for(int pass = 0; pass < 3; pass++) {
        bool any_high = false;
        
        #pragma unroll
        for(int i = 0; i < 8; i++) {
            if(wide[8 + i] != 0) {
                any_high = true;
                break;
            }
        }
        
        if(!any_high) break;
        
        #pragma unroll
        for(int i = 0; i < 8; i++) {
            uint32_t h = wide[8 + i];
            if(h == 0) continue;
            
            wide[8 + i] = 0;
            
            // h * 977
            uint32_t lo977, hi977;
            asm("mul.lo.u32 %0, %1, 977;" : "=r"(lo977) : "r"(h));
            asm("mul.hi.u32 %0, %1, 977;" : "=r"(hi977) : "r"(h));
            
            // Add h*977 to wide[i] with full carry chain
            asm("add.cc.u32 %0, %0, %1;" : "+r"(wide[i]) : "r"(lo977));
            
            if(i + 1 < 16) {
                asm("addc.cc.u32 %0, %0, %1;" : "+r"(wide[i+1]) : "r"(hi977));
                asm("addc.cc.u32 %0, %0, %1;" : "+r"(wide[i+1]) : "r"(h));
                
                // Propagate carries
                #pragma unroll
                for(int j = i + 2; j < 16; j++) {
                    asm("addc.cc.u32 %0, %0, 0;" : "+r"(wide[j]));
                }
                
                uint32_t final_carry;
                asm("addc.u32 %0, 0, 0;" : "=r"(final_carry));
                
                if(final_carry) {
                    wide[8] += final_carry;
                }
            }
        }
    }
    
    // Copy to accumulator
    uint32_t acc[8];
    #pragma unroll
    for(int i = 0; i < 8; i++) acc[i] = wide[i];
    
    // Final conditional subtraction (unrolled twice)
    #pragma unroll 2
    for(int k = 0; k < 2; k++) {
        uint32_t s[8];
        uint32_t borrow;
        
        asm("sub.cc.u32 %0, %1, %2;" : "=r"(s[0]) : "r"(acc[0]), "r"(MODULUS[0]));
        #pragma unroll
        for(int i = 1; i < 8; i++) {
            asm("subc.cc.u32 %0, %1, %2;" : "=r"(s[i]) : "r"(acc[i]), "r"(MODULUS[i]));
        }
        asm("subc.u32 %0, 0, 0;" : "=r"(borrow));
        
        if(borrow == 0) {
            #pragma unroll
            for(int i = 0; i < 8; i++) acc[i] = s[i];
        }
    }
    
    #pragma unroll
    for(int i = 0; i < 8; i++) r->limbs[i] = acc[i];
}

// Optimized field multiplication
__device__ __forceinline__ void field_mul(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint32_t wide[16];
    mul_256_512_opt(a->limbs, b->limbs, wide);
    field_reduce_opt(wide, r);
}

// Optimized field squaring (reuses optimized mul)
__device__ __forceinline__ void field_sqr(const FieldElement* a, FieldElement* r) {
    uint32_t wide[16];
    mul_256_512_opt(a->limbs, a->limbs, wide);
    field_reduce_opt(wide, r);
}

// Optimized field addition with PTX
__device__ __forceinline__ void field_add(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint32_t sum[8];
    uint32_t carry;
    
    asm("add.cc.u32 %0, %1, %2;" : "=r"(sum[0]) : "r"(a->limbs[0]), "r"(b->limbs[0]));
    #pragma unroll
    for(int i = 1; i < 8; i++) {
        asm("addc.cc.u32 %0, %1, %2;" : "=r"(sum[i]) : "r"(a->limbs[i]), "r"(b->limbs[i]));
    }
    asm("addc.u32 %0, 0, 0;" : "=r"(carry));
    
    // Conditional subtraction
    uint32_t sub[8];
    uint32_t borrow;
    
    asm("sub.cc.u32 %0, %1, %2;" : "=r"(sub[0]) : "r"(sum[0]), "r"(MODULUS[0]));
    #pragma unroll
    for(int i = 1; i < 8; i++) {
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(sub[i]) : "r"(sum[i]), "r"(MODULUS[i]));
    }
    asm("subc.u32 %0, 0, 0;" : "=r"(borrow));
    
    // Use subtracted value if no borrow (sum >= P)
    if(carry || borrow == 0) {
        #pragma unroll
        for(int i = 0; i < 8; i++) r->limbs[i] = sub[i];
    } else {
        #pragma unroll
        for(int i = 0; i < 8; i++) r->limbs[i] = sum[i];
    }
}

// Optimized field subtraction
__device__ __forceinline__ void field_sub(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    uint32_t diff[8];
    uint32_t borrow;
    
    asm("sub.cc.u32 %0, %1, %2;" : "=r"(diff[0]) : "r"(a->limbs[0]), "r"(b->limbs[0]));
    #pragma unroll
    for(int i = 1; i < 8; i++) {
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(diff[i]) : "r"(a->limbs[i]), "r"(b->limbs[i]));
    }
    asm("subc.u32 %0, 0, 0;" : "=r"(borrow));
    
    // Add P if borrow
    if(borrow) {
        asm("add.cc.u32 %0, %0, %1;" : "+r"(diff[0]) : "r"(MODULUS[0]));
        #pragma unroll
        for(int i = 1; i < 8; i++) {
            asm("addc.cc.u32 %0, %0, %1;" : "+r"(diff[i]) : "r"(MODULUS[i]));
        }
    }
    
    #pragma unroll
    for(int i = 0; i < 8; i++) r->limbs[i] = diff[i];
}

__device__ __forceinline__ bool field_is_zero(const FieldElement* a) {
    uint32_t result = 0;
    #pragma unroll
    for(int i = 0; i < 8; i++) {
        result |= a->limbs[i];
    }
    return result == 0;
}

// Field inverse using same addition chain as before
__device__ __forceinline__ void field_inv(const FieldElement* a, FieldElement* r) {
    if(field_is_zero(a)) {
        #pragma unroll
        for(int i = 0; i < 8; i++) r->limbs[i] = 0;
        return;
    }

    FieldElement x_0, x_1, x_2, x_3, x_4, x_5;
    FieldElement t;

    // Same addition chain as 64-bit version
    field_sqr(a, &x_0);
    field_mul(&x_0, a, &x_0);

    field_sqr(&x_0, &x_1);
    field_mul(&x_1, a, &x_1);

    field_sqr(&x_1, &x_2);
    field_sqr(&x_2, &x_2);
    field_sqr(&x_2, &x_2);
    field_mul(&x_2, &x_1, &x_2);

    field_sqr(&x_2, &x_3);
    for (int i = 0; i < 5; i++) field_sqr(&x_3, &x_3);
    field_mul(&x_3, &x_2, &x_3);

    t = x_3;
    for (int i = 0; i < 12; i++) field_sqr(&t, &t);
    field_mul(&t, &x_3, &x_3);

    t = x_3;
    for (int i = 0; i < 24; i++) field_sqr(&t, &t);
    field_mul(&t, &x_3, &x_4);

    t = x_4;
    for (int i = 0; i < 48; i++) field_sqr(&t, &t);
    field_mul(&t, &x_4, &x_4);

    t = x_4;
    for (int i = 0; i < 96; i++) field_sqr(&t, &t);
    field_mul(&t, &x_4, &x_4);

    field_sqr(&x_2, &x_5);
    field_mul(&x_5, a, &x_5);

    t = x_3;
    for (int i = 0; i < 7; i++) field_sqr(&t, &t);
    field_mul(&t, &x_5, &x_5);

    t = x_4;
    for (int i = 0; i < 31; i++) field_sqr(&t, &t);
    field_mul(&t, &x_5, &x_5);

    t = x_1;
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_mul(&t, &x_0, &x_0);

    t = x_2;
    for (int i = 0; i < 5; i++) field_sqr(&t, &t);
    field_mul(&t, &x_0, &x_1);

    t = x_1;
    for (int i = 0; i < 11; i++) field_sqr(&t, &t);
    field_mul(&t, &x_1, &x_1);

    field_sqr(&x_5, &t);

    for (int i = 0; i < 22; i++) field_sqr(&t, &t);
    field_mul(&t, &x_1, &t);

    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);

    field_sqr(&t, &t);
    field_mul(&t, a, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_mul(&t, a, &t);
    field_sqr(&t, &t);
    field_mul(&t, a, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_mul(&t, a, r);
}

// Point doubling (optimized with reduced field operations)
__device__ __forceinline__ void jacobian_double(const JacobianPoint* p, JacobianPoint* r) {
    if(p->infinity) {
        *r = *p;
        return;
    }

    FieldElement s, m, t, t2;
    
    // S = 4*X*Y^2
    field_sqr(&p->y, &t);
    field_add(&t, &t, &t2);
    field_add(&t2, &t2, &t);
    field_mul(&t, &p->x, &s);
    field_add(&s, &s, &t);
    field_add(&t, &t, &s);
    
    // M = 3*X^2
    field_sqr(&p->x, &t);
    field_add(&t, &t, &t2);
    field_add(&t2, &t, &m);
    
    // X' = M^2 - 2*S
    field_sqr(&m, &r->x);
    field_sub(&r->x, &s, &r->x);
    field_sub(&r->x, &s, &r->x);
    
    // Y' = M*(S - X') - 8*Y^4
    field_sub(&s, &r->x, &t);
    field_mul(&m, &t, &r->y);
    field_sqr(&p->y, &t);
    field_sqr(&t, &t);
    field_add(&t, &t, &t2);
    field_add(&t2, &t2, &t2);
    field_add(&t2, &t2, &t);
    field_sub(&r->y, &t, &r->y);
    
    // Z' = 2*Y*Z
    field_mul(&p->y, &p->z, &r->z);
    field_add(&r->z, &r->z, &r->z);
    
    r->infinity = false;
}

// ============================================================================
// Hybrid wrapper functions for use with 64-bit FieldElement
// ============================================================================

__device__ __forceinline__ void field_mul_hybrid(const secp256k1::cuda::FieldElement* a, 
                                                  const secp256k1::cuda::FieldElement* b, 
                                                  secp256k1::cuda::FieldElement* r) {
    const FieldElement* a32 = toOptField(a);
    const FieldElement* b32 = toOptField(b);
    FieldElement* r32 = toOptField(r);
    
    field_mul(a32, b32, r32);
}

__device__ __forceinline__ void field_sqr_hybrid(const secp256k1::cuda::FieldElement* a,
                                                  secp256k1::cuda::FieldElement* r) {
    const FieldElement* a32 = toOptField(a);
    FieldElement* r32 = toOptField(r);
    
    field_sqr(a32, r32);
}

} // namespace secp_opt
