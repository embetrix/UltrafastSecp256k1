#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "secp256k1.cuh"

// ============================================================================
// Fast 32-bit multiplication using native PTX mad.lo.cc instructions
// Compatible with 64-bit FieldElement via reinterpret_cast
// ============================================================================

namespace secp256k1 {
namespace cuda {
namespace fast32 {

// P = 2^256 - 2^32 - 977  
__constant__ static const uint32_t MODULUS_32[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// ============================================================================
// 256x256 -> 512 multiplication using Comba's method with PTX
// ============================================================================

__device__ __forceinline__ void mul_256_512_fast(const uint32_t* a, const uint32_t* b, uint32_t* r) {
    uint32_t r0 = 0, r1 = 0, r2 = 0;

    #define MUL32_ACC(ai, bj) { \
        asm( \
            "mad.lo.cc.u32 %0, %3, %4, %0; \n\t" \
            "madc.hi.cc.u32 %1, %3, %4, %1; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) \
            : "r"(a[ai]), "r"(b[bj]) \
        ); \
    }

    // Col 0
    MUL32_ACC(0, 0);
    r[0] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 1
    MUL32_ACC(0, 1); MUL32_ACC(1, 0);
    r[1] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 2
    MUL32_ACC(0, 2); MUL32_ACC(1, 1); MUL32_ACC(2, 0);
    r[2] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 3
    MUL32_ACC(0, 3); MUL32_ACC(1, 2); MUL32_ACC(2, 1); MUL32_ACC(3, 0);
    r[3] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 4
    MUL32_ACC(0, 4); MUL32_ACC(1, 3); MUL32_ACC(2, 2); MUL32_ACC(3, 1); MUL32_ACC(4, 0);
    r[4] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 5
    MUL32_ACC(0, 5); MUL32_ACC(1, 4); MUL32_ACC(2, 3); MUL32_ACC(3, 2); MUL32_ACC(4, 1); MUL32_ACC(5, 0);
    r[5] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 6
    MUL32_ACC(0, 6); MUL32_ACC(1, 5); MUL32_ACC(2, 4); MUL32_ACC(3, 3); MUL32_ACC(4, 2); MUL32_ACC(5, 1); MUL32_ACC(6, 0);
    r[6] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 7
    MUL32_ACC(0, 7); MUL32_ACC(1, 6); MUL32_ACC(2, 5); MUL32_ACC(3, 4); MUL32_ACC(4, 3); MUL32_ACC(5, 2); MUL32_ACC(6, 1); MUL32_ACC(7, 0);
    r[7] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 8
    MUL32_ACC(1, 7); MUL32_ACC(2, 6); MUL32_ACC(3, 5); MUL32_ACC(4, 4); MUL32_ACC(5, 3); MUL32_ACC(6, 2); MUL32_ACC(7, 1);
    r[8] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 9
    MUL32_ACC(2, 7); MUL32_ACC(3, 6); MUL32_ACC(4, 5); MUL32_ACC(5, 4); MUL32_ACC(6, 3); MUL32_ACC(7, 2);
    r[9] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 10
    MUL32_ACC(3, 7); MUL32_ACC(4, 6); MUL32_ACC(5, 5); MUL32_ACC(6, 4); MUL32_ACC(7, 3);
    r[10] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 11
    MUL32_ACC(4, 7); MUL32_ACC(5, 6); MUL32_ACC(6, 5); MUL32_ACC(7, 4);
    r[11] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 12
    MUL32_ACC(5, 7); MUL32_ACC(6, 6); MUL32_ACC(7, 5);
    r[12] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 13
    MUL32_ACC(6, 7); MUL32_ACC(7, 6);
    r[13] = r0; r0 = r1; r1 = r2; r2 = 0;

    // Col 14
    MUL32_ACC(7, 7);
    r[14] = r0;
    r[15] = r1;

    #undef MUL32_ACC
}

// ============================================================================
// Reduction: 512 -> 256 mod P using 2^256 ≡ 2^32 + 977
// ============================================================================

__device__ __forceinline__ void field_reduce_fast(uint32_t* wide, uint32_t* result) {
    // Multi-pass reduction
    for(int pass = 0; pass < 4; pass++) {
        bool any_high = false;
        
        #pragma unroll
        for(int i = 0; i < 8; i++) {
            if (wide[8 + i] != 0) {
                any_high = true;
                break;
            }
        }
        
        if (!any_high) break;
        
        for(int i = 0; i < 8; i++) {
            uint32_t h = wide[8 + i];
            if (h == 0) continue;
            
            wide[8 + i] = 0;
            
            // h * 977
            uint32_t lo977, hi977;
            asm("mul.lo.u32 %0, %1, 977;" : "=r"(lo977) : "r"(h));
            asm("mul.hi.u32 %0, %1, 977;" : "=r"(hi977) : "r"(h));
            
            asm("add.cc.u32 %0, %0, %1;" : "+r"(wide[i]) : "r"(lo977));
            
            if (i + 1 < 16) {
                asm("addc.cc.u32 %0, %0, %1;" : "+r"(wide[i+1]) : "r"(hi977));
                asm("addc.cc.u32 %0, %0, %1;" : "+r"(wide[i+1]) : "r"(h));
                
                #pragma unroll
                for(int j = i + 2; j < 16; j++) {
                    asm("addc.cc.u32 %0, %0, 0;" : "+r"(wide[j]));
                }
                uint32_t final_carry;
                asm("addc.u32 %0, 0, 0;" : "=r"(final_carry));
                
                if (final_carry) {
                    wide[8] += final_carry;
                }
            }
        }
    }
    
    // Copy to result
    uint32_t acc[8];
    #pragma unroll
    for(int i = 0; i < 8; i++) acc[i] = wide[i];
    
    // Final conditional subtraction (2 passes)
    #pragma unroll
    for(int k = 0; k < 2; k++) {
        uint32_t s[8];
        uint32_t borrow;
        
        asm("sub.cc.u32 %0, %1, %2;" : "=r"(s[0]) : "r"(acc[0]), "r"(MODULUS_32[0]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(s[1]) : "r"(acc[1]), "r"(MODULUS_32[1]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(s[2]) : "r"(acc[2]), "r"(MODULUS_32[2]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(s[3]) : "r"(acc[3]), "r"(MODULUS_32[3]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(s[4]) : "r"(acc[4]), "r"(MODULUS_32[4]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(s[5]) : "r"(acc[5]), "r"(MODULUS_32[5]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(s[6]) : "r"(acc[6]), "r"(MODULUS_32[6]));
        asm("subc.cc.u32 %0, %1, %2;" : "=r"(s[7]) : "r"(acc[7]), "r"(MODULUS_32[7]));
        asm("subc.u32 %0, 0, 0;" : "=r"(borrow));
        
        if (borrow == 0) {
            #pragma unroll
            for(int i = 0; i < 8; i++) acc[i] = s[i];
        }
    }
    
    #pragma unroll
    for(int i = 0; i < 8; i++) result[i] = acc[i];
}

// ============================================================================
// Wrapper functions for 64-bit FieldElement
// ============================================================================

__device__ __forceinline__ void field_mul_fast(const FieldElement* a, const FieldElement* b, FieldElement* r) {
    const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a);
    const uint32_t* b32 = reinterpret_cast<const uint32_t*>(b);
    uint32_t* r32 = reinterpret_cast<uint32_t*>(r);
    
    uint32_t wide[16];
    mul_256_512_fast(a32, b32, wide);
    field_reduce_fast(wide, r32);
}

__device__ __forceinline__ void field_sqr_fast(const FieldElement* a, FieldElement* r) {
    const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a);
    uint32_t* r32 = reinterpret_cast<uint32_t*>(r);
    
    uint32_t wide[16];
    mul_256_512_fast(a32, a32, wide);
    field_reduce_fast(wide, r32);
}

} // namespace fast32
} // namespace cuda
} // namespace secp256k1
