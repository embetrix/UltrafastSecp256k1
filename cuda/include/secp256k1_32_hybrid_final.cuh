#pragma once

// Smart Hybrid: 32-bit multiplication (native PTX) + 64-bit reduction (proven)
// This file is included AFTER reduce_512_to_256 is defined
// Does NOT redefine FieldElement - uses existing 64-bit FieldElement

// ============================================================================
// 32-bit multiplication using proven Comba's method
// Input: 64-bit FieldElement (4×64) viewed as 32-bit (8×32)
// Output: 512-bit result for reduce_512_to_256
// ============================================================================

__device__ __forceinline__ void mul_256_512_hybrid(
    const secp256k1::cuda::FieldElement* a,
    const secp256k1::cuda::FieldElement* b, 
    uint64_t t[8]
) {
    // Explicit split to avoid aliasing UB (compiled to register moves, zero-cost)
    // Extract 32-bit limbs from 64-bit storage
    uint32_t a32[8], b32[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a32[2*i]   = (uint32_t)(a->limbs[i]);       // low 32 bits
        a32[2*i+1] = (uint32_t)(a->limbs[i] >> 32); // high 32 bits
        b32[2*i]   = (uint32_t)(b->limbs[i]);
        b32[2*i+1] = (uint32_t)(b->limbs[i] >> 32);
    }
    
    // Comba multiplication with 3 accumulators
    uint32_t r0 = 0, r1 = 0, r2 = 0;
    uint32_t t32[16];  // Intermediate 32-bit result
    
    #define MUL32_ACC(ai, bj) { \
        asm volatile( \
            "mad.lo.cc.u32 %0, %3, %4, %0; \n\t" \
            "madc.hi.cc.u32 %1, %3, %4, %1; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) \
            : "r"(a32[ai]), "r"(b32[bj]) \
        ); \
    }
    
    // Column 0
    MUL32_ACC(0, 0);
    t32[0] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 1
    MUL32_ACC(0, 1); MUL32_ACC(1, 0);
    t32[1] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 2
    MUL32_ACC(0, 2); MUL32_ACC(1, 1); MUL32_ACC(2, 0);
    t32[2] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 3
    MUL32_ACC(0, 3); MUL32_ACC(1, 2); MUL32_ACC(2, 1); MUL32_ACC(3, 0);
    t32[3] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 4
    MUL32_ACC(0, 4); MUL32_ACC(1, 3); MUL32_ACC(2, 2); MUL32_ACC(3, 1); MUL32_ACC(4, 0);
    t32[4] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 5
    MUL32_ACC(0, 5); MUL32_ACC(1, 4); MUL32_ACC(2, 3); MUL32_ACC(3, 2); MUL32_ACC(4, 1); MUL32_ACC(5, 0);
    t32[5] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 6
    MUL32_ACC(0, 6); MUL32_ACC(1, 5); MUL32_ACC(2, 4); MUL32_ACC(3, 3); MUL32_ACC(4, 2); MUL32_ACC(5, 1); MUL32_ACC(6, 0);
    t32[6] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 7
    MUL32_ACC(0, 7); MUL32_ACC(1, 6); MUL32_ACC(2, 5); MUL32_ACC(3, 4); MUL32_ACC(4, 3); MUL32_ACC(5, 2); MUL32_ACC(6, 1); MUL32_ACC(7, 0);
    t32[7] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 8
    MUL32_ACC(1, 7); MUL32_ACC(2, 6); MUL32_ACC(3, 5); MUL32_ACC(4, 4); MUL32_ACC(5, 3); MUL32_ACC(6, 2); MUL32_ACC(7, 1);
    t32[8] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 9
    MUL32_ACC(2, 7); MUL32_ACC(3, 6); MUL32_ACC(4, 5); MUL32_ACC(5, 4); MUL32_ACC(6, 3); MUL32_ACC(7, 2);
    t32[9] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 10
    MUL32_ACC(3, 7); MUL32_ACC(4, 6); MUL32_ACC(5, 5); MUL32_ACC(6, 4); MUL32_ACC(7, 3);
    t32[10] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 11
    MUL32_ACC(4, 7); MUL32_ACC(5, 6); MUL32_ACC(6, 5); MUL32_ACC(7, 4);
    t32[11] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 12
    MUL32_ACC(5, 7); MUL32_ACC(6, 6); MUL32_ACC(7, 5);
    t32[12] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 13
    MUL32_ACC(6, 7); MUL32_ACC(7, 6);
    t32[13] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 14
    MUL32_ACC(7, 7);
    t32[14] = r0;
    t32[15] = r1;
    
    #undef MUL32_ACC
    
    // Pack 32-bit result into 64-bit output (explicit, UB-safe)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        t[i] = ((uint64_t)t32[2*i+1] << 32) | t32[2*i];
    }
}

// ============================================================================
// Optimized 32-bit squaring using Comba's method
// Exploits symmetry: a[i]*a[j] computed once, added twice (except diagonal)
// ~40% fewer multiplications than generic multiplication
// ============================================================================

__device__ __forceinline__ void sqr_256_512_hybrid(
    const secp256k1::cuda::FieldElement* a,
    uint64_t t[8]
) {
    // Explicit split to avoid aliasing UB
    uint32_t a32[8];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        a32[2*i]   = (uint32_t)(a->limbs[i]);
        a32[2*i+1] = (uint32_t)(a->limbs[i] >> 32);
    }
    
    uint32_t t32[16];  // Intermediate 32-bit result
    uint32_t r0 = 0, r1 = 0, r2 = 0;
    
    // Diagonal multiplication (no doubling)
    #define SQR32_DIAG(ai) { \
        asm volatile( \
            "mad.lo.cc.u32 %0, %3, %3, %0; \n\t" \
            "madc.hi.cc.u32 %1, %3, %3, %1; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) \
            : "r"(a32[ai]) \
        ); \
    }
    
    // Off-diagonal multiplication (doubled: a[i]*a[j] added twice)
    #define SQR32_MUL2(ai, aj) { \
        uint32_t lo, hi; \
        asm volatile( \
            "mul.lo.u32 %0, %2, %3; \n\t" \
            "mul.hi.u32 %1, %2, %3; \n\t" \
            : "=r"(lo), "=r"(hi) \
            : "r"(a32[ai]), "r"(a32[aj]) \
        ); \
        asm volatile( \
            "add.cc.u32 %0, %0, %3; \n\t" \
            "addc.cc.u32 %1, %1, %4; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            "add.cc.u32 %0, %0, %3; \n\t" \
            "addc.cc.u32 %1, %1, %4; \n\t" \
            "addc.u32 %2, %2, 0; \n\t" \
            : "+r"(r0), "+r"(r1), "+r"(r2) \
            : "r"(lo), "r"(hi) \
        ); \
    }
    
    // Column 0: a[0]*a[0]
    SQR32_DIAG(0);
    t32[0] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 1: 2*a[0]*a[1]
    SQR32_MUL2(0, 1);
    t32[1] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 2: 2*a[0]*a[2] + a[1]*a[1]
    SQR32_MUL2(0, 2);
    SQR32_DIAG(1);
    t32[2] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 3: 2*(a[0]*a[3] + a[1]*a[2])
    SQR32_MUL2(0, 3);
    SQR32_MUL2(1, 2);
    t32[3] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 4: 2*(a[0]*a[4] + a[1]*a[3]) + a[2]*a[2]
    SQR32_MUL2(0, 4);
    SQR32_MUL2(1, 3);
    SQR32_DIAG(2);
    t32[4] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 5: 2*(a[0]*a[5] + a[1]*a[4] + a[2]*a[3])
    SQR32_MUL2(0, 5);
    SQR32_MUL2(1, 4);
    SQR32_MUL2(2, 3);
    t32[5] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 6: 2*(a[0]*a[6] + a[1]*a[5] + a[2]*a[4]) + a[3]*a[3]
    SQR32_MUL2(0, 6);
    SQR32_MUL2(1, 5);
    SQR32_MUL2(2, 4);
    SQR32_DIAG(3);
    t32[6] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 7: 2*(a[0]*a[7] + a[1]*a[6] + a[2]*a[5] + a[3]*a[4])
    SQR32_MUL2(0, 7);
    SQR32_MUL2(1, 6);
    SQR32_MUL2(2, 5);
    SQR32_MUL2(3, 4);
    t32[7] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 8: 2*(a[1]*a[7] + a[2]*a[6] + a[3]*a[5]) + a[4]*a[4]
    SQR32_MUL2(1, 7);
    SQR32_MUL2(2, 6);
    SQR32_MUL2(3, 5);
    SQR32_DIAG(4);
    t32[8] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 9: 2*(a[2]*a[7] + a[3]*a[6] + a[4]*a[5])
    SQR32_MUL2(2, 7);
    SQR32_MUL2(3, 6);
    SQR32_MUL2(4, 5);
    t32[9] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 10: 2*(a[3]*a[7] + a[4]*a[6]) + a[5]*a[5]
    SQR32_MUL2(3, 7);
    SQR32_MUL2(4, 6);
    SQR32_DIAG(5);
    t32[10] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 11: 2*(a[4]*a[7] + a[5]*a[6])
    SQR32_MUL2(4, 7);
    SQR32_MUL2(5, 6);
    t32[11] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 12: 2*a[5]*a[7] + a[6]*a[6]
    SQR32_MUL2(5, 7);
    SQR32_DIAG(6);
    t32[12] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 13: 2*a[6]*a[7]
    SQR32_MUL2(6, 7);
    t32[13] = r0; r0 = r1; r1 = r2; r2 = 0;
    
    // Column 14: a[7]*a[7]
    SQR32_DIAG(7);
    t32[14] = r0;
    t32[15] = r1;
    
    #undef SQR32_DIAG
    #undef SQR32_MUL2
    
    // Pack 32-bit result into 64-bit output (explicit, UB-safe)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        t[i] = ((uint64_t)t32[2*i+1] << 32) | t32[2*i];
    }
}

// ============================================================================
// Hybrid field operations: 32-bit mul/sqr + proven 64-bit reduce
// ============================================================================

__device__ __forceinline__ void field_mul_hybrid(
    const secp256k1::cuda::FieldElement* a,
    const secp256k1::cuda::FieldElement* b,
    secp256k1::cuda::FieldElement* r
) {
    uint64_t t[8];
    mul_256_512_hybrid(a, b, t);
    reduce_512_to_256(t, r);  // Use proven 64-bit reduction
}

__device__ __forceinline__ void field_sqr_hybrid(
    const secp256k1::cuda::FieldElement* a,
    secp256k1::cuda::FieldElement* r
) {
    uint64_t t[8];
    sqr_256_512_hybrid(a, t);  // Optimized squaring!
    reduce_512_to_256(t, r);
}

// ============================================================================
// Montgomery hybrid operations: 32-bit mul/sqr + Montgomery reduction
// These use fast 32-bit multiplication but Montgomery-specific reduction
// ============================================================================

__device__ __forceinline__ void field_mul_mont_hybrid(
    const secp256k1::cuda::FieldElement* a,
    const secp256k1::cuda::FieldElement* b,
    secp256k1::cuda::FieldElement* r
) {
    uint64_t t[8];
    mul_256_512_hybrid(a, b, t);  // Fast 32-bit PTX multiplication!
    mont_reduce_512(t, r);         // Montgomery reduction
}

__device__ __forceinline__ void field_sqr_mont_hybrid(
    const secp256k1::cuda::FieldElement* a,
    secp256k1::cuda::FieldElement* r
) {
    uint64_t t[8];
    sqr_256_512_hybrid(a, t);  // Fast optimized 32-bit squaring!
    mont_reduce_512(t, r);      // Montgomery reduction
}
