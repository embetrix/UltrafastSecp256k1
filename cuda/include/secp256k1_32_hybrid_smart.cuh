#pragma once

#include <cstdint>

// Smart hybrid: 32-bit multiplication (native) + 64-bit reduction (simple)
// Uses both interpretations in the same function where each is optimal

// No namespace - will be included inside secp256k1::cuda namespace

// ============================================================================
// Smart hybrid multiplication: 32-bit mul → 64-bit reduce
// ============================================================================

__device__ __forceinline__ void field_mul_smart_hybrid(
    const FieldElement* a, 
    const FieldElement* b, 
    FieldElement* r
) {
    // Create both views (zero-cost reinterpret_cast)
    const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a);
    const uint32_t* b32 = reinterpret_cast<const uint32_t*>(b);
    uint64_t* r64 = reinterpret_cast<uint64_t*>(r);
    
    // Temporary 512-bit result (16 × 32-bit or 8 × 64-bit)
    uint64_t t[8];
    uint32_t* t32 = reinterpret_cast<uint32_t*>(t);
    
    // ========================================================================
    // Step 1: 256×256→512 multiplication using 32-bit operations (NATIVE!)
    // ========================================================================
    // Use Comba's method with 32-bit PTX (mad.lo.cc.u32 is hardware native)
    
    uint32_t c0 = 0, c1 = 0, c2 = 0;
    
    #define MULADD(ai, bi) \
        asm("mad.lo.cc.u32 %0, %2, %3, %0;" : "+r"(c0) : "r"(c0), "r"(a32[ai]), "r"(b32[bi])); \
        asm("madc.hi.cc.u32 %0, %2, %3, %0;" : "+r"(c1) : "r"(c1), "r"(a32[ai]), "r"(b32[bi])); \
        asm("addc.u32 %0, %0, 0;" : "+r"(c2) : "r"(c2));
    
    // Column 0
    asm("mul.lo.u32 %0, %1, %2;" : "=r"(c0) : "r"(a32[0]), "r"(b32[0]));
    asm("mul.hi.u32 %0, %1, %2;" : "=r"(c1) : "r"(a32[0]), "r"(b32[0]));
    t32[0] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 1
    MULADD(0, 1); MULADD(1, 0);
    t32[1] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 2
    MULADD(0, 2); MULADD(1, 1); MULADD(2, 0);
    t32[2] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 3
    MULADD(0, 3); MULADD(1, 2); MULADD(2, 1); MULADD(3, 0);
    t32[3] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 4
    MULADD(0, 4); MULADD(1, 3); MULADD(2, 2); MULADD(3, 1); MULADD(4, 0);
    t32[4] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 5
    MULADD(0, 5); MULADD(1, 4); MULADD(2, 3); MULADD(3, 2); MULADD(4, 1); MULADD(5, 0);
    t32[5] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 6
    MULADD(0, 6); MULADD(1, 5); MULADD(2, 4); MULADD(3, 3); MULADD(4, 2); MULADD(5, 1); MULADD(6, 0);
    t32[6] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 7
    MULADD(0, 7); MULADD(1, 6); MULADD(2, 5); MULADD(3, 4); MULADD(4, 3); MULADD(5, 2); MULADD(6, 1); MULADD(7, 0);
    t32[7] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 8
    MULADD(1, 7); MULADD(2, 6); MULADD(3, 5); MULADD(4, 4); MULADD(5, 3); MULADD(6, 2); MULADD(7, 1);
    t32[8] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 9
    MULADD(2, 7); MULADD(3, 6); MULADD(4, 5); MULADD(5, 4); MULADD(6, 3); MULADD(7, 2);
    t32[9] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 10
    MULADD(3, 7); MULADD(4, 6); MULADD(5, 5); MULADD(6, 4); MULADD(7, 3);
    t32[10] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 11
    MULADD(4, 7); MULADD(5, 6); MULADD(6, 5); MULADD(7, 4);
    t32[11] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 12
    MULADD(5, 7); MULADD(6, 6); MULADD(7, 5);
    t32[12] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 13
    MULADD(6, 7); MULADD(7, 6);
    t32[13] = c0; c0 = c1; c1 = c2; c2 = 0;
    
    // Column 14
    MULADD(7, 7);
    t32[14] = c0;
    t32[15] = c1;
    
    #undef MULADD
    
    // ========================================================================
    // Step 2: 512-bit → 256-bit reduction using proven 64-bit function
    // ========================================================================
    // Use standard reduce_512_to_256 (already defined in secp256k1.cuh)
    reduce_512_to_256(t, r);
}

__device__ __forceinline__ void field_sqr_smart_hybrid(
    const FieldElement* a, 
    FieldElement* r
) {
    field_mul_smart_hybrid(a, a, r);
}
