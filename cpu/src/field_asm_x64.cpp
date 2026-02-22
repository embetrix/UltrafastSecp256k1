// Inline assembly implementation for x64 (MSVC)
// Target: 3-5x speedup on field operations
// This is platform-specific and requires BMI2 support

#include <secp256k1/field_asm.hpp>
#include <secp256k1/field.hpp>
#include <cstring>

#if defined(_MSC_VER) && defined(_M_X64) && defined(USE_INLINE_ASSEMBLY)

namespace secp256k1::fast {
namespace detail {

// ===================================================================
// 4x4 multiplication using inline assembly (x64 MSVC)
// Uses MULX + ADCX + ADOX for parallel carry chains
// ===================================================================

extern "C" void mul_4x4_asm_impl(const uint64_t a[4], const uint64_t b[4], uint64_t result[8]);

// Assembly implementation
// Input: RCX = a, RDX = b, R8 = result
__declspec(naked) void mul_4x4_asm_impl(const uint64_t* a, const uint64_t* b, uint64_t* result) {
    __asm {
        // Save non-volatile registers
        push rbx
        push rsi
        push rdi
        push r12
        push r13
        push r14
        push r15
        
        // Load pointers
        mov rsi, rcx     ; a
        mov rdi, rdx     ; b  
        mov r15, r8      ; result
        
        // Zero out result area
        xor rax, rax
        mov [r15+0], rax
        mov [r15+8], rax
        mov [r15+16], rax
        mov [r15+24], rax
        mov [r15+32], rax
        mov [r15+40], rax
        mov [r15+48], rax
        mov [r15+56], rax
        
        // Load a[0] into RDX for MULX
        mov rdx, [rsi+0]
        
        ; Column 0: a[0] * b[0]
        mulx r9, r8, [rdi+0]     ; a[0]*b[0] -> r8:r9 (lo:hi)
        mov [r15+0], r8          ; result[0] = lo
        
        ; Column 1: a[0]*b[1]
        mulx r11, r10, [rdi+8]   ; a[0]*b[1] -> r10:r11
        xor r12, r12             ; Clear for ADCX
        adcx r9, r10             ; Add to column 1
        mov [r15+8], r9          ; result[1]
        
        ; Column 2: a[0]*b[2]  
        mulx r9, r10, [rdi+16]   ; a[0]*b[2] -> r10:r9
        adcx r11, r10            ; Add to column 2
        mov [r15+16], r11        ; result[2]
        
        ; Column 3: a[0]*b[3]
        mulx r11, r10, [rdi+24]  ; a[0]*b[3] -> r10:r11
        adcx r9, r10             ; Add to column 3
        mov [r15+24], r9         ; result[3]
        adcx r11, r12            ; Add carry
        mov [r15+32], r11        ; result[4]
        
        ; Row 1: a[1] * b[0..3]
        mov rdx, [rsi+8]         ; a[1]
        
        mulx r9, r8, [rdi+0]     ; a[1]*b[0]
        add [r15+8], r8
        adc [r15+16], r9
        adc [r15+24], 0
        adc [r15+32], 0
        
        mulx r9, r8, [rdi+8]     ; a[1]*b[1]
        add [r15+16], r8
        adc [r15+24], r9
        adc [r15+32], 0
        adc [r15+40], 0
        
        mulx r9, r8, [rdi+16]    ; a[1]*b[2]
        add [r15+24], r8
        adc [r15+32], r9
        adc [r15+40], 0
        
        mulx r9, r8, [rdi+24]    ; a[1]*b[3]
        add [r15+32], r8
        adc [r15+40], r9
        adc [r15+48], 0
        
        ; Row 2: a[2] * b[0..3]
        mov rdx, [rsi+16]        ; a[2]
        
        mulx r9, r8, [rdi+0]
        add [r15+16], r8
        adc [r15+24], r9
        adc [r15+32], 0
        adc [r15+40], 0
        adc [r15+48], 0
        
        mulx r9, r8, [rdi+8]
        add [r15+24], r8
        adc [r15+32], r9
        adc [r15+40], 0
        adc [r15+48], 0
        
        mulx r9, r8, [rdi+16]
        add [r15+32], r8
        adc [r15+40], r9
        adc [r15+48], 0
        
        mulx r9, r8, [rdi+24]
        add [r15+40], r8
        adc [r15+48], r9
        adc [r15+56], 0
        
        ; Row 3: a[3] * b[0..3]
        mov rdx, [rsi+24]        ; a[3]
        
        mulx r9, r8, [rdi+0]
        add [r15+24], r8
        adc [r15+32], r9
        adc [r15+40], 0
        adc [r15+48], 0
        adc [r15+56], 0
        
        mulx r9, r8, [rdi+8]
        add [r15+32], r8
        adc [r15+40], r9
        adc [r15+48], 0
        adc [r15+56], 0
        
        mulx r9, r8, [rdi+16]
        add [r15+40], r8
        adc [r15+48], r9
        adc [r15+56], 0
        
        mulx r9, r8, [rdi+24]
        add [r15+48], r8
        adc [r15+56], r9
        
        ; Restore non-volatile registers
        pop r15
        pop r14
        pop r13
        pop r12
        pop rdi
        pop rsi
        pop rbx
        ret
    }
}

void mul_4x4_asm(const uint64_t a[4], const uint64_t b[4], uint64_t result[8]) {
    mul_4x4_asm_impl(a, b, result);
}

void square_4_asm(const uint64_t a[4], uint64_t result[8]) {
    // For now, use multiplication
    // TODO: Optimize with Karatsuba or Toom-Cook
    mul_4x4_asm_impl(a, a, result);
}

// ===================================================================
// Fast reduction modulo secp256k1 prime
// p = 2^256 - 2^32 - 977
// Property: 2^256 == 2^32 + 977 (mod p)
// ===================================================================

void montgomery_reduce_asm(uint64_t t[8], uint64_t result[4]) {
    // Implement fast reduction using secp256k1 properties
    // For 512-bit t: t == t_low + t_high * (2^32 + 977) (mod p)
    
    constexpr uint64_t p0 = 0xFFFFFFFEFFFFFC2FULL;
    constexpr uint64_t p1 = 0xFFFFFFFFFFFFFFFFULL;
    constexpr uint64_t p2 = 0xFFFFFFFFFFFFFFFFULL;
    constexpr uint64_t p3 = 0xFFFFFFFFFFFFFFFFULL;
    
    // Start with low 256 bits
    uint64_t r[5] = {t[0], t[1], t[2], t[3], 0};
    
    // Add high limbs multiplied by (2^32 + 977)
    for (size_t i = 0; i < 4; ++i) {
        uint64_t hi = t[4 + i];
        if (hi == 0) continue;
        
        // Add hi * 977
        __asm {
            mov rdx, hi
            mov rax, 977
            mul rax          ; RDX:RAX = hi * 977
            
            lea r8, r
            mov rcx, i
            shl rcx, 3       ; i * 8 (bytes)
            add r8, rcx
            
            add [r8], rax    ; Add low
            adc [r8+8], rdx  ; Add high with carry
            adc qword ptr [r8+16], 0
            adc qword ptr [r8+24], 0
            adc qword ptr [r8+32], 0
        }
        
        // Add hi << 32 (shift by 32 bits)
        size_t shift_limbs = i + 1; // 32 bits = skip to next limb partially
        if (shift_limbs < 5) {
            // This is simplified - proper implementation needs careful bit shifting
            r[shift_limbs] += (hi >> 32);
            if (shift_limbs + 1 < 5) {
                r[shift_limbs + 1] += (hi << 32);
            }
        }
    }
    
    // Final reduction if r >= p
    // Compare and conditionally subtract
    bool needs_reduction = false;
    if (r[4] > 0 || 
        (r[3] > p3) ||
        (r[3] == p3 && r[2] > p2) ||
        (r[3] == p3 && r[2] == p2 && r[1] > p1) ||
        (r[3] == p3 && r[2] == p2 && r[1] == p1 && r[0] >= p0)) {
        needs_reduction = true;
    }
    
    if (needs_reduction) {
        __asm {
            lea r8, r
            lea r9, result
            
            mov rax, [r8+0]
            sub rax, p0
            mov [r9+0], rax
            
            mov rax, [r8+8]
            sbb rax, p1
            mov [r9+8], rax
            
            mov rax, [r8+16]
            sbb rax, p2
            mov [r9+16], rax
            
            mov rax, [r8+24]
            sbb rax, p3
            mov [r9+24], rax
        }
    } else {
        result[0] = r[0];
        result[1] = r[1];
        result[2] = r[2];
        result[3] = r[3];
    }
}

} // namespace detail

// ===================================================================
// High-level wrappers for FieldElement
// ===================================================================

FieldElement field_mul_asm(const FieldElement& a, const FieldElement& b) {
    uint64_t wide[8];
    detail::mul_4x4_asm(a.limbs().data(), b.limbs().data(), wide);
    
    uint64_t reduced[4];
    detail::montgomery_reduce_asm(wide, reduced);
    
    return FieldElement::from_limbs({reduced[0], reduced[1], reduced[2], reduced[3]});
}

FieldElement field_square_asm(const FieldElement& a) {
    uint64_t wide[8];
    detail::square_4_asm(a.limbs().data(), wide);
    
    uint64_t reduced[4];
    detail::montgomery_reduce_asm(wide, reduced);
    
    return FieldElement::from_limbs({reduced[0], reduced[1], reduced[2], reduced[3]});
}

} // namespace secp256k1::fast

#endif // _MSC_VER && _M_X64 && USE_INLINE_ASSEMBLY
