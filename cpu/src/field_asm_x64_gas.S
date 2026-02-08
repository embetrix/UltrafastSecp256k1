# ==============================================================================
# SECP256K1 Field Operations - x64 Assembly (GAS/AT&T Syntax)
# ==============================================================================
# Ultra-optimized field arithmetic using manual register allocation
# Target: 3-5× speedup over BMI2 intrinsics (6-8 ns per operation)
#
# Platform: Linux x64, GAS (GNU Assembler)
# Required: BMI2 instruction set (Intel Haswell 2013+, AMD Excavator 2015+)
# ==============================================================================

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    #define ELF_TYPE(name)
    #define ELF_SIZE(name)
    #define ELF_SECTION(...)

    /* Windows (Clang with sysv_abi) Compatibility Layer */
    /* C++ calls with __attribute__((sysv_abi)), so regs are RDI, RSI, RDX... */
    /* No shims needed for registers! Just hide ELF directives. */
    #define PROLOGUE_3ARG
    #define PROLOGUE_2ARG
    #define PROLOGUE_1ARG
    #define EPILOGUE
#else
    #define ELF_TYPE(name) .type name, @function
    #define ELF_SIZE(name) .size name, .-name
    /* Accept commas by taking varargs and emitting them verbatim. */
    #define ELF_SECTION(...) __VA_ARGS__

    #define PROLOGUE_3ARG
    #define PROLOGUE_2ARG
    #define PROLOGUE_1ARG
    #define EPILOGUE
#endif

.text
.intel_syntax noprefix

# ==============================================================================
# mul_4x4_asm: 256-bit × 256-bit → 512-bit multiplication
# ==============================================================================
# Ultra-fast bigint multiplication using BMI2 MULX + parallel carry chains
#
# Parameters (x64 System V calling convention - Linux):
#   RDI = const uint64_t* a  (pointer to 4 limbs)
#   RSI = const uint64_t* b  (pointer to 4 limbs)
#   RDX = uint64_t* result   (pointer to 8 limbs output)
#
# Note: Different from Windows! System V uses RDI, RSI, RDX, RCX, R8, R9
#
# Performance: Target 6-8 ns (vs 33-37 ns with BMI2 intrinsics)
# ==============================================================================

.globl mul_4x4_asm
ELF_TYPE(mul_4x4_asm)
mul_4x4_asm:
    PROLOGUE_3ARG
    # Preserve non-volatile registers (callee-saved: rbx, rbp, r12-r15)
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    # Save result pointer (rdx will be used by mulx)
    mov rcx, rdx         # rcx = result pointer
    
    # rdi = a pointer (already in place)
    # rsi = b pointer (already in place)
    # rcx = result pointer

    # Zero accumulators res0..res7 in r8..r15
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15

    # ---------------- i = 0 (multiply by a[0]) ----------------
    mov rbp, [rdi+0]     # rbp = a[0]

    mov rdx, [rsi+0]     # rdx = b[0] (mulx source)
    mulx rbx, rax, rbp   # rbx:rax = b[0] * a[0]
    add r8, rax
    adc r9, rbx
    adc r10, 0

    mov rdx, [rsi+8]     # rdx = b[1]
    mulx rbx, rax, rbp
    add r9, rax
    adc r10, rbx
    adc r11, 0

    mov rdx, [rsi+16]    # rdx = b[2]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0

    mov rdx, [rsi+24]    # rdx = b[3]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    # ---------------- i = 1 (multiply by a[1]) ----------------
    mov rbp, [rdi+8]     # rbp = a[1]

    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r9, rax
    adc r10, rbx
    adc r11, 0
    adc r12, 0
    adc r13, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    # ---------------- i = 2 (multiply by a[2]) ----------------
    mov rbp, [rdi+16]    # rbp = a[2]

    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r13, rax
    adc r14, rbx
    adc r15, 0

    # ---------------- i = 3 (multiply by a[3]) ----------------
    mov rbp, [rdi+24]    # rbp = a[3]

    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r13, rax
    adc r14, rbx
    adc r15, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r14, rax
    adc r15, rbx

    # Store result (8 limbs)
    mov [rcx+0],  r8
    mov [rcx+8],  r9
    mov [rcx+16], r10
    mov [rcx+24], r11
    mov [rcx+32], r12
    mov [rcx+40], r13
    mov [rcx+48], r14
    mov [rcx+56], r15

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    EPILOGUE
    ret
ELF_SIZE(mul_4x4_asm)

# ==============================================================================
# sqr_4x4_asm: 256-bit Squaring -> 512-bit result
# ==============================================================================
# Optimized squaring using cross-products (10 muls instead of 16)
#
# Parameters:
#   RDI = const uint64_t* a
#   RSI = uint64_t* result
# ==============================================================================

.globl sqr_4x4_asm
ELF_TYPE(sqr_4x4_asm)
sqr_4x4_asm:
    PROLOGUE_2ARG
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    # rdi = a
    # rsi = result
    
    # Clear accumulators
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15

    # ----------------------------------------------------------------
    # 1. Cross Products (accumulate in r9..r15)
    # ----------------------------------------------------------------
    
    # a[0] * a[1] -> r9, r10
    mov rdx, [rdi+8]     # a[1]
    mulx r10, r9, [rdi]  # a[0] * a[1]
    
    # a[0] * a[2] -> r10, r11
    mov rdx, [rdi+16]    # a[2]
    mulx rbx, rax, [rdi] # a[0] * a[2]
    add r10, rax
    adc r11, rbx
    
    # a[0] * a[3] -> r11, r12
    mov rdx, [rdi+24]    # a[3]
    mulx rbx, rax, [rdi] # a[0] * a[3]
    add r11, rax
    adc r12, rbx
    
    # a[1] * a[2] -> r11, r12
    mov rdx, [rdi+16]    # a[2]
    mulx rbx, rax, [rdi+8] # a[1] * a[2]
    add r11, rax
    adc r12, rbx
    adc r13, 0
    
    # a[1] * a[3] -> r12, r13
    mov rdx, [rdi+24]    # a[3]
    mulx rbx, rax, [rdi+8] # a[1] * a[3]
    add r12, rax
    adc r13, rbx
    adc r14, 0
    
    # a[2] * a[3] -> r13, r14
    mov rdx, [rdi+24]    # a[3]
    mulx rbx, rax, [rdi+16] # a[2] * a[3]
    add r13, rax
    adc r14, rbx
    adc r15, 0
    
    # ----------------------------------------------------------------
    # 2. Double the cross products (r9..r15)
    # ----------------------------------------------------------------
    # r8 is 0.
    add r9, r9
    adc r10, r10
    adc r11, r11
    adc r12, r12
    adc r13, r13
    adc r14, r14
    adc r15, r15
    # Carry from r15 is lost? No, result is 512-bit.
    # Max value of cross products sum is roughly (2^256)^2 / 2.
    # Doubling it gives (2^256)^2.
    # It fits in 512 bits.
    
    # ----------------------------------------------------------------
    # 3. Add Squares
    # ----------------------------------------------------------------
    
    # a[0]^2 -> r8, r9
    mov rdx, [rdi]
    mulx rbx, rax, rdx
    add r8, rax
    adc r9, rbx
    adc r10, 0
    adc r11, 0
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0
    
    # a[1]^2 -> r10, r11
    mov rdx, [rdi+8]
    mulx rbx, rax, rdx
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0
    
    # a[2]^2 -> r12, r13
    mov rdx, [rdi+16]
    mulx rbx, rax, rdx
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0
    
    # a[3]^2 -> r14, r15
    mov rdx, [rdi+24]
    mulx rbx, rax, rdx
    add r14, rax
    adc r15, rbx
    
    # Store result
    mov [rsi+0],  r8
    mov [rsi+8],  r9
    mov [rsi+16], r10
    mov [rsi+24], r11
    mov [rsi+32], r12
    mov [rsi+40], r13
    mov [rsi+48], r14
    mov [rsi+56], r15

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    EPILOGUE
    ret
ELF_SIZE(sqr_4x4_asm)

# ==============================================================================
# add_4_asm: 256-bit Modular Addition (a + b mod p)
# ==============================================================================
# p = 2^256 - 2^32 - 977
#
# Parameters:
#   RDI = const uint64_t* a
#   RSI = const uint64_t* b
#   RDX = uint64_t* result
# ==============================================================================

.globl add_4_asm
ELF_TYPE(add_4_asm)
add_4_asm:
    PROLOGUE_3ARG
    push rbx
    push rbp
    
    # Load a
    mov r8, [rdi]
    mov r9, [rdi+8]
    mov r10, [rdi+16]
    mov r11, [rdi+24]
    
    # Add b
    add r8, [rsi]
    adc r9, [rsi+8]
    adc r10, [rsi+16]
    adc r11, [rsi+24]
    
    # Capture carry of S
    setc cl  # cl = 1 if S overflowed
    
    # Calculate T = S + K
    # K = 0x1000003D1
    mov rax, 0x1000003D1
    
    mov rdi, r8
    add rdi, rax
    mov rsi, r9
    adc rsi, 0
    mov rbx, r10
    adc rbx, 0
    mov rbp, r11
    adc rbp, 0
    
    # Capture carry of T
    setc al # al = 1 if T overflowed
    
    # Condition: S_carry OR T_carry
    or al, cl
    
    # If true, use T (rdi, rsi, rbx, rbp)
    # If false, use S (r8, r9, r10, r11)
    
    test al, al
    cmovnz r8, rdi
    cmovnz r9, rsi
    cmovnz r10, rbx
    cmovnz r11, rbp
    
    # Store result
    mov [rdx], r8
    mov [rdx+8], r9
    mov [rdx+16], r10
    mov [rdx+24], r11
    
    pop rbp
    pop rbx
    EPILOGUE
    ret

ELF_SIZE(add_4_asm)

# ==============================================================================
# sub_4_asm: 256-bit Modular Subtraction (a - b mod p)
# ==============================================================================
# Parameters:
#   RDI = const uint64_t* a
#   RSI = const uint64_t* b
#   RDX = uint64_t* result
# ==============================================================================

.globl sub_4_asm
ELF_TYPE(sub_4_asm)
sub_4_asm:
    PROLOGUE_3ARG
    push rbx
    push rbp
    
    # Load a
    mov r8, [rdi]
    mov r9, [rdi+8]
    mov r10, [rdi+16]
    mov r11, [rdi+24]
    
    # Subtract b
    sub r8, [rsi]
    sbb r9, [rsi+8]
    sbb r10, [rsi+16]
    sbb r11, [rsi+24]
    
    # Capture borrow
    setc cl
    
    # Calculate T = S - K
    # K = 0x1000003D1
    mov rax, 0x1000003D1
    
    mov rdi, r8
    sub rdi, rax
    mov rsi, r9
    sbb rsi, 0
    mov rbx, r10
    sbb rbx, 0
    mov rbp, r11
    sbb rbp, 0
    
    # If borrow (cl=1), use T
    test cl, cl
    cmovnz r8, rdi
    cmovnz r9, rsi
    cmovnz r10, rbx
    cmovnz r11, rbp
    
    # Store
    mov [rdx], r8
    mov [rdx+8], r9
    mov [rdx+16], r10
    mov [rdx+24], r11
    
    pop rbp
    pop rbx
    EPILOGUE
    ret

ELF_SIZE(sub_4_asm)

.text

# ==============================================================================

# ==============================================================================
# reduce_4_asm: 512-bit -> 256-bit Modular Reduction (Strictly Normalized)
# ==============================================================================
# Reduces a 512-bit number modulo P = 2^256 - 2^32 - 977
# Ensures result < P
#
# Parameters:
#   RDI = uint64_t* data (pointer to 8 limbs, result stored in first 4)
# ==============================================================================

.globl reduce_4_asm
ELF_TYPE(reduce_4_asm)
reduce_4_asm:
    PROLOGUE_1ARG
    push rbx
    push r12
    push r13
    push r14
    push r15

    # Load low 256 bits
    mov r8, [rdi]
    mov r9, [rdi+8]
    mov r10, [rdi+16]
    mov r11, [rdi+24]
    xor r12, r12  # 5th limb (carry)

    mov rdx, 977  # Constant for mulx

    # === Limb 4 (Offset 0) ===
    mov rax, [rdi+32]
    
    # 1. mul by 977
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc r12, 0

    # 2. shift by 32
    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc r12, 0

    # === Limb 5 (Offset 1) ===
    mov rax, [rdi+40]
    
    # 1. mul by 977
    mulx rcx, rbx, rax
    add r9, rbx
    adc r10, rcx
    adc r11, 0
    adc r12, 0

    # 2. shift by 32
    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r9, rbx
    adc r10, rax
    adc r11, 0
    adc r12, 0

    # === Limb 6 (Offset 2) ===
    mov rax, [rdi+48]
    
    # 1. mul by 977
    mulx rcx, rbx, rax
    add r10, rbx
    adc r11, rcx
    adc r12, 0

    # 2. shift by 32
    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r10, rbx
    adc r11, rax
    adc r12, 0

    # === Limb 7 (Offset 3) ===
    mov rax, [rdi+56]
    
    # 1. mul by 977
    mulx rcx, rbx, rax
    add r11, rbx
    adc r12, rcx

    # 2. shift by 32
    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r11, rbx
    adc r12, rax

.Lreduce_loop_strict:
    test r12, r12
    jz .Lcheck_mod_p

    mov rax, r12
    xor r12, r12 # Clear carry

    # 1. mul by 977
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc r12, 0

    # 2. shift by 32
    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc r12, 0
    
    jmp .Lreduce_loop_strict

.Lcheck_mod_p:
    # Check if result >= P
    # P = 2^256 - K, where K = 0x1000003D1
    # result >= P <=> result + K >= 2^256 (Carry out)
    
    mov r13, 0x1000003D1 # K
    
    mov rax, r8
    add rax, r13
    mov rbx, r9
    adc rbx, 0
    mov rcx, r10
    adc rcx, 0
    mov rdx, r11
    adc rdx, 0
    
    # If Carry (CF=1), then result >= P
    # And (rax, rbx, rcx, rdx) contains (result + K) mod 2^256
    # Which is exactly (result - P)
    
    cmovc r8, rax
    cmovc r9, rbx
    cmovc r10, rcx
    cmovc r11, rdx

.Ldone_strict:
    mov [rdi], r8
    mov [rdi+8], r9
    mov [rdi+16], r10
    mov [rdi+24], r11

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
ELF_SIZE(reduce_4_asm)

# ==============================================================================
# field_mul_full_asm: 256x256 -> 512 -> 256-bit Modular Multiplication
# ==============================================================================
# Combines mul_4x4_asm and reduce_4_asm into a single function.
# Avoids memory round-trip for the 512-bit intermediate result.
#
# Parameters:
#   RDI = const uint64_t* a
#   RSI = const uint64_t* b
#   RDX = uint64_t* result
# ==============================================================================

.globl field_mul_full_asm
ELF_TYPE(field_mul_full_asm)
field_mul_full_asm:
    PROLOGUE_3ARG
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    # Save result pointer (rdx will be used by mulx)
    push rdx

    # ----------------------------------------------------------------
    # 1. Multiplication (256x256 -> 512)
    #    Result in r8..r15
    # ----------------------------------------------------------------

    # Zero accumulators
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15

    # --- i = 0 ---
    mov rbp, [rdi+0]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r8, rax
    adc r9, rbx
    adc r10, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r9, rax
    adc r10, rbx
    adc r11, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    # --- i = 1 ---
    mov rbp, [rdi+8]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r9, rax
    adc r10, rbx
    adc r11, 0
    adc r12, 0
    adc r13, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    # --- i = 2 ---
    mov rbp, [rdi+16]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r13, rax
    adc r14, rbx
    adc r15, 0

    # --- i = 3 ---
    mov rbp, [rdi+24]
    mov rdx, [rsi+0]
    mulx rbx, rax, rbp
    add r11, rax
    adc r12, rbx
    adc r13, 0
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+8]
    mulx rbx, rax, rbp
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0

    mov rdx, [rsi+16]
    mulx rbx, rax, rbp
    add r13, rax
    adc r14, rbx
    adc r15, 0

    mov rdx, [rsi+24]
    mulx rbx, rax, rbp
    add r14, rax
    adc r15, rbx

    # ----------------------------------------------------------------
    # 2. Reduction (512 -> 256)
    #    Input: r8..r15
    #    Output: r8..r11
    # ----------------------------------------------------------------
    
    # We need a temporary register for carry.
    # rdi and rsi are free now (we loaded a and b).
    # Let's use rdi as carry accumulator (instead of r12 in reduce_4_asm, which is occupied).
    
    xor rdi, rdi # Carry for reduction
    mov rdx, 977 # Constant

    # === Limb 4 (r12) ===
    mov rax, r12
    
    # 1. mul by 977
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    # 2. shift by 32
    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    # === Limb 5 (r13) ===
    mov rax, r13
    
    # 1. mul by 977
    mulx rcx, rbx, rax
    add r9, rbx
    adc r10, rcx
    adc r11, 0
    adc rdi, 0

    # 2. shift by 32
    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r9, rbx
    adc r10, rax
    adc r11, 0
    adc rdi, 0

    # === Limb 6 (r14) ===
    mov rax, r14
    
    # 1. mul by 977
    mulx rcx, rbx, rax
    add r10, rbx
    adc r11, rcx
    adc rdi, 0

    # 2. shift by 32
    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r10, rbx
    adc r11, rax
    adc rdi, 0

    # === Limb 7 (r15) ===
    mov rax, r15
    
    # 1. mul by 977
    mulx rcx, rbx, rax
    add r11, rbx
    adc rdi, rcx

    # 2. shift by 32
    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r11, rbx
    adc rdi, rax

.Lfull_reduce_loop:
    test rdi, rdi
    jz .Lfull_check_mod_p

    mov rax, rdi
    xor rdi, rdi # Clear carry

    # 1. mul by 977
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    # 2. shift by 32
    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc rdi, 0
    
    jmp .Lfull_reduce_loop

.Lfull_check_mod_p:
    # Check if result >= P
    mov r13, 0x1000003D1 # K
    
    mov rax, r8
    add rax, r13
    mov rbx, r9
    adc rbx, 0
    mov rcx, r10
    adc rcx, 0
    mov rdx, r11
    adc rdx, 0
    
    cmovc r8, rax
    cmovc r9, rbx
    cmovc r10, rcx
    cmovc r11, rdx

    # Restore result pointer
    pop rdx

    # Store result
    mov [rdx], r8
    mov [rdx+8], r9
    mov [rdx+16], r10
    mov [rdx+24], r11

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    EPILOGUE
    ret
ELF_SIZE(field_mul_full_asm)

# ==============================================================================
# field_sqr_full_asm: 256 -> 512 -> 256-bit Modular Squaring
# ==============================================================================
# Combines sqr_4x4_asm and reduce_4_asm.
#
# Parameters:
#   RDI = const uint64_t* a
#   RSI = uint64_t* result
# ==============================================================================

.globl field_sqr_full_asm
ELF_TYPE(field_sqr_full_asm)
field_sqr_full_asm:
    PROLOGUE_2ARG
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15

    # Save result pointer
    push rsi

    # Clear accumulators
    xor r8, r8
    xor r9, r9
    xor r10, r10
    xor r11, r11
    xor r12, r12
    xor r13, r13
    xor r14, r14
    xor r15, r15

    # ----------------------------------------------------------------
    # 1. Cross Products (accumulate in r9..r15)
    # ----------------------------------------------------------------
    
    # a[0] * a[1] -> r9, r10
    mov rdx, [rdi+8]     # a[1]
    mulx r10, r9, [rdi]  # a[0] * a[1]
    
    # a[0] * a[2] -> r10, r11
    mov rdx, [rdi+16]    # a[2]
    mulx rbx, rax, [rdi] # a[0] * a[2]
    add r10, rax
    adc r11, rbx
    
    # a[0] * a[3] -> r11, r12
    mov rdx, [rdi+24]    # a[3]
    mulx rbx, rax, [rdi] # a[0] * a[3]
    add r11, rax
    adc r12, rbx
    
    # a[1] * a[2] -> r11, r12
    mov rdx, [rdi+16]    # a[2]
    mulx rbx, rax, [rdi+8] # a[1] * a[2]
    add r11, rax
    adc r12, rbx
    adc r13, 0
    
    # a[1] * a[3] -> r12, r13
    mov rdx, [rdi+24]    # a[3]
    mulx rbx, rax, [rdi+8] # a[1] * a[3]
    add r12, rax
    adc r13, rbx
    adc r14, 0
    
    # a[2] * a[3] -> r13, r14
    mov rdx, [rdi+24]    # a[3]
    mulx rbx, rax, [rdi+16] # a[2] * a[3]
    add r13, rax
    adc r14, rbx
    adc r15, 0
    
    # ----------------------------------------------------------------
    # 2. Double the cross products
    # ----------------------------------------------------------------
    add r9, r9
    adc r10, r10
    adc r11, r11
    adc r12, r12
    adc r13, r13
    adc r14, r14
    adc r15, r15
    # No carry out from r15 possible for 256-bit inputs
    
    # ----------------------------------------------------------------
    # 3. Add Squares
    # ----------------------------------------------------------------
    
    # a[0]^2 -> r8, r9
    mov rdx, [rdi]
    mulx rbx, rax, rdx
    add r8, rax
    adc r9, rbx
    adc r10, 0
    adc r11, 0
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0
    
    # a[1]^2 -> r10, r11
    mov rdx, [rdi+8]
    mulx rbx, rax, rdx
    add r10, rax
    adc r11, rbx
    adc r12, 0
    adc r13, 0
    adc r14, 0
    adc r15, 0
    
    # a[2]^2 -> r12, r13
    mov rdx, [rdi+16]
    mulx rbx, rax, rdx
    add r12, rax
    adc r13, rbx
    adc r14, 0
    adc r15, 0
    
    # a[3]^2 -> r14, r15
    mov rdx, [rdi+24]
    mulx rbx, rax, rdx
    add r14, rax
    adc r15, rbx

    # ----------------------------------------------------------------
    # 4. Reduction (512 -> 256)
    # ----------------------------------------------------------------
    
    xor rdi, rdi # Carry
    mov rdx, 977
    
    # === Limb 4 (r12) ===
    mov rax, r12
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    # === Limb 5 (r13) ===
    mov rax, r13
    mulx rcx, rbx, rax
    add r9, rbx
    adc r10, rcx
    adc r11, 0
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r9, rbx
    adc r10, rax
    adc r11, 0
    adc rdi, 0

    # === Limb 6 (r14) ===
    mov rax, r14
    mulx rcx, rbx, rax
    add r10, rbx
    adc r11, rcx
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r10, rbx
    adc r11, rax
    adc rdi, 0

    # === Limb 7 (r15) ===
    mov rax, r15
    mulx rcx, rbx, rax
    add r11, rbx
    adc rdi, rcx

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r11, rbx
    adc rdi, rax

.Lsqr_reduce_loop:
    test rdi, rdi
    jz .Lsqr_check_mod_p

    mov rax, rdi
    xor rdi, rdi
    
    mulx rcx, rbx, rax
    add r8, rbx
    adc r9, rcx
    adc r10, 0
    adc r11, 0
    adc rdi, 0

    mov rbx, rax
    shl rbx, 32
    shr rax, 32
    add r8, rbx
    adc r9, rax
    adc r10, 0
    adc r11, 0
    adc rdi, 0
    
    jmp .Lsqr_reduce_loop

.Lsqr_check_mod_p:
    mov r13, 0x1000003D1
    
    mov rax, r8
    add rax, r13
    mov rbx, r9
    adc rbx, 0
    mov rcx, r10
    adc rcx, 0
    mov rdx, r11
    adc rdx, 0
    
    cmovc r8, rax
    cmovc r9, rbx
    cmovc r10, rcx
    cmovc r11, rdx

    # Restore result pointer
    pop rsi

    mov [rsi], r8
    mov [rsi+8], r9
    mov [rsi+16], r10
    mov [rsi+24], r11

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    EPILOGUE
    ret
ELF_SIZE(field_sqr_full_asm)

/* Mark stack as non-executable (ELF). No-op on Windows. */
#if !defined(_WIN32)
.section .note.GNU-stack,"",@progbits
#endif

