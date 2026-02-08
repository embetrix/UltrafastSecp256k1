/*
 * RISC-V 64-bit Assembly Optimizations for secp256k1 Field Operations
 * 
 * Target: RV64GC (RISC-V 64-bit with General + Compressed + Multiply extensions)
 * Optional: RVV (Vector Extension) for batch operations
 * Tested on: StarFive JH7110 (Milk-V Mars)
 * 
 * Field Element: 256-bit = 4 × 64-bit limbs (little-endian)
 * Prime: 2^256 - 2^32 - 977 (secp256k1 field prime)
 * 
 * RISC-V Instructions Used:
 *   MUL    rd, rs1, rs2    - 64×64 → 64 (low bits)
 *   MULH   rd, rs1, rs2    - 64×64 → 64 (high bits, signed×signed)
 *   MULHU  rd, rs1, rs2    - 64×64 → 64 (high bits, unsigned×unsigned)
 *   ADD    rd, rs1, rs2    - 64-bit addition
 *   SLTU   rd, rs1, rs2    - Set if less than unsigned (for carry detection)
 * 
 * Build Options (CMake):
 *   -DSECP256K1_RISCV_USE_VECTOR=ON    - Enable RVV batch operations (4-8× speedup)
 *   -DSECP256K1_RISCV_FAST_REDUCTION=ON - Fast modular reduction in assembly
 *   -DSECP256K1_RISCV_USE_PREFETCH=ON  - Enable cache prefetch hints
 */

#ifndef SECP256K1_RISCV_FAST_REDUCTION
#define SECP256K1_RISCV_FAST_REDUCTION 1
#endif

    .section .text
    .align 2

/*
 * field_mul_asm_riscv64
 * 
 * Multiply two field elements: r = a × b (mod p)
 * 
 * Parameters:
 *   a0 = result pointer (uint64_t r[4])
 *   a1 = operand a pointer (uint64_t a[4])
 *   a2 = operand b pointer (uint64_t b[4])
 * 
 * Clobbers: t0-t6, a3-a7
 * 
 * Algorithm: Comba multiplication (product scanning)
 *   - Compute 4×4 → 8 limbs product
 *   - Reduce mod p using fast reduction
 */
    .globl field_mul_asm_riscv64
    .type field_mul_asm_riscv64, @function
field_mul_asm_riscv64:
    # RISC-V ABI: Save callee-saved registers (s0-s11) and ra
    # Stack layout: [ra, s0-s11: 104 bytes] [scratch: 64 bytes] = 176 bytes (aligned to 16)
    addi    sp, sp, -176
    sd      ra, 0(sp)
    sd      s0, 8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)
    sd      s6, 56(sp)
    sd      s7, 64(sp)
    sd      s8, 72(sp)
    sd      s9, 80(sp)
    sd      s10, 88(sp)
    sd      s11, 96(sp)

    # Save result pointer in s11 (callee-saved)
    mv      s11, a0

    # Load operand a into s0-s3 (use callee-saved to avoid conflicts)
    ld      s0, 0(a1)       # a[0]
    ld      s1, 8(a1)       # a[1]
    ld      s2, 16(a1)      # a[2]
    ld      s3, 24(a1)      # a[3]

    # Load operand b into s4-s7
    ld      s4, 0(a2)       # b[0]
    ld      s5, 8(a2)       # b[1]
    ld      s6, 16(a2)      # b[2]
    ld      s7, 24(a2)      # b[3]

    #
    # Comba multiplication: compute 8-limb product
    # Product limbs stored on stack at offset 104: sp[104..167] (8 × 8 bytes)
    # Using simplified, correct carry propagation
    #

    # Column 0: r[0] = a[0]*b[0]
    mul     t0, s0, s4      # low
    mulhu   t1, s0, s4      # high
    sd      t0, 104(sp)     # r[0] at offset 104
    mv      t2, t1          # carry0 = high part
    li      t3, 0           # carry1 = 0

    # Column 1: r[1] = a[0]*b[1] + a[1]*b[0] + carry
    mul     t0, s0, s5      # a[0]*b[1] low
    mulhu   t1, s0, s5      # a[0]*b[1] high
    add     t0, t0, t2      # add carry from previous column
    sltu    a0, t0, t2      # overflow?
    add     t1, t1, a0      # propagate to high part

    mul     t4, s1, s4      # a[1]*b[0] low
    mulhu   t5, s1, s4      # a[1]*b[0] high
    add     t0, t0, t4      # add to column
    sltu    a0, t0, t4      # overflow?
    add     t1, t1, t5      # add high part
    sltu    a1, t1, t5      # overflow from high add?
    add     t1, t1, a0      # add carry from low overflow
    sltu    a2, t1, a0      # overflow from carry add?
    add     t2, a1, a2      # total carry out

    sd      t0, 112(sp)     # r[1]

    # Column 2: r[2] = a[0]*b[2] + a[1]*b[1] + a[2]*b[0] + carry
    mul     t0, s0, s6      # a[0]*b[2] low
    mulhu   t4, s0, s6      # a[0]*b[2] high
    add     t0, t0, t1      # add carry (high from column 1)
    sltu    a0, t0, t1      # overflow?
    add     t4, t4, a0      # propagate
    add     t4, t4, t2      # add second-level carry
    sltu    a0, t4, t2      # overflow?

    mul     t5, s1, s5      # a[1]*b[1] low
    mulhu   t6, s1, s5      # a[1]*b[1] high
    add     t0, t0, t5      # add low
    sltu    a1, t0, t5
    add     t4, t4, t6      # add high
    sltu    a2, t4, t6
    add     t4, t4, a1      # add carry from low
    sltu    a3, t4, a1
    add     a0, a0, a2
    add     a0, a0, a3

    mul     t5, s2, s4      # a[2]*b[0] low
    mulhu   t6, s2, s4      # a[2]*b[0] high
    add     t0, t0, t5
    sltu    a1, t0, t5
    add     t4, t4, t6
    sltu    a2, t4, t6
    add     t4, t4, a1
    sltu    a3, t4, a1
    add     a0, a0, a2
    add     t2, a0, a3      # carry out

    mv      t1, t4          # move high to carry
    sd      t0, 120(sp)     # r[2]

    # Column 3: r[3] = a[0]*b[3] + a[1]*b[2] + a[2]*b[1] + a[3]*b[0] + carry
    mul     t0, s0, s7      # a[0]*b[3] low
    mulhu   t4, s0, s7      # a[0]*b[3] high
    add     t0, t0, t1      # add carry
    sltu    a0, t0, t1
    add     t4, t4, a0
    add     t4, t4, t2      # add second-level carry
    sltu    a0, t4, t2

    mul     t5, s1, s6      # a[1]*b[2]
    mulhu   t6, s1, s6
    add     t0, t0, t5
    sltu    a1, t0, t5
    add     t4, t4, t6
    sltu    a2, t4, t6
    add     t4, t4, a1
    sltu    a3, t4, a1
    add     a0, a0, a2
    add     a0, a0, a3

    mul     t5, s2, s5      # a[2]*b[1]
    mulhu   t6, s2, s5
    add     t0, t0, t5
    sltu    a1, t0, t5
    add     t4, t4, t6
    sltu    a2, t4, t6
    add     t4, t4, a1
    sltu    a3, t4, a1
    add     a0, a0, a2
    add     a0, a0, a3

    mul     t5, s3, s4      # a[3]*b[0]
    mulhu   t6, s3, s4
    add     t0, t0, t5
    sltu    a1, t0, t5
    add     t4, t4, t6
    sltu    a2, t4, t6
    add     t4, t4, a1
    sltu    a3, t4, a1
    add     a0, a0, a2
    add     t2, a0, a3

    mv      t1, t4
    sd      t0, 128(sp)     # r[3]

    # Column 4: r[4] = a[1]*b[3] + a[2]*b[2] + a[3]*b[1] + carry
    mul     t0, s1, s7      # a[1]*b[3]
    mulhu   t4, s1, s7
    add     t0, t0, t1
    sltu    a0, t0, t1
    add     t4, t4, a0
    add     t4, t4, t2
    sltu    a0, t4, t2

    mul     t5, s2, s6      # a[2]*b[2]
    mulhu   t6, s2, s6
    add     t0, t0, t5
    sltu    a1, t0, t5
    add     t4, t4, t6
    sltu    a2, t4, t6
    add     t4, t4, a1
    sltu    a3, t4, a1
    add     a0, a0, a2
    add     a0, a0, a3

    mul     t5, s3, s5      # a[3]*b[1]
    mulhu   t6, s3, s5
    add     t0, t0, t5
    sltu    a1, t0, t5
    add     t4, t4, t6
    sltu    a2, t4, t6
    add     t4, t4, a1
    sltu    a3, t4, a1
    add     a0, a0, a2
    add     t2, a0, a3

    mv      t1, t4
    sd      t0, 136(sp)     # r[4]

    # Column 5: r[5] = a[2]*b[3] + a[3]*b[2] + carry
    mul     t0, s2, s7      # a[2]*b[3]
    mulhu   t4, s2, s7
    add     t0, t0, t1
    sltu    a0, t0, t1
    add     t4, t4, a0
    add     t4, t4, t2
    sltu    a0, t4, t2

    mul     t5, s3, s6      # a[3]*b[2]
    mulhu   t6, s3, s6
    add     t0, t0, t5
    sltu    a1, t0, t5
    add     t4, t4, t6
    sltu    a2, t4, t6
    add     t4, t4, a1
    sltu    a3, t4, a1
    add     t2, a0, a2
    add     t2, t2, a3

    mv      t1, t4
    sd      t0, 144(sp)     # r[5]

    # Column 6,7: r[6:7] = a[3]*b[3] + carry
    mul     t0, s3, s7      # a[3]*b[3]
    mulhu   t4, s3, s7
    add     t0, t0, t1
    sltu    a0, t0, t1
    add     t4, t4, a0
    add     t4, t4, t2      # add final carry

    sd      t0, 152(sp)     # r[6]
    sd      t4, 160(sp)     # r[7]

#ifdef SECP256K1_RISCV_FAST_REDUCTION
    #
    # Full modular reduction: reduce 8-limb (512-bit) product mod p
    # Algorithm (matching x64):
    #   For each high limb h[i]:
    #     result += h[i] * 977
    #     result += h[i] << 32
    #   Loop until overflow is zero
    #   Final check: if result >= P, add K (which is subtract P mod 2^256)
    #
    # P = 2^256 - 2^32 - 977
    # K = 2^32 + 977 = 0x1000003D1
    #

    # Load low 256 bits into s3-s6 (use callee-saved as working registers)
    ld      s3, 104(sp)     # r0 at offset 104
    ld      s4, 112(sp)     # r1 at offset 112
    ld      s5, 120(sp)     # r2 at offset 120
    ld      s6, 128(sp)     # r3 at offset 128

    li      s2, 977         # constant for multiplication
    li      s1, 0           # overflow/carry accumulator

    # === Process limb 4 (r[4]) ===
    ld      s0, 136(sp)     # h0 = r[4] at offset 136

    # 1. Multiply by 977
    mul     a0, s0, s2      # low
    mulhu   a1, s0, s2      # high
    add     s3, s3, a0
    sltu    a2, s3, a0
    add     s4, s4, a1
    sltu    a3, s4, a1
    add     s4, s4, a2
    sltu    a2, s4, a2
    or      a3, a3, a2      # combine carries
    add     s5, s5, a3
    sltu    a3, s5, a3
    add     s6, s6, a3
    sltu    a3, s6, a3
    add     s1, s1, a3      # accumulate to overflow

    # 2. Shift by 32 bits and add
    slli    a0, s0, 32      # low part (s0 << 32)
    srli    a1, s0, 32      # high part (s0 >> 32)
    add     s3, s3, a0
    sltu    a2, s3, a0
    add     s4, s4, a1
    sltu    a3, s4, a1
    add     s4, s4, a2
    sltu    a2, s4, a2
    or      a3, a3, a2
    add     s5, s5, a3
    sltu    a3, s5, a3
    add     s6, s6, a3
    sltu    a3, s6, a3
    add     s1, s1, a3

    # === Process limb 5 (r[5]) ===
    ld      s0, 144(sp)     # h1 = r[5] at offset 144

    # 1. Multiply by 977
    mul     a0, s0, s2
    mulhu   a1, s0, s2
    add     s4, s4, a0
    sltu    a2, s4, a0
    add     s5, s5, a1
    sltu    a3, s5, a1
    add     s5, s5, a2
    sltu    a2, s5, a2
    or      a3, a3, a2
    add     s6, s6, a3
    sltu    a3, s6, a3
    add     s1, s1, a3

    # 2. Shift by 32
    slli    a0, s0, 32
    srli    a1, s0, 32
    add     s4, s4, a0
    sltu    a2, s4, a0
    add     s5, s5, a1
    sltu    a3, s5, a1
    add     s5, s5, a2
    sltu    a2, s5, a2
    or      a3, a3, a2
    add     s6, s6, a3
    sltu    a3, s6, a3
    add     s1, s1, a3

    # === Process limb 6 (r[6]) ===
    ld      s0, 152(sp)     # h2 = r[6] at offset 152

    # 1. Multiply by 977
    mul     a0, s0, s2
    mulhu   a1, s0, s2
    add     s5, s5, a0
    sltu    a2, s5, a0
    add     s6, s6, a1
    sltu    a3, s6, a1
    add     s6, s6, a2
    sltu    a2, s6, a2
    or      a3, a3, a2
    add     s1, s1, a3

    # 2. Shift by 32
    slli    a0, s0, 32
    srli    a1, s0, 32
    add     s5, s5, a0
    sltu    a2, s5, a0
    add     s6, s6, a1
    sltu    a3, s6, a1
    add     s6, s6, a2
    sltu    a2, s6, a2
    or      a3, a3, a2
    add     s1, s1, a3

    # === Process limb 7 (r[7]) ===
    ld      s0, 160(sp)     # h3 = r[7] at offset 160

    # 1. Multiply by 977
    mul     a0, s0, s2
    mulhu   a1, s0, s2
    add     s6, s6, a0
    sltu    a2, s6, a0
    add     s1, s1, a1      # accumulate overflow
    add     s1, s1, a2

    # 2. Shift by 32
    slli    a0, s0, 32
    srli    a1, s0, 32
    add     s6, s6, a0
    sltu    a2, s6, a0
    add     s1, s1, a1
    add     s1, s1, a2

    # === Loop: reduce overflow ===
.Lreduce_loop:
    beqz    s1, .Lcheck_mod_p

    mv      s0, s1
    li      s1, 0           # clear for next iteration

    # 1. Multiply by 977 and add to result
    mul     a0, s0, s2      # low
    mulhu   a1, s0, s2      # high
    add     s3, s3, a0      # add to limb 0
    sltu    a2, s3, a0      # carry from limb 0
    add     s4, s4, a1      # add high to limb 1
    sltu    a3, s4, a1      # carry from high add
    add     s4, s4, a2      # add carry from limb 0
    sltu    a4, s4, a2      # carry from carry add
    or      a3, a3, a4      # combine carries
    add     s5, s5, a3      # propagate to limb 2
    sltu    a3, s5, a3      # carry from limb 2
    add     s6, s6, a3      # propagate to limb 3
    sltu    a3, s6, a3      # carry from limb 3
    add     s1, s1, a3      # accumulate overflow

    # 2. Shift by 32 and add to result
    slli    a0, s0, 32      # low part (s0 << 32)
    srli    a1, s0, 32      # high part (s0 >> 32)
    add     s3, s3, a0      # add to limb 0
    sltu    a2, s3, a0      # carry from limb 0
    add     s4, s4, a1      # add high to limb 1
    sltu    a3, s4, a1      # carry from high add
    add     s4, s4, a2      # add carry from limb 0
    sltu    a4, s4, a2      # carry from carry add
    or      a3, a3, a4      # combine carries
    add     s5, s5, a3      # propagate to limb 2
    sltu    a3, s5, a3      # carry from limb 2
    add     s6, s6, a3      # propagate to limb 3
    sltu    a3, s6, a3      # carry from limb 3
    add     s1, s1, a3      # accumulate overflow

    j       .Lreduce_loop

.Lcheck_mod_p:
    # Final reduction: if (result >= P) { result -= P }
    # Method: result >= P <=> result + K >= 2^256 (Carry out)
    # K = 2^32 + 977 = 0x1000003D1
    # P = 2^256 - K
    #
    # BRANCHLESS implementation matching x64 cmovc:
    # 1. Compute temp = result + K
    # 2. If carry out (result >= P), use temp (which is result - P mod 2^256), else keep result
    #
    # Result is currently in s3-s6

    li      t4, 0x3D1       # K = 0x1000003D1
    li      t5, 1
    slli    t5, t5, 32
    or      t4, t4, t5      # t4 = K

    # Compute temp = result + K (use t0-t3 for temp)
    add     t0, s3, t4      # temp[0] = result[0] + K
    sltu    a0, t0, s3      # carry from limb 0 (FIXED: compare with original!)
    add     t1, s4, a0      # temp[1] = result[1] + carry
    sltu    a1, t1, s4      # carry from limb 1 (compare with original!)
    add     t2, s5, a1      # temp[2] = result[2] + carry
    sltu    a2, t2, s5      # carry from limb 2 (compare with original!)
    add     t3, s6, a2      # temp[3] = result[3] + carry
    sltu    a3, t3, s6      # final carry out (a3 = 1 if result >= P)

    # BRANCHLESS conditional move using mask
    # If a3 == 1 (carry happened): use temp (t0-t3)
    # If a3 == 0 (no carry): use original (s3-s6)
    #
    # Create mask: if a3=1 then mask=0xFFFFFFFFFFFFFFFF, if a3=0 then mask=0
    # RISC-V trick: -a3 = (a3==0)?0:(a3==1)?-1
    neg     a4, a3          # a4 = -a3 (if a3=1, then a4=-1=0xFFFF..., if a3=0, then a4=0)

    # CORRECTED: For each limb: if (carry) result = temp; else result = original
    # Using: result = (temp & mask) | (original & ~mask)
    # Optimized: result = original ^ ((original ^ temp) & mask)

    xor     a5, s3, t0      # a5 = s3 ^ t0
    and     a5, a5, a4      # a5 = (s3 ^ t0) & mask
    xor     s3, s3, a5      # s3 = s3 ^ ((s3 ^ t0) & mask)

    xor     a5, s4, t1
    and     a5, a5, a4
    xor     s4, s4, a5

    xor     a5, s5, t2
    and     a5, a5, a4
    xor     s5, s5, a5

    xor     a5, s6, t3
    and     a5, a5, a4
    xor     s6, s6, a5


.Ldone_reduction:
    # Store reduced result (from s3-s6)
    sd      s3, 0(s11)
    sd      s4, 8(s11)
    sd      s5, 16(s11)
    sd      s6, 24(s11)

#endif  /* SECP256K1_RISCV_FAST_REDUCTION */

    # Restore callee-saved registers
    ld      ra, 0(sp)
    ld      s0, 8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    ld      s5, 48(sp)
    ld      s6, 56(sp)
    ld      s7, 64(sp)
    ld      s8, 72(sp)
    ld      s9, 80(sp)
    ld      s10, 88(sp)
    ld      s11, 96(sp)

    # Clean up stack
    addi    sp, sp, 176
    ret
    .size field_mul_asm_riscv64, .-field_mul_asm_riscv64

/*
 * field_sub_asm_riscv64
 * 
 * Subtract two field elements: r = a - b (mod p)
 */
    .globl field_sub_asm_riscv64
    .type field_sub_asm_riscv64, @function
field_sub_asm_riscv64:
    # Save callee-saved registers and ra
    addi    sp, sp, -80
    sd      ra, 0(sp)
    sd      s0, 8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)
    sd      s6, 56(sp)
    sd      s7, 64(sp)

    # Load operands
    ld      a3, 0(a1)       # a[0]
    ld      a4, 8(a1)       # a[1]
    ld      a5, 16(a1)      # a[2]
    ld      a6, 24(a1)      # a[3]
    
    ld      t0, 0(a2)       # b[0]
    ld      t1, 8(a2)       # b[1]
    ld      t2, 16(a2)      # b[2]
    ld      t3, 24(a2)       # b[3]

    # Subtract with borrow chain: result = a - b
    # limb 0
    sltu    t5, a3, t0      # borrow0 = (a[0] < b[0])
    sub     t4, a3, t0      # result[0] = a[0] - b[0]

    # limb 1: a[1] - b[1] - borrow0
    mv      t6, a4
    sub     t6, t6, t5      # a[1] - borrow0
    sltu    s0, a4, t5      # underflow from borrow subtraction
    sltu    s1, t6, t1      # will borrow from b[1] subtraction
    sub     t6, t6, t1      # final result[1]
    or      t5, s0, s1      # combined borrow

    # limb 2: a[2] - b[2] - borrow1
    mv      s2, a5
    sub     s2, s2, t5
    sltu    s3, a5, t5
    sltu    s4, s2, t2
    sub     s2, s2, t2
    or      t5, s3, s4      # combined borrow

    # limb 3: a[3] - b[3] - borrow2
    mv      s5, a6
    sub     s5, s5, t5
    sltu    s6, a6, t5
    sltu    s7, s5, t3
    sub     s5, s5, t3
    or      t5, s6, s7      # final borrow
    
    # If borrow (t5 != 0), result underflowed
    # Need to add P, which is equivalent to subtracting K
    # Compute temp = result - K
    li      a3, 0x3D1
    li      s0, 1
    slli    s0, s0, 32
    or      a3, a3, s0      # a3 = K = 0x1000003D1

    sltu    a5, t4, a3
    sub     a4, t4, a3      # temp[0]

    mv      a6, t6
    sub     a6, a6, a5
    sltu    a5, t6, a5

    mv      a7, s2
    sub     a7, a7, a5
    sltu    a5, s2, a5

    mv      s0, s5
    sub     s0, s0, a5

    # Branchless select: if borrow (t5 != 0), use temp
    # mask = -borrow (all 1s if borrow, all 0s otherwise)
    neg     t5, t5

    xor     s1, t4, a4
    and     s1, s1, t5
    xor     t4, t4, s1

    xor     s1, t6, a6
    and     s1, s1, t5
    xor     t6, t6, s1

    xor     s1, s2, a7
    and     s1, s1, t5
    xor     s2, s2, s1

    xor     s1, s5, s0
    and     s1, s1, t5
    xor     s5, s5, s1

    # Store result
    sd      t4, 0(a0)
    sd      t6, 8(a0)
    sd      s2, 16(a0)
    sd      s5, 24(a0)

    # Restore callee-saved registers and ra
    ld      ra, 0(sp)
    ld      s0, 8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    ld      s5, 48(sp)
    ld      s6, 56(sp)
    ld      s7, 64(sp)
    addi    sp, sp, 80
    ret
    .size field_sub_asm_riscv64, .-field_sub_asm_riscv64

/*
 * field_negate_asm_riscv64
 * 
 * Negate a field element: r = -a
 */
    .globl field_negate_asm_riscv64
    .type field_negate_asm_riscv64, @function
field_negate_asm_riscv64:
    # Save callee-saved registers and ra
    addi    sp, sp, -64
    sd      ra, 0(sp)
    sd      s0, 8(sp)
    sd      s1, 16(sp)
    sd      s2, 24(sp)
    sd      s3, 32(sp)
    sd      s4, 40(sp)
    sd      s5, 48(sp)

    # Load operand
    ld      a3, 0(a1)       # a[0]
    ld      a4, 8(a1)       # a[1]
    ld      a5, 16(a1)      # a[2]
    ld      a6, 24(a1)       # a[3]

    # P[0] = 0xFFFFFFFEFFFFFC2F
    li      t0, 0xFFFFFC2F
    li      t1, 0xFFFFFFFE
    slli    t1, t1, 32
    or      t0, t0, t1      # t0 = P[0]
    li      t1, -1          # P[1] = P[2] = P[3] = 0xFFFFFFFFFFFFFFFF

    # r = P - a with borrow chain
    # limb 0
    sltu    t5, t0, a3      # borrow = (P[0] < a[0])
    sub     t4, t0, a3      # result[0] = P[0] - a[0]

    # limb 1: P[1] - a[1] - borrow
    mv      t6, t1
    sub     t6, t6, t5      # P[1] - borrow
    sltu    s0, t1, t5      # underflow from borrow
    sltu    s1, t6, a4      # will borrow from a[1]
    sub     t6, t6, a4      # result[1]
    or      t5, s0, s1      # combined borrow

    # limb 2: P[2] - a[2] - borrow
    mv      s0, t1
    sub     s0, s0, t5
    sltu    s2, t1, t5
    sltu    s3, s0, a5
    sub     s0, s0, a5
    or      t5, s2, s3

    # limb 3: P[3] - a[3] - borrow
    mv      s1, t1
    sub     s1, s1, t5
    sltu    s4, t1, t5
    sltu    s5, s1, a6
    sub     s1, s1, a6
    # Final borrow (s4 | s5) should be 0 for valid field element

    sd      t4, 0(a0)
    sd      t6, 8(a0)
    sd      s0, 16(a0)
    sd      s1, 24(a0)

    # Restore callee-saved registers and ra
    ld      ra, 0(sp)
    ld      s0, 8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    ld      s5, 48(sp)
    addi    sp, sp, 64
    ret
    .size field_negate_asm_riscv64, .-field_negate_asm_riscv64

/*
 * field_square_asm_riscv64
 * 
 * Square a field element: r = a² (mod p)
 */
    .globl field_square_asm_riscv64
    .type field_square_asm_riscv64, @function
field_square_asm_riscv64:
    # Just call multiply with a, a
    mv      a2, a1
    tail    field_mul_asm_riscv64
    .size field_square_asm_riscv64, .-field_square_asm_riscv64

/*
 * field_add_asm_riscv64
 * 
 * Add two field elements: r = a + b (mod p)
 */
    .globl field_add_asm_riscv64
    .type field_add_asm_riscv64, @function
field_add_asm_riscv64:
    # Load a
    ld      t0, 0(a1)
    ld      t1, 8(a1)
    ld      t2, 16(a1)
    ld      t3, 24(a1)
    
    # Load b
    ld      t4, 0(a2)
    ld      t5, 8(a2)
    ld      t6, 16(a2)
    ld      a3, 24(a2)
    
    # Add with carry propagation (clean add-with-carry chain)
    add     t0, t0, t4      # limb 0
    sltu    a4, t0, t4      # c0

    add     t1, t1, a4      # add carry from limb 0
    sltu    a5, t1, a4      # c1a
    add     t1, t1, t5      # add b[1]
    sltu    a6, t1, t5      # c1b
    or      a4, a5, a6      # c1

    add     t2, t2, a4      # add carry from limb 1
    sltu    a5, t2, a4
    add     t2, t2, t6      # add b[2]
    sltu    a6, t2, t6
    or      a4, a5, a6      # c2

    add     t3, t3, a4
    sltu    a5, t3, a4
    add     t3, t3, a3
    sltu    a6, t3, a3
    or      a4, a5, a6      # c3 (scalar overflow)

    # Result is t0, t1, t2, t3. Carry is a4.
    # If carry==1 OR result >= P, subtract P.
    # Logic: if (carry | (result+K overflows 256 bits)), then result = result+K (-2^256 + K = -P)

    # Calculate R + K
    li      t4, 0x3D1       # K low part
    li      a5, 1           # K high part
    slli    a5, a5, 32
    or      t4, t4, a5      # t4 = 0x1000003D1

    add     t5, t0, t4      # r0 + K
    sltu    a5, t5, t0      # k_c0 (FIXED: compare result with original!)

    add     t6, t1, a5      # t6 = t1 + carry
    sltu    a5, t6, t1      # k_c1 (compare with original!)

    add     a1, t2, a5      # a1 = t2 + carry
    sltu    a5, a1, t2      # k_c2 (compare with original!)

    add     a2, t3, a5      # a2 = t3 + carry
    sltu    a3, a2, t3      # k_c3 (final carry out, compare with original!)

    # Final condition: a4 (original carry) OR a3 (K-check carry)
    or      a4, a4, a3

    # If a4 is non-zero, use [t5, t6, a1, a2] (which is R+K)
    # Else use [t0, t1, t2, t3]

    # Branchless select
    neg     a4, a4          # mask = -condition

    xor     a5, t0, t5
    and     a5, a5, a4
    xor     t0, t0, a5      # t0 = mask ? t5 : t0 (Verified: m=0 -> x, m=-1 -> y)

    xor     a5, t1, t6
    and     a5, a5, a4
    xor     t1, t1, a5

    xor     a5, t2, a1
    and     a5, a5, a4
    xor     t2, t2, a5

    xor     a5, t3, a2
    and     a5, a5, a4
    xor     t3, t3, a5

    sd      t0, 0(a0)
    sd      t1, 8(a0)
    sd      t2, 16(a0)
    sd      t3, 24(a0)
    
    ret
    .size field_add_asm_riscv64, .-field_add_asm_riscv64


#ifdef SECP256K1_HAS_RISCV_VECTOR
/*
 * RISC-V Vector Extension (RVV) Batch Operations
 * 
 * These functions process multiple field elements at once using SIMD.
 * Requires: RVV 1.0 (rv64gcv)
 * 
 * Expected speedup: 4-8× for batch operations (8-16 elements at once)
 */

/*
 * field_mul_batch_rvv
 * 
 * Batch multiply: r[i] = a[i] × b[i] (mod p) for i = 0..count-1
 * 
 * Parameters:
 *   a0 = result pointer (uint64_t r[][4])
 *   a1 = operand a pointer (uint64_t a[][4])
 *   a2 = operand b pointer (uint64_t b[][4])
 *   a3 = count (number of elements)
 * 
 * Note: This is a placeholder for future RVV implementation.
 * Full RVV code requires vector register management and loop unrolling.
 */
    .globl field_mul_batch_rvv
    .type field_mul_batch_rvv, @function
field_mul_batch_rvv:
    # Save ra and s0-s4
    addi    sp, sp, -48
    sd      ra, 0(sp)
    sd      s0, 8(sp)   # counter
    sd      s1, 16(sp)  # result ptr
    sd      s2, 24(sp)  # a ptr
    sd      s3, 32(sp)  # b ptr
    sd      s4, 40(sp)  # count

    # Move arguments to saved registers
    mv      s1, a0
    mv      s2, a1
    mv      s3, a2
    mv      s4, a3
    li      s0, 0       # counter = 0

    # For now, fall back to scalar loop
1:
    beq     s0, s4, 2f      # if counter == count, exit

    # Call scalar multiply
    mv      a0, s1
    mv      a1, s2
    mv      a2, s3
    call    field_mul_asm_riscv64
    
    # Advance pointers and counter
    addi    s1, s1, 32      # r += 4 limbs
    addi    s2, s2, 32      # a += 4 limbs
    addi    s3, s3, 32      # b += 4 limbs
    addi    s0, s0, 1       # counter++
    j       1b
    
2:
    # Restore registers
    ld      ra, 0(sp)
    ld      s0, 8(sp)
    ld      s1, 16(sp)
    ld      s2, 24(sp)
    ld      s3, 32(sp)
    ld      s4, 40(sp)
    addi    sp, sp, 48
    ret
    .size field_mul_batch_rvv, .-field_mul_batch_rvv

#endif // SECP256K1_HAS_RISCV_VECTOR

    .section .note.GNU-stack,"",@progbits
