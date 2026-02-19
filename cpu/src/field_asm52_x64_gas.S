# ═══════════════════════════════════════════════════════════════════════════
# 5×52 Field Arithmetic — x86-64 Assembly (GAS/Intel Syntax)
# ═══════════════════════════════════════════════════════════════════════════
#
# Ultra-optimized field multiplication and squaring for secp256k1 using
# MULX (BMI2) with hand-tuned register allocation.
#
# Produces output in 5×52-bit limb representation with inline modular
# reduction: p = 2^256 - 0x1000003D1.
#
# Functions:
#   fe52_mul_inner_x64(uint64_t *r, const uint64_t *a, const uint64_t *b)
#   fe52_sqr_inner_x64(uint64_t *r, const uint64_t *a)
#
# Calling convention: System V AMD64 ABI
#   Args: rdi=r, rsi=a, rdx=b
#   Callee-saved: rbx, rbp, r12-r15
#
# Required: BMI2 (MULX), Intel Haswell (2013+) / AMD Excavator (2015+)
#
# Reduction constants:
#   R      = 0x1000003D10      (2^260 mod p)
#   R >> 4 = 0x1000003D1       (2^256 mod p)
#   R << 12= 0x1000003D10000
#   M52    = 0xFFFFFFFFFFFFF   (52-bit mask)
# ═══════════════════════════════════════════════════════════════════════════

#if defined(_WIN32) || defined(__WIN32__) || defined(WIN32)
    # Windows: hide ELF directives. C++ uses __attribute__((sysv_abi))
    # so register layout is still System V (rdi, rsi, rdx).
    #define ELF_TYPE(name)
    #define ELF_SIZE(name)
#else
    #define ELF_TYPE(name) .type name, @function
    #define ELF_SIZE(name) .size name, .-name
#endif

.text
.intel_syntax noprefix

# ═══════════════════════════════════════════════════════════════════════════
# fe52_mul_inner_x64: 5×52 field multiplication with secp256k1 reduction
# ═══════════════════════════════════════════════════════════════════════════
#
# Register allocation (persistent):
#   rdi  = output pointer r
#   rcx  = input pointer b (saved from rdx)
#   r8   = a[0], r9 = a[1], r10 = a[2], r11 = a[3], r12 = a[4]
#   r13  = M52 mask (0xFFFFFFFFFFFFF)
#
# Accumulators:
#   d = rbp:rax   (128-bit, columns 3→8→4→5→6→7)
#   c = r15:rsi   (128-bit, columns 0→1→2→3→4)
#
# Scratch: rbx, r14 (MULX hi/lo temporaries)
# Stack:   [rsp+0]=t3, [rsp+8]=t4, [rsp+16]=temp
# ═══════════════════════════════════════════════════════════════════════════

.globl fe52_mul_inner_x64
ELF_TYPE(fe52_mul_inner_x64)
.p2align 4
fe52_mul_inner_x64:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 24                # Stack: [rsp]=t3, [rsp+8]=t4, [rsp+16]=temp

    # Save b pointer (rdx will be used by MULX)
    mov  rcx, rdx

    # Load a[0..4] into r8-r12
    mov  r8,  [rsi]
    mov  r9,  [rsi+8]
    mov  r10, [rsi+16]
    mov  r11, [rsi+24]
    mov  r12, [rsi+32]
    # rsi is now free for scratch

    # M52 constant
    mov  r13, 0xFFFFFFFFFFFFF

    # ─── STEP 1: Column 3 + reduced column 8 ─────────────────────────
    # d = a0*b3 + a1*b2 + a2*b1 + a3*b0
    # c = a4*b4
    # d += R * (uint64)c;  c >>= 64
    # t3 = (uint64)d & M52;  d >>= 52

    # d = a0 * b[3]
    mov  rdx, [rcx+24]
    mulx rbp, rax, r8

    # d += a1 * b[2]
    mov  rdx, [rcx+16]
    mulx rbx, r14, r9
    add  rax, r14
    adc  rbp, rbx

    # d += a2 * b[1]
    mov  rdx, [rcx+8]
    mulx rbx, r14, r10
    add  rax, r14
    adc  rbp, rbx

    # d += a3 * b[0]
    mov  rdx, [rcx]
    mulx rbx, r14, r11
    add  rax, r14
    adc  rbp, rbx               # d = (rbp:rax), ≤114 bits

    # c_full = a4 * b[4]
    mov  rdx, [rcx+32]
    mulx r15, rsi, r12          # c = (r15:rsi), ≤112 bits

    # d += R52 * c_lo
    mov  rdx, 0x1000003D10
    mulx rbx, r14, rsi
    add  rax, r14
    adc  rbp, rbx               # d ≤ 115 bits

    # c >> 64: c_remaining = r15 (≤48 bits)

    # t3 = d_lo & M52
    mov  rsi, rax
    and  rsi, r13
    mov  [rsp], rsi             # [rsp] = t3

    # d >>= 52
    shrd rax, rbp, 52
    shr  rbp, 52                # d ≤ 63 bits

    # ─── STEP 2: Column 4 + column 8 carry ───────────────────────────
    # d += a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0
    # d += (R<<12) * c_remaining
    # t4 = d & M52;  d >>= 52;  tx = t4>>48;  t4 &= M48

    # d += a0 * b[4]
    mov  rdx, [rcx+32]
    mulx rbx, r14, r8
    add  rax, r14
    adc  rbp, rbx

    # d += a1 * b[3]
    mov  rdx, [rcx+24]
    mulx rbx, r14, r9
    add  rax, r14
    adc  rbp, rbx

    # d += a2 * b[2]
    mov  rdx, [rcx+16]
    mulx rbx, r14, r10
    add  rax, r14
    adc  rbp, rbx

    # d += a3 * b[1]
    mov  rdx, [rcx+8]
    mulx rbx, r14, r11
    add  rax, r14
    adc  rbp, rbx

    # d += a4 * b[0]
    mov  rdx, [rcx]
    mulx rbx, r14, r12
    add  rax, r14
    adc  rbp, rbx               # d ≤ 115 bits

    # d += (R<<12) * c_remaining = 0x1000003D10000 * r15
    mov  rdx, 0x1000003D10000
    mulx rbx, r14, r15
    add  rax, r14
    adc  rbp, rbx               # d ≤ 116 bits

    # t4 = d_lo & M52
    mov  r14, rax
    and  r14, r13

    # d >>= 52
    shrd rax, rbp, 52
    shr  rbp, 52                # d ≤ 64 bits

    # tx = t4 >> 48
    mov  r15, r14
    shr  r15, 48                # r15 = tx (4 bits)

    # t4 &= M48 = M52 >> 4
    mov  rsi, r13
    shr  rsi, 4
    and  r14, rsi               # r14 = t4 (48 bits)
    mov  [rsp+8], r14           # [rsp+8] = t4

    # ─── STEP 3: Column 0 + reduced column 5 ─────────────────────────
    # d += a1*b4 + a2*b3 + a3*b2 + a4*b1
    # u0 = d & M52;  d >>= 52;  u0 = (u0<<4)|tx
    # c = a0*b0;  c += u0 * (R>>4)
    # r[0] = c & M52;  c >>= 52

    # d += a1 * b[4]
    mov  rdx, [rcx+32]
    mulx rbx, r14, r9
    add  rax, r14
    adc  rbp, rbx

    # d += a2 * b[3]
    mov  rdx, [rcx+24]
    mulx rbx, r14, r10
    add  rax, r14
    adc  rbp, rbx

    # d += a3 * b[2]
    mov  rdx, [rcx+16]
    mulx rbx, r14, r11
    add  rax, r14
    adc  rbp, rbx

    # d += a4 * b[1]
    mov  rdx, [rcx+8]
    mulx rbx, r14, r12
    add  rax, r14
    adc  rbp, rbx               # d ≤ 114 bits

    # u0 = d_lo & M52
    mov  r14, rax
    and  r14, r13

    # d >>= 52
    shrd rax, rbp, 52
    shr  rbp, 52                # d ≤ 62 bits

    # u0 = (u0 << 4) | tx
    shl  r14, 4
    or   r14, r15               # r14 = u0 (56 bits)
    # r15 is now free

    # c = a0 * b[0]
    mov  rdx, [rcx]
    mulx r15, rsi, r8           # c = (r15:rsi)

    # c += u0 * (R >> 4) = u0 * 0x1000003D1
    mov  [rsp+16], r14          # temp = u0
    mov  rdx, 0x1000003D1
    mulx rbx, r14, [rsp+16]    # (rbx:r14) = (R>>4) * u0
    add  rsi, r14
    adc  r15, rbx               # c += u0*(R>>4)

    # r[0] = c_lo & M52
    mov  r14, rsi
    and  r14, r13
    mov  [rdi], r14

    # c >>= 52
    shrd rsi, r15, 52
    shr  r15, 52                # c = (r15:rsi) ≤ 61 bits

    # ─── STEP 4: Column 1 + reduced column 6 ─────────────────────────
    # c += a0*b1 + a1*b0
    # d += a2*b4 + a3*b3 + a4*b2
    # c += (d & M52) * R;  d >>= 52
    # r[1] = c & M52;  c >>= 52

    # c += a0 * b[1]
    mov  rdx, [rcx+8]
    mulx rbx, r14, r8
    add  rsi, r14
    adc  r15, rbx

    # c += a1 * b[0]
    mov  rdx, [rcx]
    mulx rbx, r14, r9
    add  rsi, r14
    adc  r15, rbx               # c ≤ 114 bits

    # d += a2 * b[4]
    mov  rdx, [rcx+32]
    mulx rbx, r14, r10
    add  rax, r14
    adc  rbp, rbx

    # d += a3 * b[3]
    mov  rdx, [rcx+24]
    mulx rbx, r14, r11
    add  rax, r14
    adc  rbp, rbx

    # d += a4 * b[2]
    mov  rdx, [rcx+16]
    mulx rbx, r14, r12
    add  rax, r14
    adc  rbp, rbx               # d ≤ 114 bits

    # c += (d_lo & M52) * R52
    mov  r14, rax
    and  r14, r13               # r14 = d_lo & M52
    mov  [rsp+16], r14          # temp
    mov  rdx, 0x1000003D10
    mulx rbx, r14, [rsp+16]    # (rbx:r14) = R * (d & M52)
    add  rsi, r14
    adc  r15, rbx               # c ≤ 115 bits

    # d >>= 52
    shrd rax, rbp, 52
    shr  rbp, 52                # d ≤ 62 bits

    # r[1] = c_lo & M52
    mov  r14, rsi
    and  r14, r13
    mov  [rdi+8], r14

    # c >>= 52
    shrd rsi, r15, 52
    shr  r15, 52

    # ─── STEP 5: Column 2 + reduced column 7 ─────────────────────────
    # c += a0*b2 + a1*b1 + a2*b0
    # d += a3*b4 + a4*b3
    # c += R * (uint64)d;  d >>= 64
    # r[2] = c & M52;  c >>= 52

    # c += a0 * b[2]
    mov  rdx, [rcx+16]
    mulx rbx, r14, r8
    add  rsi, r14
    adc  r15, rbx

    # c += a1 * b[1]
    mov  rdx, [rcx+8]
    mulx rbx, r14, r9
    add  rsi, r14
    adc  r15, rbx

    # c += a2 * b[0]
    mov  rdx, [rcx]
    mulx rbx, r14, r10
    add  rsi, r14
    adc  r15, rbx               # c ≤ 114 bits

    # d += a3 * b[4]
    mov  rdx, [rcx+32]
    mulx rbx, r14, r11
    add  rax, r14
    adc  rbp, rbx

    # d += a4 * b[3]
    mov  rdx, [rcx+24]
    mulx rbx, r14, r12
    add  rax, r14
    adc  rbp, rbx               # d ≤ 114 bits

    # c += R * (uint64)d   — uses FULL 64-bit d_lo, not masked to M52!
    mov  [rsp+16], rax          # temp = d_lo
    mov  rdx, 0x1000003D10
    mulx rbx, r14, [rsp+16]    # (rbx:r14) = R * d_lo
    add  rsi, r14
    adc  r15, rbx

    # d >>= 64
    mov  rax, rbp
    xor  rbp, rbp               # d = (0:rax)

    # r[2] = c_lo & M52
    mov  r14, rsi
    and  r14, r13
    mov  [rdi+16], r14

    # c >>= 52
    shrd rsi, r15, 52
    shr  r15, 52

    # ─── STEP 6: Finalize columns 3 and 4 ────────────────────────────
    # c += (R<<12) * (uint64)d
    # c += t3;  r[3] = c & M52;  c >>= 52
    # c += t4;  r[4] = c

    # c += (R<<12) * d
    mov  [rsp+16], rax          # temp = d (remaining high bits)
    mov  rdx, 0x1000003D10000
    mulx rbx, r14, [rsp+16]    # (rbx:r14) = (R<<12) * d
    add  rsi, r14
    adc  r15, rbx

    # c += t3
    add  rsi, [rsp]
    adc  r15, 0

    # r[3] = c_lo & M52
    mov  r14, rsi
    and  r14, r13
    mov  [rdi+24], r14

    # c >>= 52
    shrd rsi, r15, 52
    shr  r15, 52

    # c += t4
    add  rsi, [rsp+8]

    # r[4] = c (fits in 64 bits, magnitude ≤ 1)
    mov  [rdi+32], rsi

    # Epilogue
    add  rsp, 24
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret
ELF_SIZE(fe52_mul_inner_x64)


# ═══════════════════════════════════════════════════════════════════════════
# fe52_sqr_inner_x64: 5×52 field squaring with secp256k1 reduction
# ═══════════════════════════════════════════════════════════════════════════
#
# Uses a[i]*a[j] == a[j]*a[i] symmetry: compute once, double via (a_i*2).
# Cross-products use LEA for the *2 (no flags, pairs with MULX).
#
# Same register allocation as mul, but no b pointer needed:
#   rdi  = output r
#   r8-r12 = a[0..4]
#   r13  = M52
#   d = rbp:rax,  c = r15:rsi,  scratch = rbx, r14, rcx
# ═══════════════════════════════════════════════════════════════════════════

.globl fe52_sqr_inner_x64
ELF_TYPE(fe52_sqr_inner_x64)
.p2align 4
fe52_sqr_inner_x64:
    push rbp
    push rbx
    push r12
    push r13
    push r14
    push r15
    sub  rsp, 24

    # Load a[0..4] into r8-r12
    mov  r8,  [rsi]
    mov  r9,  [rsi+8]
    mov  r10, [rsi+16]
    mov  r11, [rsi+24]
    mov  r12, [rsi+32]
    # rsi free, rcx free (no b pointer for squaring)

    mov  r13, 0xFFFFFFFFFFFFF   # M52

    # ─── STEP 1: d = (a0*2)*a3 + (a1*2)*a2;  c = a4²;
    #             d += R*c_lo;  c >>= 64
    #             t3 = d & M52;  d >>= 52

    # d = (a0*2) * a3
    lea  rdx, [r8+r8]          # rdx = a0*2
    mulx rbp, rax, r11         # d = (rbp:rax) = (a0*2)*a3

    # d += (a1*2) * a2
    lea  rdx, [r9+r9]          # rdx = a1*2
    mulx rbx, r14, r10         # (rbx:r14) = (a1*2)*a2
    add  rax, r14
    adc  rbp, rbx

    # c = a4 * a4
    mov  rdx, r12
    mulx r15, rsi, r12         # c = (r15:rsi) = a4²

    # d += R * c_lo
    mov  rdx, 0x1000003D10
    mulx rbx, r14, rsi
    add  rax, r14
    adc  rbp, rbx

    # c >> 64: r15 = c_remaining

    # t3 = d & M52
    mov  rsi, rax
    and  rsi, r13
    mov  [rsp], rsi

    # d >>= 52
    shrd rax, rbp, 52
    shr  rbp, 52

    # ─── STEP 2: d += (a0*2)*a4 + (a1*2)*a3 + a2²
    #             d += (R<<12) * c_remaining
    #             t4 = d & M52;  d >>= 52;  tx = t4>>48;  t4 &= M48

    # d += (a0*2) * a4
    lea  rdx, [r8+r8]
    mulx rbx, r14, r12
    add  rax, r14
    adc  rbp, rbx

    # d += (a1*2) * a3
    lea  rdx, [r9+r9]
    mulx rbx, r14, r11
    add  rax, r14
    adc  rbp, rbx

    # d += a2 * a2
    mov  rdx, r10
    mulx rbx, r14, r10
    add  rax, r14
    adc  rbp, rbx

    # d += (R<<12) * c_remaining
    mov  rdx, 0x1000003D10000
    mulx rbx, r14, r15
    add  rax, r14
    adc  rbp, rbx

    # t4 = d & M52
    mov  r14, rax
    and  r14, r13

    # d >>= 52
    shrd rax, rbp, 52
    shr  rbp, 52

    # tx = t4 >> 48;  t4 &= M48
    mov  r15, r14
    shr  r15, 48
    mov  rsi, r13
    shr  rsi, 4
    and  r14, rsi
    mov  [rsp+8], r14

    # ─── STEP 3: d += (a1*2)*a4 + (a2*2)*a3
    #             u0 = d & M52;  d >>= 52;  u0 = (u0<<4)|tx
    #             c = a0²;  c += u0 * (R>>4)
    #             r[0] = c & M52;  c >>= 52

    # d += (a1*2) * a4
    lea  rdx, [r9+r9]
    mulx rbx, r14, r12
    add  rax, r14
    adc  rbp, rbx

    # d += (a2*2) * a3
    lea  rdx, [r10+r10]
    mulx rbx, r14, r11
    add  rax, r14
    adc  rbp, rbx

    # u0 = d & M52
    mov  r14, rax
    and  r14, r13

    # d >>= 52
    shrd rax, rbp, 52
    shr  rbp, 52

    # u0 = (u0 << 4) | tx
    shl  r14, 4
    or   r14, r15

    # c = a0 * a0
    mov  rdx, r8
    mulx r15, rsi, r8           # c = (r15:rsi) = a0²

    # c += u0 * (R >> 4)
    mov  [rsp+16], r14
    mov  rdx, 0x1000003D1
    mulx rbx, r14, [rsp+16]
    add  rsi, r14
    adc  r15, rbx

    # r[0] = c & M52
    mov  r14, rsi
    and  r14, r13
    mov  [rdi], r14

    # c >>= 52
    shrd rsi, r15, 52
    shr  r15, 52

    # ─── STEP 4: c += (a0*2)*a1
    #             d += (a2*2)*a4 + a3²
    #             c += (d & M52) * R;  d >>= 52
    #             r[1] = c & M52;  c >>= 52

    # c += (a0*2) * a1
    lea  rdx, [r8+r8]
    mulx rbx, r14, r9
    add  rsi, r14
    adc  r15, rbx

    # d += (a2*2) * a4
    lea  rdx, [r10+r10]
    mulx rbx, r14, r12
    add  rax, r14
    adc  rbp, rbx

    # d += a3 * a3
    mov  rdx, r11
    mulx rbx, r14, r11
    add  rax, r14
    adc  rbp, rbx

    # c += (d & M52) * R
    mov  r14, rax
    and  r14, r13
    mov  [rsp+16], r14
    mov  rdx, 0x1000003D10
    mulx rbx, r14, [rsp+16]
    add  rsi, r14
    adc  r15, rbx

    # d >>= 52
    shrd rax, rbp, 52
    shr  rbp, 52

    # r[1] = c & M52
    mov  r14, rsi
    and  r14, r13
    mov  [rdi+8], r14

    # c >>= 52
    shrd rsi, r15, 52
    shr  r15, 52

    # ─── STEP 5: c += (a0*2)*a2 + a1²
    #             d += (a3*2)*a4
    #             c += R * (uint64)d;  d >>= 64
    #             r[2] = c & M52;  c >>= 52

    # c += (a0*2) * a2
    lea  rdx, [r8+r8]
    mulx rbx, r14, r10
    add  rsi, r14
    adc  r15, rbx

    # c += a1 * a1
    mov  rdx, r9
    mulx rbx, r14, r9
    add  rsi, r14
    adc  r15, rbx

    # d += (a3*2) * a4
    lea  rdx, [r11+r11]
    mulx rbx, r14, r12
    add  rax, r14
    adc  rbp, rbx

    # c += R * (uint64)d
    mov  [rsp+16], rax
    mov  rdx, 0x1000003D10
    mulx rbx, r14, [rsp+16]
    add  rsi, r14
    adc  r15, rbx

    # d >>= 64
    mov  rax, rbp
    xor  rbp, rbp

    # r[2] = c & M52
    mov  r14, rsi
    and  r14, r13
    mov  [rdi+16], r14

    # c >>= 52
    shrd rsi, r15, 52
    shr  r15, 52

    # ─── STEP 6: Finalize ────────────────────────────────────────────
    # c += (R<<12) * d;  c += t3;  r[3] = c & M52;  c >>= 52
    # c += t4;  r[4] = c

    mov  [rsp+16], rax
    mov  rdx, 0x1000003D10000
    mulx rbx, r14, [rsp+16]
    add  rsi, r14
    adc  r15, rbx

    add  rsi, [rsp]
    adc  r15, 0

    mov  r14, rsi
    and  r14, r13
    mov  [rdi+24], r14

    shrd rsi, r15, 52
    shr  r15, 52

    add  rsi, [rsp+8]
    mov  [rdi+32], rsi

    add  rsp, 24
    pop  r15
    pop  r14
    pop  r13
    pop  r12
    pop  rbx
    pop  rbp
    ret
ELF_SIZE(fe52_sqr_inner_x64)
