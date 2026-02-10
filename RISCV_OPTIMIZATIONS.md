# RISC-V Optimizations Report

**Date:** 2026-02-10  
**Platform:** RISC-V 64-bit (Milk-V Mars / StarFive JH7110)  
**Compiler:** Clang 21.1.8 + LTO  

---

## Summary of Optimizations

### 1. Field Square Optimization (10 mul vs 16)

**File:** `cpu/src/field_asm_riscv64.S`

**Problem:** Original `field_square` used tail call to `field_mul`, performing 16 multiplications.

**Solution:** Dedicated squaring using symmetry property:
```
a² = sum(ai²) + 2*sum(ai*aj for i<j)
```

**Implementation Details:**
- Diagonal terms (4 multiplications): `a0², a1², a2², a3²`
- Off-diagonal terms (6 multiplications): `a0*a1, a0*a2, a0*a3, a1*a2, a1*a3, a2*a3`
- Doubling: Add each off-diagonal term twice (avoids complex shift carry propagation)
- Total: 10 multiplications + 12 additions

**Key Insight:** Instead of shifting 128-bit values (which has complex carry), we add the same value twice. This is mathematically equivalent but avoids carry bugs.

**Result:**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Field Square | 186 ns | 177 ns | **5%** |

---

### 2. wNAF Window Width Increase (w=4 → w=5)

**File:** `cpu/src/point.cpp`

**Problem:** wNAF with w=4 uses 8 precomputed points, requiring more doublings.

**Solution:** Increase window width to 5:
- 16 precomputed points instead of 8
- Fewer non-zero digits in wNAF representation
- Better trade-off between precomputation and main loop iterations

**Code Change:**
```cpp
// Before
constexpr unsigned window_width = 4;
constexpr int table_size = 8;  // [1P, 3P, 5P, ..., 15P]

// After
constexpr unsigned window_width = 5;
constexpr int table_size = 16; // [1P, 3P, 5P, ..., 31P]
```

**Result:**
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Scalar Mul | 678 μs | 670 μs | **1.2%** |

---

### 3. LTO (Link-Time Optimization) Integration

**Files:** `cpu/CMakeLists.txt`, `CMakePresets.json`

**Problem:** CMake's `CheckIPOSupported` fails with RISC-V because it doesn't use our `-fuse-ld=lld` flag.

**Solution:** Direct `-flto=thin` flags for Clang without using CheckIPOSupported.

**Result:**
| Metric | Without LTO | With LTO | Improvement |
|--------|-------------|----------|-------------|
| Field Mul | 206 ns | 198 ns | **4%** |
| Field Add | 40 ns | 34 ns | **15%** |
| Field Sub | 37 ns | 31 ns | **16%** |

---

## Final Performance Summary (RISC-V)

| Operation | Time | Notes |
|-----------|------|-------|
| Field Mul | 198 ns | Assembly + LTO |
| Field Square | 177 ns | **Optimized (10 mul)** |
| Field Add | 34 ns | Inline C++ (faster than ASM wrapper) |
| Field Sub | 31 ns | Inline C++ (faster than ASM wrapper) |
| Field Inverse | 18 μs | Binary GCD |
| Point Add | 3 μs | Jacobian |
| Point Double | 1 μs | Jacobian |
| Point Scalar Mul | 672 μs | **wNAF w=5** |
| Generator Mul | 40 μs | Precomputed |
| Batch Inverse (n=100) | 765 ns | Montgomery trick |
| Batch Inverse (n=1000) | 615 ns | Montgomery trick |

### Key Learnings

1. **Assembly wrapper overhead matters**: For simple operations like add/sub (~30ns), the cost of converting between `limbs4` and `FieldElement` via wrappers exceeds the operation itself. Inline C++ is faster.

2. **Assembly wins for complex operations**: For mul (~200ns) and square (~180ns), the assembly overhead is negligible compared to the computation, so assembly optimization pays off.

3. **Window width trade-off**: wNAF w=5 provides ~1% improvement over w=4 for scalar multiplication, with acceptable precomputation cost (16 vs 8 points).

---

## Portability to Other Platforms

### x86-64 Applicability

| Optimization | Portable? | Notes |
|--------------|-----------|-------|
| Field Square (10 mul) | ✅ YES | Algorithm is platform-independent |
| wNAF w=5 | ✅ YES | Already in portable C++ code |
| LTO | ✅ YES | Standard compiler feature |

**x86-64 Implementation:**
- Use `__int128` for 128-bit arithmetic
- Or use intrinsics: `_mulx_u64`, `_addcarry_u64`
- AVX-512 IFMA can do 8 parallel 52-bit muls

### CUDA Applicability

| Optimization | Portable? | Notes |
|--------------|-----------|-------|
| Field Square (10 mul) | ✅ YES | Same algorithm in PTX/CUDA |
| wNAF w=5 | ⚠️ PARTIAL | May need tuning for GPU occupancy |
| LTO | ✅ YES | CUDA supports LTO since 11.0 |

**CUDA Implementation:**
- Use `__umul64hi()` for high 64 bits
- Use PTX inline assembly for optimal performance
- Consider using `__uint128_t` with newer CUDA

---

## Files Changed

1. **cpu/src/field_asm_riscv64.S**
   - Added optimized `field_square_asm_riscv64` function
   - ~400 lines of new assembly code

2. **cpu/src/point.cpp**
   - Changed `window_width` from 4 to 5
   - Updated comments for 31P instead of 15P

3. **cpu/CMakeLists.txt**
   - Added direct LTO flags for Clang
   - Removed dependency on CheckIPOSupported

---

## Next Steps for Cross-Platform

### Priority 1: x86-64 Field Square
```cpp
// Portable C++ implementation using __int128
inline void field_square_opt(uint64_t* r, const uint64_t* a) {
    // Diagonal terms
    __uint128_t d0 = (__uint128_t)a[0] * a[0];
    __uint128_t d1 = (__uint128_t)a[1] * a[1];
    __uint128_t d2 = (__uint128_t)a[2] * a[2];
    __uint128_t d3 = (__uint128_t)a[3] * a[3];
    
    // Off-diagonal terms (computed once, added twice)
    __uint128_t c01 = (__uint128_t)a[0] * a[1];
    __uint128_t c02 = (__uint128_t)a[0] * a[2];
    __uint128_t c03 = (__uint128_t)a[0] * a[3];
    __uint128_t c12 = (__uint128_t)a[1] * a[2];
    __uint128_t c13 = (__uint128_t)a[1] * a[3];
    __uint128_t c23 = (__uint128_t)a[2] * a[3];
    
    // Build columns and reduce...
}
```

### Priority 2: CUDA Field Square
```cuda
__device__ void field_square_opt(uint64_t* r, const uint64_t* a) {
    // Same algorithm, use __umul64hi() for high bits
    uint64_t d0_lo = a[0] * a[0];
    uint64_t d0_hi = __umul64hi(a[0], a[0]);
    // ...
}
```

---

## Conclusion

The RISC-V optimizations provide:
- **5% improvement** in field squaring
- **1.2% improvement** in scalar multiplication  
- **4-16% improvement** from LTO

All optimizations are algorithmically portable and can be implemented on x86-64 and CUDA with similar or better results due to those platforms' more advanced hardware capabilities.

